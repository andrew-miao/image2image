import argparse
import logging
import math
import os
from pathlib import Path
import json
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub.utils import insecure_hashlib
from jax.experimental.compilation_cache import compilation_cache as cc
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxUNet2DConditionModel,
)

import ddpo
from ddpo.diffusers_patch.scheduling_ddim_flax import FlaxDDIMScheduler
from ddpo.diffusers_patch.pipeline_flax_stable_diffusion import FlaxStableDiffusionPipeline

from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.utils import check_min_version
from ddpo.datasets import DreamBoothDataset, PromptDataset


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

# Cache compiled models across invocations of this script.
cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

logger = logging.getLogger(__name__)

class Parser(ddpo.utils.Parser):
    config: str = "config.base"
    
def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def main():
    args = Parser().parse_args("dream_ppo")
    transformers.set_seed(args.seed)
    compilation_cache.initialize_cache(args.cache)  # only works on TPU

    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    rng = jax.random.PRNGKey(args.seed)
    n_devices = jax.local_device_count()
    train_worker_batch_size = n_devices * args.train_batch_size
    train_pod_batch_size = train_worker_batch_size * jax.process_count()
    train_effective_batch_size = train_pod_batch_size * args.train_accumulation_steps
    sample_worker_batch_size = n_devices * args.sample_batch_size
    sample_pod_batch_size = sample_worker_batch_size * jax.process_count()
    total_samples_per_epoch = args.num_sample_batches_per_epoch * sample_pod_batch_size

    print(
        f"[ DreamBooth PPO ] local devices: {n_devices} | "
        f"number of workers: {jax.process_count()}"
    )
    print(
        f"[ DreamBooth PPO ] sample worker batch size: {sample_worker_batch_size} | "
        f"sample pod batch size: {sample_pod_batch_size}"
    )
    print(
        f"[ DreamBooth PPO ] train worker batch size: {train_worker_batch_size} | "
        f"train pod batch size: {train_pod_batch_size} | "
        f"train accumulated batch size: {train_effective_batch_size}"
    )
    print(
        f"[ DreamBooth PPO ] number of sample batches per epoch: {args.num_sample_batches_per_epoch}"
    )
    print(
        f"[ DreamBooth PPO ] total number of samples per epoch: {total_samples_per_epoch}"
    )
    print(
        f"[ DreamBooth PPO ] number of gradient updates per inner epoch: {total_samples_per_epoch // train_effective_batch_size}"
    )

    # ----------------- Setup logging and save the args ----------------- #
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)

    worker_id = jax.process_index()

    localpath = "logs/" + args.savepath.replace("gs://", "")
    os.umask(0)
    os.makedirs(localpath, exist_ok=True, mode=0o777)
    with open(f"{localpath}/args.json", "w") as f:
        json.dump(args._dict, f, indent=4)

    logging.info(f"Worker {worker_id} is logging to {localpath}")

    # Handle the repository creation
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer and add the placeholder token as a additional special token
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer"
    )

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        batch = {k: v.numpy() for k, v in batch.items()}
        return batch

    total_train_batch_size = args.train_batch_size * jax.local_device_count()
    if len(train_dataset) < total_train_batch_size:
        raise ValueError(
            f"Training batch size is {total_train_batch_size}, but your dataset only contains"
            f" {len(train_dataset)} images. Please, use a larger dataset or reduce the effective batch size. Note that"
            f" there are {jax.local_device_count()} parallel devices, so your batch size can't be smaller than that."
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=total_train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    # ----------------- Load the models ----------------- #
    print("Loading models...")
    pipeline, params = ddpo.utils.load_unet(
        None,
        epoch=args.load_epoch,
        pretrained_model=args.pretrained_model,
        dtype=args.dtype,
        cache=args.cache,
    )

    pipeline.safety_checker = None
    params = jax.device_get(params)
    
    pipeline.scheduler = FlaxDDIMScheduler(
        num_train_timesteps=pipeline.scheduler.config.num_train_timesteps,
        beta_start=pipeline.scheduler.config.beta_start,
        beta_end=pipeline.scheduler.config.beta_end,
        beta_schedule=pipeline.scheduler.config.beta_schedule,
        trained_betas=pipeline.scheduler.config.trained_betas,
        set_alpha_to_one=pipeline.scheduler.config.set_alpha_to_one,
        steps_offset=pipeline.scheduler.config.steps_offset,
        prediction_type=pipeline.scheduler.config.prediction_type,
    )

    noise_scheduler_state = pipeline.scheduler.set_timesteps(
        params["scheduler"],
        num_inference_steps=args.n_inference_steps,
        shape=(
            args.train_batch_size,
            pipeline.unet.in_channels,
            args.resolution // pipeline.vae_scale_factor,
            args.resolution // pipeline.vae_scale_factor,
        )
    )

    # ----------------- Optimization ----------------- #
    print("Initializing optimizers and train state...")
    constant_scheduler = optax.constant_schedule(args.learning_rate)

    optim = {
        "adamw": optax.adamw(
            learning_rate=constant_scheduler,
            b1=args.beta1,
            b2=args.beta2,
            eps=args.epsilon,
            weight_decay=args.weight_decay,
            mu_dtype=jnp.bfloat16,
        ),
        "adafactor": optax.adafactor(
            learning_rate=constant_scheduler,
            weight_decay_rate=args.weight_decay,
        ),
    }[args.optimizer]

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optim,
    )

    opt_state = jax.jit(optimizer.init, backend="cpu")(params["unet"])
    # NOTE(kvablack) optax.MultiSteps takes way more memory than necessary; this
    # class is a workaround that requires compiling twice. there may be a better
    # way but this works.
    train_state = ddpo.training.policy_gradient.AccumulatingTrainState(
        step=0,
        apply_fn=pipeline.unet.apply,
        params=params["unet"],
        tx=optimizer,
        opt_state=opt_state,
        grad_acc=jax.tree_map(np.zeros_like, params["unet"]),
        n_acc=0,
    )
    # ----------------- Replication ----------------- #
    train_state = jax_utils.replicate(train_state)
    sampling_scheduler_params = jax_utils.replicate(params["scheduler"])
    noise_scheduler_state = jax_utils.replicate(noise_scheduler_state)

    # -------------------------- setup ------------------------#
    timer = ddpo.utils.Timer()

    @partial(jax.pmap)
    def vae_decode(latents, vae_params):
        # expects latents in NCHW format (batch_size, 4, 64, 64)
        latents = latents / 0.18215
        images = pipeline.vae.apply(
            {"params": vae_params}, latents, method=pipeline.vae.decode
        ).sample
        images = (images / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return images

    # text encode on CPU to save memory
    @partial(jax.jit, backend="cpu")
    def text_encode(input_ids):
        return pipeline.text_encoder(input_ids, params=params["text_encoder"])[0]

    # make uncond prompts and embed them
    uncond_prompt_ids = ddpo.datasets.make_uncond_text(pipeline.tokenizer, 1)
    timer()
    uncond_prompt_embeds = text_encode(uncond_prompt_ids).squeeze()
    print(f"[ embed uncond prompts ] in {timer():.2f}s")

    sample_uncond_prompt_embeds = np.broadcast_to(
        uncond_prompt_embeds, (args.sample_batch_size, *uncond_prompt_embeds.shape)
    )
    sample_uncond_prompt_embeds = jax_utils.replicate(sample_uncond_prompt_embeds)
    train_uncond_prompt_embeds = sample_uncond_prompt_embeds[:, : args.train_batch_size]

    train_rng, sample_rng = jax.random.split(rng)

    #TODO: finish trainining loop and callbacks.

    def train_step(unet_state, text_encoder_state, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        if args.train_text_encoder:
            params = {"text_encoder": text_encoder_state.params, "unet": unet_state.params}
        else:
            params = {"unet": unet_state.params}

        def compute_loss(params):
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params}, batch["pixel_values"], deterministic=True, method=vae.encode
            )
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

            # Get the text embedding for conditioning
            if args.train_text_encoder:
                encoder_hidden_states = text_encoder_state.apply_fn(
                    batch["input_ids"], params=params["text_encoder"], dropout_rng=dropout_rng, train=True
                )[0]
            else:
                encoder_hidden_states = text_encoder(
                    batch["input_ids"], params=text_encoder_state.params, train=False
                )[0]

            # Predict the noise residual
            model_pred = unet.apply(
                {"params": params["unet"]}, noisy_latents, timesteps, encoder_hidden_states, train=True
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if args.with_prior_preservation:
                # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = jnp.split(model_pred, 2, axis=0)
                target, target_prior = jnp.split(target, 2, axis=0)

                # Compute instance loss
                loss = (target - model_pred) ** 2
                loss = loss.mean()

                # Compute prior loss
                prior_loss = (target_prior - model_pred_prior) ** 2
                prior_loss = prior_loss.mean()

                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss
            else:
                loss = (target - model_pred) ** 2
                loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(params)
        grad = jax.lax.pmean(grad, "batch")

        new_unet_state = unet_state.apply_gradients(grads=grad["unet"])
        if args.train_text_encoder:
            new_text_encoder_state = text_encoder_state.apply_gradients(grads=grad["text_encoder"])
        else:
            new_text_encoder_state = text_encoder_state

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_unet_state, new_text_encoder_state, metrics, new_train_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0, 1))

    # Replicate the train state on each device
    unet_state = jax_utils.replicate(unet_state)
    text_encoder_state = jax_utils.replicate(text_encoder_state)
    vae_params = jax_utils.replicate(vae_params)

    # Train!
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    # Scheduler and math around the number of training steps.
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def checkpoint(step=None):
        # Create the pipeline using the trained modules and save it.
        scheduler, _ = FlaxPNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker", from_pt=True
        )
        pipeline = FlaxStableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        )

        outdir = os.path.join(args.output_dir, str(step)) if step else args.output_dir
        pipeline.save_pretrained(
            outdir,
            params={
                "text_encoder": get_params_to_save(text_encoder_state.params),
                "vae": get_params_to_save(vae_params),
                "unet": get_params_to_save(unet_state.params),
                "safety_checker": safety_checker.params,
            },
        )

        

    global_step = 0

    epochs = tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================

        train_metrics = []

        steps_per_epoch = len(train_dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for batch in train_dataloader:
            batch = shard(batch)
            unet_state, text_encoder_state, train_metric, train_rngs = p_train_step(
                unet_state, text_encoder_state, vae_params, batch, train_rngs
            )
            train_metrics.append(train_metric)

            train_step_progress_bar.update(jax.local_device_count())

            global_step += 1
            if jax.process_index() == 0 and args.save_steps and global_step % args.save_steps == 0:
                checkpoint(global_step)
            if global_step >= args.max_train_steps:
                break

        train_metric = jax_utils.unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")

    if jax.process_index() == 0:
        checkpoint()


if __name__ == "__main__":
    main()
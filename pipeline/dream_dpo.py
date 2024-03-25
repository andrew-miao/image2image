import argparse
import logging
import math
import os
from pathlib import Path
import json
from functools import partial
from concurrent import futures
import time

import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache
from jax.experimental import multihost_utils
from jax.experimental.compilation_cache import compilation_cache as cc
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from flax.training.checkpoints import save_checkpoint_multiprocess

from PIL import Image
import tqdm
from transformers import CLIPTokenizer
from matplotlib import pyplot as plt

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxUNet2DConditionModel,
)

import ddpo
from ddpo.utils.stat_tracking import PerPromptStatTracker
from ddpo.diffusers_patch.scheduling_ddim_flax import FlaxDDIMScheduler
from ddpo.diffusers_patch.pipeline_flax_stable_diffusion import FlaxStableDiffusionPipeline
from ddpo.training.images import DreamBoothDataset
from ddpo.training import AccumulatingTrainState


# Cache compiled models across invocations of this script.
cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

logger = logging.getLogger(__name__)

def train_step(
    unet_state,
    text_encoder_state,
    vae,
    vae_params,
    batch,
    train_rng,
    train_text_encoder=False,
):
    dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)
    if train_text_encoder:
        params = {"text_encoder": text_encoder_state.params, "unet": unet_state.params}
    else:
        params = {"unet": unet_state.params}
    # win and lose concatenate along channel dimension
    # shape of batch["pixel_values"] = (N, 2 * C, H, W)
    feed_pixel_values = jnp.concatenate(jnp.split(batch["pixel_values"], 2, axis=1))
    # Convert images to latent space
    vae_outputs = vae.apply(
        {"params": vae_params}, batch["pixel_values"], deterministic=True, method=vae.encode,
    )
    latents = vae_outputs.latent_dist.sample(sample_rng)
    


class Parser(ddpo.utils.Parser):
    config: str = "config.base"
    dataset: str = "dreambooth"

p_train_step = jax.pmap(
    ddpo.training.ppo.train_step,
    axis_name="batch",
    donate_argnums=0,
    static_broadcasted_argnums=(3, 4, 5, 6, 7, 8),
)

def main():
    args = Parser().parse_args("dream_ppo")
    transformers.set_seed(args.seed)

    try:
        compilation_cache.initialize_cache(args.cache)
    except AssertionError as e:
        print(f"Warning: {e}")


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
    # ----------------- Load the dataset ----------------- #
    print("Loading dataset...")

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        batch = {
            "pixel_values": pixel_values,
        }
        batch = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    total_train_batch_size = args.train_batch_size * jax.local_device_count()
    if len(train_dataset) < total_train_batch_size:
        raise ValueError(
            f"Training batch size is {total_train_batch_size}, but your dataset only contains"
            f" {len(train_dataset)} images. Please, use a larger dataset or reduce the effective batch size. Note that"
            f" there are {jax.local_device_count()} parallel devices, so your batch size can't be smaller than that."
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
    unet_state = ddpo.training.ppo.AccumulatingTrainState(
        step=0,
        apply_fn=pipeline.unet.apply,
        params=params["unet"],
        tx=optimizer,
        opt_state=opt_state,
        grad_acc=jax.tree_map(np.zeros_like, params["unet"]),
        n_acc=0,
    )
    # ----------------- Replication ----------------- #
    unet_state = jax_utils.replicate(unet_state)
    sampling_scheduler_params = jax_utils.replicate(params["scheduler"])
    noise_scheduler_state = jax_utils.replicate(noise_scheduler_state)
    params["vae"] = jax_utils.replicate(params["vae"])

    # -------------------------- Setup ------------------------#
    timer = ddpo.utils.Timer()
    # -------------------------- Functions ------------------------ #
    def normalize(x, mean, std):
        """
        mirrors `torchvision.transforms.Normalize(mean, std)`
        """
        return (x - mean) / std

    @partial(jax.pmap)
    def vae_encode(images, vae_params):
        images = images.transpose(0, 3, 1, 2)
        images = normalize(images, mean=0.5, std=0.5)
        vae_outputs = pipeline.vae.apply(
            {"params": vae_params}, 
            images,
            deterministic=True, 
            method=pipeline.vae.encode
        )
        gaussian = vae_outputs.latent_dist
        return gaussian.mean, gaussian.logvar

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
        return pipeline.text_encoder(input_ids, params=params["text_encoder"], train=False)[0]

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

    rng, sample_rng = jax.random.split(rng)

    # -------------------------- Callbacks ------------------------ #
    callback_fns = {
        args.filter_field: ddpo.training.callback_fns[args.filter_field](),
    }

    executor = futures.ThreadPoolExecutor(max_workers=2)

    if args.per_prompt_stats_bufsize is not None:
        per_prompt_stats = PerPromptStatTracker(
            buffer_size=args.per_prompt_stats_bufsize,
            min_count=args.per_prompt_stats_min_count,
        )

    # -------------------------- Training Loop ------------------------ #
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {n_devices * args.sample_batch_size}")

    mean_rewards, std_rewards = [], []
    epochs = tqdm.tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ------------------------- Prepare RL experience ----------------------- #
        print(f"Epoch {epoch + 1}, preparing RL experience...")
        experience = []
        for i in range(args.num_sample_batches_per_epoch):
            image_batch = collate_fn(train_dataset.get_batch(n_devices * args.sample_batch_size))
            # ------------------------- Make prompts ----------------------- #
            instance_prompts, class_prompts, prompts_metadata = ddpo.training.prompts.make_prompts(
                args.prompt_fn,
                n_devices * args.sample_batch_size,
                identical_batch=False,
                evaluate=False,
            )
            # Encode prompts
            instance_prompt_ids = pipeline.prepare_inputs(instance_prompts)
            instance_prompt_embeds = text_encode(instance_prompt_ids)
            # ------------------------- DDIM Sampling ----------------------- #
            sample_rng, sample_seeds = jax.random.split(sample_rng)
            sample_seeds = jax.random.split(sample_seeds, n_devices)
            sampling_params = {
                "unet": unet_state.params,
                "scheduler": sampling_scheduler_params,
            }
            final_latents, latents, next_latents, log_probs, timesteps = pipeline(
                shard(instance_prompt_embeds),
                sample_uncond_prompt_embeds,
                sampling_params,
                sample_seeds,
                args.n_inference_steps,
                jit=True,
                height=args.resolution,
                width=args.resolution,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
            )
            # ---------------------- Decode latents ---------------------- #
            generate_images = vae_decode(final_latents, params["vae"])
            generate_images = jax.device_get(ddpo.utils.unshard(generate_images))
            
            # ---------------------- Evaluate callbacks ---------------------- #
            # reshape instance images (bz, 3, resolution, resolution) -> (bz, resolution, resolution, 3)
            instance_images = jnp.transpose(image_batch["pixel_values"], (0, 2, 3, 1))
            callbacks = executor.submit(
                ddpo.training.evaluate_callbacks,
                callback_fns,
                generate_images,
                instance_images,
                instance_prompts,
                metadata=None
            )
            time.sleep(0)

            # Add experience to the buffer
            experience.append(
                {   
                    "prompts": np.array(instance_prompts),
                    "prompts_embeds": np.array(instance_prompt_embeds),
                    "latents": jax.device_get(ddpo.utils.unshard(latents)),
                    "next_latents": jax.device_get(ddpo.utils.unshard(next_latents)),
                    "log_probs": jax.device_get(ddpo.utils.unshard(log_probs)),
                    "timesteps": jax.device_get(ddpo.utils.unshard(timesteps)),
                    "callbacks": callbacks,
                }
            )
            print("RL experience saved!")
            # Save a sample
            pipeline.numpy_to_pil(generate_images[0])[0].save(
                ddpo.utils.fs.join_and_create(
                    localpath, f"samples/{worker_id}_{epoch}_{i}.png"
                )
            )

        # Wait for the callbacks to finish
        for exp in experience:
            exp["rewards"], exp["callback_info"] = exp["callbacks"].result()[
                args.filter_field
            ]
            del exp["callbacks"]
        
        # Collate samples into a single dictionary
        # shape: (num_sample_batches_per_epoch * sample_batch_size * n_devices)
        experience = jax.tree_map(lambda *xs: np.concatenate(xs), *experience)
        # allgather rewards for multi-host training
        rewards = np.array(
            multihost_utils.process_allgather(experience["rewards"], tiled=True)
        )
        # -------------------------- Computing Advantages ------------------------ #
        if args.per_prompt_stats_bufsize is not None:
            prompt_ids = pipeline.tokenizer(
                experience["prompts"].tolist(), padding="max_length", return_tensors="np"
            ).input_ids
            prompt_ids = multihost_utils.process_allgather(prompt_ids, tiled=True)
            prompts = np.array(
                pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            )

            advantages = per_prompt_stats.update(prompts, rewards)

            if jax.process_index == 0:
                np.save(
                    ddpo.utils.fs.join_and_create(
                        localpath, f"per_prompt_stats/{worker_id}_{epoch}.npy"
                    ),
                    per_prompt_stats.get_stats(),
                )
        else:
            advantages = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-08)

        experience["advantages"] = advantages.reshape(jax.process_count(), -1)[worker_id]
        print(f"mean rewards: {np.mean(rewards):.4f}")

        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))
        # save data for future analysis
        np.save(
            ddpo.utils.fs.join_and_create(localpath, f"rewards/{worker_id}_{epoch}.npy"),
            experience["rewards"],
        )
        np.save(
            ddpo.utils.fs.join_and_create(localpath, f"prompts/{worker_id}_{epoch}.npy"),
            experience["prompts"],
        )
        np.save(
            ddpo.utils.fs.join_and_create(
                localpath, f"callback_info/{worker_id}_{epoch}.npy"
            ),
            experience["callback_info"],
        )
        del experience["prompts"]
        del experience["callback_info"]
        del experience["rewards"]

        for inner_epoch in range(args.num_inner_epochs):
            total_batch_size, num_timesteps = experience["log_probs"].shape
            # shuffle samples along batch dimension
            perm = np.random.permutation(total_batch_size)
            experience = jax.tree_map(lambda x: x[perm], experience)
            # shuffle along time dimension
            perms = np.array(
                [np.random.permutation(num_timesteps) for _ in range(total_batch_size)]
            )
            for key in ["log_probs", "latents", "next_latents", "timesteps"]:
                experience[key] = experience[key][np.arange(total_batch_size)[:, None], perms]
            # split experience into batches
            experience_train = jax.tree_map(
                lambda x: x.reshape(-1, n_devices, args.train_batch_size, *x.shape[1:]),
                experience,
            )
            experience_train = [
                dict(zip(experience_train, x)) for x in zip(*experience_train.values())
            ]  # Need to check
            # --------------------- PPO --------------------- #
            num_train_timesteps = int(num_timesteps * args.train_timestep_ratio)
            all_infos = []
            for i, replay in tqdm.tqdm(
                list(enumerate(experience_train)),  # Need to check
                desc=f"PPO inner Epoch {inner_epoch}...",
                leave=False,
            ):
                for j in range(num_train_timesteps):
                    batch = {
                        "prompt_embeds": replay["prompts_embeds"],
                        "uncond_embeds": train_uncond_prompt_embeds,
                        "advantages": replay["advantages"],
                        "latents": replay["latents"][:, :, j],
                        "next_latents": replay["next_latents"][:, :, j],
                        "log_probs": replay["log_probs"][:, :, j],
                        "timesteps": replay["timesteps"][:, :, j],
                    }

                    do_opt_update = (j == num_train_timesteps - 1) and ((i + 1) % args.train_accumulation_steps == 0)
                    if do_opt_update:
                        print(f"Optimizing at {i}th batch and {j}th timestep")
                    
                    unet_state, info = p_train_step(
                        unet_state,
                        batch,
                        noise_scheduler_state,
                        pipeline.scheduler,
                        args.train_cfg,
                        args.guidance_scale,
                        args.eta,
                        args.ppo_clip_range,
                        do_opt_update,
                    )
                    multihost_utils.assert_equal(info, "infos equal")
                    all_infos.append(
                        jax.tree_map(lambda x: jax.device_get(x), info)
                    )
            all_infos = jax.tree_map(lambda *xs: np.stack(xs), *all_infos)
            print(f"mean info: {jax.tree_map(np.mean, all_infos)}")
            # Save training info
            if jax.process_index() == 0:
                np.save(
                    ddpo.utils.fs.join_and_create(
                        localpath, f"train_info/{worker_id}_{epoch}_{inner_epoch}.npy"
                    ),
                    all_infos,
                )
        # -------------------------- Save the model ------------------------ #
        if (epoch + 1) % args.save_freq == 0 or epoch == args.num_train_epochs - 1:
            save_checkpoint_multiprocess(
                os.path.join(args.savepath, "checkpoints"),
                jax_utils.unreplicate(unet_state.params),
                step=epoch,
                keep=1e6,
                overwrite=True,
            )
            print(f"Model saved at epoch {epoch + 1}")
        
        # plot learning curve
        if jax.process_index() == 0:
            plt.clf()
            plt.plot(mean_rewards)
            plt.fill_between(
                range(len(mean_rewards)),
                np.array(mean_rewards) - np.array(std_rewards),
                np.array(mean_rewards) + np.array(std_rewards),
                alpha=0.2,
            )
            plt.grid()
            plt.savefig(os.path.join(localpath, f"log_{worker_id}.png"))
            ddpo.utils.async_to_bucket(localpath, args.savepath)

if __name__ == "__main__":
    main()
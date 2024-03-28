import argparse
import logging
import math
import os
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub.utils import insecure_hashlib
from huggingface_hub import create_repo, upload_folder
from jax.experimental.compilation_cache import compilation_cache as cc
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel, set_seed

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker

from config import common_args
from google.cloud import storage

from datasets import PromptDataset, DPODataset


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.27.0.dev0")

# Cache compiled models across invocations of this script.
cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

logger = logging.getLogger(__name__)

def upload_images(localpath, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    for filename in os.listdir(localpath):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            local_file = os.path.join(localpath, filename)
            blob_name = os.path.join(destination_blob_name, filename)
            blob = bucket.blob(blob_name)
            # upload the file to the cloud
            blob.upload_from_filename(local_file)
            print(f"Uploaded {local_file} to gs://{bucket_name}/{blob_name}.")


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def main():
    parser = argparse.ArgumentParser(description="Training")
    common_args.add_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    rng = jax.random.PRNGKey(args.seed)

    # ------------------------ Generate sample images from pretrained model ------------------------ #
    generated_images_dir = args.generated_data_dir
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)
    num_generated_images= len(list(Path(generated_images_dir).iterdir()))

    if num_generated_images < args.num_generated_images:
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, safety_checker=None, revision=args.revision
        )
        pipeline.set_progress_bar_config(disable=True)

        num_new_images = args.num_generated_images - num_generated_images
        logger.info(f"Number of generated images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(args.prompt, num_new_images)
        total_sample_batch_size = args.sample_batch_size * jax.local_device_count()
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=total_sample_batch_size)

        for example in tqdm(
            sample_dataloader, desc="Generating images", disable=not jax.process_index() == 0
        ):  
            prompt_ids = pipeline.prepare_inputs(example["prompt"])
            prompt_ids = shard(prompt_ids)
            p_params = jax_utils.replicate(params)
            rng = jax.random.split(rng)[0]
            sample_rng = jax.random.split(rng, jax.device_count())
            images = pipeline(prompt_ids, p_params, sample_rng, jit=True).images
            images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
            images = pipeline.numpy_to_pil(np.array(images))

            for i, image in enumerate(images):
                hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = generated_images_dir + f"/{example['index'][i] + num_generated_images}-{hash_image}.jpg"
                image.save(image_filename)

        del pipeline

        # ------------------------ Upload images to Google Cloud ------------------------ #
        upload_images(generated_images_dir, args.bucket, "class_images")

    # Handle the repository creation
    if jax.process_index() == 0:
        if args.savepath is not None:
            os.makedirs(args.savepath, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.savepath).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
    else:
        raise NotImplementedError("No tokenizer specified!")

    # ----------------------- Load the dataset ------------------------ #
    train_dataset = DPODataset(
        instance_data_root=args.instance_data_dir,
        generated_data_root=args.generated_data_dir,
        prompt=args.prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        input_ids = [example["prompt_ids"] for example in examples]
        pixel_values = [example["pixel_values"] for example in examples]

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

    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16

    vae_arg, vae_kwargs = (args.pretrained_model_name_or_path, {"subfolder": "vae", "revision": args.revision})

    # Load models and create wrapper for stable diffusion
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        dtype=weight_dtype,
        revision=args.revision,
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        vae_arg,
        dtype=weight_dtype,
        **vae_kwargs,
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        dtype=weight_dtype,
        revision=args.revision,
    )
    # Copy the parameters to have a reference
    ref_unet_params = jax.tree_map(lambda x: x.copy(), unet_params)

    # Optimization
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size

    constant_scheduler = optax.constant_schedule(args.learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )

    unet_state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)
    text_encoder_state = train_state.TrainState.create(
        apply_fn=text_encoder.__call__, params=text_encoder.params, tx=optimizer
    )

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    noise_scheduler_state = noise_scheduler.create_state()

    # Initialize our training
    train_rngs = jax.random.split(rng, jax.local_device_count())

    def train_step(unet_state, text_encoder_state, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        if args.train_text_encoder:
            params = {"text_encoder": text_encoder_state.params, "unet": unet_state.params}
        else:
            params = {"unet": unet_state.params}

        def compute_loss(params):
            # pixel_values is of shape (N, 2 * C, H, W)
            # reshape it to (2 * N, C, H, W)
            feed_pixel_values = jnp.concatenate(
                jnp.split(batch["pixel_values"], 2, axis=1), axis=0
            )
            print(f"feed_pixel_values: {feed_pixel_values.shape}")
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params}, feed_pixel_values, deterministic=True, method=vae.encode
            )
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (2 * N, H, W, C) -> (2 * N, C, H, W)
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
            # Make timesteps and noise same for both instances and generated images
            split_noise = jnp.split(noise, 2, axis=0)[0]
            noise = jnp.tile(split_noise, (2, 1, 1, 1))
            split_timesteps = jnp.split(timesteps, 2, axis=0)[0]
            timesteps = jnp.tile(split_timesteps, (2,))

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
            # Repeat the encoder hidden states for the two instances
            encoder_hidden_states = jnp.tile(encoder_hidden_states, (2, 1, 1))

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

            # Get the difference for learned model
            model_losses = jnp.mean(jnp.square(model_pred - target), axis=(1, 2, 3))
            model_losses_i, model_losses_g = jnp.split(model_losses, 2, axis=0)
            mdoel_diff = model_losses_i - model_losses_g

            # Get the reference prediction
            ref_model_pred = unet.apply(
                {"params": ref_unet_params}, noisy_latents, timesteps, encoder_hidden_states, train=False
            ).sample
            ref_losses = jnp.mean(jnp.square(ref_model_pred - target), axis=(1, 2, 3))
            ref_losses_i, ref_losses_g = jnp.split(ref_losses, 2, axis=0)
            ref_diff = ref_losses_i - ref_losses_g
            
            # Compute the loss
            scale_term = -0.5 * args.dpo_beta
            inside_term = scale_term * (mdoel_diff - ref_diff)
            loss = -jnp.mean(jax.nn.log_sigmoid(inside_term))
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

        outdir = os.path.join(args.savepath, str(step)) if step else args.savepath
        print(f"Saving model to {outdir}")
        pipeline.save_pretrained(
            outdir,
            params={
                "text_encoder": get_params_to_save(text_encoder_state.params),
                "vae": get_params_to_save(vae_params),
                "unet": get_params_to_save(unet_state.params),
                "safety_checker": safety_checker.params,
            },
        )

        if args.push_to_hub:
            message = f"checkpoint-{step}" if step is not None else "End of training"
            upload_folder(
                repo_id=repo_id,
                folder_path=args.savepath,
                commit_message=message,
                ignore_patterns=["step_*", "epoch_*"],
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
        print(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")
        epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")

    if jax.process_index() == 0:
        checkpoint()


if __name__ == "__main__":
    main()

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
import diffusers

from PIL import Image
import tqdm
from transformers import CLIPTokenizer
from matplotlib import pyplot as plt

from torchvision import transforms
from tensorflow_probability.substrates import jax as tfp

cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

def load_process_images(image_path):
    size = 512
    image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )
    image = Image.open(image_path)
    return image_transforms(image).unsqueeze(0).numpy()

def vae_generate(images):
    vae, vae_params = diffusers.FlaxAutoencoderKL.from_pretrained(
        "duongna/stable-diffusion-v1-4-flax", subfolder="vae", dtype="float32"
    )
    
    latent_dist = vae.apply(
        {"params": vae_params},
        images,
        deterministic=False,
        method=vae.encode,
    ).latent_dist
    latent_samples = latent_dist.sample(jax.random.PRNGKey(0))
    print(latent_samples.shape)

    generated_gaussian = tfp.distributions.Normal(
        loc=latent_dist.mean, scale=jnp.exp(latent_dist.logvar / 2)
    )
    latent_samples = generated_gaussian.sample(seed=jax.random.PRNGKey(0), sample_shape=(5,))
    latent_samples = latent_samples.squeeze()
    print("latent_sample.shape", latent_samples.shape)

    generated_images = vae.apply(
        {"params": vae_params},
        latent_samples,
        method=vae.decode,
    ).sample
    generated_images = (generated_images / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)

    for i, img_array in enumerate(generated_images):
        # Convert the image array to a PIL Image object
        img_array = np.array(img_array)
        img = Image.fromarray((img_array * 255).astype(np.uint8))

        # Save the image to disk
        img.save(f'./figs/generated_image_{i}.png')


def main():
    image1_path = Path('../../dreambooth/dataset/dog/00.jpg')
    image2_path = Path('../../dreambooth/dataset/dog/01.jpg')
    image3_path = Path('../../dreambooth/dataset/dog2/00.jpg')

    image1 = load_process_images(image1_path)
    image2 = load_process_images(image2_path)
    image3 = load_process_images(image3_path)

    vae_generate(image1)
    print("Done")

if __name__ == "__main__":
    main()
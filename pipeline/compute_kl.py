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
from tensorflow_probability.substrates import jax as tfp
import diffusers

from PIL import Image
import tqdm
from transformers import CLIPTokenizer
from matplotlib import pyplot as plt

from torchvision import transforms

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

def normalize(x, mean, std):
        """
        mirrors `torchvision.transforms.Normalize(mean, std)`
        """
        return (x - mean) / std

def vae_similarity_fn(generated_images, instance_images, jit=True):
    vae, vae_params = diffusers.FlaxAutoencoderKL.from_pretrained(
        "duongna/stable-diffusion-v1-4-flax", subfolder="vae", dtype="float32"
    )
    gen_images = generated_images
    ins_images = instance_images
    # gen_images = normalize(generated_images, mean=0.5, std=0.5)
    # ins_images = normalize(instance_images, mean=0.5, std=0.5)
    generated_vae_outputs = vae.apply(
        {"params": vae_params},
        gen_images,
        deterministic=True,
        method=vae.encode,
    )
    generated_latent_dist = generated_vae_outputs.latent_dist
    generated_gaussian = tfp.distributions.Normal(
        loc=generated_latent_dist.mean, scale=jnp.exp(generated_latent_dist.logvar / 2)
    )

    instance_vae_outputs = vae.apply(
        {"params": vae_params},
        ins_images,
        deterministic=True,
        method=vae.encode,
    )
    instance_latent_dist = instance_vae_outputs.latent_dist
    instance_gaussian = tfp.distributions.Normal(
        loc=instance_latent_dist.mean, scale=jnp.exp(instance_latent_dist.logvar / 2)
    )

    len_shape = len(generated_latent_dist.mean.shape)
    axis = tuple(range(1, len_shape))

    generated_mean = generated_latent_dist.mean.reshape(generated_latent_dist.mean.shape[0], -1)
    instance_mean = instance_latent_dist.mean.reshape(instance_latent_dist.mean.shape[0], -1)

    generated_logvar = generated_latent_dist.logvar.reshape(generated_latent_dist.logvar.shape[0], -1)
    instance_logvar = instance_latent_dist.logvar.reshape(instance_latent_dist.logvar.shape[0], -1)

    distance = jnp.mean((generated_mean - instance_mean) ** 2, axis=-1) \
                + jnp.mean((generated_logvar - instance_logvar) ** 2, axis=-1)
    return distance

    # kl = tfp.distributions.kl_divergence(generated_gaussian, instance_gaussian)
    # len_shape = len(kl.shape)
    # axis = tuple(range(1, len_shape))
    # kl_result = jnp.mean(kl, axis=axis)
    # return kl_result

def main():
    image1_path = Path('../../dreambooth/dataset/dog/00.jpg')
    image2_path = Path('../../dreambooth/dataset/dog/01.jpg')
    image3_path = Path('../../dreambooth/dataset/dog2/00.jpg')

    image1 = load_process_images(image1_path)
    image2 = load_process_images(image2_path)
    image3 = load_process_images(image3_path)

    kl1 = vae_similarity_fn(image1, image2)
    kl2 = vae_similarity_fn(image1, image3)
    print(f'KL 1: {kl1}')
    print(f'iteself KL: {vae_similarity_fn(image1, image1)}')
    print(f'KL 2: {kl2}')

if __name__ == "__main__":
    main()
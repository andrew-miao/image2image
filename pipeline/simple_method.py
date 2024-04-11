import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline
import os
from google.cloud import storage
import argparse
import diffusers
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tensorflow_probability.substrates import jax as tfp

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--savepath",
        type=str,
        default="path-to-save-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--bucket", type=str, default="dpo_booth_bucket", help="Google Cloud Bucket to store the data.")
    parser.add_argument("--revision", type=str, default=None, help="The revision of the model to be used.")

    args = parser.parse_args()
    return args

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

def encode(images, num_samples=4):
    vae, vae_params = diffusers.FlaxAutoencoderKL.from_pretrained(
        "duongna/stable-diffusion-v1-4-flax", subfolder="vae", dtype="float32"
    )
    
    latent_dist = vae.apply(
        {"params": vae_params},
        images,
        deterministic=False,
        method=vae.encode,
    ).latent_dist
    generated_gaussian = tfp.distributions.Normal(
        loc=latent_dist.mean, scale=jnp.exp(latent_dist.logvar / 2)
    )
    latent_samples = generated_gaussian.sample(seed=jax.random.PRNGKey(0), sample_shape=(num_samples,))
    latent_samples = latent_samples.squeeze()
    latent_samples = latent_samples.transpose(0, 3, 1, 2)
    return latent_samples

def main():
    args = parse_args()
    # ----------------------- Loading Models ----------------------- #
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "duongna/stable-diffusion-v1-4-flax", safety_checker=None, revision=args.revision
    )
    
    # ----------------------- Evaluation ----------------------- #
    local_path = "logs/simple/sampled_images/"
    os.makedirs(local_path, exist_ok=True)
    test_prompt = 'a photo of sks dogs is swimming'
    test_rng = jax.random.PRNGKey(0)
    num_samples = jax.device_count()
    prompt = num_samples * [test_prompt]
    prompt_ids = pipeline.prepare_inputs(prompt)
    image1_path = Path('../../dreambooth/dataset/dog/00.jpg')
    image1 = load_process_images(image1_path)

    params = replicate(params)
    test_seed = jax.random.split(test_rng, num_samples)
    prompt_ids = shard(prompt_ids)
    latents = encode(image1, num_samples)
    latents = shard(latents)
    images = pipeline(prompt_ids, params, test_seed, latents=latents, guidance_scale=30.0, jit=True).images
    images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
    for i, image in enumerate(images):
        image_filename = local_path + f"{test_prompt}-{i}.jpg"
        image.save(image_filename)
                
    print(f"Inference images saved to {local_path}")
    # ----------------------- Upload images to Google Cloud ----------------------- #
    upload_images(local_path, args.bucket, "simple_generated_images")

if __name__ == "__main__":
    main()
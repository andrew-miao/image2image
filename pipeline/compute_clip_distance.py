import os
import jax
import jax.numpy as jnp
from PIL import Image
from torchvision import transforms
import numpy as np

import transformers

def reward_fn(reference_images, compare_images):
    model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # min_distance = np.inf * np.ones(len(compare_images))
    max_similarity = -np.inf * np.ones(len(compare_images))
    for reference_image in reference_images:
        images = [reference_image] + compare_images
        inputs = processor(images=images, return_tensors="np")
        features = model.get_image_features(**inputs)
        reference_features = features[0]
        compare_features = features[1:]

        # L2 distance
        # distance = jnp.linalg.norm(reference_features - compare_features, axis=-1)
        # min_distance = jnp.minimum(min_distance, distance)

        # normalize image features
        reference_norm = jnp.linalg.norm(reference_features, axis=-1, keepdims=True)
        compare_norm = jnp.linalg.norm(compare_features, axis=-1, keepdims=True)

        reference_features = reference_features / reference_norm
        compare_features = compare_features / compare_norm

        # dot product
        similarity = jnp.sum(reference_features * compare_features, axis=-1)
        max_similarity = jnp.maximum(max_similarity, similarity)
    
    return max_similarity
    # reward = jnp.exp(-min_distance)
    # return reward

def read_images(directory):
    images = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file_path)
            images.append(image)
    return images

def main():
    images = read_images("../../dreambooth/dataset/dog")
    compare_images = read_images("../../dreambooth/dataset/dog6")
    same_reward = reward_fn(images, images)
    different_reward = reward_fn(images, compare_images)
    print(f"average reward for the same dog: {np.mean(same_reward):.4f}")
    print(f"average reward for different dogs: {np.mean(different_reward):.4f}")

if __name__ == "__main__":
    pass
    # main()
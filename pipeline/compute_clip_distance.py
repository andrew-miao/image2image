import os
import jax
import jax.numpy as jnp
from PIL import Image
from torchvision import transforms
import numpy as np

import transformers


def cosine_similarity_fn(reference_images, compare_images):
    model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(images=[reference_images, compare_images], return_tensors="np")
    features = model.get_image_features(**inputs)
    reference_features, compare_features = jnp.split(features, 2, axis=0)

    # normalize image features
    reference_norm = jnp.linalg.norm(reference_features, axis=-1, keepdims=True)
    compare_norm = jnp.linalg.norm(compare_features, axis=-1, keepdims=True)

    reference_features = reference_features / reference_norm
    compare_features = compare_features / compare_norm

    # dot product
    similarity = jnp.sum(reference_features * compare_features, axis=-1)
    return similarity

def l2_distance_fn(reference_images, compare_images):
    model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(images=[reference_images, compare_images], return_tensors="np")
    features = model.get_image_features(**inputs)
    reference_features, compare_features = jnp.split(features, 2, axis=0)

    # L2 distance
    distance = jnp.linalg.norm(reference_features - compare_features, axis=-1)
    return distance

def cosine_l2_fn(reference_images, compare_images):
    model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(images=[reference_images, compare_images], return_tensors="np")
    features = model.get_image_features(**inputs)
    reference_features, compare_features = jnp.split(features, 2, axis=0)
    # L2 distance
    distance = jnp.linalg.norm(reference_features - compare_features, axis=-1)

    # normalize image features
    reference_norm = jnp.linalg.norm(reference_features, axis=-1, keepdims=True)
    compare_norm = jnp.linalg.norm(compare_features, axis=-1, keepdims=True)

    reference_features = reference_features / reference_norm
    compare_features = compare_features / compare_norm

    # dot product
    similarity = jnp.sum(reference_features * compare_features, axis=-1)
    return similarity, distance

def reward_fn(reference_images, compare_images):
    model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    min_distance = np.inf * np.ones(len(compare_images))
    for reference_image in reference_images:
        images = [reference_image] + compare_images
        inputs = processor(images=images, return_tensors="np")
        features = model.get_image_features(**inputs)
        reference_features = features[0]
        compare_features = features[1:]

        # L2 distance
        distance = jnp.linalg.norm(reference_features - compare_features, axis=-1)
        min_distance = jnp.minimum(min_distance, distance)
    
    reward = jnp.exp(-min_distance)
    return reward

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

    # same_similarity, same_distance = [], []
    # for i in range(len(images)):
    #     for j in range(i, len(images)):
    #         similarity, distance = cosine_l2_fn(images[i], images[j])
    #         print(f"dog{i} vs dog{j} cosine similarity: {similarity[0]:.4f}")
    #         print(f"dog{i} vs dog{j} L2 distance: {distance[0]:.4f}")
    #         same_similarity.append(similarity[0])
    #         same_distance.append(distance[0])

    # different_similarity, different_distance = [], []

    # for i in range(len(images)):
    #     for j in range(len(compare_images)):
    #         similarity, distance = cosine_l2_fn(images[i], compare_images[j])
    #         print(f"dog{i} vs dog6{j} cosine similarity: {similarity[0]:.4f}")
    #         print(f"dog{i} vs dog6{j} L2 distance: {distance[0]:.4f}")
    #         different_similarity.append(similarity[0])
    #         different_distance.append(distance[0])

    # print("-------------------------------------------------")
    # print("Cosine Similarity Information")
    # print(f"average cosine similarity for the same dog: {np.mean(same_similarity):.4f}")
    # print(f"min cosine similarity for the same dog: {np.min(same_similarity):.4f}")
    # print(f"avergae cosine similarity for different dogs: {np.mean(different_similarity):.4f}")
    # print(f"max cosine similarity for different dogs: {np.max(different_similarity):.4f}")
    # print("-------------------------------------------------")
    # print("L2 Distance Information")
    # print(f"average L2 distance for the same dog: {np.mean(same_distance):.4f}")
    # print(f"max L2 distance for the same dog: {np.max(same_distance):.4f}")
    # print(f"average L2 distance for different dogs: {np.mean(different_distance):.4f}")
    # print(f"min L2 distance for different dogs: {np.min(different_distance):.4f}")

if __name__ == "__main__":
    main()
import torch
import timm

from PIL import Image
from torchvision import transforms
from pathlib import Path
import torch.nn.functional as F


def load_process_images(image_path):
    size = 224
    image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )
    image = Image.open(image_path)
    return image_transforms(image).unsqueeze(0)

def main():
    model = timm.create_model('resnest26d', pretrained=True, features_only=True)
    image1_path = Path('../../dreambooth/dataset/dog/00.jpg')
    image2_path = Path('../../dreambooth/dataset/dog/01.jpg')
    image3_path = Path('../../dreambooth/dataset/dog2/01.jpg')

    image1 = load_process_images(image1_path)
    image2 = load_process_images(image2_path)
    image3 = load_process_images(image3_path)

    images = torch.cat([image1, image2, image3], dim=0)
    features = model(images)
    x = features[0] # shape: (# images, 64, 112, 112)
    image_features1 = x[0].view(1, x[0].shape[1], -1)
    image_features2 = x[1].view(1, x[0].shape[1], -1)
    image_features3 = x[2].view(1, x[0].shape[1], -1)
    
    with torch.no_grad():
        cos_sim1 = torch.mean(F.cosine_similarity(image_features1, image_features2, dim=1))
        cos_sim2 = torch.mean(F.cosine_similarity(image_features1, image_features3, dim=1))
    print(f"cosine similarity between image1 and image2: {cos_sim1}")
    print(f"cosine similarity between image1 and image3: {cos_sim2}")

if __name__ == '__main__':
    main()
import torch
from torchvision.utils import save_image
from pathlib import Path
import os
from PIL import Image
from torchvision import transforms


def stack_and_save(images: list, save_path: str, max_images: int = 5):
    # Ensure there are enough images to stack
    num_images = min(len(images), max_images)

    # Select the first `num_images` and stack them vertically (dim=2 for height).
    stacked_image = torch.cat(images[:num_images], dim=1)

    # Save the stacked image to the specified path
    save_image(stacked_image, save_path, nrow=1, padding=0)
    print(f"Saved stacked image at {save_path}")


if __name__ == '__main__':
    # Define a transform to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts an image to a tensor [C, H, W] in range [0, 1]
    ])

    # Load the rest of the images as tensors
    image_folder = 'outputs/samples/run_38'
    image_ids = [7, 17, 24, 34, 48, 57, 78, 112, 114, 116]
    images = [transform(Image.open(os.path.join(image_folder, f'{p}-images.jpg')).convert('RGB')) for p in image_ids]

    # Stack and save the images
    save_path = Path('test/stacked_images.png')
    stack_and_save(images, save_path, 10)

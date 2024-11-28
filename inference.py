from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from config.models import Config, DataConfig
from main import load_checkpoint, load_config
from utils import generate_valid_permutations, denorm
from network import Generator
from dataloader import get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_input(image: Image.Image, user_attributes: list, all_attributes: dict,
                     config: DataConfig) -> tuple[torch.Tensor, list[torch.Tensor]]:

    transform = get_transform(config, 'val')
    image_tensor = transform(image)  # -> torch.Tensor Normalized image

    # Generate valid permutations
    label_org = torch.tensor([1 if attr in user_attributes else 0 for attr in attrs.values()], dtype=torch.float32)
    valid_permutations = generate_valid_permutations(label_org, all_attributes)

    return image_tensor, valid_permutations


def d(image):
    image * 0.5 + 0.5


def postprocess_output(images: list[torch.Tensor], labels: list[torch.Tensor],
                       all_attributes: dict) -> list[tuple[Image.Image, list[str]]]:
    postprocessed_images_labels = []
    for image, label in zip(images, labels):
        print('-----------------')
        image = d(image.cpu())
        print(image.shape)
        print(type(image))
        print(image.max(), image.min())
        print('-----------------')
        image = image.mul(255).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
        print(image.shape)
        print(type(image))
        print(image.max(), image.min())
        label = [all_attributes[i] for i, label in enumerate(label.cpu().numpy()) if label == 1]
        postprocessed_images_labels.append((Image.fromarray(image), label))

    return postprocessed_images_labels


def infer(G: Generator, input_tensor: torch.Tensor,
          target_labels: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    G.eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Generate images
    generated_images = []
    for label in target_labels:
        with torch.no_grad():
            target_tensor = label.unsqueeze(0).to(device)

            print(input_tensor.shape, target_tensor.shape)
            generated_image = G(input_tensor, target_tensor)
        generated_images.append(generated_image.squeeze(0))

    return generated_images


def save_images(images_labels: list[tuple[Image.Image, list[str]]], save_dir: Path):
    for idx, (image, labels) in enumerate(images_labels):
        image_path = save_dir / f'{idx}_{labels}.png'
        image.save(image_path)


if __name__ == '__main__':
    # Load config
    config: Config = load_config('config/config.yaml')

    # Load model
    model_name = '30-G'
    model_path = Path(config.folders.checkpoints) / f'{model_name}.ckpt'
    G = load_checkpoint(Path(config.folders.checkpoints) / '30-G.ckpt', config.model)

    # Preprocess input
    input_image = Image.open('test/test_image.jpg').convert('RGB')
    attrs = {0: 'Old', 1: 'Female', 2: 'Black_Hair', 3: 'Blond_Hair', 4: 'Brown_Hair', 5: 'Male', 6: 'Young'}
    user_attributes = ['Male', 'Black_Hair', 'Young']
    input_tensor, target_labels = preprocess_input(input_image, user_attributes, attrs, config.data)

    # Inference
    generated_images = infer(G, input_tensor, target_labels)

    # Postprocess output
    postprocessed_images_labels = postprocess_output(generated_images, target_labels, attrs)

    # Save images
    save_images(postprocessed_images_labels, Path('test'))

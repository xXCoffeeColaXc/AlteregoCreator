from pathlib import Path
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from config.models import Config, DataConfig
from main import load_checkpoint, load_config
from utils import generate_valid_permutations, denorm
from network import Generator
from dataloader import get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Load config and model
    config: Config = load_config('config/config.yaml')
    G = load_checkpoint(Path(config.folders.checkpoints) / '25-G.ckpt', config.model)
    G.to(device)
    G.eval()

    # Prepare the original label tensor
    #attrs = {0: 'Old', 1: 'Female', 2: 'Black_Hair', 3: 'Blond_Hair', 4: 'Brown_Hair', 5: 'Male', 6: 'Young'}
    attrs = {
        -2: 'Old',
        -1: 'Female',
        8: 'Black_Hair',
        9: 'Blond_Hair',
        11: 'Brown_Hair',  #18: 'Heavy_Makeup',
        20: 'Male',  #32: 'Straight_Hair',
        #33: 'Wavy_Hair',
        #36: 'Wearing_Lipstick',
        39: 'Young'
    }
    user_attributes = ['Male', 'Black_Hair', 'Young']
    #c_org = torch.tensor([1 if attr in user_attributes else 0 for attr in attrs.values()], dtype=torch.float32)

    c_org = torch.tensor([0, 0, 0, 1, 0, 1, 1], dtype=torch.float32)
    # Run inference and save images
    infer_and_save(G, 'test/test_image.jpg', c_org, config)

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
import os
from pathlib import Path
from PIL import Image
import numpy as np
from config.models import DataConfig, TrainingConfig
from utils import load_config

all_attrs_map = {
    0: '5_o_Clock_Shadow',
    1: 'Arched_Eyebrows',
    2: 'Attractive',
    3: 'Bags_Under_Eyes',
    4: 'Bald',
    5: 'Bangs',
    6: 'Big_Lips',
    7: 'Big_Nose',
    8: 'Black_Hair',
    9: 'Blond_Hair',
    10: 'Blurry',
    11: 'Brown_Hair',
    12: 'Bushy_Eyebrows',
    13: 'Chubby',
    14: 'Double_Chin',
    15: 'Eyeglasses',
    16: 'Goatee',
    17: 'Gray_Hair',
    18: 'Heavy_Makeup',
    19: 'High_Cheekbones',
    20: 'Male',
    21: 'Mouth_Slightly_Open',
    22: 'Mustache',
    23: 'Narrow_Eyes',
    24: 'No_Beard',
    25: 'Oval_Face',
    26: 'Pale_Skin',
    27: 'Pointy_Nose',
    28: 'Receding_Hairline',
    29: 'Rosy_Cheeks',
    30: 'Sideburns',
    31: 'Smiling',
    32: 'Straight_Hair',
    33: 'Wavy_Hair',
    34: 'Wearing_Earrings',
    35: 'Wearing_Hat',
    36: 'Wearing_Lipstick',
    37: 'Wearing_Necklace',
    38: 'Wearing_Necktie',
    39: 'Young'
}

attr2id_map = {name: idx for idx, name in all_attrs_map.items()}


class CelebA(Dataset):

    def __init__(self, config: DataConfig, transform: transforms.Compose):
        self.root_dir = Path(config.root_dir)
        self.selected_attrs = config.selected_attrs
        self.transform = transform

        self.img_dir = self.root_dir / config.image_dir
        self.attr_path = self.root_dir / config.attributes_file

        self.images = []
        self.labels = []

        self.preprocess()

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        lines = lines[2:]  # actual images with label information

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []  # [Old, Young, Female, Male, Black_Hair, Blond_Hair, Brown_Hair]
            for attr in self.selected_attrs:
                if attr == 'Old':
                    label.append(1 if values[39] != '1' else 0)  # If not young then old
                elif attr == 'Female':
                    label.append(1 if values[20] != '1' else 0)  # If not male then female
                else:
                    label.append(1 if values[attr2id_map[attr]] == '1' else 0)

            if np.sum(label) == 0:
                continue

            self.images.append(filename)
            self.labels.append(label)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image = Image.open(self.img_dir / str(image)).convert('RGB')
        label = torch.FloatTensor(label)

        return self.transform(image), label

    def __len__(self):
        return len(self.images)


def get_transform(data_config: DataConfig, split: str = 'train'):
    if split == 'train':
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(data_config.crop_size),
                transforms.Resize(data_config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )
    elif split == 'val' or split == 'test':
        transform = transforms.Compose(
            [
                transforms.CenterCrop(data_config.crop_size),
                transforms.Resize(data_config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )

    return transform


def get_loader(data_config: DataConfig, train_config: TrainingConfig, split: str = 'train'):
    transform = get_transform(data_config, split)

    full_dataset = CelebA(data_config, transform)

    total_size = len(full_dataset)
    train_size = train_config.batch_size * 10  #int(0.8 * total_size)
    val_size = train_config.batch_size * 10  # int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(train_config.random_seed))

    if split == 'train':
        train_dataset.dataset.transform = transform
        data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=train_config.num_workers,
            pin_memory=True
        )
    elif split == 'val':
        val_dataset.dataset.transform = transform
        data_loader = DataLoader(
            dataset=val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=True
        )
    else:
        test_dataset.dataset.transform = transform
        data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=True
        )

    return data_loader


if __name__ == '__main__':
    config = load_config('config/config.yaml')

    data_config = config.data
    train_config = config.training

    train_loader = get_loader(data_config, train_config, 'train')
    print(len(train_loader.dataset))
    val_loader = get_loader(data_config, train_config, 'val')
    print(len(val_loader.dataset))
    test_loader = get_loader(data_config, train_config, 'test')
    print(len(test_loader.dataset))

    for i, (images, labels) in enumerate(train_loader):
        print(images.shape, labels.shape)
        print(labels)
        break

    for i, (images, labels) in enumerate(val_loader):
        print(images.shape, labels.shape)
        break

    for i, (images, labels) in enumerate(test_loader):
        print(images.shape, labels.shape)
        break

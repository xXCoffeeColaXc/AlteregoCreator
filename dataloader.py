import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
from pathlib import Path
from PIL import Image
import numpy as np
from config.models import DataConfig, TrainingConfig

# NOTE: handle train/val/test somehow


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

            label = []
            for idx in self.selected_attrs.keys():
                if idx == -2:
                    label.append(1 if values[39] != '1' else 0)  # If not young then old
                elif idx == -1:
                    label.append(1 if values[20] != '1' else 0)  # If not male then female
                else:
                    label.append(1 if values[idx] == '1' else 0)

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


def get_loader(data_config: DataConfig, train_config: TrainingConfig, split: str = 'train'):
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
    elif split == 'val':
        transform = transforms.Compose(
            [
                transforms.CenterCrop(data_config.crop_size),
                transforms.Resize(data_config.image_size),
                transforms.ToTensor(),  #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )

    dataset = CelebA(data_config, transform)
    return DataLoader(
        dataset=dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=train_config.num_workers
    )


if __name__ == '__main__':
    pass

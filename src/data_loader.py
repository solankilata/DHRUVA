import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class AI4MarsDataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_transform=None, size=(256, 256)):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_transform = image_transform
        self.size = size
        self.label_files = sorted(self.labels_dir.glob("*.png"))

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_path = self.label_files[idx]
        image_name = label_path.stem + ".JPG"
        image_path = self.images_dir / image_name

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        image = image.resize(self.size, Image.BILINEAR)
        label = label.resize(self.size, Image.NEAREST)

        image = self.image_transform(image) if self.image_transform else T.ToTensor()(image)
        label = torch.from_numpy(np.array(label)).long()

        return image, label
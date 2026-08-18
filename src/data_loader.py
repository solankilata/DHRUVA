import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class AI4MarsDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.label_files = sorted(self.labels_dir.glob("*.png"))

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_path = self.label_files[idx]
        image_name = label_path.stem + ".JPG"
        image_path = self.images_dir / image_name

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        label = np.array(label)

        if self.transform:
            image = self.transform(image)

        label = torch.from_numpy(label).long()

        return image, label


if __name__ == "__main__":
    images_dir = "../data/raw/ai4mars-dataset-merged-0.1/msl/images/edr"
    labels_dir = "../data/raw/ai4mars-dataset-merged-0.1/msl/labels/train"

    dataset = AI4MarsDataset(images_dir, labels_dir)
    print(f"Total samples found: {len(dataset)}")

    image, label = dataset[0]
    print(f"Image size: {image.size}")
    print(f"Label shape: {label.shape}, unique values: {label.unique()}")

    print("\nChecking a few more samples:")
    for i in [1, 5, 50, 500, 5000]:
        _, lbl = dataset[i]
        print(f"Sample {i}: unique values = {lbl.unique()}")
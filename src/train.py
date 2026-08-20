import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import time

from data_loader import AI4MarsDataset
from unet_model import UNet


def main():
    # NOTE: this script is written to reference a LOCAL copy of the dataset.
    # Training was actually performed on Kaggle (GPU), using this same logic,
    # since this laptop has no dedicated GPU. This file documents/reproduces
    # that exact training logic locally.
    base = "../data/raw/ai4mars-dataset-merged-0.1"
    images_dir = f"{base}/msl/images/edr"
    labels_dir = f"{base}/msl/labels/train"

    image_transform = T.Compose([T.ToTensor()])
    full_dataset = AI4MarsDataset(images_dir, labels_dir, image_transform=image_transform)

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=3, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 15

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.1f}s")

    torch.save(model.state_dict(), "../outputs/dhruva_unet.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()
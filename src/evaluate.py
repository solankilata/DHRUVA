import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data_loader import AI4MarsDataset
from unet_model import UNet
import torchvision.transforms as T


def compute_metrics(model, loader, device, num_classes=4):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            preds_flat = preds.cpu().numpy().flatten()
            labels_flat = labels.cpu().numpy().flatten()

            valid_mask = labels_flat != 255
            all_preds.append(preds_flat[valid_mask])
            all_labels.append(labels_flat[valid_mask])

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    ious = []
    class_names = ["Soil", "Bedrock", "Sand", "Big Rock"]
    print("Per-class IoU:")
    for cls in range(num_classes):
        pred_cls = (all_preds == cls)
        label_cls = (all_labels == cls)
        intersection = np.logical_and(pred_cls, label_cls).sum()
        union = np.logical_or(pred_cls, label_cls).sum()
        iou = intersection / union if union > 0 else float('nan')
        ious.append(iou)
        print(f"  {class_names[cls]}: {iou:.4f}")

    miou = np.nanmean(ious)
    print(f"\nMean IoU (mIoU): {miou:.4f}")

    f1_scores = f1_score(all_labels, all_preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    print("\nPer-class F1:")
    for cls, f1 in zip(class_names, f1_scores):
        print(f"  {cls}: {f1:.4f}")

    macro_f1 = f1_score(all_labels, all_preds, labels=[0, 1, 2, 3], average='macro', zero_division=0)
    print(f"\nMacro F1: {macro_f1:.4f}")

    return miou, macro_f1


if __name__ == "__main__":
    base = "../data/raw/ai4mars-dataset-merged-0.1"
    images_dir = f"{base}/msl/images/edr"
    labels_dir = f"{base}/msl/labels/train"

    image_transform = T.Compose([T.ToTensor()])
    dataset = AI4MarsDataset(images_dir, labels_dir, image_transform=image_transform)

    from torch.utils.data import random_split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=5).to(device)
    model.load_state_dict(torch.load("../outputs/dhruva_unet.pth", map_location=device))

    compute_metrics(model, val_loader, device)
import torch
import numpy as np
from unet_model import UNet


def predict_with_uncertainty(model, image_tensor, device, n_samples=15):
    """
    Run the model n_samples times with dropout active, to estimate
    per-pixel prediction uncertainty (Monte Carlo Dropout).
    """
    model.eval()
    model.enable_mc_dropout()  # force dropout ON despite eval() mode

    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            all_preds.append(probs.cpu().numpy())

    all_preds = np.stack(all_preds, axis=0)  # shape: (n_samples, 1, num_classes, H, W)

    mean_pred = all_preds.mean(axis=0)
    final_class = np.argmax(mean_pred, axis=1).squeeze(0)  # most likely class overall

    # Uncertainty = variance across samples, averaged over classes
    variance = all_preds.var(axis=0).mean(axis=1).squeeze(0)  # shape: (H, W)

    # Normalize to 0-1 for easy interpretation
    if variance.max() > 0:
        confidence_map = 1 - (variance / variance.max())
    else:
        confidence_map = np.ones_like(variance)

    return final_class, confidence_map


if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=5).to(device)
    model.load_state_dict(torch.load("../outputs/dhruva_unet.pth", map_location=device))
    print("Loaded trained weights (dropout layer initialized fresh, untrained but functional for MC sampling).")

    base = "../data/raw/ai4mars-dataset-merged-0.1/msl/images/edr"
    sample_files = sorted(Path(base).glob("*.JPG"))
    img = Image.open(sample_files[500]).convert("RGB")
    img_resized = img.resize((256, 256), Image.BILINEAR)
    img_tensor = T.ToTensor()(img_resized).unsqueeze(0).to(device)

    final_class, confidence_map = predict_with_uncertainty(model, img_tensor, device, n_samples=15)

    print(f"Confidence map range: {confidence_map.min():.3f} to {confidence_map.max():.3f}")
    print(f"Mean confidence: {confidence_map.mean():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_resized)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(final_class, cmap="tab10", vmin=0, vmax=4)
    axes[1].set_title("Terrain Prediction")
    axes[1].axis("off")

    im = axes[2].imshow(confidence_map, cmap="RdYlGn", vmin=0, vmax=1)
    axes[2].set_title("Confidence Map (MC Dropout)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("../outputs/confidence_map_sample.png", dpi=150)
    print("Saved confidence_map_sample.png")
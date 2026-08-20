import numpy as np
import torch
from scipy.ndimage import generic_filter

# ---- Base risk per terrain class ----
# Grounded in documented rover mobility incidents (e.g. Spirit rover sand
# entrapment) and general terramechanics reasoning, NOT real slip telemetry.
# 0 = Soil, 1 = Bedrock, 2 = Sand, 3 = Big Rock
BASE_RISK = {
    0: 0.15,   # Soil - generally stable
    1: 0.10,   # Bedrock - solid, lowest slip risk
    2: 0.70,   # Sand - documented entrapment risk
    3: 0.65,   # Big Rock - obstacle/damage risk, not slip but still high danger
}


def compute_local_roughness(gray_image, window_size=9):
    """Local standard deviation as a texture/roughness proxy."""
    def local_std(values):
        return np.std(values)

    roughness = generic_filter(gray_image.astype(float), local_std, size=window_size)
    if roughness.max() > 0:
        roughness = roughness / roughness.max()
    return roughness


def compute_risk_map(terrain_pred, gray_image, roughness_weight=0.3):
    """Combine base class risk with local roughness into a 0-1 risk map."""
    base_risk_map = np.zeros_like(terrain_pred, dtype=float)
    for cls, risk in BASE_RISK.items():
        base_risk_map[terrain_pred == cls] = risk

    roughness = compute_local_roughness(gray_image)

    risk_map = (1 - roughness_weight) * base_risk_map + roughness_weight * roughness
    risk_map = np.clip(risk_map, 0, 1)

    return risk_map


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # Force non-interactive backend so saving always works
    import matplotlib.pyplot as plt
    import torchvision.transforms as T
    from PIL import Image
    from pathlib import Path
    from unet_model import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=5).to(device)
    model.load_state_dict(torch.load("../outputs/dhruva_unet.pth", map_location=device))
    model.eval()

    base = "../data/raw/ai4mars-dataset-merged-0.1/msl/images/edr"
    sample_files = sorted(Path(base).glob("*.JPG"))
    img = Image.open(sample_files[500]).convert("RGB")
    img_resized = img.resize((256, 256), Image.BILINEAR)

    img_tensor = T.ToTensor()(img_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        terrain_pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    gray_image = np.array(img_resized.convert("L"))
    risk_map = compute_risk_map(terrain_pred, gray_image)

    print(f"Risk map shape: {risk_map.shape}")
    print(f"Risk range: {risk_map.min():.3f} to {risk_map.max():.3f}")
    print(f"Mean risk: {risk_map.mean():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_resized)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(terrain_pred, cmap="tab10", vmin=0, vmax=4)
    axes[1].set_title("Terrain Classification (Module 1)")
    axes[1].axis("off")

    im = axes[2].imshow(risk_map, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[2].set_title("Slip-Risk Map (Module 2)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = "../outputs/risk_map_sample.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved visualization to {output_path}")
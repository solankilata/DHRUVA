import numpy as np
import heapq


def a_star_search(risk_map, start, goal, risk_weight=5.0):
    """
    Find safest path from start to goal on a 2D risk_map (values 0-1).
    start, goal: (row, col) tuples
    risk_weight: how strongly to penalize high-risk cells vs. pure distance
    """
    rows, cols = risk_map.shape

    def heuristic(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def neighbors(node):
        r, c = node
        candidates = [
            (r-1, c), (r+1, c), (r, c-1), (r, c+1),
            (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)
        ]
        return [(nr, nc) for nr, nc in candidates if 0 <= nr < rows and 0 <= nc < cols]

    def step_cost(node):
        r, c = node
        base = 1.0
        risk_penalty = (risk_map[r, c] ** 2) * risk_weight
        return base + risk_penalty

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in neighbors(current):
            tentative_g = g_score[current] + step_cost(neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None


if __name__ == "__main__":
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from unet_model import UNet
    from slip_risk import compute_risk_map
    from uncertainty import predict_with_uncertainty

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=5).to(device)
    model.load_state_dict(torch.load("../outputs/dhruva_unet.pth", map_location=device))

    base = "../data/raw/ai4mars-dataset-merged-0.1/msl/images/edr"
    sample_files = sorted(Path(base).glob("*.JPG"))
    img = Image.open(sample_files[500]).convert("RGB")
    img_resized = img.resize((256, 256), Image.BILINEAR)
    img_tensor = T.ToTensor()(img_resized).unsqueeze(0).to(device)

    terrain_pred, confidence_map = predict_with_uncertainty(model, img_tensor, device, n_samples=15)
    gray_image = np.array(img_resized.convert("L"))
    risk_map = compute_risk_map(terrain_pred, gray_image)

    start = (10, 250)
    goal = (250, 10)

    def evaluate_path(path, risk_map):
        risks = [risk_map[r, c] for r, c in path]
        return {
            "length": len(path),
            "mean_risk": np.mean(risks),
            "max_risk": np.max(risks),
            "total_risk_exposure": np.sum(risks),
        }

    print("=== NAIVE (shortest-path, risk-blind) ===")
    naive_path = a_star_search(risk_map, start, goal, risk_weight=0.0)
    naive_stats = evaluate_path(naive_path, risk_map)
    for k, v in naive_stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== DHRUVA (risk-aware) ===")
    dhruva_path = a_star_search(risk_map, start, goal, risk_weight=30.0)
    dhruva_stats = evaluate_path(dhruva_path, risk_map)
    for k, v in dhruva_stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== COMPARISON ===")
    risk_reduction = (1 - dhruva_stats["mean_risk"] / naive_stats["mean_risk"]) * 100
    length_increase = (dhruva_stats["length"] / naive_stats["length"] - 1) * 100
    print(f"  Mean risk reduced by: {risk_reduction:.1f}%")
    print(f"  Path length increased by: {length_increase:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, path, title in [(axes[0], naive_path, "Naive (Shortest Path)"),
                              (axes[1], dhruva_path, "DHRUVA (Risk-Aware)")]:
        ax.imshow(risk_map, cmap="RdYlGn_r", vmin=0, vmax=1)
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, color="blue", linewidth=2)
        ax.scatter([start[1]], [start[0]], color="lime", s=100, marker="o", zorder=5)
        ax.scatter([goal[1]], [goal[0]], color="cyan", s=100, marker="*", zorder=5)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("../outputs/baseline_comparison.png", dpi=150)
    print("\nSaved baseline_comparison.png")
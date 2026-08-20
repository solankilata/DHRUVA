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
        risk_penalty = risk_map[r, c] * risk_weight
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

    return None  # no path found


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

    start = (10, 10)
    goal = (240, 240)

    path = a_star_search(risk_map, start, goal, risk_weight=5.0)

    if path is None:
        print("No path found.")
    else:
        print(f"Path found with {len(path)} steps.")

        path_confidences = [confidence_map[r, c] for r, c in path]
        low_conf_points = [(r, c) for (r, c), conf in zip(path, path_confidences) if conf < 0.5]
        print(f"Mean path confidence: {np.mean(path_confidences):.3f}")
        print(f"Low-confidence points flagged for review: {len(low_conf_points)}")

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(risk_map, cmap="RdYlGn_r", vmin=0, vmax=1)
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        ax.plot(path_cols, path_rows, color="blue", linewidth=2, label="Planned path")
        ax.scatter([start[1]], [start[0]], color="lime", s=100, marker="o", label="Start", zorder=5)
        ax.scatter([goal[1]], [goal[0]], color="cyan", s=100, marker="*", label="Goal", zorder=5)

        if low_conf_points:
            lc_rows = [p[0] for p in low_conf_points]
            lc_cols = [p[1] for p in low_conf_points]
            ax.scatter(lc_cols, lc_rows, color="black", s=30, marker="x", label="Low confidence", zorder=6)

        ax.legend()
        ax.set_title("DHRUVA: Planned Safe Path with Risk Map")
        plt.tight_layout()
        plt.savefig("../outputs/path_planning_sample.png", dpi=150)
        print("Saved path_planning_sample.png")
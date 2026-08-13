import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import PINN


device = torch.device("cpu")

save_dir = "./checkpoints"
split_dir = "./splits"
plot_dir = "./plots"

os.makedirs(plot_dir, exist_ok=True)

checkpoint_path = os.path.join(save_dir, "final_best.pt")
split_path = os.path.join(split_dir, "train_test_split.pt")

batch_size = 1
threads_per_fold = 8


def collect_predictions(pinn, loader, device):
    pinn.eval()

    y_true = []
    y_pred = []

    for x1, x2, y1, y2 in loader:
        x1 = x1.float().to(device)
        y1 = y1.float().view(-1, 1).to(device)

        u1, _ = pinn(x1)

        y_true.append(y1.detach().cpu().numpy().reshape(-1))
        y_pred.append(u1.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return y_true, y_pred


def make_scatter_plot(y_true, y_pred, title, out_path):
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=12, alpha=0.7)
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.5)

    ax.set_xlabel("True SoH")
    ax.set_ylabel("Predicted SoH")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(threads_per_fold)
    torch.set_num_interop_threads(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    split_data = torch.load(split_path, map_location=device)

    test_dataset = TensorDataset(*split_data["test_tensors"])

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    pinn = PINN(checkpoint["config"]).to(device)
    pinn.load_state_dict(checkpoint["model_state_dict"])

    test_true, test_pred = collect_predictions(pinn, test_loader, device)

    out_path = os.path.join(plot_dir, "test_soh_pred_vs_true.png")
    make_scatter_plot(
        test_true,
        test_pred,
        "Test: Predicted SoH vs True SoH",
        out_path,
    )

    print(f"Saved: {out_path}")


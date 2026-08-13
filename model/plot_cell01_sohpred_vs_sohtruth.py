import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataloader import FeatureDataset
from model import PINN


device = torch.device("cpu")

data_path = "../data/processed_battery_data_features_cleaned/"
save_dir = "./checkpoints"
plot_dir = "./plots"

os.makedirs(plot_dir, exist_ok=True)

checkpoint_path = os.path.join(save_dir, "cell01_holdout_last.pt")

test_conditions = [
    "AG_25_100_A_01_features_cleaned_normalized__pchip.csv",
    "AG_25_100_B_01_features_cleaned_normalized__pchip.csv",
    "AG_25_100_C_01_features_cleaned_normalized__pchip.csv",
    "AG_25_30_A_01_features_cleaned_normalized__pchip.csv",
    "AG_45_100_A_01_features_cleaned_normalized__pchip.csv",
]

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


def make_condition_label(filename):
    return filename.replace("_features_cleaned_normalized__pchip.csv", "")


def make_scatter_plot(predictions_by_condition, title, out_path):
    all_true = np.concatenate([v["y_true"] for v in predictions_by_condition.values()])
    all_pred = np.concatenate([v["y_pred"] for v in predictions_by_condition.values()])

    vmin = min(all_true.min(), all_pred.min())
    vmax = max(all_true.max(), all_pred.max())

    fig, ax = plt.subplots(figsize=(7, 7))

    for label, data in predictions_by_condition.items():
        ax.scatter(
            data["y_true"],
            data["y_pred"],
            s=12,
            alpha=0.7,
            label=label,
        )

    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.5)

    ax.set_xlabel("True SoH")
    ax.set_ylabel("Predicted SoH")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(threads_per_fold)
    torch.set_num_interop_threads(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    pinn = PINN(checkpoint["config"]).to(device)
    pinn.load_state_dict(checkpoint["model_state_dict"])

    predictions_by_condition = {}

    for condition in test_conditions:
        test_dat = FeatureDataset(path=data_path, conditions=[condition])
        x1_test, x2_test, y1_test, y2_test = test_dat.load_all_cells()

        test_dataset = TensorDataset(
            torch.as_tensor(x1_test),
            torch.as_tensor(x2_test),
            torch.as_tensor(y1_test),
            torch.as_tensor(y2_test),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        y_true, y_pred = collect_predictions(pinn, test_loader, device)
        label = make_condition_label(condition)

        predictions_by_condition[label] = {
            "y_true": y_true,
            "y_pred": y_pred,
        }

    out_path = os.path.join(
        plot_dir,
        "cell01_holdout_test_soh_pred_vs_true_by_condition.png",
    )

    make_scatter_plot(
        predictions_by_condition,
        "Cell holdout test: Predicted SoH vs True SoH",
        out_path,
    )

    print(f"Saved: {out_path}")


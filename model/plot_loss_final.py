
import os
import pandas as pd
import matplotlib.pyplot as plt


history_dir = "./history"
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

history_path = os.path.join(history_dir, "cell03_holdout_history.csv")

show_plots = True
save_plots = True

metrics = [
    ("loss", "Total loss"),
    ("mse", "Data loss (MSE)"),
    ("mono", "Monotonicity loss"),
    ("pde", "PDE loss"),
]


def load_history_csv(path):
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"'epoch' column not found in {path}")
    return df.sort_values("epoch").reset_index(drop=True)


def plot_metric(df, metric_key, metric_title):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["epoch"], df[f"train_{metric_key}"], linewidth=2, label="Train")
    ax.plot(df["epoch"], df[f"test_{metric_key}"], linewidth=2, linestyle="--", label="Test")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_title)
    ax.set_yscale("log")
    ax.set_title(metric_title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_plots:
        out_name = metric_key + "_train_test.png"
        plt.savefig(os.path.join(output_dir, out_name), dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    df = load_history_csv(history_path)

    for metric_key, metric_title in metrics:
        plot_metric(df, metric_key, metric_title)

    print(f"Saved plots to: {output_dir}")

import os
import re
import pandas as pd
import matplotlib.pyplot as plt


log_dir = "./logs"
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

show_plots = True
save_plots = True

metrics = [
    ("loss", "Total loss"),
    ("mse", "Data loss (MSE)"),
    ("mono", "Monotonicity loss"),
    ("pde", "PDE loss"),
]

log_pattern = re.compile(
    r"Fold\s+(?P<fold>\d+)\s+\|\s+Epoch\s+(?P<epoch>\d+)/(?P<n_epochs>\d+)\s+\|\s+"
    r"train=(?P<train_loss>[-+eE0-9\.]+)\s+"
    r"\(mse=(?P<train_mse>[-+eE0-9\.]+),\s+mono=(?P<train_mono>[-+eE0-9\.]+),\s+pde=(?P<train_pde>[-+eE0-9\.]+)\)\s+\|\s+"
    r"val=(?P<val_loss>[-+eE0-9\.]+)\s+"
    r"\(mse=(?P<val_mse>[-+eE0-9\.]+),\s+mono=(?P<val_mono>[-+eE0-9\.]+),\s+pde=(?P<val_pde>[-+eE0-9\.]+)\)"
)


def list_fold_logs(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if re.fullmatch(r"fold_\d+\.log", f)
    )


def parse_log_file(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = log_pattern.search(line)
            if m:
                row = {
                    "fold": int(m.group("fold")),
                    "epoch": int(m.group("epoch")),
                    "n_epochs": int(m.group("n_epochs")),
                    "train_loss": float(m.group("train_loss")),
                    "train_mse": float(m.group("train_mse")),
                    "train_mono": float(m.group("train_mono")),
                    "train_pde": float(m.group("train_pde")),
                    "val_loss": float(m.group("val_loss")),
                    "val_mse": float(m.group("val_mse")),
                    "val_mono": float(m.group("val_mono")),
                    "val_pde": float(m.group("val_pde")),
                }
                rows.append(row)

    if not rows:
        raise ValueError(f"No epoch lines parsed from {path}")

    df = pd.DataFrame(rows).sort_values(["fold", "epoch"]).reset_index(drop=True)
    return df


def load_all_histories(folder):
    paths = list_fold_logs(folder)
    if not paths:
        raise RuntimeError(f"No fold log files found in {folder}")

    dfs = []
    for path in paths:
        try:
            dfs.append(parse_log_file(path))
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not dfs:
        raise RuntimeError("No valid fold logs could be parsed.")

    return pd.concat(dfs, ignore_index=True)


def plot_metric(df, metric_key, metric_title):
    fig, ax = plt.subplots(figsize=(10, 6))

    for fold, grp in df.groupby("fold"):
        grp = grp.sort_values("epoch")
        ax.plot(grp["epoch"], grp[f"train_{metric_key}"], linewidth=1.3, alpha=0.7, label=f"Train fold {fold}")
        ax.plot(grp["epoch"], grp[f"val_{metric_key}"], linewidth=1.3, alpha=0.7, linestyle="--", label=f"Val fold {fold}")

    mean_df = (
        df.groupby("epoch")[[f"train_{metric_key}", f"val_{metric_key}"]]
        .mean()
        .reset_index()
    )

    ax.plot(mean_df["epoch"], mean_df[f"train_{metric_key}"], linewidth=3, label="Train mean")
    ax.plot(mean_df["epoch"], mean_df[f"val_{metric_key}"], linewidth=3, linestyle="--", label="Val mean")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_title)
    ax.set_yscale("log")
    ax.set_title(metric_title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    plt.tight_layout()

    if save_plots:
        out_name = metric_key + "_train_val.png"
        plt.savefig(os.path.join(output_dir, out_name), dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    df = load_all_histories(log_dir)

    csv_out = os.path.join(output_dir, "parsed_training_history.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved parsed history to: {csv_out}")

    for metric_key, metric_title in metrics:
        plot_metric(df, metric_key, metric_title)

    print(f"Saved plots to: {output_dir}")


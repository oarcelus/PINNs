import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
input_dir = "./processed_battery_data"
output_path = os.path.join(input_dir, "battery_01_voltage_current_grid.png")

conditions = [
    "AG_25_100_A_01",
    "AG_25_100_B_01",
    "AG_25_100_C_01",
    "AG_25_30_A_01",
    "AG_25_80_A_01",
    "AG_45_100_A_01",
]

# Examples:
# step_types = "All"
# step_types = ["CV"]
# step_types = ["Chg", "CV"]
step_types = "All"

# Examples:
# experiment_types = "All"
# experiment_types = ["CU"]
experiment_types = ["Aging"]
# experiment_types = "All"


# =========================
# Helpers
# =========================
def load_condition_csv(folder, condition):
    file_path = os.path.join(folder, f"{condition}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV not found: {file_path}")
    return pd.read_csv(file_path)


def normalize_filter_arg(value):
    if value == "All":
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    raise ValueError(f"Unsupported filter value: {value}")


def filter_dataframe(df, step_types="All", experiment_types="All"):
    step_filter = normalize_filter_arg(step_types)
    exp_filter = normalize_filter_arg(experiment_types)

    out = df.copy()

    if step_filter is not None:
        if "StepType" not in out.columns:
            raise ValueError("Column 'StepType' not found in CSV.")
        out = out[out["StepType"].isin(step_filter)]

    if exp_filter is not None:
        if "ExperimentType" not in out.columns:
            raise ValueError("Column 'ExperimentType' not found in CSV.")
        out = out[out["ExperimentType"].isin(exp_filter)]

    return out


def build_suffix(step_types="All", experiment_types="All"):
    def fmt(x):
        if x == "All":
            return "All"
        if isinstance(x, str):
            return x
        return "-".join(x)

    return f"StepType_{fmt(step_types)}__ExperimentType_{fmt(experiment_types)}"


def plot_condition(ax_v, ax_i, df, condition, cmap_name="plasma"):
    required_cols = ["CycleID", "Voltage", "Current"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{condition}: missing columns {missing}")

    df = df.copy()
    df["CycleID"] = pd.to_numeric(df["CycleID"], errors="coerce")
    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")
    df["Current"] = pd.to_numeric(df["Current"], errors="coerce")
    df = df.dropna(subset=["CycleID", "Voltage", "Current"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{condition}: no rows left after filtering.")

    unique_cycles = np.sort(df["CycleID"].unique())
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=unique_cycles.min(), vmax=unique_cycles.max())

    for cyc in unique_cycles:
        cyc_df = df[df["CycleID"] == cyc]
        x = np.arange(len(cyc_df))
        color = cmap(norm(cyc))

        ax_v.plot(x, cyc_df["Voltage"].to_numpy(), color=color, linewidth=0.6, alpha=0.9)
        ax_i.plot(x, cyc_df["Current"].to_numpy(), color=color, linewidth=0.6, alpha=0.9)

    ax_v.set_title(f"{condition} - Voltage")
    ax_i.set_title(f"{condition} - Current")
    ax_v.set_ylabel("Voltage (V)")
    ax_i.set_ylabel("Current (A)")
    ax_i.set_xlabel("Datapoint")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


# =========================
# Main
# =========================
suffix = build_suffix(step_types=step_types, experiment_types=experiment_types)
output_path = os.path.join(input_dir, f"battery_01_voltage_current_grid__{suffix}.png")

fig, axes = plt.subplots(
    nrows=6,
    ncols=2,
    figsize=(16, 24),
    constrained_layout=True
)

scalar_mappables = []

for row_idx, condition in enumerate(conditions):
    try:
        df = load_condition_csv(input_dir, condition)
        df = filter_dataframe(
            df,
            step_types=step_types,
            experiment_types=experiment_types,
        )

        sm = plot_condition(
            axes[row_idx, 0],
            axes[row_idx, 1],
            df,
            condition,
            cmap_name="plasma",
        )
        scalar_mappables.append((row_idx, sm))

    except Exception as e:
        axes[row_idx, 0].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center")
        axes[row_idx, 1].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center")
        axes[row_idx, 0].set_title(f"{condition} - Voltage")
        axes[row_idx, 1].set_title(f"{condition} - Current")

for row_idx, sm in scalar_mappables:
    cbar = fig.colorbar(
        sm,
        ax=[axes[row_idx, 0], axes[row_idx, 1]],
        fraction=0.02,
        pad=0.01
    )
    cbar.set_label("Cycle number")

for ax in axes.flat:
    ax.grid(True, alpha=0.25)

fig.suptitle(
    f"Voltage vs Datapoint and Current vs Datapoint for *_01 Batteries\n"
    f"StepType={step_types}, ExperimentType={experiment_types}",
    fontsize=16
)

fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {output_path}")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
input_dir = "./processed_battery_data_distilled"

conditions = [
    "AG_25_100_A_01",
    "AG_25_100_B_01",
    "AG_25_100_C_01",
    "AG_25_30_A_01",
    "AG_25_80_A_01",
    "AG_45_100_A_01",
]

# Experiment filter only
# Examples:
# experiment_types = "All"
# experiment_types = ["CU"]
# experiment_types = ["Aging"]
experiment_types = ["Aging"]


# =========================
# Helpers
# =========================
def load_condition_csv(folder, condition):
    file_path = os.path.join(folder, f"{condition}_distilled.csv")
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


def filter_experiment_type(df, experiment_types="All"):
    exp_filter = normalize_filter_arg(experiment_types)
    out = df.copy()

    if exp_filter is not None:
        if "ExperimentType" not in out.columns:
            raise ValueError("Column 'ExperimentType' not found in CSV.")
        out = out[out["ExperimentType"].isin(exp_filter)]

    return out


def build_suffix(experiment_types="All"):
    def fmt(x):
        if x == "All":
            return "All"
        if isinstance(x, str):
            return x
        return "-".join(x)

    return f"ExperimentType_{fmt(experiment_types)}"


def plot_condition(ax_v, ax_i, df, condition, cmap_name="plasma"):
    required_cols = ["CycleID", "Voltage", "Current", "StepType"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{condition}: missing columns {missing}")

    df = df.copy()
    df["CycleID"] = pd.to_numeric(df["CycleID"], errors="coerce")
    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")
    df["Current"] = pd.to_numeric(df["Current"], errors="coerce")
    df["StepType"] = df["StepType"].astype(str).str.strip()

    # Split by step type for each panel
    df_cc = df[df["StepType"] == "CC Chg"].dropna(subset=["CycleID", "Voltage"]).reset_index(drop=True)
    df_cv = df[df["StepType"] == "CV Chg"].dropna(subset=["CycleID", "Current"]).reset_index(drop=True)

    if df_cc.empty:
        raise ValueError(f"{condition}: no CC Chg rows left for voltage plot.")
    if df_cv.empty:
        raise ValueError(f"{condition}: no CV Chg rows left for current plot.")

    unique_cycles = np.sort(
        np.unique(
            np.concatenate([
                df_cc["CycleID"].unique(),
                df_cv["CycleID"].unique()
            ])
        )
    )

    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=unique_cycles.min(), vmax=unique_cycles.max())

    # Voltage panel: only CC Chg
    for cyc in np.sort(df_cc["CycleID"].unique()):
        cyc_df = df_cc[df_cc["CycleID"] == cyc]
        volt = cyc_df["Voltage"].to_numpy()

        if volt[0] > 3.9:
            print(condition, cyc)
        x = np.arange(len(cyc_df))
        color = cmap(norm(cyc))
        ax_v.plot(x, cyc_df["Voltage"].to_numpy(), color=color, linewidth=0.8, alpha=0.95)

    # Current panel: only CV Chg
    for cyc in np.sort(df_cv["CycleID"].unique()):
        cyc_df = df_cv[df_cv["CycleID"] == cyc]
        curr = cyc_df["Current"].to_numpy()
        if curr[0] < 1e-3:
            print(condition, cyc)
        x = np.arange(len(cyc_df))
        color = cmap(norm(cyc))
        ax_i.plot(x, cyc_df["Current"].to_numpy(), color=color, linewidth=0.8, alpha=0.95)

    ax_v.set_title(f"{condition} - Voltage (CC Chg)")
    ax_i.set_title(f"{condition} - Current (CV Chg)")
    ax_v.set_ylabel("Voltage (V)")
    ax_i.set_ylabel("Current (A)")
    ax_i.set_xlabel("Datapoint")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


# =========================
# Main
# =========================
suffix = build_suffix(experiment_types=experiment_types)
output_path = os.path.join(
    input_dir,
    f"battery_01_voltage_current_grid_distilled__{suffix}.png"
)

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
        df = filter_experiment_type(df, experiment_types=experiment_types)

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
        axes[row_idx, 0].set_title(f"{condition} - Voltage (CC Chg)")
        axes[row_idx, 1].set_title(f"{condition} - Current (CV Chg)")

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
    f"Distilled data for *_01 batteries\n"
    f"Voltage from CC Chg, Current from CV Chg, ExperimentType={experiment_types}",
    fontsize=16
)

fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {output_path}")


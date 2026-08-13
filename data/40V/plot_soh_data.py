import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
input_dir = "./processed_battery_data"

nominal_capacity_ah = 1.3
save_figure = True
figure_path = os.path.join(input_dir, "soh_trajectories_cu_cycles.png")

# Only repetitions 01, 02, 03
battery_suffixes = {"01", "02", "03"}


# =========================
# Helpers
# =========================
def list_condition_files(folder):
    return sorted(
        f for f in os.listdir(folder)
        if f.endswith(".csv") and re.match(r"^AG_.*_(01|02|03)\.csv$", f)
    )


def split_condition_name(filename):
    """
    Example:
        AG_25_100_A_01.csv
    -> stem           = AG_25_100_A_01
       base_condition = AG_25_100_A
       battery_id     = 01
    """
    stem = filename[:-4]
    parts = stem.split("_")
    battery_id = parts[-1]
    base_condition = "_".join(parts[:-1])
    return stem, base_condition, battery_id


def get_discharge_rows(df):
    """
    Prefer StepType containing discharge information.
    Otherwise fall back to Current < 0.
    """
    if "StepType" in df.columns:
        step = df["StepType"].astype(str).str.strip().str.lower()
        dch = df[
            step.isin(["dch", "cc dchg", "cv dchg"]) |
            step.str.contains("dchg", na=False)
        ].copy()
        if not dch.empty:
            return dch

    if "Current" in df.columns:
        cur = pd.to_numeric(df["Current"], errors="coerce")
        dch = df[cur < 0].copy()
        if not dch.empty:
            return dch

    raise ValueError("Could not identify discharge rows. Need StepType discharge labels or Current<0.")


def extract_cu_blocks(df):
    """
    Returns list of contiguous CU blocks.
    """
    if "ExperimentType" not in df.columns:
        raise ValueError("Column 'ExperimentType' not found.")

    exp = df["ExperimentType"].astype(str).str.strip()
    is_cu = exp.eq("CU")

    if not is_cu.any():
        return []

    blocks = []
    idx = np.flatnonzero(is_cu.to_numpy())

    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            blocks.append(df.iloc[start:prev + 1].copy())
            start = i
            prev = i
    blocks.append(df.iloc[start:prev + 1].copy())

    return blocks


def capacity_from_cycle(cycle_df):
    """
    Return the discharge capacity at the end of 'CC DChg'
    immediately before the first 'Rest'.

    If the 'CC DChg' -> 'Rest' transition is not found,
    fall back to the last capacity value in 'CC DChg'.
    """
    required_cols = {"StepType", "Capacity"}
    missing = required_cols - set(cycle_df.columns)
    if missing:
        raise ValueError(f"cycle_df must contain columns: {sorted(missing)}")

    tmp = cycle_df.copy()
    tmp["StepType"] = tmp["StepType"].astype(str).str.strip()
    tmp["Capacity"] = pd.to_numeric(tmp["Capacity"], errors="coerce")
    tmp = tmp.dropna(subset=["Capacity"]).reset_index(drop=True)

    if tmp.empty:
        return np.nan

    step = tmp["StepType"].str.lower()

    for i in range(len(tmp) - 1):
        if step.iloc[i] == "cc dchg" and step.iloc[i + 1] == "rest":
            return abs(tmp["Capacity"].iloc[i])

    cc_dchg_rows = tmp[step == "cc dchg"]
    if not cc_dchg_rows.empty:
        return abs(cc_dchg_rows["Capacity"].iloc[-1])

    return abs(tmp["Capacity"].iloc[-1])


def extract_soh_trajectory_from_file(file_path, nominal_capacity_ah=1.3):
    """
    Rules:
    - For the 1st CU block: use discharge capacity from the 5th cycle
    - For all subsequent CU blocks: use discharge capacity from the 3rd cycle
    - Capacity is taken at the end of 'CC DChg', just before 'Rest'
    """
    df = pd.read_csv(file_path)

    required_cols = {"ExperimentType", "CycleID", "Capacity", "StepType"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["CycleID"] = pd.to_numeric(df["CycleID"], errors="coerce")
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df = df.dropna(subset=["CycleID", "Capacity"]).reset_index(drop=True)

    cu_blocks = extract_cu_blocks(df)
    records = []

    for cu_idx, cu_block in enumerate(cu_blocks, start=1):
        dch_block = get_discharge_rows(cu_block)

        cycle_ids = pd.Index(
            dch_block["CycleID"].dropna().astype(int)
        ).drop_duplicates().tolist()

        target_local_cycle = 5 if cu_idx == 1 else 3

        if len(cycle_ids) < target_local_cycle:
            print(
                f"Warning: {os.path.basename(file_path)} | CU {cu_idx} "
                f"has only {len(cycle_ids)} discharge cycles. Skipping."
            )
            continue

        target_cycle_id = cycle_ids[target_local_cycle - 1]
        cycle_df = cu_block[cu_block["CycleID"].astype(int) == int(target_cycle_id)].copy()

        discharge_capacity_ah = capacity_from_cycle(cycle_df)
        soh = discharge_capacity_ah / nominal_capacity_ah if pd.notna(discharge_capacity_ah) else np.nan

        records.append({
            "Checkupnumber": cu_idx,
            "SoH": soh,
            "Capacity": discharge_capacity_ah,
            "Cycle Number": int(target_cycle_id),
        })

    return pd.DataFrame(records)


# =========================
# Build all trajectories + cache one CSV per cell
# =========================
files = list_condition_files(input_dir)

all_trajs = []
for fname in files:
    stem, base_condition, battery_id = split_condition_name(fname)

    if battery_id not in battery_suffixes:
        continue

    source_file_path = os.path.join(input_dir, fname)
    cache_file_path = os.path.join(input_dir, f"{stem}_soh.csv")

    try:
        if os.path.exists(cache_file_path):
            traj = pd.read_csv(cache_file_path)
            print(f"Loaded cached table: {cache_file_path}")
        else:
            traj = extract_soh_trajectory_from_file(
                file_path=source_file_path,
                nominal_capacity_ah=nominal_capacity_ah,
            )
            traj["Condition"] = stem

            export_df = traj[["Checkupnumber", "SoH", "Capacity", "Cycle Number", "Condition"]].copy()
            export_df.to_csv(cache_file_path, index=False)
            print(f"Computed and saved table: {cache_file_path}")

        traj["Condition"] = stem
        traj["BaseCondition"] = base_condition
        traj["BatteryID"] = battery_id
        all_trajs.append(traj)

    except Exception as e:
        print(f"Error processing {fname}: {e}")

if not all_trajs:
    raise RuntimeError("No trajectories could be extracted.")

soh_df = pd.concat(all_trajs, ignore_index=True)


# =========================
# Plot
# =========================
base_conditions = sorted(soh_df["BaseCondition"].unique())

cmap = plt.get_cmap("tab10")
color_map = {
    base_cond: cmap(i % 10)
    for i, base_cond in enumerate(base_conditions)
}

style_map = {
    "01": {"linestyle": "-",  "marker": "o"},
    "02": {"linestyle": "--", "marker": "s"},
    "03": {"linestyle": ":",  "marker": "^"},
}

fig, ax = plt.subplots(figsize=(13, 8))

for (base_cond, battery_id), grp in soh_df.groupby(["BaseCondition", "BatteryID"]):
    grp = grp.sort_values("Cycle Number")

    ax.plot(
        grp["Cycle Number"],
        grp["SoH"],
        color=color_map[base_cond],
        linestyle=style_map[battery_id]["linestyle"],
        marker=style_map[battery_id]["marker"],
        linewidth=2,
        markersize=5,
        alpha=0.95,
        label=f"{base_cond}_{battery_id}",
    )

ax.set_xlabel("Cycle number")
ax.set_ylabel("State of Health (SOH)")
ax.set_title("SOH trajectories from selected CU discharge cycles")
ax.set_ylim(0.75, 1.0)
ax.grid(True, alpha=0.3)

ax.axhline(0.8, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax.text(
    0.99, 0.802, "80% SOH",
    color="red",
    ha="right", va="bottom",
    transform=ax.get_yaxis_transform()
)

ax.legend(ncol=3, fontsize=9, frameon=True, loc="best")

plt.tight_layout()

if save_figure:
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {figure_path}")

plt.show()


# =========================
# Save combined SOH table
# =========================
combined_cols = ["Checkupnumber", "SoH", "Capacity", "Cycle Number", "Condition"]
soh_csv_path = os.path.join(input_dir, "soh_trajectories_selected_cu_cycles.csv")
soh_df[combined_cols].to_csv(soh_csv_path, index=False)
print(f"Combined SOH table saved to: {soh_csv_path}")


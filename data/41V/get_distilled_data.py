import os
import pandas as pd


# =========================
# User settings
# =========================
input_dir = "./processed_battery_data"
output_dir = "./processed_battery_data_distilled"
os.makedirs(output_dir, exist_ok=True)

conditions = [
    "AG_25_100_A_01",
    "AG_25_100_A_02",
    "AG_25_100_A_03",
    "AG_25_100_B_01",
    "AG_25_100_B_02",
    "AG_25_100_B_03",
    "AG_25_100_C_01",
    "AG_25_100_C_02",
    "AG_25_100_C_03",
    "AG_25_30_A_01",
    "AG_25_30_A_02",
    "AG_25_30_A_03",
    "AG_25_80_A_01",
    "AG_25_80_A_02",
    "AG_25_80_A_03",
    "AG_45_100_A_01",
    "AG_45_100_A_02",
    "AG_45_100_A_03",
]

cc_voltage_window = 0.3   # keep CC Chg rows with Voltage in [Vmax-0.3, Vmax]
cv_current_window = 0.25  # keep CV Chg rows with Current in [Imin, Imin+0.25]


# =========================
# Helpers
# =========================
def load_condition_df(input_dir, condition):
    csv_path = os.path.join(input_dir, f"{condition}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def validate_columns(df, condition):
    required_cols = [
        "CycleID",
        "StepType",
        "Current",
        "Capacity",
        "Voltage",
        "Time",
        "TotalTime",
        "Condition",
        "ExperimentType",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{condition} missing required columns: {missing}")


def distill_cycle(
    cycle_df,
    cc_voltage_window=0.3,
    cv_current_window=0.25,
    v_max_min_threshold=4.0,
    i_min_threshold=0.011,
    cv_current_offset=0.01,
):
    """
    Keep only cycles that pass global validity checks:
    - CC Chg must reach at least v_max_min_threshold
    - CV Chg must have minimum current at least i_min_threshold

    If either check fails, drop the whole cycle.

    For valid cycles, keep only:
    - CC Chg rows with Voltage in [Vmax - cc_voltage_window, Vmax]
    - CV Chg rows with Current in [Imin + cv_current_offset, Imin + cv_current_window]

    Returns a dataframe with the same columns as input.
    """
    step = cycle_df["StepType"].astype(str).str.strip()

    cc_df = cycle_df[step.eq("CC Chg")].copy()
    cv_df = cycle_df[step.eq("CV Chg")].copy()

    # Must have both parts
    if cc_df.empty or cv_df.empty:
        return cycle_df.iloc[0:0].copy()

    cc_df["Voltage"] = pd.to_numeric(cc_df["Voltage"], errors="coerce")
    cv_df["Current"] = pd.to_numeric(cv_df["Current"], errors="coerce")

    cc_df = cc_df.dropna(subset=["Voltage"])
    cv_df = cv_df.dropna(subset=["Current"])

    if cc_df.empty or cv_df.empty:
        return cycle_df.iloc[0:0].copy()

    vmax = cc_df["Voltage"].max()
    imin = cv_df["Current"].min()

    # -------- global cycle validity checks --------
    if vmax < v_max_min_threshold:
        return cycle_df.iloc[0:0].copy()

    if imin < i_min_threshold:
        return cycle_df.iloc[0:0].copy()

    # -------- keep only distilled windows --------
    cc_keep = cc_df[
        (cc_df["Voltage"] >= vmax - cc_voltage_window) &
        (cc_df["Voltage"] <= vmax)
    ].copy()

    cv_keep = cv_df[
        (cv_df["Current"] >= imin + cv_current_offset) &
        (cv_df["Current"] <= imin + cv_current_window)
    ].copy()

    out_parts = []
    if not cc_keep.empty:
        out_parts.append(cc_keep)
    if not cv_keep.empty:
        out_parts.append(cv_keep)

    if out_parts:
        distilled = pd.concat(out_parts, ignore_index=False)
        distilled = distilled.sort_index().reset_index(drop=True)
    else:
        distilled = cycle_df.iloc[0:0].copy()

    return distilled


def distill_condition_df(df, condition, cc_voltage_window=0.3, cv_current_window=0.25):
    """
    Distill one condition while ensuring all cycles from 1..Nmax are iterated over.
    Cycles with no matching rows will simply contribute no rows to the distilled output.
    """
    validate_columns(df, condition)

    df = df.copy()
    df["CycleID"] = pd.to_numeric(df["CycleID"], errors="coerce")
    df["Current"] = pd.to_numeric(df["Current"], errors="coerce")
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["TotalTime"] = pd.to_numeric(df["TotalTime"], errors="coerce")
    df = df.dropna(subset=["CycleID"])

    nmax = int(df["CycleID"].max())
    distilled_cycles = []

    iaging = 0
    for cyc in range(1, nmax + 1):
        cycle_df = df[df["CycleID"].astype(int) == cyc].copy()

        if cycle_df["ExperimentType"].iloc[0] == "Aging" and iaging == 0:
            iaging += 1
            continue
        
        if cycle_df.empty:
            # cycle number is included in iteration, but there is no data for it
            continue

        distilled_cycle_df = distill_cycle(
            cycle_df,
            cc_voltage_window=cc_voltage_window,
            cv_current_window=cv_current_window,
        )
        distilled_cycles.append(distilled_cycle_df)

    if distilled_cycles:
        distilled_df = pd.concat(distilled_cycles, ignore_index=True)
    else:
        distilled_df = df.iloc[0:0].copy()

    return distilled_df, nmax


# =========================
# Main
# =========================
for condition in conditions:
    try:
        df = load_condition_df(input_dir, condition)

        distilled_df, nmax = distill_condition_df(
            df,
            condition,
            cc_voltage_window=cc_voltage_window,
            cv_current_window=cv_current_window,
        )

        # Save with same column structure
        pkl_path = os.path.join(output_dir, f"{condition}_distilled.pkl")
        csv_path = os.path.join(output_dir, f"{condition}_distilled.csv")

        distilled_df.to_pickle(pkl_path)
        distilled_df.to_csv(csv_path, index=False)

        print(f"{condition}: original rows={len(df)}, distilled rows={len(distilled_df)}, cycles checked=1..{nmax}")
        print(f"  saved: {pkl_path}")
        print(f"  saved: {csv_path}")

    except Exception as e:
        print(f"Error processing {condition}: {e}")

print("Done.")

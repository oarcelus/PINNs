import os
import re
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


# =========================
# User settings
# =========================
distilled_dir = "./processed_battery_data_distilled"
soh_interp_dir = "./processed_battery_data/"
output_dir = "./processed_battery_data_features"
os.makedirs(output_dir, exist_ok=True)

# Choose the interpolated SOH file suffix
# examples:
# "pchip", "linear", "cubic", "spline", "exp_decay"
soh_method = "pchip"


# =========================
# Helpers
# =========================
def list_distilled_files(folder):
    return sorted(
        f for f in os.listdir(folder)
        if f.endswith("_distilled.csv") and re.match(r"^AG_.*_(01|02|03)_distilled\.csv$", f)
    )


def stem_from_distilled_filename(fname):
    # AG_25_100_A_01_distilled.csv -> AG_25_100_A_01
    return fname[:-14]


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").dropna()


def safe_kurtosis(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    if np.allclose(x, x[0]):
        return 0.0
    return float(kurtosis(x, fisher=True, bias=False))


def safe_skew(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    if np.allclose(x, x[0]):
        return 0.0
    return float(skew(x, bias=False))


def safe_entropy(x, bins=20):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    if np.allclose(x, x[0]):
        return 0.0

    hist, _ = np.histogram(x, bins=min(bins, max(2, len(np.unique(x)))))
    p = hist.astype(float) / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def slope_vs_index(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan

    idx = np.arange(n, dtype=float)
    coeffs = np.polyfit(idx, x, 1)
    return float(coeffs[0])


def last_minus_first(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return float(x[-1] - x[0])


def load_soh_interpolated(stem, soh_interp_dir, soh_method):
    path = os.path.join(
        soh_interp_dir,
        f"{stem}_soh_interpolated__{soh_method}.csv"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Interpolated SOH file not found: {path}")

    df = pd.read_csv(path)
    required = ["Cycle Number", "SoH"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df["Cycle Number"] = pd.to_numeric(df["Cycle Number"], errors="coerce")
    df["SoH"] = pd.to_numeric(df["SoH"], errors="coerce")
    df = df.dropna(subset=["Cycle Number", "SoH"]).copy()
    df["Cycle Number"] = df["Cycle Number"].astype(int)
    return df[["Cycle Number", "SoH"]]


def compute_cycle_features(cycle_df):
    """
    Uses:
    - CC Chg rows for voltage-side descriptors
    - CV Chg rows for current-side descriptors
    """
    cycle_df = cycle_df.copy()
    cycle_df["StepType"] = cycle_df["StepType"].astype(str).str.strip()

    cc = cycle_df[cycle_df["StepType"] == "CC Chg"].copy()
    cv = cycle_df[cycle_df["StepType"] == "CV Chg"].copy()

    # numeric conversions
    for df_part in (cc, cv):
        if not df_part.empty:
            for col in ["Voltage", "Current", "Capacity", "Time", "TotalTime"]:
                if col in df_part.columns:
                    df_part[col] = pd.to_numeric(df_part[col], errors="coerce")

    # -------- Voltage-side features from CC Chg --------
    v = safe_numeric(cc["Voltage"]) if not cc.empty else pd.Series(dtype=float)
    cap_cc = safe_numeric(cc["Capacity"]) if not cc.empty else pd.Series(dtype=float)
    time_cc = safe_numeric(cc["Time"]) if not cc.empty else pd.Series(dtype=float)

    voltage_mean = float(v.mean()) if len(v) else np.nan
    voltage_std = float(v.std(ddof=1)) if len(v) > 1 else np.nan
    voltage_kurtosis = safe_kurtosis(v)
    voltage_skewness = safe_skew(v)
    capacity = float(cap_cc.max()) if len(cap_cc) else np.nan
    charge_time = float(time_cc.max() - time_cc.min()) if len(time_cc) > 1 else np.nan
    voltage_slope = last_minus_first(v) / last_minus_first(time_cc)
    voltage_entropy = safe_entropy(v)

    # -------- Current-side features from CV Chg --------
    i = safe_numeric(cv["Current"]) if not cv.empty else pd.Series(dtype=float)
    cap_cv = safe_numeric(cv["Capacity"]) if not cv.empty else pd.Series(dtype=float)
    time_cv = safe_numeric(cv["Time"]) if not cv.empty else pd.Series(dtype=float)

    current_mean = float(i.mean()) if len(i) else np.nan
    current_std = float(i.std(ddof=1)) if len(i) > 1 else np.nan
    current_kurtosis = safe_kurtosis(i)
    current_skewness = safe_skew(i)
    capacity_cv = float(cap_cv.max() - cap_cv.min()) if len(cap_cv) > 1 else (float(cap_cv.max()) if len(cap_cv) else np.nan)
    cv_time = float(time_cv.max() - time_cv.min()) if len(time_cv) > 1 else np.nan
    current_slope = last_minus_first(i) / last_minus_first(time_cv)
    current_entropy = safe_entropy(i)

    return {
        "Voltage mean": voltage_mean,
        "Voltage standard deviation": voltage_std,
        "Voltage kurtosis": voltage_kurtosis,
        "Voltage skewness": voltage_skewness,
        "Capacity": capacity,
        "Charge time": charge_time,
        "Voltage slope": voltage_slope,
        "Voltage entropy": voltage_entropy,
        "Current mean": current_mean,
        "Current standard deviation": current_std,
        "Current kurtosis": current_kurtosis,
        "Current skewness": current_skewness,
        "Capacity of CV": capacity_cv,
        "CV time": cv_time,
        "Current slope": current_slope,
        "Current entropy": current_entropy,
    }


# =========================
# Main
# =========================
distilled_files = list_distilled_files(distilled_dir)
if not distilled_files:
    raise RuntimeError(f"No distilled CSV files found in {distilled_dir}")

for fname in distilled_files:
    stem = stem_from_distilled_filename(fname)
    distilled_path = os.path.join(distilled_dir, fname)

    print(f"Processing {stem} ...")

    df = pd.read_csv(distilled_path)
    required_cols = ["CycleID", "StepType", "Voltage", "Current", "Capacity", "Time"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Skipping {stem}: missing columns {missing}")
        continue

    df["CycleID"] = pd.to_numeric(df["CycleID"], errors="coerce")
    df = df.dropna(subset=["CycleID"]).copy()
    df["CycleID"] = df["CycleID"].astype(int)

    # Compute descriptors cycle by cycle
    feature_rows = []
    for cyc in sorted(df["CycleID"].unique()):
        cycle_df = df[df["CycleID"] == cyc].copy()
        feats = compute_cycle_features(cycle_df)
        feats["Cycle Number"] = int(cyc)
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows)

    # Load interpolated SOH and merge
    try:
        soh_df = load_soh_interpolated(stem, soh_interp_dir, soh_method)
    except Exception as e:
        print(f"Skipping SOH merge for {stem}: {e}")
        continue

    out_df = pd.merge(
        feat_df,
        soh_df,
        on="Cycle Number",
        how="left"
    )

    # Reorder columns
    ordered_cols = [
        "Cycle Number",
        "SoH",
        "Voltage mean",
        "Voltage standard deviation",
        "Voltage kurtosis",
        "Voltage skewness",
        "Capacity",
        "Charge time",
        "Voltage slope",
        "Voltage entropy",
        "Current mean",
        "Current standard deviation",
        "Current kurtosis",
        "Current skewness",
        "Capacity of CV",
        "CV time",
        "Current slope",
        "Current entropy",
    ]
    out_df = out_df[ordered_cols]

    # -------------------------
    # Min-max normalize features to [-1, 1]
    # -------------------------
    feature_cols = [
        "Voltage mean",
        "Voltage standard deviation",
        "Voltage kurtosis",
        "Voltage skewness",
        "Capacity",
        "Charge time",
        "Voltage slope",
        "Voltage entropy",
        "Current mean",
        "Current standard deviation",
        "Current kurtosis",
        "Current skewness",
        "Capacity of CV",
        "CV time",
        "Current slope",
        "Current entropy",
    ]

    norm_df = out_df.copy()

    for col in feature_cols:
        col_values = pd.to_numeric(norm_df[col], errors="coerce")
        col_min = col_values.min()
        col_max = col_values.max()

        if pd.isna(col_min) or pd.isna(col_max):
            norm_df[col] = np.nan
        elif np.isclose(col_max, col_min):
            # constant feature -> map to 0
            norm_df[col] = 0.0
        else:
            norm_df[col] = 2.0 * (col_values - col_min) / (col_max - col_min) - 1.0

    # Save
    out_csv = os.path.join(output_dir, f"{stem}_features__{soh_method}.csv")
    out_pkl = os.path.join(output_dir, f"{stem}_features__{soh_method}.pkl")

    out_df.to_csv(out_csv, index=False)
    out_df.to_pickle(out_pkl)

    # Save normalized descriptors
    norm_csv = os.path.join(output_dir, f"{stem}_features_normalized__{soh_method}.csv")
    norm_pkl = os.path.join(output_dir, f"{stem}_features_normalized__{soh_method}.pkl")

    norm_df.to_csv(norm_csv, index=False)
    norm_df.to_pickle(norm_pkl)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_pkl}")
    print(f"Saved: {norm_csv}")
    print(f"Saved: {norm_pkl}")



import os
import re
import numpy as np
import pandas as pd


# =========================
# User settings
# =========================
feature_dir = "./processed_battery_data_features"
output_dir = "./processed_battery_data_features_cleaned"
os.makedirs(output_dir, exist_ok=True)

soh_method = "pchip"


# =========================
# Feature list
# =========================
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

base_cols = ["Cycle Number", "SoH"] + feature_cols


# =========================
# Helpers
# =========================
def list_feature_files(folder, soh_method):
    suffix = f"_features__{soh_method}.csv"
    pattern = r"^AG_.*_(01|02|03)_features"
    return sorted(
        f for f in os.listdir(folder)
        if f.endswith(suffix) and re.match(pattern, f)
    )


def three_sigma_indices(series):
    """
    Return row indices where values lie outside mean ± 3*std.
    """
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std()

    if pd.isna(mean) or pd.isna(std) or np.isclose(std, 0.0):
        return []

    mask = (s < mean - 3 * std) | (s > mean + 3 * std)
    return list(np.flatnonzero(mask.to_numpy()))


def delete_3_sigma(df):
    """
    Drop rows that are outliers in any column using the 3-sigma rule.
    """
    work = df.copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna().reset_index(drop=True)

    outlier_idx = []
    for col in work.columns:
        outlier_idx.extend(three_sigma_indices(work[col]))

    outlier_idx = sorted(set(outlier_idx))

    if len(outlier_idx) > 0:
        work = work.drop(index=outlier_idx).reset_index(drop=True)

    return work


def recompute_minmax_normalized(df, feature_cols):
    norm_df = df.copy()

    for col in feature_cols:
        values = pd.to_numeric(norm_df[col], errors="coerce")
        vmin = values.min()
        vmax = values.max()

        if pd.isna(vmin) or pd.isna(vmax):
            norm_df[col] = np.nan
        elif np.isclose(vmin, vmax):
            norm_df[col] = 0.0
        else:
            norm_df[col] = 2.0 * (values - vmin) / (vmax - vmin) - 1.0

    return norm_df


# =========================
# Main
# =========================
raw_files = list_feature_files(feature_dir, soh_method)

if not raw_files:
    raise RuntimeError(f"No raw feature files found in {feature_dir}")

for fname in raw_files:
    stem = fname.replace(f"_features__{soh_method}.csv", "")
    raw_path = os.path.join(feature_dir, fname)

    print(f"Cleaning {stem} ...")

    df = pd.read_csv(raw_path)

    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        print(f"Skipping {stem}: missing columns {missing}")
        continue

    df = df[base_cols].copy()
    df["Cycle Number"] = pd.to_numeric(df["Cycle Number"], errors="coerce")
    df = df.dropna(subset=["Cycle Number"]).copy()
    df = df.sort_values("Cycle Number").reset_index(drop=True)

    # 3-sigma cleaning
    cleaned_df = delete_3_sigma(df)

    cleaned_raw_csv = os.path.join(
        output_dir,
        f"{stem}_features_cleaned__{soh_method}.csv"
    )
    cleaned_raw_pkl = os.path.join(
        output_dir,
        f"{stem}_features_cleaned__{soh_method}.pkl"
    )

    cleaned_df.to_csv(cleaned_raw_csv, index=False)
    cleaned_df.to_pickle(cleaned_raw_pkl)

    print(f"Saved cleaned raw features: {cleaned_raw_csv}")
    print(f"Saved cleaned raw features: {cleaned_raw_pkl}")

    # Normalized cleaned version
    cleaned_norm_df = recompute_minmax_normalized(cleaned_df, feature_cols)

    cleaned_norm_csv = os.path.join(
        output_dir,
        f"{stem}_features_cleaned_normalized__{soh_method}.csv"
    )
    cleaned_norm_pkl = os.path.join(
        output_dir,
        f"{stem}_features_cleaned_normalized__{soh_method}.pkl"
    )

    cleaned_norm_df.to_csv(cleaned_norm_csv, index=False)
    cleaned_norm_df.to_pickle(cleaned_norm_pkl)

    print(f"Saved cleaned normalized features: {cleaned_norm_csv}")
    print(f"Saved cleaned normalized features: {cleaned_norm_pkl}")

print("Done.")


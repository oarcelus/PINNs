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

# -------- Pass 1 cleaning hyperparameters --------
rolling_window_pass1 = 11
mad_threshold_pass1 = 4.0
min_flagged_features_pass1 = 4

# -------- Pass 2 cleaning hyperparameters --------
rolling_window_pass2 = 11
mad_threshold_pass2 = 4.0
min_flagged_features_pass2 = 4


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


def rolling_median_and_mad(series, window):
    med = series.rolling(window=window, center=True, min_periods=3).median()

    def mad_func(x):
        x = pd.Series(x).dropna()
        if len(x) == 0:
            return np.nan
        m = np.median(x)
        return np.median(np.abs(x - m))

    mad = series.rolling(window=window, center=True, min_periods=3).apply(mad_func, raw=False)
    med = med.bfill().ffill()
    mad = mad.bfill().ffill()
    return med, mad


def detect_outlier_cycles(df, feature_cols, rolling_window=11, mad_threshold=4.0, min_flagged_features=4):
    """
    Return a boolean mask of cycles to keep.
    """
    work = df.copy()
    n_flags = np.zeros(len(work), dtype=int)

    for col in feature_cols:
        s = pd.to_numeric(work[col], errors="coerce")
        med, mad = rolling_median_and_mad(s, rolling_window)

        mad = mad.where(mad > 1e-12, np.nan)
        score = (s - med).abs() / mad
        flag = (score > mad_threshold).fillna(False).to_numpy()

        n_flags += flag.astype(int)

    keep_mask = n_flags < min_flagged_features
    return keep_mask


def run_cleaning_pass(df, feature_cols, rolling_window, mad_threshold, min_flagged_features):
    keep_mask = detect_outlier_cycles(
        df=df,
        feature_cols=feature_cols,
        rolling_window=rolling_window,
        mad_threshold=mad_threshold,
        min_flagged_features=min_flagged_features,
    )
    return df.loc[keep_mask, base_cols].copy()


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

    df["Cycle Number"] = pd.to_numeric(df["Cycle Number"], errors="coerce")
    df = df.dropna(subset=["Cycle Number"]).copy()
    df = df.sort_values("Cycle Number").reset_index(drop=True)
    df = df[base_cols].copy()

    # Pass 1
    cleaned_df_pass1 = run_cleaning_pass(
        df=df,
        feature_cols=feature_cols,
        rolling_window=rolling_window_pass1,
        mad_threshold=mad_threshold_pass1,
        min_flagged_features=min_flagged_features_pass1,
    )

    cleaned_raw_pass1_csv = os.path.join(
        output_dir,
        f"{stem}_features_cleaned_pass1__{soh_method}.csv"
    )
    cleaned_raw_pass1_pkl = os.path.join(
        output_dir,
        f"{stem}_features_cleaned_pass1__{soh_method}.pkl"
    )

    cleaned_df_pass1.to_csv(cleaned_raw_pass1_csv, index=False)
    cleaned_df_pass1.to_pickle(cleaned_raw_pass1_pkl)

    print(f"Saved pass-1 cleaned raw features: {cleaned_raw_pass1_csv}")
    print(f"Saved pass-1 cleaned raw features: {cleaned_raw_pass1_pkl}")

    # Pass 2
    cleaned_df_pass2 = run_cleaning_pass(
        df=cleaned_df_pass1,
        feature_cols=feature_cols,
        rolling_window=rolling_window_pass2,
        mad_threshold=mad_threshold_pass2,
        min_flagged_features=min_flagged_features_pass2,
    )

    cleaned_raw_pass2_csv = os.path.join(
        output_dir,
        f"{stem}_features_cleaned__{soh_method}.csv"
    )
    cleaned_raw_pass2_pkl = os.path.join(
        output_dir,
        f"{stem}_features_cleaned__{soh_method}.pkl"
    )

    cleaned_df_pass2.to_csv(cleaned_raw_pass2_csv, index=False)
    cleaned_df_pass2.to_pickle(cleaned_raw_pass2_pkl)

    print(f"Saved final cleaned raw features: {cleaned_raw_pass2_csv}")
    print(f"Saved final cleaned raw features: {cleaned_raw_pass2_pkl}")

    # Final normalized cleaned version
    cleaned_norm_df = recompute_minmax_normalized(cleaned_df_pass2, feature_cols)

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

    print(f"Saved final cleaned normalized features: {cleaned_norm_csv}")
    print(f"Saved final cleaned normalized features: {cleaned_norm_pkl}")

print("Done.")


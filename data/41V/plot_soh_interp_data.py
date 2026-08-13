import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d, PchipInterpolator, UnivariateSpline
from scipy.optimize import curve_fit


# =========================
# User settings
# =========================
soh_input_dir = "./processed_battery_data"
full_data_dir = "./processed_battery_data"   # original per-cell full CSVs, used to get Ncycl_max
output_dir = soh_input_dir

# Available:
# "linear"
# "pchip"
# "cubic"
# "spline"
# "exp_decay"
interpolation_method = "pchip"

save_plots = True
plot_path = os.path.join(output_dir, f"soh_interpolation_grid__{interpolation_method}.png")


# =========================
# Helpers
# =========================
def list_soh_files(folder):
    return sorted(
        f for f in os.listdir(folder)
        if f.endswith("_soh.csv") and re.match(r"^AG_.*_(01|02|03)_soh\.csv$", f)
    )


def stem_from_soh_filename(fname):
    # AG_25_100_A_01_soh.csv -> AG_25_100_A_01
    return fname[:-8]


def split_condition_name(stem):
    parts = stem.split("_")
    battery_id = parts[-1]
    base_condition = "_".join(parts[:-1])
    return stem, base_condition, battery_id


def get_ncycle_max(stem, full_data_dir):
    source_csv = os.path.join(full_data_dir, f"{stem}.csv")
    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Original processed CSV not found: {source_csv}")

    df = pd.read_csv(source_csv, usecols=["CycleID"])
    cycle_id = pd.to_numeric(df["CycleID"], errors="coerce").dropna()
    if cycle_id.empty:
        raise ValueError(f"No valid CycleID in {source_csv}")
    return int(cycle_id.max())


def clean_xy(df):
    x = pd.to_numeric(df["Cycle Number"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["SoH"], errors="coerce").to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # sort and deduplicate x
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if len(x) == 0:
        raise ValueError("No valid SOH points found.")

    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) != len(x):
        y_avg = np.zeros_like(unique_x, dtype=float)
        counts = np.zeros_like(unique_x, dtype=float)
        for i, idx in enumerate(inverse):
            y_avg[idx] += y[i]
            counts[idx] += 1
        y = y_avg / counts
        x = unique_x

    return x.astype(float), y.astype(float)


def exp_decay_model(x, a, b, c):
    # flexible monotone-ish decay if b > 0
    return c + a * np.exp(-b * x)


def interpolate_soh(x, y, x_new, method="pchip"):
    if len(x) == 1:
        return np.full_like(x_new, y[0], dtype=float)

    if method == "linear":
        f = interp1d(x, y, kind="linear", fill_value="extrapolate", bounds_error=False)
        y_new = f(x_new)

    elif method == "pchip":
        f = PchipInterpolator(x, y, extrapolate=True)
        y_new = f(x_new)

    elif method == "cubic":
        if len(x) < 4:
            f = interp1d(x, y, kind="linear", fill_value="extrapolate", bounds_error=False)
        else:
            f = interp1d(x, y, kind="cubic", fill_value="extrapolate", bounds_error=False)
        y_new = f(x_new)

    elif method == "spline":
        k = min(3, len(x) - 1)
        if k < 1:
            return np.full_like(x_new, y[0], dtype=float)
        # light smoothing; adjust if you want stronger smoothing
        s = max(1e-6, 0.0005 * len(x))
        f = UnivariateSpline(x, y, k=k, s=s)
        y_new = f(x_new)

    elif method == "exp_decay":
        if len(x) < 3:
            f = interp1d(x, y, kind="linear", fill_value="extrapolate", bounds_error=False)
            y_new = f(x_new)
        else:
            a0 = max(y[0] - y[-1], 1e-3)
            b0 = 1e-3
            c0 = max(min(y[-1], 1.0), 0.0)
            try:
                popt, _ = curve_fit(
                    exp_decay_model,
                    x,
                    y,
                    p0=[a0, b0, c0],
                    maxfev=20000
                )
                y_new = exp_decay_model(x_new, *popt)
            except Exception:
                f = interp1d(x, y, kind="linear", fill_value="extrapolate", bounds_error=False)
                y_new = f(x_new)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return np.asarray(y_new, dtype=float)


def maybe_clip_soh(y):
    # conservative clipping to keep values in a reasonable SOH range
    return np.clip(y, 0.0, 1.2)


# =========================
# Main processing
# =========================
soh_files = list_soh_files(soh_input_dir)
if not soh_files:
    raise RuntimeError(f"No *_soh.csv files found in {soh_input_dir}")

all_interpolated = []

for fname in soh_files:
    stem = stem_from_soh_filename(fname)
    soh_path = os.path.join(soh_input_dir, fname)

    df = pd.read_csv(soh_path)
    required_cols = ["Checkupnumber", "SoH", "Capacity", "Cycle Number", "Condition"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{fname} is missing columns: {missing}")

    ncycle_max = get_ncycle_max(stem, full_data_dir)
    x, y = clean_xy(df)

    x_new = np.arange(1, ncycle_max + 1, dtype=float)
    y_new = interpolate_soh(x, y, x_new, method=interpolation_method)
    y_new = maybe_clip_soh(y_new)

    # Build output with same columns as original data
    out = pd.DataFrame({
        "Checkupnumber": np.nan,
        "SoH": y_new,
        "Capacity": y_new * 1.3,
        "Cycle Number": x_new.astype(int),
        "Condition": df["Condition"].iloc[0],
    })

    # Put original check-up numbers back on exact measured cycles
    checkup_map = dict(zip(pd.to_numeric(df["Cycle Number"], errors="coerce"),
                           pd.to_numeric(df["Checkupnumber"], errors="coerce")))
    out["Checkupnumber"] = out["Cycle Number"].map(checkup_map)

    out_path = os.path.join(output_dir, f"{stem}_soh_interpolated__{interpolation_method}.csv")
    out.to_csv(out_path, index=False)

    out["BaseStem"] = stem
    all_interpolated.append((df.copy(), out.copy(), stem))

    print(f"Saved interpolated CSV: {out_path}")


# =========================
# Plotting
# =========================
n_files = len(all_interpolated)
ncols = 3
nrows = int(np.ceil(n_files / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4.5 * nrows), squeeze=False)

for ax, item in zip(axes.flat, all_interpolated):
    original_df, interp_df, stem = item

    x_meas = pd.to_numeric(original_df["Cycle Number"], errors="coerce")
    y_meas = pd.to_numeric(original_df["SoH"], errors="coerce")

    x_interp = pd.to_numeric(interp_df["Cycle Number"], errors="coerce")
    y_interp = pd.to_numeric(interp_df["SoH"], errors="coerce")

    ax.plot(x_interp, y_interp, linewidth=2, label=f"{interpolation_method} interpolation")
    ax.scatter(x_meas, y_meas, s=30, zorder=3, label="Measured SOH points")

    ax.set_title(stem)
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("SoH")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0.0, min(y_interp.min(), y_meas.min()) - 0.05),
                top=min(1.2, max(y_interp.max(), y_meas.max()) + 0.05))

    # keep legend compact
    ax.legend(fontsize=8)

# Hide unused axes
for ax in axes.flat[n_files:]:
    ax.axis("off")

fig.suptitle(f"SOH interpolation per cell/condition - method: {interpolation_method}", fontsize=16)
plt.tight_layout()

if save_plots:
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot grid: {plot_path}")

plt.show()


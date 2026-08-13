import os
import re
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
soh_method = "pchip"

# "raw" or "cleaned"
data_version = "cleaned"

# Only used when data_version == "cleaned"
# "pass1" -> *_features_cleaned_pass1__<method>.csv
# "final" -> *_features_cleaned__<method>.csv
cleaned_pass = "final"

use_normalized = True   # False -> raw feature values, True -> normalized feature values

# Include filters
# conditions = "All"
conditions = ["AG_25_100_A_01", "AG_25_100_A_02", "AG_25_100_A_03"]
batteries = "All"

# Exclude filters
exclude_conditions = "AG_25_80_A_03"
exclude_batteries = "None"

# Color mode:
# "condition" -> color by full condition name
# "battery"   -> color by battery id 01/02/03
# "base"      -> color by base condition (e.g. AG_25_100_A)
color_by = "battery"

save_figure = True


# =========================
# Resolve feature directory
# =========================
if data_version == "raw":
    feature_dir = "./processed_battery_data_features"
elif data_version == "cleaned":
    feature_dir = "./processed_battery_data_features_cleaned"
else:
    raise ValueError("data_version must be 'raw' or 'cleaned'")

if cleaned_pass not in {"pass1", "final"}:
    raise ValueError("cleaned_pass must be 'pass1' or 'final'")


# =========================
# Feature layout
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

feature_grid = [
    ["Voltage mean", "Voltage standard deviation", "Voltage kurtosis", "Voltage skewness"],
    ["Capacity", "Charge time", "Voltage slope", "Voltage entropy"],
    ["Current mean", "Current standard deviation", "Current kurtosis", "Current skewness"],
    ["Capacity of CV", "CV time", "Current slope", "Current entropy"],
]


# =========================
# Helpers
# =========================
def normalize_filter_arg(value, none_token="None"):
    if value == "All":
        return None
    if value == none_token:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    raise ValueError(f"Unsupported filter value: {value}")


def get_feature_suffix(soh_method, data_version="raw", use_normalized=False, cleaned_pass="final"):
    if data_version == "raw":
        return f"_features_normalized__{soh_method}.csv" if use_normalized else f"_features__{soh_method}.csv"

    if data_version == "cleaned":
        if cleaned_pass == "pass1":
            return (
                f"_features_cleaned_pass1_normalized__{soh_method}.csv"
                if use_normalized
                else f"_features_cleaned_pass1__{soh_method}.csv"
            )
        elif cleaned_pass == "final":
            return (
                f"_features_cleaned_normalized__{soh_method}.csv"
                if use_normalized
                else f"_features_cleaned__{soh_method}.csv"
            )

    raise ValueError("Invalid combination of data_version / cleaned_pass")


def list_feature_files(folder, soh_method, data_version="raw", use_normalized=False, cleaned_pass="final"):
    suffix = get_feature_suffix(
        soh_method,
        data_version=data_version,
        use_normalized=use_normalized,
        cleaned_pass=cleaned_pass,
    )
    return sorted(
        f for f in os.listdir(folder)
        if f.endswith(suffix) and re.match(r"^AG_.*_(01|02|03)_features", f)
    )


def stem_from_feature_filename(fname, soh_method, data_version="raw", use_normalized=False, cleaned_pass="final"):
    suffix = get_feature_suffix(
        soh_method,
        data_version=data_version,
        use_normalized=use_normalized,
        cleaned_pass=cleaned_pass,
    )
    return fname[:-len(suffix)]


def split_condition_name(stem):
    parts = stem.split("_")
    battery_id = parts[-1]
    base_condition = "_".join(parts[:-1])
    return stem, base_condition, battery_id


def load_feature_dataframe(folder, fname, soh_method, data_version="raw", use_normalized=False, cleaned_pass="final"):
    path = os.path.join(folder, fname)
    df = pd.read_csv(path)

    stem = stem_from_feature_filename(
        fname,
        soh_method,
        data_version=data_version,
        use_normalized=use_normalized,
        cleaned_pass=cleaned_pass,
    )
    condition, base_condition, battery_id = split_condition_name(stem)

    df["Condition"] = condition
    df["BaseCondition"] = base_condition
    df["BatteryID"] = battery_id
    return df


def filter_feature_data(
    df,
    conditions="All",
    batteries="All",
    exclude_conditions="None",
    exclude_batteries="None",
):
    cond_filter = normalize_filter_arg(conditions)
    batt_filter = normalize_filter_arg(batteries)
    excl_cond_filter = normalize_filter_arg(exclude_conditions, none_token="None")
    excl_batt_filter = normalize_filter_arg(exclude_batteries, none_token="None")

    out = df.copy()

    if cond_filter is not None:
        out = out[out["Condition"].isin(cond_filter)]

    if batt_filter is not None:
        out = out[out["BatteryID"].isin(batt_filter)]

    if len(excl_cond_filter) > 0:
        out = out[~out["Condition"].isin(excl_cond_filter)]

    if len(excl_batt_filter) > 0:
        out = out[~out["BatteryID"].isin(excl_batt_filter)]

    return out


def build_output_suffix(
    conditions="All",
    batteries="All",
    exclude_conditions="None",
    exclude_batteries="None",
    color_by="base",
    use_normalized=False,
    data_version="raw",
    cleaned_pass="final",
):
    def fmt(x):
        if x == "All":
            return "All"
        if x == "None":
            return "None"
        if isinstance(x, str):
            return x
        return "-".join(x)

    norm_txt = "normalized" if use_normalized else "rawvalues"
    if data_version == "cleaned":
        version_txt = f"{data_version}_{cleaned_pass}"
    else:
        version_txt = data_version

    return (
        f"{version_txt}"
        f"__{norm_txt}"
        f"__conditions_{fmt(conditions)}"
        f"__batteries_{fmt(batteries)}"
        f"__exclude_conditions_{fmt(exclude_conditions)}"
        f"__exclude_batteries_{fmt(exclude_batteries)}"
        f"__colorby_{color_by}"
    )

def build_color_map(df, color_by="base"):
    if color_by == "condition":
        groups = sorted(df["Condition"].dropna().unique())
        key_col = "Condition"
    elif color_by == "battery":
        groups = sorted(df["BatteryID"].dropna().unique())
        key_col = "BatteryID"
    elif color_by == "base":
        groups = sorted(df["BaseCondition"].dropna().unique())
        key_col = "BaseCondition"
    else:
        raise ValueError("color_by must be one of: 'condition', 'battery', 'base'")

    cmap = plt.get_cmap("tab20")
    color_map = {g: cmap(i % 20) for i, g in enumerate(groups)}
    return key_col, color_map


# =========================
# Load and merge all files
# =========================
feature_files = list_feature_files(
    feature_dir,
    soh_method,
    data_version=data_version,
    use_normalized=use_normalized,
    cleaned_pass=cleaned_pass,
)
if not feature_files:
    raise RuntimeError(f"No feature CSV files found in {feature_dir}")

all_dfs = []
for fname in feature_files:
    try:
        df = load_feature_dataframe(
            feature_dir,
            fname,
            soh_method,
            data_version=data_version,
            use_normalized=use_normalized,
            cleaned_pass=cleaned_pass,
        )
        all_dfs.append(df)
    except Exception as e:
        print(f"Skipping {fname}: {e}")

if not all_dfs:
    raise RuntimeError("No valid feature data could be loaded.")

full_df = pd.concat(all_dfs, ignore_index=True)

required_cols = ["Cycle Number", "SoH"] + feature_cols + ["Condition", "BaseCondition", "BatteryID"]
missing = [c for c in required_cols if c not in full_df.columns]
if missing:
    raise ValueError(f"Missing required columns in merged feature dataframe: {missing}")

plot_df = filter_feature_data(
    full_df,
    conditions=conditions,
    batteries=batteries,
    exclude_conditions=exclude_conditions,
    exclude_batteries=exclude_batteries,
)

if plot_df.empty:
    raise RuntimeError("No rows left after filtering.")

plot_df["SoH"] = pd.to_numeric(plot_df["SoH"], errors="coerce")
for col in feature_cols:
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

plot_df = plot_df.dropna(subset=["SoH"]).copy()

key_col, color_map = build_color_map(plot_df, color_by=color_by)


# =========================
# Plot 4x4 tile
# =========================
fig, axes = plt.subplots(4, 4, figsize=(18, 18), constrained_layout=True)

for r in range(4):
    for c in range(4):
        ax = axes[r, c]
        feat = feature_grid[r][c]

        sub = plot_df.dropna(subset=[feat]).copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(feat)
            ax.set_xlabel("SoH")
            ax.set_ylabel(feat)
            ax.grid(True, alpha=0.25)
            continue

        for group_name, grp in sub.groupby(key_col):
            ax.scatter(
                grp["SoH"],
                grp[feat],
                s=12,
                alpha=0.65,
                color=color_map[group_name],
                label=str(group_name),
            )

        ax.set_title(feat)
        ax.set_xlabel("SoH")
        ax.set_ylabel(feat)
        ax.grid(True, alpha=0.25)

handles_dict = {}
for group_name, color in color_map.items():
    handles_dict[group_name] = plt.Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        color=color,
        markersize=6,
        label=str(group_name),
    )

fig.legend(
    handles=list(handles_dict.values()),
    labels=[str(k) for k in handles_dict.keys()],
    loc="upper center",
    ncol=min(6, len(handles_dict)),
    frameon=True
)

norm_txt = "normalized" if use_normalized else "raw feature values"
version_txt = f"{data_version} ({cleaned_pass})" if data_version == "cleaned" else data_version

fig.suptitle(
    f"SoH vs descriptors\n"
    f"data_version={version_txt}, values={norm_txt}\n"
    f"conditions={conditions}, batteries={batteries}, "
    f"exclude_conditions={exclude_conditions}, exclude_batteries={exclude_batteries}, "
    f"color_by={color_by}",
    fontsize=16
)

if save_figure:
    suffix = build_output_suffix(
        conditions=conditions,
        batteries=batteries,
        exclude_conditions=exclude_conditions,
        exclude_batteries=exclude_batteries,
        color_by=color_by,
        use_normalized=use_normalized,
        data_version=data_version,
        cleaned_pass=cleaned_pass,
    )
    out_path = os.path.join(feature_dir, f"soh_vs_features_4x4__{suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

plt.show()


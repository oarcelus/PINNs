import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
soh_method = "pchip"

# "raw" or "cleaned"
data_version = "cleaned"

# Only used when data_version == "cleaned"
# "pass1" or "final"
cleaned_pass = "final"

use_normalized = True   # False -> raw feature values, True -> normalized feature values

# Include filters
conditions = "All"
batteries = "All"

# Exclude filters
exclude_conditions = "None"
exclude_batteries = "None"

# Correlation method: "pearson", "spearman", "kendall"
corr_method = "pearson"

group_mode = "base"
# options:
# "auto"          -> Condition, unless only one condition remains, then BatteryID
# "condition"     -> exact cell-condition, e.g. AG_25_100_A_01
# "battery"       -> battery id only, e.g. 01 / 02 / 03
# "base"          -> aggregate 01/02/03 together, e.g. AG_25_100_A
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


def choose_grouping_column(df, group_mode="auto"):
    if group_mode == "condition":
        return "Condition"
    elif group_mode == "battery":
        return "BatteryID"
    elif group_mode == "base":
        return "BaseCondition"
    elif group_mode == "auto":
        unique_conditions = sorted(df["Condition"].dropna().unique())
        if len(unique_conditions) <= 1:
            return "BatteryID"
        return "Condition"
    else:
        raise ValueError("group_mode must be one of: 'auto', 'condition', 'battery', 'base'")


def build_output_suffix(
    conditions="All",
    batteries="All",
    exclude_conditions="None",
    exclude_batteries="None",
    use_normalized=False,
    data_version="raw",
    cleaned_pass="final",
    corr_method="pearson",
    group_col="Condition",
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
    version_txt = f"{data_version}_{cleaned_pass}" if data_version == "cleaned" else data_version

    return (
        f"{version_txt}"
        f"__{norm_txt}"
        f"__corr_{corr_method}"
        f"__groupby_{group_col}"
        f"__conditions_{fmt(conditions)}"
        f"__batteries_{fmt(batteries)}"
        f"__exclude_conditions_{fmt(exclude_conditions)}"
        f"__exclude_batteries_{fmt(exclude_batteries)}"
    )


def compute_group_correlations(df, feature_cols, group_col="Condition", corr_method="pearson"):
    rows = []
    for group_name, grp in df.groupby(group_col):
        grp = grp.copy()
        grp["SoH"] = pd.to_numeric(grp["SoH"], errors="coerce")

        corr_row = {"Group": group_name}
        for feat in feature_cols:
            x = pd.to_numeric(grp[feat], errors="coerce")
            y = grp["SoH"]

            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                corr = np.nan
            else:
                corr = x[valid].corr(y[valid], method=corr_method)

            corr_row[feat] = corr
        rows.append(corr_row)

    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        raise RuntimeError("No correlation rows could be computed.")

    corr_df = corr_df.set_index("Group")
    return corr_df


def draw_heatmap(corr_df, title, save_path=None):
    data = corr_df.to_numpy(dtype=float)

    fig_w = max(14, 0.8 * corr_df.shape[1])
    fig_h = max(4, 0.7 * corr_df.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(corr_df.shape[1]))
    ax.set_yticks(np.arange(corr_df.shape[0]))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)

    # annotate values
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            val = data[i, j]
            txt = "nan" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation with SoH")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()


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

group_col = choose_grouping_column(plot_df, group_mode=group_mode)

corr_df = compute_group_correlations(
    plot_df,
    feature_cols=feature_cols,
    group_col=group_col,
    corr_method=corr_method,
)

version_txt = f"{data_version} ({cleaned_pass})" if data_version == "cleaned" else data_version
norm_txt = "normalized" if use_normalized else "raw feature values"

title = (
    f"Descriptor–SoH correlation map\n"
    f"data_version={version_txt}, values={norm_txt}, corr={corr_method}, group_by={group_col}\n"
    f"group_mode={group_mode}, conditions={conditions}, batteries={batteries}, "
    f"conditions={conditions}, batteries={batteries}, "
    f"exclude_conditions={exclude_conditions}, exclude_batteries={exclude_batteries}"
)

save_path = None
if save_figure:
    suffix = build_output_suffix(
        conditions=conditions,
        batteries=batteries,
        exclude_conditions=exclude_conditions,
        exclude_batteries=exclude_batteries,
        use_normalized=use_normalized,
        data_version=data_version,
        cleaned_pass=cleaned_pass,
        corr_method=corr_method,
        group_col=group_col,
    )
    save_path = os.path.join(feature_dir, f"descriptor_soh_correlation_map__{suffix}.png")

draw_heatmap(corr_df, title=title, save_path=save_path)

# Optional: save correlation table too
corr_csv_path = os.path.join(
    feature_dir,
    f"descriptor_soh_correlation_map__{build_output_suffix(conditions, batteries, exclude_conditions, exclude_batteries, use_normalized, data_version, cleaned_pass, corr_method, group_col)}.csv"
)
corr_df.to_csv(corr_csv_path)
print(f"Saved correlation table to: {corr_csv_path}")


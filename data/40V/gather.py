import os
import pandas as pd


aging_path = "path to cycling data"
checkup_path = "path to refference periodic checkup data"

# In this script we get the full cell history by concatenating different aging and checkup data files

age_number = [
    "/Aging 1",
    "/Aging 2",
    "/Aging 3",
    "/Aging 4",
    "/Aging 5",
    "/Aging 6",
    "/Aging 7",
    "/Aging 8",
    "/Aging 9",
    "/Aging 10",
    "/Aging 11",
    "/Aging 12",
    "/Aging 13",
    "/Aging 14",
    "/Aging 15",
    "/Aging 16",
    "/Aging 17",
    "/Aging 18",
    "/Aging 19",
    "/Aging 20",
    "/Aging 21",
    "/Aging 22",
    "/Aging 23",
    "/Aging 24",
]

cu_path = [
    "/Initial check up",
    "/Check up 1",
    "/Check up 2",
    "/Check up 3",
    "/Check up 4",
    "/Check up 5",
    "/Check up 6",
    "/Check up 7",
    "/Check up 8",
    "/Check up 9",
    "/Check up 10",
    "/Check up 11",
    "/Check up 12",
    "/Check up 13",
    "/Check up 14",
    "/Check up 15",
    "/Check up 16",
    "/Check up 17",
    "/Check up 18",
    "/Check up 19",
    "/Check up 20",
    "/Check up 21",
    "/Check up 22",
    "/Check up 23",
    "/Check up 24",
]

aging_join_path = [aging_path + folder for folder in age_number]
cu_join_path = [checkup_path + folder for folder in cu_path]

intercalated = []
for cu_folder, aging_folder in zip(cu_join_path[:-1], aging_join_path):
    intercalated.append((cu_folder, "CU"))
    intercalated.append((aging_folder, "Aging"))
intercalated.append((cu_join_path[-1], "CU"))

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


def find_column(df, candidates):
    return next((c for c in candidates if c in df.columns), None)


def normalize_current_to_A(df, current_col):
    if current_col == "Current(mA)":
        return pd.to_numeric(df[current_col], errors="coerce") / 1000.0
    if current_col == "Current(A)":
        return pd.to_numeric(df[current_col], errors="coerce")
    raise ValueError(f"Unsupported current column: {current_col}")


def normalize_capacity_to_Ah(df, capacity_col):
    if capacity_col == "Capacity(mAh)":
        return pd.to_numeric(df[capacity_col], errors="coerce") / 1000.0
    if capacity_col == "Capacity(Ah)":
        return pd.to_numeric(df[capacity_col], errors="coerce")
    raise ValueError(f"Unsupported capacity column: {capacity_col}")


all_conditions_data = []
output_dir = "./processed_battery_data"
os.makedirs(output_dir, exist_ok=True)
condition_data_dict = {}

for condition in conditions:
    condition_frames = []
    cycle_offset = 0
    total_time_offset = 0.0

    for path, experiment_type in intercalated:
        try:
            filename = next(
                file for file in os.listdir(path)
                if condition in file and file.endswith(".xlsx")
            )
            full_file_path = os.path.join(path, filename)

            data = pd.read_excel(full_file_path, sheet_name="record")

            cycle_col = find_column(data, ["Cycle ID", "Cycle Index"])
            curr_col = find_column(data, ["Current(mA)", "Current(A)"])
            cap_col = find_column(data, ["Capacity(mAh)", "Capacity(Ah)"])
            vol_col = find_column(data, ["Voltage(V)"])
            time_col = find_column(data, ["Time"])
            step_col = find_column(data, ["Step Type"])
            tottime_col = find_column(data, ["Total Time"])

            cols = [cycle_col, curr_col, cap_col, vol_col, step_col, time_col, tottime_col]
            if any(col is None for col in cols):
                raise ValueError(f"Missing mandatory column(s) in file: {full_file_path}")

            df = data[cols].copy()

            # Normalize units
            current_A = normalize_current_to_A(df, curr_col)
            capacity_Ah = normalize_capacity_to_Ah(df, cap_col)

            # Build standardized dataframe
            out = pd.DataFrame({
                "CycleID": pd.to_numeric(df[cycle_col], errors="coerce"),
                "StepType": df[step_col],
                "Current": current_A,
                "Capacity": capacity_Ah,
                "Voltage": pd.to_numeric(df[vol_col], errors="coerce"),
                "Time": pd.to_timedelta(df[time_col], errors="coerce").dt.total_seconds(),
                "TotalTime": pd.to_timedelta(df[tottime_col], errors="coerce").dt.total_seconds(),
            })

            # Make CycleID cumulative across files for the same condition
            out["CycleID"] = out["CycleID"] + cycle_offset

            # Make TotalTime cumulative across files for the same condition
            out["TotalTime"] = out["TotalTime"] + total_time_offset

            # Add metadata
            out["Condition"] = condition
            out["ExperimentType"] = experiment_type

            condition_frames.append(out)

            # Update offsets
            cycle_offset = int(out["CycleID"].max())
            total_time_offset = float(out["TotalTime"].max())

            print(f"Loaded: {full_file_path}")

        except StopIteration:
            print(f"File not found for {condition} in {path}")
        except Exception as e:
            print(f"Error in {path} - {condition}: {e}")

    if condition_frames:
        condition_df = pd.concat(condition_frames, ignore_index=True)
        condition_data_dict[condition] = condition_df

        safe_condition = condition.replace("/", "_").replace(" ", "_")
        pkl_path = os.path.join(output_dir, f"{safe_condition}.pkl")
        csv_path = os.path.join(output_dir, f"{safe_condition}.csv")

        condition_df.to_pickle(pkl_path)
        condition_df.to_csv(csv_path, index=False)

        print(f"Saved {condition} to:")
        print(f"  {pkl_path}")
        print(f"  {csv_path}")

if all_conditions_data:
    full_df = pd.concat(all_conditions_data, ignore_index=True)
else:
    full_df = pd.DataFrame()

print("Done.")
print(f"Full dataframe shape: {full_df.shape}")


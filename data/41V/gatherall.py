import numpy as np
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

directory_path = [
    "path to data",
]

output_file = "capacities.csv"

# Load already processed files if output exists
if os.path.exists(output_file):
    existing = pd.read_csv(output_file)
    processed_files = set(existing["file"].tolist())
else:
    existing = pd.DataFrame(columns=["file", "dch_cap"])
    processed_files = set()

for j, path in enumerate(directory_path):
    xlsxpaths = [
        os.path.join(path, file)
        for file in os.listdir(path)
        if file.endswith(".xlsx")
    ]

    for i, excel in enumerate(xlsxpaths):
        if xlsxpaths[i] in processed_files:
            continue

        try:
            data = pd.read_excel(excel, sheet_name="step")
            step_col = next((c for c in ["Cycle ID", "Cycle Index"] if c in data.columns), None)

            if step_col is None:
                raise ValueError(f"No cycle column found in {excel}")

            capacity_data = data[data[step_col].isin([5])]

            if len(capacity_data) < 3:
                raise ValueError(f"Not enough matching rows in {excel}")

            dch_cap = capacity_data["Capacity(mAh)"].iloc[-3]

            # Save this file's result immediately
            row = pd.DataFrame([{"file": xlsxpaths[i], "dch_cap": dch_cap}])
            row.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)

            processed_files.add(excel)
            print(f"Saved: {excel}")

        except Exception as e:
            print(f"Failed on {excel}: {e}")

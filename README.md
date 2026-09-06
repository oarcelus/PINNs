# PINNs for lithium-ion battery state-of-health estimation

This repository prepares battery cycling records, derives near-full-charge features, creates cycle-level state-of-health (SoH) labels, and trains a PyTorch physics-informed neural network (PINN) to estimate SoH. It is a script-oriented research codebase: source workbooks and generated datasets are intentionally not committed, and there is not yet a single command that runs the entire workflow.

> **Scope of this README.** This is a code-review-derived guide to the behavior currently implemented in the repository. It explains the intended workflow and calls out material implementation caveats; it does not claim that a result is a complete reproduction of the source paper or scientifically validated without the original data and experiment protocol.

## Scientific reference

The high-level approach is inspired by Wang et al., who combine a feature-to-SoH neural network with a learned degradation-dynamics network and use automatic differentiation to impose a physics-informed residual. Their features come from a short constant-current/constant-voltage charging interval near full charge, and their model uses cycle as its time coordinate. See:

> Wang, F., Zhai, Z., Zhao, Z. et al. *Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis*. **Nature Communications 15**, 4332 (2024). [https://doi.org/10.1038/s41467-024-48779-z](https://doi.org/10.1038/s41467-024-48779-z)

The repository follows that broad design but is an adaptation, not an asserted line-for-line reproduction. Its data windows, preprocessing thresholds, loss implementation, and current derivative-coordinate handling differ in important ways described below.

## Repository map

| Path | Purpose |
| --- | --- |
| `data/40V/`, `data/41V/` | Parallel, mostly identical data-preparation and exploratory-analysis scripts. Their directory names are not accompanied by a repository-level description of their scientific distinction. |
| `cfg/pinn_config.json` | Network widths, depths, dropout, and tensor-dimension configuration. |
| `model/dataloader.py` | Reads final feature CSVs and converts each cell trajectory into adjacent-cycle training pairs. |
| `model/model.py` | PyTorch MLPs, SoH predictor, dynamics network, autograd residual, and JSON configuration loader. |
| `model/train.py`, `model/train_final.py` | Random paired-sample split, cross-validation, and final training on a saved split. |
| `model/train_cell_0{1,2,3}_holdout.py` | Explicit file-list cell/repetition holdout experiments. |
| `model/plot_*.py` | Training-loss and predicted-versus-true SoH plots. |
| `agents/` | Reusable Architect, Implementer, Code Reviewer, and Tester role specifications. |

## End-to-end flow

```text
external Aging and Check-up Excel workbooks
  -> standardized per-condition cycling records
  -> (a) sparse SoH measurements at check-ups -> cycle-level interpolated SoH labels
  -> (b) distilled CC/CV charge windows -> 16 cycle descriptors
  -> merge descriptors with interpolated SoH
  -> clean and normalize descriptors
  -> adjacent-cycle pairs per cell
  -> PINN training, checkpoints, histories, and plots
```

The two middle branches are independent after collection: SoH interpolation starts from check-up discharge capacity, while feature extraction starts from distilled charge segments. They meet in `extract_features.py`.

## Data preparation pipeline

Run one pipeline variant at a time. All scripts use paths relative to the **current working directory**, not the script location.

- Running from `data/` (for example, `python 40V/gather.py`) writes the `data/processed_*` locations expected by the default model scripts, but the 40V and 41V variants then share and can overwrite the same outputs.
- Running inside `data/40V/` or `data/41V/` keeps their generated directories separate, but you must update the model scripts' `data_path` to point to that chosen cleaned-output directory.
- `gather.py` starts from a hard-coded external `/mnt/o/...` source location. Configure those paths and condition lists for the data available in your environment before running it.

### Suggested order

From the chosen data-pipeline working directory, use the following order. The plotting scripts are optional diagnostics.

```text
gather.py
plot_initial_data.py                         # optional raw-data QA
get_distilled_data.py
plot_initial_data_distilled.py               # optional window QA
plot_soh_data.py
plot_soh_interp_data.py
extract_features.py
clean_features_morepasses.py                 # choose this OR clean_features_3sigma.py
plot_feature_relations.py                    # optional
plot_correlation_map.py                      # optional
```

### What each data script does

| Stage | Script(s) | Current behavior and output |
| --- | --- | --- |
| Collect and standardize | `gather.py` | Reads `record` worksheets from interleaved Aging and Check-up Excel folders for configured `AG_*` conditions. It standardizes `CycleID`, `StepType`, current in A, capacity in Ah, voltage, elapsed time, total time, condition, and `ExperimentType`; it offsets cycle and total time across source files and writes `processed_battery_data/<condition>.csv` and `.pkl`. |
| Optional check-up scan | `gatherall.py` | Reads `step` worksheets from the initial check-up, extracts one discharge-capacity value per file, and appends `file,dch_cap` rows to `capacities.csv`. No other checked-in script consumes this file, so it is not part of the training path. |
| Inspect raw records | `plot_initial_data.py` | Plots voltage/current traces by cycle, with optional step-type and CU/Aging filtering. It is diagnostic only. |
| Distill near-full-charge data | `get_distilled_data.py` | Keeps valid cycles with both `CC Chg` and `CV Chg`: CC must reach at least 4.0 V and CV minimum current must be at least 0.011 A. It retains CC voltage in `[Vmax - 0.3, Vmax]` and CV current in `[Imin + 0.01, Imin + 0.25]`, then writes `processed_battery_data_distilled/<condition>_distilled.csv` and `.pkl`. |
| Inspect distilled data | `plot_initial_data_distilled.py` | Plots the selected CC-voltage and CV-current windows. It is diagnostic only. |
| Make sparse measured SoH points | `plot_soh_data.py` | Finds contiguous check-up (CU) blocks in each standardized record. It takes the fifth discharge cycle in the initial CU block and the third in later blocks, obtains discharge capacity at the `CC DChg` to `Rest` boundary (with fallbacks), and calculates `SoH = abs(capacity_Ah) / 1.3`. It writes `<condition>_soh.csv` and `soh_trajectories_selected_cu_cycles.csv`. |
| Interpolate SoH by cycle | `plot_soh_interp_data.py` | Cleans and deduplicates sparse check-up points, then creates an SoH label for every integer cycle from 1 through the maximum `CycleID`. The default is PCHIP; linear, cubic, spline, and exponential-decay alternatives are available. Output is `<condition>_soh_interpolated__pchip.csv` by default, plus a comparison plot. |
| Extract descriptors | `extract_features.py` | Computes 16 descriptors per distilled cycle, merges them with the interpolated SoH label, and writes raw plus per-file min-max-normalized feature CSV/PKL files in `processed_battery_data_features/`. |
| Clean descriptors | `clean_features_3sigma.py` or `clean_features_morepasses.py` | The 3-sigma option removes rows outlying in any column. The multi-pass option uses two rolling median/MAD passes and removes cycles flagged in at least four descriptors. Both write the same final `processed_battery_data_features_cleaned/*_features_cleaned[_normalized]__pchip.*` names, so run exactly one cleaner unless outputs are isolated. |
| Explore relationships | `plot_feature_relations.py`, `plot_correlation_map.py` | Produce scatter plots and correlation tables for descriptor/SoH analysis. They do not change model inputs. |

### Feature-table contract

The final CSV is positional as well as named. `model/dataloader.py` expects this order:

| Columns | Meaning | Used as |
| --- | --- | --- |
| 0 | `Cycle Number` | Model input |
| 1 | `SoH` | Supervised target |
| 2-9 | CC-voltage descriptors: mean, standard deviation, kurtosis, skewness, capacity, charge time, slope, entropy | Model inputs |
| 10-17 | CV-current descriptors: mean, standard deviation, kurtosis, skewness, capacity of CV, CV time, slope, entropy | Model inputs |

Only the 16 descriptors are min-max normalized to `[-1, 1]`; cycle number and SoH are intentionally retained as separate columns. For one cell with rows ordered by cycle, the loader makes overlapping adjacent pairs:

```text
(x[0], x[1], y[0], y[1]), (x[1], x[2], y[1], y[2]), ...
```

Consequently, every input file must be sorted and must contain at least two valid rows.

### Interpolation and data-quality interpretation

The PCHIP values between check-ups are generated labels, not directly measured SoH values. The current implementation may extrapolate at the ends, clips outputs to `[0, 1.2]`, and does not explicitly enforce monotonic degradation. Treat interpolation method, capacity reference, and cleaning rule as experimental choices that need to be recorded with any result.

The paper selects a near-full-charge interval too, but its reported windows are not the same as this code's `Vmax - 0.3` and relative-CV-current thresholds. Do not describe the local preprocessing as identical to the paper without checking the intended experiment.

## Model architecture

For each feature row, the current configuration has 17 inputs: cycle number plus 16 descriptors. `SoHNetwork` maps those inputs to a scalar prediction `u`; it consists of:

- an encoder MLP, configured as `17 -> 64 -> 64 -> 64 -> 32` with Tanh activations and dropout in intermediate hidden layers;
- a predictor MLP, configured as `32 -> 32 -> 1`, with input dropout and Tanh; and
- Xavier initialization for linear layers.

The PINN also has a dynamics MLP, configured as `35 -> 64 -> 64 -> 64 -> 1`. In `PINN.forward`, the code calculates:

```text
u       = SoHNetwork(xt)
du_dxt  = autograd.grad(sum(u), xt)
ux      = du_dxt[:, :-1]
ut      = du_dxt[:, -1:]
dynamics_input = concat(xt, u, ux, ut)      # 17 + 1 + 16 + 1 = 35 columns
residual = ut - DynamicsNetwork(dynamics_input)
```

The training code calls the residual `l` and minimizes it toward zero. It uses two Adam optimizers: one for the SoH network (`lr_u = 1e-4`) and one for the dynamics network (`lr_f = 1e-3`). Default runs use CPU, batch size 1, 2,000 epochs, and the architecture in `cfg/pinn_config.json`.

For a pair of adjacent cycles, the implementation computes:

```text
L_data = 0.5 * MSE(u1, y1) + 0.5 * MSE(u2, y2)
L_mono = sum(ReLU((u2 - u1) * (y2 - y1)))
L_pde  = 0.5 * MSE(l1, 0) + 0.5 * MSE(l2, 0)
L      = L_data + 0.7 * L_mono + 0.2 * L_pde
```

Evaluation must leave autograd enabled because `PINN.forward` calculates a derivative even in `eval()` mode. Do not wrap residual evaluation in `torch.no_grad()` or `torch.inference_mode()`.

### Important model-review caveat

The cited paper describes cycle as `t`, the time coordinate. The feature CSV and loader place `Cycle Number` in the **first** input column. The current implementation, however, defines `ut` as the derivative with respect to the **last** input column, which is currently `Current entropy`. Therefore the implemented residual is presently a derivative with respect to the final descriptor, not a derivative with respect to cycle number.

This must be resolved explicitly before calling the residual a battery-degradation equation in the sense of the paper. It requires a documented choice of input ordering or explicit coordinate selection, compatible schema changes, and targeted tests.

The monotonicity expression also needs scientific confirmation. For declining ground-truth SoH, both `u2-u1` and `y2-y1` are negative when the prediction declines correctly, making their product positive and therefore adding a penalty. That differs from the paper's described penalty for predicted SoH increases. The paper describes the PDE and monotonicity terms with the opposite `alpha`/`beta` assignment to the current code. These are review findings, not silently corrected behavior.

## Training, random splits, and cell holdouts

Run model scripts from `model/` because imports and paths are cwd-relative. The default `data_path` in the training scripts is `../data/processed_battery_data_features_cleaned/`; point it at the selected pipeline output if you generated data inside `data/40V/` or `data/41V/`.

### Random paired-sample workflow

```text
cd model
python train.py
python train_final.py
python plot_sohpred_vs_sohtruth.py
```

`train.py`:

1. Loads a hard-coded list of cleaned normalized condition files.
2. Concatenates adjacent-cycle pairs from all listed cells.
3. Uses `sklearn.model_selection.train_test_split` with `test_size=0.2` and `random_state=1` on those pairs.
4. Saves the resulting tensors in `splits/train_test_split.pt`.
5. Runs 10-fold shuffled K-fold validation (`random_state=42`) over the 80% training portion, with parallel fold workers, and writes checkpoints/logs.

`train_final.py` reloads the saved 80/20 tensors, trains a new model, writes history, and saves `final_best.pt` (best nominal test loss) and `final_last.pt`. `train1.py` is an older, shorter exploratory variant: it uses five folds, 10 epochs, learning rates of `0.1`, and checkpoint names that overlap with the main workflow. Do not treat it as the standard experiment entry point.

This is a **random pair split**, not an unseen-cell or future-cycle evaluation. Consecutive pairs overlap: for example, `(cycle 1, cycle 2)` and `(cycle 2, cycle 3)` can fall on opposite sides of a random split. This exposes the same raw observation to training and test/fold data. Use it for within-population experimentation only, and use grouped or time-forward splitting for stronger generalization claims.

### Explicit cell/repetition holdout workflow

```text
cd model
python train_cell_01_holdout.py
python train_cell_02_holdout.py
python train_cell_03_holdout.py
```

Each holdout script names separate `train_conditions` and `test_conditions` rather than using `train_test_split`:

| Script | Held-out filename suffix in its explicit test list | Checkpoint / history |
| --- | --- | --- |
| `train_cell_01_holdout.py` | `_01_features_cleaned_normalized__pchip.csv` | `cell01_holdout_last.pt`, `cell01_holdout_history.csv` |
| `train_cell_02_holdout.py` | `_02_features_cleaned_normalized__pchip.csv` | `cell02_holdout_last.pt`, `cell02_holdout_history.csv` |
| `train_cell_03_holdout.py` | `_03_features_cleaned_normalized__pchip.csv` | `cell03_holdout_last.pt`, `cell03_holdout_history.csv` |

This is the appropriate starting point for a leave-one-cell/repetition-out experiment because the selected test files are not combined with the selected training files. The lists are manually maintained and do not cover every nominal condition uniformly: for example, `AG_25_80` is absent from all holdout test lists and is inconsistently included across training lists. Inspect them before interpreting the result as a complete cross-cell benchmark.

The corresponding `plot_cell0{1,2,3}_sohpred_vs_sohtruth.py` scripts reload the last checkpoint and generate per-condition predicted-versus-true SoH scatter plots. `train.sl` is a Slurm submission file that currently launches `train_cell_03_holdout.py` with CPU-thread environment variables.

### Evaluation protocol caveat

`train_final.py` evaluates the nominal test split every epoch and saves its best checkpoint by that test loss. The holdout scripts also log held-out loss every epoch, although they save the last epoch rather than selecting a best held-out checkpoint. For a strict final-test protocol, introduce a separate validation split and reserve test cells for one final evaluation. Reported MAE/RMSE also use the first endpoint of each adjacent pair (`x1`, `y1`), while the training data loss uses both endpoints.

## Practical notes

- The repository has no pinned dependency file or automated test suite. The scripts import PyTorch, NumPy, pandas, SciPy, scikit-learn, matplotlib, and Excel-reading support; `gatherall.py` additionally imports `cicemok`.
- Generated CSVs, pickles, checkpoints, histories, plots, and logs are ignored by Git. Preserve data provenance, preprocessing settings, interpolation method, split file, config, and random seeds outside the source tree when reporting an experiment.
- Many data scripts execute their work at import time. Treat them as command-line scripts rather than reusable import-safe modules.
- `plot_soh_data.py` reuses a cached `<condition>_soh.csv` if it exists. Delete or isolate stale cache files when changing the capacity-extraction rule.
- Input validation is uneven. For example, `FeatureDataset` selects columns by position and does not validate schema, finite values, ordering, duplicates, or cycle continuity; a missing integer cycle can also reach an `iloc[0]` access in `get_distilled_data.py` before its empty-cycle check. The feature merge does not enforce a complete one-to-one SoH join. Validate source-data completeness before long training runs.

## Recommended interpretation checklist

Before relying on a result, record and review:

- which of the 40V/41V pipelines and source workbooks were used;
- CC/CV window thresholds, interpolation method, capacity reference, and cleaner;
- whether generated labels were interpolated or direct check-up measurements;
- the exact feature schema and the selected physics/time coordinate;
- split unit (pair, cycle, cell, or time) and how normalization was fitted;
- configuration, seed/thread settings, checkpoint choice, and whether test loss influenced model selection.

These details are necessary to distinguish a useful engineering experiment from a reproducible battery-SOH benchmark.

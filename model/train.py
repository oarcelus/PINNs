import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dataloader import FeatureDataset
from model import PINN, load_pinn_config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold
import copy
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

device = torch.device("cpu")

conditions = [
    "AG_25_100_A_01_features_cleaned_normalized__pchip.csv",
    "AG_25_100_A_02_features_cleaned_normalized__pchip.csv",
    "AG_25_100_A_03_features_cleaned_normalized__pchip.csv",
    "AG_25_100_B_01_features_cleaned_normalized__pchip.csv",
    "AG_25_100_B_02_features_cleaned_normalized__pchip.csv",
    "AG_25_100_B_03_features_cleaned_normalized__pchip.csv",
    "AG_25_100_C_01_features_cleaned_normalized__pchip.csv",
    "AG_25_100_C_02_features_cleaned_normalized__pchip.csv",
    "AG_25_100_C_03_features_cleaned_normalized__pchip.csv",
    "AG_25_30_A_01_features_cleaned_normalized__pchip.csv",
    "AG_25_30_A_02_features_cleaned_normalized__pchip.csv",
    "AG_25_30_A_03_features_cleaned_normalized__pchip.csv",
    "AG_25_80_A_01_features_cleaned_normalized__pchip.csv",
    "AG_25_80_A_02_features_cleaned_normalized__pchip.csv",
    "AG_45_100_A_01_features_cleaned_normalized__pchip.csv",
    "AG_45_100_A_02_features_cleaned_normalized__pchip.csv",
    "AG_45_100_A_03_features_cleaned_normalized__pchip.csv",
]

# -----------------
# Save options
# -----------------
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

split_dir = "./splits"
os.makedirs(split_dir, exist_ok=True)
split_path = os.path.join(split_dir, "train_test_split.pt")

save_best_only = True
save_every_epoch = False
save_last_epoch = True

# -----------------
# Training settings
# -----------------
alpha = 0.7
beta = 0.2

n_splits = 10
n_epochs = 2000

lr_u = 1e-4
lr_f = 1e-3

n_fold_workers = 10
threads_per_fold = 2


def setup_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def compute_losses(pinn, batch, mse, relu, alpha, beta, device):
    x1, x2, y1, y2 = batch

    x1 = x1.float().to(device)
    x2 = x2.float().to(device)
    y1 = y1.float().view(-1, 1).to(device)
    y2 = y2.float().view(-1, 1).to(device)

    u1, l1 = pinn(x1)
    u2, l2 = pinn(x2)

    loss_mse = 0.5 * mse(u1, y1) + 0.5 * mse(u2, y2)
    loss_mono = relu(torch.mul(u2 - u1, y1 - y2)).sum()

    zero_ref1 = torch.zeros_like(l1)
    zero_ref2 = torch.zeros_like(l2)
    loss_pde = 0.5 * mse(l1, zero_ref1) + 0.5 * mse(l2, zero_ref2)

    loss = loss_mse + alpha * loss_pde + beta * loss_mono

    return loss, loss_mse, loss_mono, loss_pde


def run_one_epoch(pinn, loader, optimizer_u, optimizer_f, mse, relu, alpha, beta, device, training=True):
    if training:
        pinn.train()
    else:
        pinn.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_mono = 0.0
    total_pde = 0.0

    for batch in loader:
        if training:
            optimizer_u.zero_grad()
            optimizer_f.zero_grad()

        loss, loss_mse, loss_mono, loss_pde = compute_losses(
            pinn, batch, mse, relu, alpha, beta, device
        )

        if training:
            loss.backward()
            optimizer_u.step()
            optimizer_f.step()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_mono += loss_mono.item()
        total_pde += loss_pde.item()

    n_batches = len(loader)
    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "mono": total_mono / n_batches,
        "pde": total_pde / n_batches,
    }


def evaluate_mae_rmse(pinn, loader, device):
    pinn.eval()

    y_true = []
    y_pred = []

    for x1, x2, y1, y2 in loader:
        x1 = x1.float().to(device)
        y1 = y1.float().view(-1, 1).to(device)

        u1, _ = pinn(x1)

        y_true.append(y1.detach().cpu())
        y_pred.append(u1.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy().reshape(-1)
    y_pred = torch.cat(y_pred, dim=0).numpy().reshape(-1)

    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    return {"mae": float(mae), "rmse": float(rmse)}


def run_fold(
    ifold,
    train_idx,
    val_idx,
    train_tensors,
    batch_size,
    config,
    alpha,
    beta,
    n_epochs,
    lr_u,
    lr_f,
    save_dir,
    save_best_only,
    save_every_epoch,
    save_last_epoch,
    threads_per_fold,
    log_dir,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(threads_per_fold)
    torch.set_num_interop_threads(1)

    fold_logger = setup_logger(
        name=f"fold_{ifold}",
        log_path=os.path.join(log_dir, f"fold_{ifold}.log"),
    )

    device = torch.device("cpu")
    mse = nn.MSELoss()
    relu = nn.ReLU()

    train_dataset = TensorDataset(*train_tensors)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        Subset(train_dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    pinn = PINN(config).to(device)

    optimizer_u = torch.optim.Adam(pinn.u.parameters(), lr=lr_u)
    optimizer_f = torch.optim.Adam(pinn.f.parameters(), lr=lr_f)

    best_val_loss = np.inf
    best_state = None
    best_epoch = -1

    history = []

    fold_logger.info(f"========== Fold {ifold}/{n_splits} ==========")

    for iepoch in range(n_epochs):
        train_metrics = run_one_epoch(
            pinn, train_loader, optimizer_u, optimizer_f,
            mse, relu, alpha, beta, device, training=True
        )

        val_metrics = run_one_epoch(
            pinn, val_loader, optimizer_u, optimizer_f,
            mse, relu, alpha, beta, device, training=False
        )

        history.append({
            "epoch": iepoch + 1,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_mono": train_metrics["mono"],
            "train_pde": train_metrics["pde"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_mono": val_metrics["mono"],
            "val_pde": val_metrics["pde"],
        })

        fold_logger.info(
            f"Fold {ifold} | Epoch {iepoch + 1:03d}/{n_epochs} | "
            f"train={train_metrics['loss']:.6f} "
            f"(mse={train_metrics['mse']:.6f}, mono={train_metrics['mono']:.6f}, pde={train_metrics['pde']:.6f}) | "
            f"val={val_metrics['loss']:.6f} "
            f"(mse={val_metrics['mse']:.6f}, mono={val_metrics['mono']:.6f}, pde={val_metrics['pde']:.6f})"
        )

        checkpoint = {
            "fold": ifold,
            "epoch": iepoch + 1,
            "model_state_dict": pinn.state_dict(),
            "optimizer_u_state_dict": optimizer_u.state_dict(),
            "optimizer_f_state_dict": optimizer_f.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": config,
            "history": history,
        }

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(pinn.state_dict())
            best_epoch = iepoch + 1

            torch.save(
                checkpoint,
                os.path.join(save_dir, f"fold_{ifold}_best.pt")
            )

        if (not save_best_only) and save_every_epoch:
            torch.save(
                checkpoint,
                os.path.join(save_dir, f"fold_{ifold}_epoch_{iepoch + 1:03d}.pt"),
            )

    if save_last_epoch:
        torch.save(
            {
                "fold": ifold,
                "epoch": len(history),
                "model_state_dict": pinn.state_dict(),
                "optimizer_u_state_dict": optimizer_u.state_dict(),
                "optimizer_f_state_dict": optimizer_f.state_dict(),
                "history": history,
                "config": config,
            },
            os.path.join(save_dir, f"fold_{ifold}_last.pt"),
        )

    pinn.load_state_dict(best_state)

    train_reg = evaluate_mae_rmse(pinn, train_loader, device)
    val_reg = evaluate_mae_rmse(pinn, val_loader, device)

    fold_logger.info(
        f"Fold {ifold} finished | "
        f"best epoch={best_epoch:03d} | "
        f"best val={best_val_loss:.6f} | "
        f"train_mae={train_reg['mae']:.6f} | "
        f"train_rmse={train_reg['rmse']:.6f} | "
        f"val_mae={val_reg['mae']:.6f} | "
        f"val_rmse={val_reg['rmse']:.6f}"
    )

    return {
        "fold": ifold,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "train_mae": train_reg["mae"],
        "train_rmse": train_reg["rmse"],
        "val_mae": val_reg["mae"],
        "val_rmse": val_reg["rmse"],
        "history": history,
    }


if __name__ == "__main__":
    main_logger = setup_logger(
        name="main",
        log_path=os.path.join(log_dir, "main.log"),
    )

    dat = FeatureDataset(
        path="../data/processed_battery_data_features_cleaned/",
        conditions=conditions,
    )

    x1, x2, y1, y2 = dat.load_all_cells()
    dat.set_train_test_data([x1, x2, y1, y2])

    torch.save(
        {
            "conditions": conditions,
            "train_tensors": dat.loader["train"].dataset.tensors,
            "test_tensors": dat.loader["test"].dataset.tensors,
        },
        split_path,
    )

    config = load_pinn_config("../cfg/pinn_config.json")

    train_dataset = dat.loader["train"].dataset
    batch_size = dat.loader["train"].batch_size
    train_tensors = train_dataset.tensors

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_jobs = []

    for ifold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset))), start=1):
        fold_jobs.append((
            ifold,
            train_idx,
            val_idx,
            train_tensors,
            batch_size,
            config,
            alpha,
            beta,
            n_epochs,
            lr_u,
            lr_f,
            save_dir,
            save_best_only,
            save_every_epoch,
            save_last_epoch,
            threads_per_fold,
            log_dir,
        ))

    fold_results = []

    main_logger.info("Starting cross-validation")

    with ProcessPoolExecutor(max_workers=n_fold_workers) as executor:
        futures = [executor.submit(run_fold, *job) for job in fold_jobs]

        for future in as_completed(futures):
            result = future.result()
            fold_results.append(result)
            main_logger.info(
                f"Finished fold {result['fold']} | "
                f"best epoch={result['best_epoch']:03d} | "
                f"best val={result['best_val_loss']:.6f} | "
                f"train_mae={result['train_mae']:.6f} | "
                f"train_rmse={result['train_rmse']:.6f} | "
                f"val_mae={result['val_mae']:.6f} | "
                f"val_rmse={result['val_rmse']:.6f}"
            )

    fold_results = sorted(fold_results, key=lambda x: x["fold"])
    fold_val_losses = [r["best_val_loss"] for r in fold_results]
    fold_val_mae = [r["val_mae"] for r in fold_results]
    fold_val_rmse = [r["val_rmse"] for r in fold_results]

    main_logger.info("========== Summary ==========")
    main_logger.info(f"CV val loss mean: {np.mean(fold_val_losses):.6f}")
    main_logger.info(f"CV val loss std : {np.std(fold_val_losses):.6f}")
    main_logger.info(f"CV val MAE mean : {np.mean(fold_val_mae):.6f}")
    main_logger.info(f"CV val MAE std  : {np.std(fold_val_mae):.6f}")
    main_logger.info(f"CV val RMSE mean: {np.mean(fold_val_rmse):.6f}")
    main_logger.info(f"CV val RMSE std : {np.std(fold_val_rmse):.6f}")



import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import PINN, load_pinn_config

device = torch.device("cpu")

# -----------------
# Paths
# -----------------
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

split_dir = "./splits"
os.makedirs(split_dir, exist_ok=True)
split_path = os.path.join(split_dir, "train_test_split.pt")

history_dir = "./history"
os.makedirs(history_dir, exist_ok=True)
history_csv_path = os.path.join(history_dir, "final_train_history.csv")

# -----------------
# Save options
# -----------------
save_last_epoch = True
save_best_epoch = True

# -----------------
# Training settings
# -----------------
alpha = 0.7
beta = 0.2

n_epochs = 2000

lr_u = 1e-4
lr_f = 1e-3

threads_per_fold = 8
batch_size = 1


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


def run_one_epoch(
    pinn, loader, optimizer_u, optimizer_f, mse, relu, alpha, beta, device
):
    pinn.train()

    total_loss = 0.0
    total_mse = 0.0
    total_mono = 0.0
    total_pde = 0.0

    for batch in loader:
        optimizer_u.zero_grad()
        optimizer_f.zero_grad()

        loss, loss_mse, loss_mono, loss_pde = compute_losses(
            pinn, batch, mse, relu, alpha, beta, device
        )

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


def evaluate_loss(pinn, loader, mse, relu, alpha, beta, device):
    pinn.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_mono = 0.0
    total_pde = 0.0

    for batch in loader:
        loss, loss_mse, loss_mono, loss_pde = compute_losses(
            pinn, batch, mse, relu, alpha, beta, device
        )

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


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(threads_per_fold)
    torch.set_num_interop_threads(1)

    logger = setup_logger(
        name="final_train",
        log_path=os.path.join(log_dir, "final_train.log"),
    )

    split_data = torch.load(split_path)

    train_dataset = TensorDataset(*split_data["train_tensors"])
    test_dataset = TensorDataset(*split_data["test_tensors"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    config = load_pinn_config("../cfg/pinn_config.json")
    pinn = PINN(config).to(device)

    optimizer_u = torch.optim.Adam(pinn.u.parameters(), lr=lr_u)
    optimizer_f = torch.optim.Adam(pinn.f.parameters(), lr=lr_f)

    mse = nn.MSELoss()
    relu = nn.ReLU()

    history = []
    best_test_loss = np.inf
    best_epoch = -1
    best_state = None

    logger.info("Starting final training on saved train/test split")

    for iepoch in range(n_epochs):
        train_metrics = run_one_epoch(
            pinn, train_loader, optimizer_u, optimizer_f, mse, relu, alpha, beta, device
        )

        test_metrics = evaluate_loss(pinn, test_loader, mse, relu, alpha, beta, device)

        history.append(
            {
                "epoch": iepoch + 1,
                "train_loss": train_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "train_mono": train_metrics["mono"],
                "train_pde": train_metrics["pde"],
                "test_loss": test_metrics["loss"],
                "test_mse": test_metrics["mse"],
                "test_mono": test_metrics["mono"],
                "test_pde": test_metrics["pde"],
            }
        )

        pd.DataFrame(history).to_csv(
            history_csv_path, index=False, float_format="%.16e"
        )

        logger.info(
            f"Epoch {iepoch + 1:03d}/{n_epochs} | "
            f"train={train_metrics['loss']:.6f} "
            f"(mse={train_metrics['mse']:.6f}, mono={train_metrics['mono']:.6f}, pde={train_metrics['pde']:.6f}) | "
            f"test={test_metrics['loss']:.6f} "
            f"(mse={test_metrics['mse']:.6f}, mono={test_metrics['mono']:.6f}, pde={test_metrics['pde']:.6f})"
        )

        if test_metrics["loss"] < best_test_loss:
            best_test_loss = test_metrics["loss"]
            best_epoch = iepoch + 1
            best_state = copy.deepcopy(pinn.state_dict())

            if save_best_epoch:
                torch.save(
                    {
                        "epoch": iepoch + 1,
                        "model_state_dict": pinn.state_dict(),
                        "optimizer_u_state_dict": optimizer_u.state_dict(),
                        "optimizer_f_state_dict": optimizer_f.state_dict(),
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "history": history,
                        "config": config,
                    },
                    os.path.join(save_dir, "final_best.pt"),
                )

    if save_last_epoch:
        torch.save(
            {
                "epoch": len(history),
                "model_state_dict": pinn.state_dict(),
                "optimizer_u_state_dict": optimizer_u.state_dict(),
                "optimizer_f_state_dict": optimizer_f.state_dict(),
                "history": history,
                "config": config,
            },
            os.path.join(save_dir, "final_last.pt"),
        )

    pinn.load_state_dict(best_state)

    train_reg = evaluate_mae_rmse(pinn, train_loader, device)
    test_reg = evaluate_mae_rmse(pinn, test_loader, device)

    logger.info("========== Final Summary ==========")
    logger.info(f"Best epoch : {best_epoch:03d}")
    logger.info(f"Best test loss : {best_test_loss:.6f}")
    logger.info(f"Train MAE  : {train_reg['mae']:.6f}")
    logger.info(f"Train RMSE : {train_reg['rmse']:.6f}")
    logger.info(f"Test MAE   : {test_reg['mae']:.6f}")
    logger.info(f"Test RMSE  : {test_reg['rmse']:.6f}")

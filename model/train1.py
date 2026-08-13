import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dataloader import FeatureDataset
from model import PINN, load_pinn_config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import copy
import numpy as np

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

save_best_only = True  # if False, also save every epoch
save_every_epoch = False  # only used if save_best_only = False
save_last_epoch = True

dat = FeatureDataset(
    path="../data/processed_battery_data_features_cleaned/",
    conditions=conditions,
)

x1, x2, y1, y2 = dat.load_all_cells()
dat.set_train_test_data([x1, x2, y1, y2])

config = load_pinn_config("../cfg/pinn_config.json")

# Loss functions
mse = nn.MSELoss()
relu = nn.ReLU()

alpha = 0.7
beta = 0.2

n_splits = 5
n_epochs = 10

train_dataset = dat.loader["train"].dataset
test_loader = dat.loader["test"]

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_val_losses = []

for ifold, (train_idx, val_idx) in enumerate(
    kf.split(range(len(train_dataset))), start=1
):
    print(f"\n========== Fold {ifold}/{n_splits} ==========")

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=dat.loader["train"].batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        Subset(train_dataset, val_idx),
        batch_size=dat.loader["train"].batch_size,
        shuffle=False,
    )

    pinn = PINN(config).to(device)

    optimizer_u = torch.optim.Adam(pinn.u.parameters(), lr=0.1)
    optimizer_f = torch.optim.Adam(pinn.f.parameters(), lr=0.1)

    best_val_loss = np.inf
    best_state = None
    best_epoch = -1

    for iepoch in range(n_epochs):
        # -----------------
        # Train
        # -----------------
        pinn.train()
        train_loss_epoch = 0.0

        for idx, (x1, x2, y1, y2) in enumerate(train_loader):
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)
            y1 = y1.float().view(-1, 1).to(device)
            y2 = y2.float().view(-1, 1).to(device)

            optimizer_u.zero_grad()
            optimizer_f.zero_grad()

            u1, l1 = pinn(x1)
            u2, l2 = pinn(x2)

            loss_mse = 0.5 * mse(u1, y1) + 0.5 * mse(u2, y2)
            loss_mono = relu(torch.mul(u2 - u1, y2 - y1)).sum()

            zero_ref1 = torch.zeros_like(l1)
            zero_ref2 = torch.zeros_like(l2)
            loss_pde = 0.5 * mse(l1, zero_ref1) + 0.5 * mse(l2, zero_ref2)

            loss = loss_mse + alpha * loss_mono + beta * loss_pde

            loss.backward()

            optimizer_u.step()
            optimizer_f.step()

            train_loss_epoch += loss.item()

        train_loss_epoch /= len(train_loader)

        # -----------------
        # Validation
        # -----------------
        pinn.eval()
        val_loss_epoch = 0.0

        for idx, (x1, x2, y1, y2) in enumerate(val_loader):
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)
            y1 = y1.float().view(-1, 1).to(device)
            y2 = y2.float().view(-1, 1).to(device)

            u1, l1 = pinn(x1)
            u2, l2 = pinn(x2)

            loss_mse = 0.5 * mse(u1, y1) + 0.5 * mse(u2, y2)
            loss_mono = relu(torch.mul(u2 - u1, y2 - y1)).sum()

            zero_ref1 = torch.zeros_like(l1)
            zero_ref2 = torch.zeros_like(l2)
            loss_pde = 0.5 * mse(l1, zero_ref1) + 0.5 * mse(l2, zero_ref2)

            loss = loss_mse + alpha * loss_mono + beta * loss_pde

            val_loss_epoch += loss.item()

        val_loss_epoch /= len(val_loader)

        print(
            f"Fold {ifold} | Epoch {iepoch + 1:03d}/{n_epochs} | "
            f"train={train_loss_epoch:.6f} | val={val_loss_epoch:.6f}"
        )

        checkpoint = {
            "fold": ifold,
            "epoch": iepoch + 1,
            "model_state_dict": pinn.state_dict(),
            "optimizer_u_state_dict": optimizer_u.state_dict(),
            "optimizer_f_state_dict": optimizer_f.state_dict(),
            "train_loss": train_loss_epoch,
            "val_loss": val_loss_epoch,
            "config": config,
        }

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_state = copy.deepcopy(pinn.state_dict())
            best_epoch = iepoch + 1

            torch.save(checkpoint, os.path.join(save_dir, f"fold_{ifold}_best.pt"))

        if (not save_best_only) and save_every_epoch:
            torch.save(
                checkpoint,
                os.path.join(save_dir, f"fold_{ifold}_epoch_{iepoch + 1:03d}.pt"),
            )

    fold_val_losses.append(best_val_loss)

    if save_last_epoch:
        torch.save(
            {
                "fold": ifold,
                "epoch": n_epochs,
                "model_state_dict": pinn.state_dict(),
                "optimizer_u_state_dict": optimizer_u.state_dict(),
                "optimizer_f_state_dict": optimizer_f.state_dict(),
                "train_loss": train_loss_epoch,
                "val_loss": val_loss_epoch,
                "config": config,
            },
            os.path.join(save_dir, f"fold_{ifold}_last.pt"),
        )

print("\n========== Summary ==========")
print(f"CV val loss mean: {np.mean(fold_val_losses):.6f}")
print(f"CV val loss std : {np.std(fold_val_losses):.6f}")

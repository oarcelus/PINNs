import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class FeatureDataset:
    def __init__(self, path: str, conditions: list[str]):
        self.path = path
        self.conditions = conditions

        self.loader = {}

    def read_one_csv(self, condition: str):
        csvpath = os.path.join(self.path, condition)
        df = pd.read_csv(csvpath)

        return df

    def load_one_cell(self, condition: str):
        df = self.read_one_csv(condition)

        x = df.iloc[:, [0] + list(range(2, df.shape[1]))].values
        y = df.iloc[:, 1].values

        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]

        return x1, x2, y1, y2

    def load_all_cells(self):
        X1, X2, Y1, Y2 = [], [], [], []
        for condition in self.conditions:
            x1, x2, y1, y2 = self.load_one_cell(condition)

            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)

        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        Y1 = np.concatenate(Y1, axis=0)
        Y2 = np.concatenate(Y2, axis=0)

        X1_tensor = torch.from_numpy(X1)
        X2_tensor = torch.from_numpy(X2)
        Y1_tensor = torch.from_numpy(Y1)
        Y2_tensor = torch.from_numpy(Y2)

        return X1_tensor, X2_tensor, Y1_tensor, Y2_tensor

    def set_train_test_data(
        self,
        tensors: list[torch.Tensor],
        size: float = 0.2,
        seed: int = 1,
        batch_size: int = 1,
    ):
        train_X1, test_X1, train_X2, test_X2, train_Y1, test_Y1, train_Y2, test_Y2 = (
            train_test_split(*tensors, test_size=size, random_state=seed)
        )

        train_loader = DataLoader(
            TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = DataLoader(
            TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
            batch_size=batch_size,
            shuffle=True,
        )

        self.loader["train"] = train_loader
        self.loader["test"] = test_loader

    def set_validation_data(self, tensors: list[torch.Tensor]):
        valid_loader = DataLoader(
            TensorDataset(*tensors),
            shuffle=True,
        )

        self.loader["valid"] = valid_loader

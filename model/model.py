import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
import os
import json


def load_pinn_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        isize: int = 17,
        osize: int = 1,
        nlayer: int = 4,
        hsize: int = 20,
        dropout: float = 0.2,
    ):
        super(MultiLayerPerceptron, self).__init__()

        self.isize = isize
        self.osize = osize
        self.nlayer = nlayer
        self.hsize = hsize
        self.droput = dropout

        layers = []

        for i in range(nlayer):
            if i == 0:
                layers.append(nn.Linear(isize, hsize))
                layers.append(nn.Tanh())
            elif i == nlayer - 1:
                layers.append(nn.Linear(hsize, osize))
            else:
                layers.append(nn.Linear(hsize, hsize))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.net.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class PredictorNetwork(nn.Module):
    def __init__(self, isize: int = 17, hsize: int = 20, dropout: float = 0.2):
        super(PredictorNetwork, self).__init__()

        self.isize = isize
        self.hsize = hsize
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(isize, hsize), nn.Tanh(), nn.Linear(hsize, 1)
        )

        self.net.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class SoHNetwork(nn.Module):
    def __init__(
        self,
        eisize: int = 17,
        eosize: int = 1,
        enlayer: int = 4,
        ehsize: int = 20,
        edropout: float = 0.2,
        phsize: int = 20,
        pdropout: float = 0.2,
    ):

        super(SoHNetwork, self).__init__()

        self.encoder = MultiLayerPerceptron(eisize, eosize, enlayer, ehsize, edropout)
        self.predictor = PredictorNetwork(eosize, phsize, pdropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)

        return x


class PINN(nn.Module):
    def __init__(self, input_params):
        super(PINN, self).__init__()

        assert "encoder" in input_params
        assert "predictor" in input_params
        assert "dynamics" in input_params

        self.input_params = input_params

        self.u = SoHNetwork(
            **self.input_params["encoder"], **self.input_params["predictor"]
        )
        self.f = MultiLayerPerceptron(**self.input_params["dynamics"])

    def forward(self, xt):
        xt = xt.float().requires_grad_(True)

        u = self.u(xt)
        uxt = grad(
            u.sum(), xt, create_graph=True, only_inputs=True
        )[0]
        ux = uxt[:, 1:]
        ut = uxt[:, 0:1]

        f = self.f(torch.cat([xt, u, ux, ut], dim=1))
        l = ut - f

        return u, l

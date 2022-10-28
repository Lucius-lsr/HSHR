# -*- coding: utf-8 -*-
"""
@Time    : 2021/12/30 15:40
@Author  : Lucius
@FileName: base_model.py
@Software: PyCharm
"""
import torch
from torch import nn


class HashLayer(nn.Module):
    def __init__(self, feature_in, feature_out, depth) -> None:
        super().__init__()
        if depth == 1:
            self.fc = nn.Linear(feature_in, feature_out)
        elif depth == 2:
            self.fc = nn.Sequential(
                nn.Linear(feature_in, 2 * feature_out),
                nn.Linear(2*feature_out, feature_out),
            )

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)
        return x


class SqueezeOp(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = x.squeeze()
        return x
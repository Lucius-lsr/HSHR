# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/29 14:18
@Author  : Lucius
@FileName: hgrnet.py
@Software: PyCharm
"""

import torch.nn.functional as F
from torch import nn

from models.HyperG.conv import *


class HGRNet(nn.Module):
    def __init__(self, in_ch, n_target, hiddens: list, dropout, sensitive, pooling_strategy, k_neighbors: list) -> None:
        super().__init__()
        _in = in_ch
        self.hyconvs = []
        for _h in hiddens:
            _out = _h
            self.hyconvs.append(HyConv(_in, _out))
            _in = _out
        self.hyconvs = nn.ModuleList(self.hyconvs)
        self.last_fc = nn.Linear(_in, n_target)

        self.dropout = dropout
        self.sensitive = sensitive
        self.pooling_strategy = pooling_strategy
        self.k_neighbors = k_neighbors

    def forward(self, x_H):
        # H = self.get_H(x, self.k_neighbors)
        x, H = x_H
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
            x = F.dropout(x, self.dropout)

        feats_pool = None
        if self.pooling_strategy == 'mean':
            if self.sensitive == 'attribute':
                # N x C -> 1 x C
                x = x.mean(dim=-2)
            if self.sensitive == 'pattern':
                # N x C -> N x 1
                x = x.mean(dim=-1)
            # C -> 1 x C
            feats_pool = x
        if self.pooling_strategy == 'max':
            feats_pool = x.max(dim=0)[0]

        final_feature = self.last_fc(feats_pool)
        return final_feature

    # def get_H(self, fts, k_nearest):
    #     H = hyedge_concat([neighbor_distance(fts, k) for k in k_nearest])
    #     return H
    #
    # def random_hyperedge_generate(self, coordinates):
    #     pass


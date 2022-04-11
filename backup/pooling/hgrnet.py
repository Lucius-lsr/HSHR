# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/29 14:18
@Author  : Lucius
@FileName: hgrnet.py
@Software: PyCharm
"""
import torch
import torch.nn.functional as F
from torch import nn

from models.HyperG.conv import *
from models.HyperG.hyedge import neighbor_distance


class HGRNet(nn.Module):
    def __init__(self, in_ch, n_target, hiddens: list, dropout, sensitive, pooling_strategy, k) -> None:
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
        self.k = k

    def forward(self, x_nl, loc=True):
        # H = self.get_H(x, self.k_neighbors)
        x, nn_idx = x_nl

        if loc:
            H = self.generate_random_H_loc(nn_idx)
        else:
            H = self.generate_H_feature(x)

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

    def generate_random_H_loc(self, nn_idx):
        device = nn_idx.device
        self_idx = (torch.arange(nn_idx.shape[1]).reshape(1, -1, 1)).repeat(nn_idx.shape[0], 1, 1)  # bs * ps * 1
        hyedge_idx = torch.arange(nn_idx.shape[1]).unsqueeze(0).repeat(self.k, 1).transpose(1, 0).reshape(1, 1, -1)
        self_idx, hyedge_idx = self_idx.to(device), hyedge_idx.to(device)
        sample_nn_idx = nn_idx[:, :, torch.randperm(2 * self.k - 1)]
        sample_nn_idx = sample_nn_idx[:, :, :self.k - 1]  # batch_size * patch_size * k-1
        sample_nn_idx = torch.cat((self_idx, sample_nn_idx), dim=2)  # batch_size * patch_size * k
        H = torch.cat(
            (sample_nn_idx.reshape(nn_idx.shape[0], 1, -1), hyedge_idx.repeat(nn_idx.shape[0], 1, 1)), dim=1)

        return H

    def generate_H_feature(self, b_x):
        h_list = []
        for x in b_x:
            h = neighbor_distance(x, self.k)
            h_list.append(h.unsqueeze(0))
        H = torch.cat(h_list, dim=0)
        return H


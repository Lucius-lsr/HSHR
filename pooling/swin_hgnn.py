# -*- coding: utf-8 -*-
"""
@Time    : 2021/11/3 14:42
@Author  : Lucius
@FileName: swin_hgnn.py
@Software: PyCharm
"""
import torch
from torch import nn
import torch.nn.functional as F

from models.HyperG.conv import HyConv, hyconv
from models.HyperG.hyedge import neighbor_distance


def batch_pairwise_distance(m: torch.Tensor, n: torch.Tensor):
    """
    Args:
        m: b x m x d
        n: b x n x d

    Returns:
    """
    assert isinstance(m, torch.Tensor) and isinstance(n, torch.Tensor)
    assert len(m.shape) == 3 and len(n.shape) == 3
    assert m.shape[-1] == n.shape[-1]
    m, n = m.float(), n.float()

    n_transpose = torch.transpose(n, dim0=1, dim1=2)
    mn_inner = torch.matmul(m, n_transpose)
    mn_inner = - 2 * mn_inner
    m_square = torch.sum(m ** 2, dim=2, keepdim=True).repeat(1, 1, n.shape[1])
    n_square = torch.sum(n ** 2, dim=2, keepdim=True).repeat(1, 1, m.shape[1])
    n_square_transpose = torch.transpose(n_square, dim0=1, dim1=2)
    dis = mn_inner + m_square + n_square_transpose
    return dis


def get_related_extra(nearest, extra_feature):
    """

    Args:
        nearest: batch x 2000 x 1
        extra_feature: batch x num_sample_above x feature_extra

    Returns: batch x 2000 x feature_extra

    """
    batch_num = extra_feature.shape[0]
    nearest = nearest.squeeze()
    batch_list = []
    for b in range(batch_num):
        # patch_list = []
        # for patch in nearest[b]:
        #     patch_list.append(extra_feature[b][patch].unsqueeze(0))
        # patches = torch.cat(patch_list, dim=0)
        patches = extra_feature[b][nearest[b]]
        batch_list.append(patches.unsqueeze(0))
    extra = torch.cat(batch_list, dim=0)

    return extra


class SWinHGNNet(nn.Module):
    def __init__(self, num_level, level_dims: list, hidden_upper_dims: list, hgnn_dims: list, target_dim, k, dropout):
        super().__init__()
        self.mode = False
        assert num_level > 1 and num_level == len(level_dims) == len(hidden_upper_dims) + 1 == len(hgnn_dims)
        self.num_level = num_level
        self.k = k
        self.dropout = dropout

        self.input_fc_layers = list()
        for in_dim, hidden_dim in zip(level_dims[1:], hidden_upper_dims):
            self.input_fc_layers.append(nn.Linear(in_dim, hidden_dim))
        self.input_fc_layers = nn.ModuleList(self.input_fc_layers)

        self.hgnn_layers = list()
        self.hgnn_layers.append(HyConv(level_dims[0], hgnn_dims[0]))
        origin_hidden = hgnn_dims[0]
        for extra_hidden, target_hidden in zip(hidden_upper_dims, hgnn_dims[1:]):
            self.hgnn_layers.append(HyConv(origin_hidden+extra_hidden, target_hidden))
            origin_hidden = target_hidden
        self.hgnn_layers = nn.ModuleList(self.hgnn_layers)

        self.last_fc = nn.Linear(hgnn_dims[-1], target_dim)

    def forward(self, xs: list, coordinates: list):
        assert len(xs) == len(coordinates) == self.num_level

        x_0 = xs[0]
        H_0 = self.generate_H_feature(x_0)
        x_1 = self.hgnn_layers[0](x_0, H_0)
        x_1 = F.leaky_relu(x_1, inplace=True)
        x_1 = F.dropout(x_1, self.dropout)

        c_0 = coordinates[0]
        hidden = x_1  # batch x 2000 x feature_hidden
        output_list = []
        if self.mode:
            output_list.append(hidden.mean(dim=-2))
        for x_i, c_i, layer, hgnn_layer in zip(xs[1:], coordinates[1:], self.input_fc_layers, self.hgnn_layers[1:]):
            e_i = layer(x_i)  # batch x num_sample_above x feature_extra
            dis = batch_pairwise_distance(c_0, c_i)  # batch x 2000 x num_sample_above
            _, nearest = torch.topk(dis, 1, largest=False)  # batch x 2000 x 1
            related_extra = get_related_extra(nearest, e_i)  # batch x 2000 x feature_extra
            hidden = torch.cat((hidden, related_extra), dim=-1)  # batch x 2000 x (feature_hidden+feature_extra)
            H = self.generate_H_feature(hidden)
            hidden = hgnn_layer(hidden, H)  # batch x 2000 x new_feature_hidden
            hidden = F.leaky_relu(hidden, inplace=True)
            hidden = F.dropout(hidden, self.dropout)
            if self.mode:
                output_list.append(hidden.mean(dim=-2))
        pool = hidden.mean(dim=-2)
        final_feature = self.last_fc(pool)
        if self.mode:
            output_list.append(final_feature)
            return output_list
        return final_feature

    def set_complex_output(self, mode):
        self.mode = mode

    def generate_H_feature(self, b_x):
        h_list = []
        for x in b_x:
            h = neighbor_distance(x, self.k)
            h_list.append(h.unsqueeze(0))
        H = torch.cat(h_list, dim=0)
        return H


class NaiveSWinHGNNet(nn.Module):
    def __init__(self, num_level, level_dim, hidden_dim, target_dim, k, dropout):
        super().__init__()
        self.hgnn_layers = []
        for i in range(num_level):
            self.hgnn_layers.append(HyConv(level_dim, hidden_dim))
        self.hgnn_layers = nn.ModuleList(self.hgnn_layers)
        self.last_fc = nn.Linear(num_level*hidden_dim, target_dim)
        self.k = k
        self.dropout = dropout

    def forward(self, x_c):
        xs, coordinates = x_c
        feature_list = []
        for x_i, c_i, hgnn_layer in zip(xs, coordinates, self.hgnn_layers):
            H_i = self.generate_H_feature(x_i)
            hidden_i = hgnn_layer(x_i, H_i)
            hidden_i = F.leaky_relu(hidden_i, inplace=True)
            hidden_i = F.dropout(hidden_i, self.dropout)
            feature_list.append(hidden_i.mean(dim=-2))
        feature = torch.cat(feature_list, dim=-1)
        output = self.last_fc(feature)
        return output

    def generate_H_feature(self, b_x):
        h_list = []
        for x in b_x:
            h = neighbor_distance(x, self.k)
            h_list.append(h.unsqueeze(0))
        H = torch.cat(h_list, dim=0)
        return H


class NaNaiveSWinHGNNet(nn.Module):
    def __init__(self, num_level, level_dim, hidden_dim, k, dropout):
        super().__init__()
        self.hgnn_layers = []
        for i in range(num_level):
            self.hgnn_layers.append(HyConv(level_dim, hidden_dim))
        self.hgnn_layers = nn.ModuleList(self.hgnn_layers)
        self.k = k
        self.dropout = dropout

    def forward(self, x_c):
        xs, coordinates = x_c
        feature_list = []
        for x_i, c_i, hgnn_layer in zip(xs, coordinates, self.hgnn_layers):
            H_i = self.generate_H_feature(x_i)
            hidden_i = hgnn_layer(x_i, H_i)
            hidden_i = F.leaky_relu(hidden_i, inplace=True)
            hidden_i = F.dropout(hidden_i, self.dropout)
            feature_list.append(hidden_i.mean(dim=-2))
        feature = torch.cat(feature_list, dim=-1)
        return feature

    def generate_H_feature(self, b_x):
        h_list = []
        for x in b_x:
            h = neighbor_distance(x, self.k)
            h_list.append(h.unsqueeze(0))
        H = torch.cat(h_list, dim=0)
        return H
import torch.nn.functional as F
from torch import nn
import torch

from models.HyperG.conv import *
from models.HyperG.hyedge import *
from models.HyperG.hygraph import hyedge_concat


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

    def forward(self, x):
        H = self.get_H(x, self.k_neighbors)
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
            x = F.dropout(x, self.dropout)

        feats = x
        feats_pool = None
        if self.pooling_strategy == 'mean':
            if self.sensitive == 'attribute':
                # N x C -> 1 x C
                x = x.mean(dim=0)
            if self.sensitive == 'pattern':
                # N x C -> N x 1
                x = x.mean(dim=1)
            # C -> 1 x C
            feats_pool = x.unsqueeze(0)
        if self.pooling_strategy == 'max':
            feats_pool = x.max(dim=0)[0].unsqueeze(0)

        # # 1 x C -> 1 x n_target
        # x = self.last_fc(feats_pool)
        # # 1 x n_target -> n_target
        # x = x.squeeze(0)
        # return torch.sigmoid(x), feats, feats_pool
        return feats_pool

    def get_H(self, fts, k_nearest):
        H = hyedge_concat([neighbor_distance(fts, k) for k in k_nearest])
        return H


class MLP(nn.Module):
    def __init__(self, in_feat, out_features):
        super().__init__()
        net = [
            nn.Linear(in_feat, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),

            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),

            nn.Linear(32, out_features)
        ]
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        return self.net(inputs)

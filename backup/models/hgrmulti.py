import torch
import torch.nn.functional as F
from torch import nn

from models.HyperG.conv import *
from models.HyperG.hyedge import *
from train_config import *


class CIndexBP(nn.Module):
    def __init__(self):
        super(CIndexBP, self).__init__()
        return

    def forward(self, output, target):
        assert output.ndim == target.ndim, 'target and output do not match'
        assert output.ndim == 1

        output = output.unsqueeze(0)
        target = target.unsqueeze(0)

        num_sample = output.shape[-1]
        num_hit = (~((output.T > output) ^ (target.T > target))).sum()

        return float(num_hit - num_sample) / float(num_sample * num_sample - num_sample)


criterion = CIndexBP()


class HGRMultiCase(nn.Module):
    def __init__(self, in_ch, n_target, hiddens=hiddens, dropout=dropout,
                 dropmax_ratio=dropmax_ratio, sensitive=sensitive, pooling_strategy=pooling_strategy) -> None:
        super().__init__()
        self.dropout = dropout
        _in = in_ch
        self.hyconvs = []
        for _h in hiddens:
            _out = _h
            self.hyconvs.append(HyConv(_in, _out))
            _in = _out
        self.hyconvs = nn.ModuleList(self.hyconvs)
        self.last_fc = nn.Linear(_in, n_target)

    def forward(self, x, hyedge_weight=None):
        global feats_pool
        H = self.get_H(x)
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
            x = F.dropout(x, self.dropout)

        feats = x
        if pooling_strategy == 'mean':
            if sensitive == 'attribute':
                # N x C -> 1 x C
                x = x.mean(dim=0)
            if sensitive == 'pattern':
                # N x C -> N x 1
                x = x.mean(dim=1)
            # C -> 1 x C
            feats_pool = x.unsqueeze(0)
        if pooling_strategy == 'max':
            feats_pool = x.max(dim=0)[0].unsqueeze(0)

        # 1 x C -> 1 x n_target
        x = self.last_fc(feats_pool)
        # 1 x n_target -> n_target
        x = x.squeeze(0)

        return torch.sigmoid(x), feats, feats_pool

    def get_H(self, fts, k_nearest=k_nearest):
        return neighbor_distance(fts, k_nearest)


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

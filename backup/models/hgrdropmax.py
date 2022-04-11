import torch.nn.functional as F
from torch import nn

from models.HyperG.conv import HyConv
from train_config import *

criterion = torch.nn.MSELoss()


class HGRDpMax(nn.Module):
    def __init__(self, in_ch, n_target, hiddens=hiddens, dropout=dropout,
                 dropmax_ratio=dropmax_ratio, sensitive=sensitive, pooling_strategy=pooling_strategy) -> None:
        super().__init__()

        if sensitive in 'attribute':
            self.pooling_dim = 0
            focus_dim = 1
        else:
            self.pooling_dim = 1
            focus_dim = 0
        self.pooling_strategy = pooling_strategy

        self.dropout = dropout
        _in = in_ch
        self.hyconvs = []
        for _h in hiddens:
            _out = _h
            self.hyconvs.append(HyConv(_in, _out))
            _in = _out
        self.hyconvs = nn.ModuleList(self.hyconvs)

        # self.fc1 = nn.Linear(_in, int(_in / 2))
        # self.fc2 = nn.Linear(int(_in / 2), int(_in / 4))
        # self.last_fc = nn.Linear(int(_in / 4), n_target)

        self.drop_max = DropMax(_in, dropmax_ratio, focus_dim)
        self.last_fc = nn.Linear(_in, n_target)

        print("sensitive: {}, pooling_strategy: {}, dropmax_ratio: {}".
              format(sensitive, pooling_strategy, dropmax_ratio))

    def forward(self, x, H, hyedge_weight=None):
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
            x = F.dropout(x, self.dropout)

        feats = x
        if self.pooling_strategy == 'mean':
            x = self.drop_max(x)
            x = x.mean(dim=self.pooling_dim)
        if self.pooling_strategy == 'max':
            x = self.drop_max(x)
            x = x.max(dim=self.pooling_dim)[0]
        feats_pool = x.unsqueeze(0)

        # 1 x C -> 1 x n_target
        # x = self.fc1(feats_pool)
        # x = F.leaky_relu(x)
        # x = self.fc2(x)
        # x = F.leaky_relu(x)
        # x = self.last_fc(x)

        x = self.last_fc(feats_pool)

        # 1 x n_target -> n_target
        x = x.squeeze(0)

        return torch.sigmoid(x), feats, feats_pool


class DropMax(nn.Module):
    def __init__(self, in_ch, dropmax_ratio=0.1, focus_dim=0) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = in_ch
        self.dropmax_ratio = dropmax_ratio
        self.focus_dim = focus_dim

    def forward(self, x):
        if self.training:
            topk_res, idx = x.topk(max((x.shape[self.focus_dim],)), self.focus_dim, True, True)
            if self.focus_dim == 1:
                idx = idx[:, : int(self.dropmax_ratio * x.shape[self.focus_dim])]
            else:
                idx = idx[: int(self.dropmax_ratio * x.shape[self.focus_dim]), :]
            x.scatter_(self.focus_dim, idx, 0.)
        return x


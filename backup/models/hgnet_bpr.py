import torch.nn.functional as F
from torch import nn

from models.HyperG.conv import *
from models.HyperG.hyedge import *
from models.HyperG.hygraph.fusion import hyedge_concat
from train_config import *

criterion = torch.nn.MSELoss()


class HGNet(nn.Module):
    def __init__(self, in_ch, n_target, hiddens=hiddens, dropout=dropout,
                 dropmax_ratio=dropmax_ratio, sensitive=sensitive, pooling_strategy=pooling_strategy) -> None:
        super().__init__()
        self.dropout = dropout
        _in = in_ch
        self.hyconvs = []
        for _h in hiddens:
            _out = _h
            self.hyconvs.append(HyConv(_in, _out))
            self.hyconvs.append(HyConv(_out, _out))  # two layers
            _in = _out
        self.hyconvs = nn.ModuleList(self.hyconvs)

    def forward(self, x, hyedge_weight=None):
        global feats_pool
        H = self.get_H(x)
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
            x = F.dropout(x, self.dropout)

        return x

    def get_H(self, fts, k_nearest=k_nearest):
        return neighbor_distance(fts, k_nearest)


class DiffRankNet(nn.Module):
    def __init__(self, feat_model, feat_size):
        super().__init__()
        self.feat_model = feat_model
        self.last_fc = nn.Linear(feat_size, n_target)

    def forward(self, fts_gt, fts1, k):
        if k == -1:
            H_gt = hyedge_concat([neighbor_distance(fts_gt, 5),
                                  neighbor_distance(fts_gt, 10), neighbor_distance(fts_gt, 15)])
            H1 = hyedge_concat([neighbor_distance(fts1, 5),
                                neighbor_distance(fts1, 10), neighbor_distance(fts1, 15)])
        else:
            H_gt = neighbor_distance(fts_gt, k)
            H1 = neighbor_distance(fts1, k)
        feat_gt = self.feat_model(fts_gt, H_gt)
        feat1 = self.feat_model(fts1, H1)

        if sensitive == 'attribute':
            # N x C -> 1 x C
            feat_gt = feat_gt.mean(dim=0)
            feat1 = feat1.mean(dim=0)
        if sensitive == 'pattern':
            # N x C -> N x 1
            feat_gt = feat_gt.mean(dim=1)
            feat1 = feat1.mean(dim=1)
        # C -> 1 x C
        feat_gt_pool = feat_gt.unsqueeze(0)
        feat1_pool = feat1.unsqueeze(0)

        # method 1
        # x_gt = self.last_fc(feat_gt_pool)
        # x1 = self.last_fc(feat1_pool)
        # x_gt.squeeze(0)
        # x1.squeeze(0)

        # method 0
        y0 = feat_gt_pool.sum().unsqueeze(0).cpu().detach().numpy()[0]
        y1 = feat1_pool.sum().unsqueeze(0).cpu().detach().numpy()[0]

        feats_diff = feat_gt_pool - feat1_pool
        y_compare = feats_diff.sum().unsqueeze(0)

        return torch.sigmoid(y_compare), y0, y1  # method 0

        # return torch.sigmoid(y_compare), torch.sigmoid(x_gt), torch.sigmoid(x1)  # method 1

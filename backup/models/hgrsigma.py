import torch.nn.functional as F
from torch import nn

from models.HyperG.conv import *
from models.HyperG.hyedge import *
from models.hgrmc import MC_Dropout_Layer, MC_Linear
from train_config import *
import numpy as np


class HGRSigma(nn.Module):
    def __init__(self, in_ch, n_target,layers, hiddens=hiddens, dropout=dropout,
                 dropmax_ratio=dropmax_ratio, sensitive=sensitive, pooling_strategy=pooling_strategy) -> None:
        super().__init__()
        self.dropout = dropout
        self.layers = layers
        _in = in_ch
        self.hyconvs = []
        for _h in hiddens:
            _out = _h
            self.hyconvs.append(HyConv(_in, _out))
            for _ in range(self.layers-1):
                self.hyconvs.append(HyConv(_out, _out))  # add additional layers!!!!!!!!!!!
            _in = _out
        self.hyconvs = nn.ModuleList(self.hyconvs)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.last_fc_mean = nn.Linear(_in, n_target)
        self.last_fc_alpha = nn.Linear(_in, n_target)
        # here we use alpha instead of sigma:
        # http://openaccess.thecvf.com/
        # content_CVPR_2019/papers/He_Bounding_Box_Regression_With_Uncertainty_for_Accurate_Object_Detection_CVPR_2019_paper.pdf

    def forward(self, x, hyedge_weight=None):
        global feats_pool
        H = self.get_H(x)
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
            # x = F.dropout(x, self.dropout)
            x = self.dropout_layer(x)

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

        # 2 x C -> 2 x n_target
        mean = self.last_fc_mean(feats_pool)
        alpha = self.last_fc_alpha(feats_pool)  # this output need no activation
        # 2 x 1 x n_target -> 2 x n_target
        mean = mean.squeeze(0)
        alpha = alpha.squeeze(0)
        sig_alpha = torch.sigmoid(alpha)
        log_alpha = torch.log(sig_alpha)  # use log to get -infinity to 0

        return torch.sigmoid(mean), log_alpha, feats, feats_pool

    def get_H(self, fts, k_nearest = k_nearest):
        return neighbor_distance(fts, k_nearest)


class VarianceLossL2(nn.Module):
    def __init__(self):
        super(VarianceLossL2, self).__init__()
        return

    def forward(self, gd_time, mean, alpha):  # longer_st_time_data is a list of hazard
        loss_mse = nn.MSELoss()
        MSE = loss_mse(gd_time, mean)
        loss = 0.5 * torch.exp(-alpha) * MSE + 0.5 * alpha
        return loss


class VarianceLossL1(nn.Module):
    def __init__(self):
        super(VarianceLossL1, self).__init__()
        return

    def forward(self, gd_time, mean, alpha):  # longer_st_time_data is a list of hazard
        loss_L1 = nn.L1Loss()
        L1 = loss_L1(gd_time, mean)
        loss = 0.5 * torch.exp(-alpha) * L1 + 0.5 * alpha
        return loss


class VarianceLossL3(nn.Module):
    def __init__(self):
        super(VarianceLossL3, self).__init__()
        return

    def forward(self, gd_time, mean, alpha):  # longer_st_time_data is a list of hazard
        loss_L1 = nn.L1Loss()
        L1 = loss_L1(gd_time, mean)
        loss = 0.5 * torch.exp(-alpha) * L1 + 0.5 * alpha
        return loss


class ASO_Data():
    def __init__(self, relative_mean_variance, pred_mean, st):
        self.relative_mean_variance = relative_mean_variance
        self.pred_mean = pred_mean
        self.st = st


class ASO_Optimizer():
    def __init__(self):
        self.data = []
        self.sorted = False

    def append_data(self, relative_mean_variance, pred_mean, st):
        self.data.append([relative_mean_variance, pred_mean, st])
        self.sorted = False

    def sort_by_rmv(self):
        self.data.sort(key=lambda d: d[0])
        self.sorted = True

    def get_C_index(self, ratio):
        c_index = CIndexMeter_Notensor()
        selected_data = self.data[:int(len(self.data) * ratio)]
        for d in selected_data:
            c_index.add(d[1], d[2])
        c_index_v = c_index.value()
        return c_index_v


class CIndexMeter_Notensor:
    def __init__(self):
        super(CIndexMeter_Notensor, self).__init__()
        self.reset()

    def reset(self):
        self.output = np.array([])
        self.target = np.array([])

    def add(self, output, target):
        output = np.array([output])
        target = np.array([target])

        assert output.ndim == target.ndim, 'target and output do not match'
        assert output.ndim == 1

        self.output = np.hstack([self.output, output])
        self.target = np.hstack([self.target, target])

    def value(self):
        output = self.output[np.newaxis]
        target = self.target[np.newaxis]

        num_sample = output.shape[-1]
        num_hit = (~((output.T > output) ^ (target.T > target))).sum()

        return float(num_hit - num_sample) / float(num_sample * num_sample - num_sample)
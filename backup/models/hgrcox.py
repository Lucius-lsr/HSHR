import torch.nn.functional as F
from torch import nn

from models.HyperG.conv import *
from models.HyperG.hyedge import *
from train_config import *


class CoxLossFunc(nn.Module):
    def __init__(self):
        super(CoxLossFunc, self).__init__()
        return

    def forward(self, cur_hazard, longer_hazards):  # longer_st_time_data is a list of hazard
        difference = [h - cur_hazard for h in longer_hazards]
        sum_exp_hazards = 0.
        for diff in difference:
            sum_exp_hazards += torch.exp(diff)

        # sum_exp_hazards=torch.from_numpy(sum_exp_hazards)
        loss = torch.log(sum_exp_hazards)
        return loss


criterion = CoxLossFunc()


def cox_cc_loss(g_case, g_control, shrink=0., clamp=(-3e+38, 80.)):
    """Torch loss function for the Cox case-control models.
    For only one control, see `cox_cc_loss_single_ctrl` instead.

    Arguments:
        g_case {torch.Tensor} -- Result of net(input_case)
        g_control {torch.Tensor} -- Results of [net(input_ctrl1), net(input_ctrl2), ...]

    Keyword Arguments:
        shrink {float} -- Shrinkage that encourage the net got give g_case and g_control
            closer to zero (a regularizer in a sense). (default: {0.})
        clamp {tuple} -- See code (default: {(-3e+38, 80.)})

    Returns:
        [type] -- [description]
    """
    control_sum = 0.
    shrink_control = 0.
    if g_case.shape != g_control.shape:
        raise ValueError(f"Need `g_case` and `g_control` to have same shape. Got {g_case.shape}" +
                         f" and {g_control.shape}")
    for ctr in g_control:
        shrink_control += ctr.abs().mean()
        ctr = ctr - g_case
        ctr = torch.clamp(ctr, *clamp)  # Kills grads for very bad cases (should instead cap grads!!!).
        control_sum += torch.exp(ctr)
    loss = torch.log(1. + control_sum)
    shrink_zero = shrink * (g_case.abs().mean() + shrink_control) / len(g_control)
    return torch.mean(loss) + shrink_zero.abs()


class HGRCox(nn.Module):
    def __init__(self, in_ch, n_target, hiddens=hiddens, dropout=dropout,
                 dropmax_ratio=dropmax_ratio, sensitive=sensitive, pooling_strategy=pooling_strategy) -> None:
        super().__init__()
        if hiddens is None:
            hiddens = [16]
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

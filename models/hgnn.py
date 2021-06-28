import torch.nn.functional as F
from torch import nn

from models.HyperG.conv import HyConv


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, hiddens=None, dropout=0.5) -> None:
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
        self.last_hyconv = HyConv(_in, n_class)

    def forward(self, x, H, hyedge_weight=None):
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
            x = F.dropout(x, self.dropout)
        x = self.last_hyconv(x, H)
        return F.log_softmax(x, dim=1)

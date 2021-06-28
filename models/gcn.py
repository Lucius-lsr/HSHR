import math

import torch
import torch.nn.functional as F
from torch import nn

from models.HyperG.hyedge.distance_metric import pairwise_euclidean_distance


def gen_adjacency(feature_matrix, threshold=0.8):
    dis = pairwise_euclidean_distance(feature_matrix)
    adj = torch.zeros(dis.shape)
    adj[dis <= threshold] = 1
    return adj, 8  #, Counter(adj.flatten())[1.0]


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet_reg(nn.Module):
    def __init__(self, fts, n_target, hiddens=None, dropout=0.5):
        super().__init__()
        if hiddens is None:
            hiddens = [2]
        self.dropout = dropout
        _in = fts
        self.convs = []
        for _h in hiddens:
            _out = _h
            self.convs.append(GraphConvolution(_in, _out))
            _in = _out
        self.convs = nn.ModuleList(self.convs)
        self.last_fc = nn.Linear(_in, n_target)

    def forward(self, x, H, hyedge_weight=None):
        for conv in self.convs:
            x = conv(x, H)
            x = F.leaky_relu(x, inplace=True)
            x = F.dropout(x, self.dropout)

        # N x C -> C
        x = x.mean(dim=0)

        # C -> 1 x C
        x = x.unsqueeze(0)
        # 1 x C -> 1 x n_target
        x = self.last_fc(x)
        # 1 x n_target -> n_target
        x = x.squeeze(0)

        return torch.sigmoid(x)

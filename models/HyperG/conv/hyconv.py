import torch
from torch import nn
from torch.nn import Parameter

from models.HyperG.hyedge import degree_hyedge, degree_node, count_hyedge, count_node


class HyConv(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True) -> None:
        super().__init__()
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))

        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def gen_hyedge_ft(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        hyedge_num = count_hyedge(H)

        # a vector to normalize hyperedge feature
        hyedge_norm = 1.0 / degree_hyedge(H).float()
        if hyedge_weight is not None:
            hyedge_norm *= hyedge_weight
        hyedge_norm = hyedge_norm[hyedge_idx]

        x = x[node_idx] * hyedge_norm.unsqueeze(1)
        x = torch.zeros(hyedge_num, ft_dim).to(x.device).scatter_add(0, hyedge_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x

    def gen_node_ft(self, x: torch.Tensor, H: torch.Tensor):
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        node_num = count_node(H)

        # a vector to normalize node feature
        node_norm = 1.0 / degree_node(H).float()
        node_norm = node_norm[node_idx]

        x = x[hyedge_idx] * node_norm.unsqueeze(1)
        x = torch.zeros(node_num, ft_dim).to(x.device).scatter_add(0, node_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x

    def forward(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        assert len(x.shape) == 3, 'the input of HyperConv should be B x N x C'
        x = x.matmul(self.theta)

        x_out_list = []
        for x_in, H_in in zip(x, H):
            # feature transform

            # generate hyperedge feature from node feature
            x_edge = self.gen_hyedge_ft(x_in, H_in, hyedge_weight)
            # generate node feature from hyperedge feature
            x_out = self.gen_node_ft(x_edge, H_in)
            if self.bias is not None:
                x_out = x_out + self.bias
            else:
                x_out = x_out
            assert x_out.shape[0] == x.shape[1]
            x_out_list.append(x_out.unsqueeze(0))
        x_out = torch.cat(x_out_list, dim=0)
        return x_out

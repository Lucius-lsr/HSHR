import torch.nn.functional as F

from models.HyperG.conv import HyConv
from models.HyperG.hyedge import neighbor_distance
from models.cnn import *


class ResNet_HGNN(nn.Module):
    def __init__(self, n_class, depth, k_nearest, hiddens=None, dropout=0.5, pretrained=True):
        super().__init__()
        if hiddens is None:
            hiddens = [512]
        self.dropout = dropout
        self.k_nearest = k_nearest
        self.ft_layers = ResNetFeature(depth=depth, pretrained=pretrained)

        # hypergraph convolution for feature refine
        self.hyconvs = []
        dim_in = self.ft_layers.len_feature
        for h in hiddens:
            dim_out = h
            self.hyconvs.append(HyConv(dim_in, dim_out))
            dim_in = dim_out
        self.hyconvs = nn.ModuleList(self.hyconvs)

        self.cls_layers = ResNetClassifier(n_class=n_class, len_feature=dim_in)

    def forward(self, x):
        x = self.ft_layers(x)

        assert x.size(0) == 1, 'when construct hypergraph, only support batch size = 1!'
        x = x.view(x.size(1), x.size(2) * x.size(3))
        # -> N x C
        x = x.permute(1, 0)
        H = neighbor_distance(x, k_nearest=self.k_nearest)
        # Hypergraph Convs
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
            x = F.dropout(x, self.dropout)
        # N x C -> 1 x C x N
        x = x.permute(1, 0).unsqueeze(0)

        x = self.cls_layers(x)

        return x

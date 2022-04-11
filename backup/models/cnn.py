import torch
from torch import nn
from models.base_cnns import ResNetClassifier, ResNetFeature


class ResNetRegression(nn.Module):

    def __init__(self, len_feature, n_target, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.len_feature = len_feature
        self.last_fc = nn.Linear(self.len_feature, n_target)

    def forward(self, x):
        # N x C -> C
        x = x.mean(dim=0)

        # C -> 1 x C
        x = x.unsqueeze(0)
        # 1 x C -> 1 x n_target
        x = self.last_fc(x)
        # 1 x n_target -> n_target
        x = x.squeeze(0)

        return torch.sigmoid(x)


class ResNet(nn.Module):
    def __init__(self, n_class, depth=34, pretrained=True):
        super().__init__()

        self.ft_layers = ResNetFeature(depth=depth, pretrained=pretrained)
        self.cls_layers = ResNetClassifier(n_class=n_class, len_feature=self.ft_layers.len_feature)

    def forward(self, x):
        x = self.ft_layers(x)
        x = self.cls_layers(x)

        return x

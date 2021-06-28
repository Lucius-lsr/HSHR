import torch.nn as nn
import torch.nn.functional as F

from models.HyperG.conv import *
from models.HyperG.hyedge import *
from train_config import *

criterion = torch.nn.MSELoss()


def MC_dropout(act_vec, p=0.5, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=True)


class MC_Dropout_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob):
        super(MC_Dropout_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.biases = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))

    def forward(self, x):
        dropout_mask = torch.bernoulli((1 - self.dropout_prob) * torch.ones(self.weights.shape)).cuda()

        return torch.mm(x, self.weights * dropout_mask) + self.biases


class MC_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid, dropout_prob):
        super(MC_Linear, self).__init__()

        self.pdrop = dropout_prob

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, output_dim)

        # choose your non linearity
        # self.activate = nn.Tanh()
        # self.activate = nn.Sigmoid()
        self.activate = nn.LeakyReLU(inplace=True)
        # self.activate = nn.ReLU(inplace=True)
        # self.activate = nn.ELU(inplace=True)
        # self.activate = nn.SELU(inplace=True)

    def forward(self, x, sample=True):
        mask = self.training or sample
        # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)

        x = self.fc1(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        x = self.activate(x)

        x = self.fc2(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        x = self.activate(x)

        y = self.fc3(x)

        return y

    def sample_predict(self, x, N_samples):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(N_samples, x.shape[0], self.output_dim)

        for i in range(N_samples):
            y = self.forward(x, sample=True)
            predictions[i] = y

        return predictions


class HGRMC(nn.Module):
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
        self.dropout_layer = torch.nn.Dropout(self.dropout)

    def forward(self, x, hyedge_weight=None):
        global feats_pool
        H = self.get_H(x)
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
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

        # 1 x C -> 1 x n_target
        x = self.last_fc(feats_pool)
        # 1 x n_target -> n_target
        x = x.squeeze(0)

        return torch.sigmoid(x), feats, feats_pool

    def get_H(self, fts, k_nearest=k_nearest):
        return neighbor_distance(fts, k_nearest)

import copy
import pickle

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import os
import torch

from baseline_supervised import ClassifyLayer, Labeler
from data.utils import get_files_type
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import random

from evaluate import Evaluator
from self_supervision.call import get_moco

feature_and_coordinate_dir = '/home2/lishengrui/all_tcga'

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, num_patches, num_classes, dim, pos_dim, depth, heads, mlp_dim, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # image_size = 256,
        # patch_size = 32,
        # num_classes = 1000,
        # dim = 1024,
        # depth = 6,
        # heads = 16,
        # mlp_dim = 2048,
        # dropout = 0.1,
        # emb_dropout = 0.1

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     nn.Linear(patch_dim, dim),
        # )

        self.pos_embedding_x = nn.Parameter(torch.randn(1, num_patches + 1, pos_dim))
        self.pos_embedding_y = nn.Parameter(torch.randn(1, num_patches + 1, pos_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        dim = dim+2*pos_dim

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, input):
        feature, x_idx, y_idx = input
        # x = self.to_patch_embedding(feature)
        x = feature
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        e_x = self.pos_embedding_x[:, 1:, :].repeat(b, 1, 1)
        for i in range(b):
            e_x[i] = e_x[i, x_idx[i], :]
        e_x = torch.cat((self.pos_embedding_x[:, 0, :].repeat(b, 1, 1), e_x), dim=1)
        x = torch.cat((x, e_x), dim=2)

        e_y = self.pos_embedding_y[:, 1:, :].repeat(b, 1, 1)
        for i in range(b):
            e_y[i] = e_y[i, y_idx[i], :]
        e_y = torch.cat((self.pos_embedding_y[:, 0, :].repeat(b, 1, 1), e_y), dim=1)
        x = torch.cat((x, e_y), dim=2)

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class TransformerDataset(Dataset):

    def __init__(self, data_from=0., data_to=1.) -> None:
        super().__init__()
        self.data = list()
        feature_list = get_files_type(feature_and_coordinate_dir, 'npy')
        feature_list.sort()
        # shuffle
        r = random.random
        random.seed(6)
        random.shuffle(feature_list, r)
        size = len(feature_list)
        for feature_path in feature_list[int(data_from * size):int(data_to * size)]:
            base_name = os.path.basename(feature_path)
            dir_name = os.path.join(feature_and_coordinate_dir, os.path.dirname(feature_path))
            if base_name == '0.npy':
                files = os.listdir(dir_name)
                if '1.npy' in files and '0.pkl' in files and '1.pkl' in files:
                    feature_coordinate = dir_name
                    self.data.append(feature_coordinate)

    def __getitem__(self, idx: int):
        dir_name = self.data[idx]

        feature_0 = np.load(dir_name + '/0.npy')
        with open(dir_name + '/0.pkl', 'rb') as f:
            coordinate_0 = pickle.load(f)
        x_idx_0, y_idx_0 = self.get_permutation(coordinate_0)

        feature_1 = np.load(dir_name + '/0.npy')
        with open(dir_name + '/0.pkl', 'rb') as f:
            coordinate_1 = pickle.load(f)
        x_idx_1, y_idx_1 = self.get_permutation(coordinate_1)

        return feature_0, x_idx_0, y_idx_0, feature_1, x_idx_1, y_idx_1, dir_name

    def get_permutation(self, coordinate):
        x_list, y_list = list(), list()
        for x, y, _, _ in coordinate:
            x_list.append(x)
            y_list.append(y)
        x, y = np.array(x_list), np.array(y_list)
        x_idx = torch.argsort(torch.from_numpy(x))
        y_idx = torch.argsort(torch.from_numpy(y))
        return x_idx, y_idx

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':
    batch_size = 16
    feature_out = 128
    lr = 0.03
    momentum = 0.9
    weight_decay = 1e-4
    class_num = 30
    criterion = nn.CrossEntropyLoss().cuda(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ViT(num_patches=2000, num_classes=128, dim=512, pos_dim=24, depth=2, heads=4, mlp_dim=128, dropout=0.5)

    last_layer = ClassifyLayer(feature_out, class_num)
    last_layer = last_layer.to(device)
    labeler = Labeler(class_num)

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    train_dataset = TransformerDataset(0, 0.8)
    test_dataset = TransformerDataset(0.8, 1)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    evaluator = Evaluator()
    for epoch in range(500):
        print('*' * 5, 'epoch: ', epoch, '*' * 5)
        loss_sum = 0
        loss_count = 0
        for feature_0, x_idx_0, y_idx_0, feature_1, x_idx_1, y_idx_1, path in train_dataloader:
            feature_0, x_idx_0, y_idx_0 = feature_0.to(device), x_idx_0.to(device), y_idx_0.to(device)
            feature = model((feature_0, x_idx_0, y_idx_0))
            output = last_layer(feature)
            label = labeler.get_label(path).to(device)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            loss_count += 1
        loss_ave = loss_sum / loss_count
        print("loss: ", loss_ave)

        # ----------------val-----------------
        evaluator.reset()
        for feature_0, x_idx_0, y_idx_0, feature_1, x_idx_1, y_idx_1, path in test_dataloader:
            feature_0, x_idx_0, y_idx_0 = feature_0.to(device), x_idx_0.to(device), y_idx_0.to(device)
            feature_1, x_idx_1, y_idx_1 = feature_1.to(device), x_idx_1.to(device), y_idx_1.to(device)
            with torch.no_grad():
                f_0 = model((feature_0, x_idx_0, y_idx_0)).cpu().detach().numpy()
                f_1 = model((feature_1, x_idx_1, y_idx_1)).cpu().detach().numpy()
            evaluator.add_result(f_0, f_1, path)
        top1, top3, top5, top10, top1_, top3_, top5_, top10_ = evaluator.report()
        print('class acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(top1, top3, top5, top10))

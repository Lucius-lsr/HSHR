# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/24 13:03
@Author  : Lucius
@FileName: pooling.py
@Software: PyCharm
"""

import os
import torch
from models.hgrnet import *

# tmp
len_ft = 512
n_target = 1
hiddens = [128]
dropout = 0.5
pooling_strategy = 'mean'
k_neighbors = [10]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_pooling_model():
    pooling_model = HGRNet(in_ch=len_ft, n_target=n_target, hiddens=hiddens, dropout=dropout, sensitive='attribute',
                           pooling_strategy=pooling_strategy, k_neighbors=k_neighbors)
    pooling_model = pooling_model.to(device)
    return pooling_model

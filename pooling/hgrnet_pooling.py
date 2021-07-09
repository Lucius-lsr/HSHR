# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/24 13:03
@Author  : Lucius
@FileName: hgrnet_pooling.py
@Software: PyCharm
"""

import os
import torch
from hgrnet import HGRNet

# tmp
len_ft = 512
n_target = 1
hiddens = [128]
dropout = 0.5
pooling_strategy = 'mean'
k_neighbors = [10]


def get_pooling_model(dim):
    pooling_model = HGRNet(in_ch=len_ft, n_target=n_target, hiddens=[dim], dropout=dropout, sensitive='attribute',
                           pooling_strategy=pooling_strategy, k_neighbors=k_neighbors)
    return pooling_model

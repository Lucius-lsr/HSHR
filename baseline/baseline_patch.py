# -*- coding: utf-8 -*-
"""
@Time    : 2021/10/6 11:28
@Author  : Lucius
@FileName: baseline_patch.py
@Software: PyCharm
"""
import torch
from sklearn.metrics.pairwise import cosine_similarity

from data.utils import get_files_type
import os
from tqdm import tqdm
import numpy as np
from evaluate import accuracy
import random

FEATURE_COORDINATE_DIR = '/home2/lishengrui/tcga_result/all_tcga'

K = 100
patch_feature_list_map = dict()
path_list_map = dict()

for i in range(K):
    patch_feature_list_map[i] = list()
    path_list_map[i] = list()

feature_list = get_files_type(FEATURE_COORDINATE_DIR, 'npy')
random.shuffle(feature_list)
for feature_path in tqdm(feature_list):
    base_name = os.path.basename(feature_path)
    dir_name = os.path.join(FEATURE_COORDINATE_DIR, os.path.dirname(feature_path))
    if base_name == '0.npy':
        files = os.listdir(dir_name)
        if '1.npy' in files:
            path = os.path.dirname(feature_path)
            feature_0 = np.load(os.path.join(dir_name, '0.npy'))
            for i in range(K):
                patch_feature_list_map[i].append(feature_0[i])
                path_list_map[i].append(path)

all_top1, all_top3, all_top5, all_top10 = 0, 0, 0, 0
for i in range(K):
    x = np.array(patch_feature_list_map[i])
    dis_matrix = -cosine_similarity(x)  # better
    _, top_idx = torch.topk(torch.from_numpy(dis_matrix), 11, dim=1, largest=False)

    top1, top3, top5, top10 = accuracy(path_list_map[i], top_idx)
    all_top1 += top1
    all_top3 += top3
    all_top5 += top5
    all_top10 += top10
all_top1, all_top3, all_top5, all_top10 = all_top1 / K, all_top3 / K, all_top5 / K, all_top10 / K
print(all_top1, all_top3, all_top5, all_top10)

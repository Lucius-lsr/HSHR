# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/29 14:27
@Author  : Lucius
@FileName: dataloader.py
@Software: PyCharm
"""

import os
import torch
from data.utils import get_files_type
from torch.utils.data import Dataset

from models.HyperG.hyedge import pairwise_euclidean_distance
import numpy as np
import random


class RandomHyperGraph(Dataset):

    def __init__(self, feature_and_coordinate_dir, k, data_from=0, data_to=1) -> None:
        super().__init__()
        self.data_pairs = list()
        self.k = k
        feature_list = get_files_type(feature_and_coordinate_dir, '0.npy')
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
                    feature_coordinate_0 = (os.path.join(dir_name, '0.npy'), os.path.join(dir_name, '0.pkl'))
                    feature_coordinate_1 = (os.path.join(dir_name, '1.npy'), os.path.join(dir_name, '1.pkl'))
                    self.data_pairs.append((feature_coordinate_0, feature_coordinate_1))

    def __getitem__(self, idx: int):
        dir_name = os.path.dirname(self.data_pairs[idx][0][0])

        feature_0 = np.load(dir_name + '/0.npy')
        nearest_idx_0 = np.load(dir_name + '/0_nearest_idx.npy')
        # with open(self.data_pairs[idx][0][1], 'rb') as f:
        #     coordinate_0 = pickle.load(f)
        # H_0 = self.get_random_H(coordinate_0, feature_0.shape[0], dir_name+'/0_nearest_idx.npy')

        feature_1 = np.load(dir_name + '/1.npy')
        nearest_idx_1 = np.load(dir_name + '/1_nearest_idx.npy')
        # with open(self.data_pairs[idx][1][1], 'rb') as f:
        #     coordinate_1 = pickle.load(f)
        # H_1 = self.get_random_H(coordinate_1, feature_1.shape[0], dir_name+'/1_nearest_idx.npy')

        return feature_0, nearest_idx_0, feature_1, nearest_idx_1, dir_name
        # return feature_0, H_0, feature_1, H_1, dir_name

    def get_random_H(self, coordinate, size, file_name):
        loc = []
        for x, y, _, _ in coordinate:
            loc.append([x, y])
        loc = np.array(loc)
        loc = loc / np.max(loc)

        dis_matrix = pairwise_euclidean_distance(torch.from_numpy(loc))
        _, nn_idx = torch.topk(dis_matrix, 2 * self.k, dim=1, largest=False)

        # random choose k in 2 * k

        nn_idx = nn_idx[:, 1:]

        nearest_idx = nn_idx
        np.save(file_name, nearest_idx.numpy())

        return [1]

        self_idx = torch.arange(nn_idx.shape[0]).reshape(-1, 1)
        sample_nn_idx = nn_idx[:, torch.randperm(2 * self.k - 1)]
        sample_nn_idx = sample_nn_idx[:, :self.k - 1]
        sample_nn_idx = torch.cat((self_idx, sample_nn_idx), dim=1)
        hyedge_idx = torch.arange(size).unsqueeze(0).repeat(self.k, 1).transpose(1, 0).reshape(-1)
        H = torch.stack([sample_nn_idx.reshape(-1), hyedge_idx])

        return H

    def __len__(self) -> int:
        return len(self.data_pairs)

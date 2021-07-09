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
from torch.utils.data import Dataset, DataLoader

from models.HyperG.hyedge import pairwise_euclidean_distance
import numpy as np
import pickle

FEATURE_DIR = '/Users/lishengrui/client/tmp'
COORDINATE_DIR = '/Users/lishengrui/client/tmp'


class RandomHyperGraph(Dataset):

    def __init__(self, feature_dir, coordinate_dir, k, with_name=False) -> None:
        super().__init__()
        self.feature_coordinate_pairs = list()
        self.k = k
        self.with_name = with_name
        feature_list = get_files_type(feature_dir, 'npy')
        coordinate_list = get_files_type(coordinate_dir, 'pkl')
        for feature_path in feature_list:
            coordinate_path = feature_path.replace('.npy', '.pkl')
            if coordinate_path in coordinate_list:
                self.feature_coordinate_pairs.append(
                    (os.path.join(feature_dir, feature_path), (os.path.join(coordinate_dir, coordinate_path))))

    def __getitem__(self, idx: int):
        feature = np.load(self.feature_coordinate_pairs[idx][0])
        with open(self.feature_coordinate_pairs[idx][1], 'rb') as f:
            coordinate = pickle.load(f)

        loc = []
        for x, y, _, _ in coordinate:
            loc.append([x, y])
        loc = np.array(loc)
        loc = loc / np.max(loc)
        dis_matrix = pairwise_euclidean_distance(torch.from_numpy(loc))
        _, nn_idx = torch.topk(dis_matrix, 2*self.k, dim=1, largest=False)

        hyedge_idx = torch.arange(feature.shape[0]).unsqueeze(0).repeat(self.k, 1).transpose(1, 0).reshape(-1)

        # random choose k in 2 * k
        self_idx = torch.arange(nn_idx.shape[0]).reshape(-1, 1)
        nn_idx = nn_idx[:, 1:]

        sample_nn_idx1 = nn_idx[:, torch.randperm(2*self.k-1)]
        sample_nn_idx1 = sample_nn_idx1[:, :self.k-1]
        sample_nn_idx1 = torch.cat((self_idx, sample_nn_idx1), dim=1)
        H1 = torch.stack([sample_nn_idx1.reshape(-1), hyedge_idx])

        sample_nn_idx2 = nn_idx[:, torch.randperm(2*self.k-1)]
        sample_nn_idx2 = sample_nn_idx2[:, :self.k-1]
        sample_nn_idx2 = torch.cat((self_idx, sample_nn_idx2), dim=1)
        H2 = torch.stack([sample_nn_idx2.reshape(-1), hyedge_idx])

        if self.with_name:
            return feature, H1, H2, self.feature_coordinate_pairs[idx]
        else:
            return feature, H1, H2

    def __len__(self) -> int:
        return len(self.feature_coordinate_pairs)


def get_dataset(feature_dir, coordinate_dir, k, with_name=False):
    dataset = RandomHyperGraph(feature_dir, coordinate_dir, k, with_name)
    return dataset


if __name__=='__main__':

    dataset = get_dataset(FEATURE_DIR, COORDINATE_DIR, 10)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, (feature, H1, H2) in enumerate(dataloader):

        print(i)
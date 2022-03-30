# -*- coding: utf-8 -*-
"""
@Time    : 2021/11/4 19:08
@Author  : Lucius
@FileName: swin_dataset.py
@Software: PyCharm
"""

import os
import pickle

from tqdm import tqdm

from data.utils import get_files_type
from torch.utils.data import Dataset

import numpy as np
import random


def numpy_coordinate(coordinate_file):
    with open(coordinate_file, 'rb') as f:
        coordinate = pickle.load(f)
    loc = []
    for x, y, w, h in coordinate:
        loc.append([x + w / 2, y + h / 2])
    loc = np.array(loc)
    return loc


class PatchesInLevels(Dataset):

    def __init__(self, feature_and_coordinate_dir, ignore: list = None, require: list = None, data_from=0,
                 data_to=1) -> None:
        super().__init__()
        self.feature_1x_0 = []
        self.feature_1x_1 = []
        self.feature_2x = []
        self.feature_4x = []
        self.coordinate_1x_0 = []
        self.coordinate_1x_1 = []
        self.coordinate_2x = []
        self.coordinate_4x = []
        self.dir_names = []

        feature_list = get_files_type(feature_and_coordinate_dir, '0.npy')
        feature_list.sort()
        r = random.random
        random.seed(6)
        random.shuffle(feature_list, r)
        size = len(feature_list)
        requires = ['1.npy', '0.pkl', '1.pkl', '2x.npy', '2x.pkl', '4x.npy', '4x.pkl']
        for feature_path in tqdm(feature_list[int(data_from * size):int(data_to * size)]):
            base_name = os.path.basename(feature_path)
            dir_name = os.path.join(feature_and_coordinate_dir, os.path.dirname(feature_path))
            if base_name == '0.npy':
                files = os.listdir(dir_name)
                ok = True
                for r in requires:
                    if r not in files:
                        ok = False
                        break
                tcga = dir_name.split('/')[-2]
                if ignore is not None and tcga in ignore:
                    continue
                if require is not None and tcga not in require:
                    continue
                if ok:
                    self.feature_1x_0.append(os.path.join(dir_name, '0.npy'))
                    self.feature_1x_1.append(os.path.join(dir_name, '1.npy'))
                    self.feature_2x.append(os.path.join(dir_name, '2x.npy'))
                    self.feature_4x.append(os.path.join(dir_name, '4x.npy'))
                    loc_0 = numpy_coordinate(os.path.join(dir_name, '0.pkl'))
                    loc_1 = numpy_coordinate(os.path.join(dir_name, '1.pkl'))
                    loc_2x = numpy_coordinate(os.path.join(dir_name, '2x.pkl'))
                    loc_4x = numpy_coordinate(os.path.join(dir_name, '4x.pkl'))
                    max_l = np.max(np.concatenate((loc_0, loc_1, loc_2x, loc_4x)))
                    loc_0, loc_1, loc_2x, loc_4x = loc_0 / max_l, loc_1 / max_l, loc_2x / max_l, loc_4x / max_l
                    self.coordinate_1x_0.append(loc_0)
                    self.coordinate_1x_1.append(loc_1)
                    self.coordinate_2x.append(loc_2x)
                    self.coordinate_4x.append(loc_4x)
                    self.dir_names.append(dir_name)

    def __getitem__(self, index):
        return np.load(self.feature_1x_0[index]), \
               np.load(self.feature_1x_1[index]), \
               np.load(self.feature_2x[index]), \
               np.load(self.feature_4x[index]), \
               self.coordinate_1x_0[index], \
               self.coordinate_1x_1[index], \
               self.coordinate_2x[index], \
               self.coordinate_4x[index], \
               self.dir_names[index]

    def __len__(self) -> int:
        return len(self.feature_1x_0)

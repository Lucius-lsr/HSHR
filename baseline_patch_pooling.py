# -*- coding: utf-8 -*-
"""
@Time    : 2021/9/22 13:53
@Author  : Lucius
@FileName: baseline_patch_pooling.py
@Software: PyCharm
"""
import os
import random

import numpy as np

from data.utils import get_files_type
from evaluate import Evaluator
from tqdm import tqdm

FEATURE_COORDINATE_DIR = '/home2/lishengrui/all_tcga'


def raw_sample_result(evaluator, data_from, data_to):
    feature_list = get_files_type(FEATURE_COORDINATE_DIR, 'npy')
    feature_list.sort()
    size = len(feature_list)
    r = random.random
    random.seed(6)
    random.shuffle(feature_list, r)
    for feature_path in tqdm(feature_list[int(data_from*size):int(data_to*size)]):
        base_name = os.path.basename(feature_path)
        dir_name = os.path.join(FEATURE_COORDINATE_DIR, os.path.dirname(feature_path))
        if base_name == '0.npy':
            files = os.listdir(dir_name)
            if '1.npy' in files:
                path = os.path.dirname(feature_path)
                feature_0 = np.load(os.path.join(dir_name, '0.npy'))
                ave_feature_0 = feature_0.mean(axis=0)
                feature_1 = np.load(os.path.join(dir_name, '1.npy'))
                ave_feature_1 = feature_1.mean(axis=0)
                evaluator.add_result(ave_feature_0, ave_feature_1, path)


if __name__ == '__main__':
    evaluator = Evaluator()
    raw_sample_result(evaluator, 0.8, 1)
    top1, top3, top5, top10, top1_, top3_, top5_, top10_ = evaluator.report()
    print('class acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(top1, top3, top5, top10))
    # print('pair acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(top1_, top3_, top5_, top10_))


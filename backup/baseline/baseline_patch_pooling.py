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
import sys
sys.path.append("..")

from utils.data_utils import get_files_type
from utils.evaluate import Evaluator
from tqdm import tqdm


FEATURE_COORDINATE_DIR = '/home2/lishengrui/all_tcga'

target_feature = '0.npy'


def raw_sample_result(evaluator, data_from, data_to):
    feature_list = get_files_type(FEATURE_COORDINATE_DIR, '0.npy')
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
            if target_feature in files:
                path = os.path.dirname(feature_path)
                feature_0 = np.load(os.path.join(dir_name, target_feature))
                ave_feature_0 = feature_0.mean(axis=0)
                while len(ave_feature_0.shape) > 1:
                    ave_feature_0 = ave_feature_0.squeeze()
                evaluator.add_result(ave_feature_0, path)


if __name__ == '__main__':
    evaluator = Evaluator()
    raw_sample_result(evaluator, 0, 1)
    print(evaluator.report_mmv(['gbm_tcga', 'lgg_tcga']))
    print(evaluator.report_mmv(['coad_tcga', 'esca_tcga', 'read_tcga', 'stad_tcga']))
    print(evaluator.report_mmv(['ucec_tcga', 'cesc_tcga', 'ucs_tcga', 'ov_tcga']))
    print(evaluator.report_mmv(['dlbc_tcga', 'thym_tcga']))
    print(evaluator.report_mmv(['uvm_tcga', 'skcm_tcga']))
    print(evaluator.report_mmv(['lusc_tcga', 'luad_tcga', 'meso_tcga']))
    print(evaluator.report_mmv(['blca_tcga', 'kirc_tcga', 'kich_tcga', 'kirp_tcga']))
    print(evaluator.report_mmv(['tgct_tcga', 'prad_tcga']))


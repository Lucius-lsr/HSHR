# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/29 15:32
@Author  : Lucius
@FileName: utils.py
@Software: PyCharm
"""
import os
import pickle

import torch
from torch import nn

TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp'


def get_files_type(directory, file_suffix):
    tmp_file = TMP + '/{}_{}.pkl'.format(directory.replace('/', '*'), file_suffix.replace('.', '*'))
    if os.path.exists(tmp_file):
        with open(tmp_file, 'rb') as f:
            svs_list = pickle.load(f)
    else:
        svs_list = list()
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(file_suffix):
                    relative_root = root[len(directory) + 1:]
                    svs_list.append(os.path.join(relative_root, file))
        with open(tmp_file, 'wb') as fp:
            pickle.dump(svs_list, fp)
    return svs_list


def check_todo(result_root, svs_list, to_dos):
    to_do_list = list()
    for svs_relative_path in svs_list:
        file_relative_dir = os.path.dirname(svs_relative_path)
        result_dir = os.path.join(result_root, file_relative_dir)
        if not os.path.exists(result_dir):
            to_do_list.append(svs_relative_path)
        else:
            files = os.listdir(result_dir)
            for to_do in to_dos:
                if to_do not in files:
                    to_do_list.append(svs_relative_path)
                    break

    return to_do_list


def check_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return file_path


def get_save_path(root_dir, svs_relative_path, file_name):
    fake_path = os.path.join(root_dir, svs_relative_path)
    file_dir = os.path.dirname(fake_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return os.path.join(file_dir, file_name)


class ClassifyLayer(nn.Module):
    def __init__(self, dim_feature, num_class) -> None:
        super().__init__()
        self.last_fc = nn.Linear(dim_feature, num_class)

    def forward(self, f):
        return self.last_fc(f)


class Labeler:
    def __init__(self):
        self.class_type = [
            'acc_tcga',
            'cesc_tcga',
            'dlbc_tcga',
            'hnsc_tcga',
            'kirp_tcga',
            'luad_tcga',
            'ov_tcga',
            'read_tcga',
            'stad_tcga',
            'ucec_tcga',
            'blca_tcga',
            'chol_tcga',
            'esca_tcga',
            'kich_tcga',
            'lgg_tcga',
            'lusc_tcga',
            'pcpg_tcga',
            'sarc_tcga',
            'tgct_tcga',
            'ucs_tcga',
            'brca_tcga',
            'coad_tcga',
            'gbm_tcga',
            'kirc_tcga',
            'lihc_tcga',
            'meso_tcga',
            'prad_tcga',
            'skcm_tcga',
            'thym_tcga',
            'uvm_tcga'
        ]
        self.class_record = dict()
        for i, c in enumerate(self.class_type):
            self.class_record[c] = i

    def get_label(self, path_batch):
        label = []
        for p in path_batch:
            class_name = p.split("/")[-2]
            label.append(self.class_record[class_name])
        return torch.LongTensor(label)

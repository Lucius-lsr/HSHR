# -*- coding: utf-8 -*-
"""
@Time    : 2021/7/9 10:55
@Author  : Lucius
@FileName: evaluate.py
@Software: PyCharm
"""
import shutil
import time

from data.dataloader import get_dataset
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import numpy as np

FEATURE_DIR = '/home/lishengrui/TCGA_experiment/lusc_tcga'
COORDINATE_DIR = '/home/lishengrui/TCGA_experiment/lusc_tcga'
DATABASE_DIR = '/home/lishengrui/TCGA_experiment/result_lusc_tcga'
MODEL_DIR = '/home/lishengrui/TCGA_experiment/result_lusc_tcga/models'
MODEL_ID = '2021-07-07-15-44-20'
RETRIEVAL_DIR = '/home/lishengrui/TCGA_experiment/result_lusc_tcga/retrieval'

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_feature_database(feature_dir, coordinate_dir, database_dir, model_dir, model_id):
    dataset = get_dataset(feature_dir, coordinate_dir, 10, True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0)

    model_path = os.path.join(model_dir, model_id, 'model_best.pth')
    model = torch.load(model_path)
    encoder_q = model.encoder_q

    result_dict = {}
    with torch.no_grad():
        for feature, H1, H2, path in tqdm(dataloader):
            # hidden_1 = encoder_q((feature.to(device), H1.to(device)))
            # result_dict['1_'+path[0][0]] = hidden_1.cpu().detach().numpy()
            # hidden_2 = encoder_q((feature.to(device), H2.to(device)))
            # result_dict['2_'+path[0][0]] = hidden_2.cpu().detach().numpy()
            hidden = encoder_q((feature.to(device), H1.to(device)))
            result_dict[path[0][0]] = hidden.cpu().detach().numpy()
    np.save(os.path.join(database_dir, 'database.npy'), result_dict)


def get_distance_matrix(x):
    G = np.dot(x, x.T)
    # 把G对角线元素拎出来，列不变，行复制n遍。
    H = np.tile(np.diag(G), (x.shape[0], 1))
    D = H + H.T - G * 2
    return D


def retrieval(database_dict):
    key_list = []
    feature_list = []
    for key in database_dict.keys():
        key_list.append(key)
        feature_list.append(database_dict[key])
    x = np.concatenate(feature_list)
    dis_matrix = get_distance_matrix(x)
    _, top_idx = torch.topk(torch.from_numpy(dis_matrix), 11, dim=1, largest=False)
    return key_list, top_idx


def collect_image(key_list, idxs, save_to):
    result_id = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    os.mkdir(os.path.join(save_to, result_id))
    for i, idx in enumerate(idxs):
        npy_file = key_list[idx]
        png_file = npy_file.replace('.npy', '.png')
        shutil.copy(png_file, os.path.join(save_to, result_id, '{}.png'.format(i)))


if __name__ == '__main__':
    database_path = os.path.join(DATABASE_DIR, 'database.npy')
    if not os.path.exists(database_path):
        save_feature_database(FEATURE_DIR, COORDINATE_DIR, DATABASE_DIR, MODEL_DIR, MODEL_ID)
    npy_dict = np.load(database_path, allow_pickle=True)
    key_list, top_idx = retrieval(npy_dict.item())

    # optional
    collect_image(key_list, top_idx[1], RETRIEVAL_DIR)


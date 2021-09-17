# -*- coding: utf-8 -*-
"""
@Time    : 2021/7/9 10:55
@Author  : Lucius
@FileName: evaluate.py
@Software: PyCharm
"""
import shutil
import time

from data.dataloader import RandomHyperGraph
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import numpy as np
import sys

sys.path.append('pooling')

FEATURE_COORDINATE_DIR = '/home2/lishengrui/tcga_result/all_tcga'
DATABASE_DIR = '/home2/lishengrui/TCGA_experiment/result_all_tcga'
MODEL_DIR = '/home2/lishengrui/TCGA_experiment/result_all_tcga/models'
MODEL_ID = '2021-08-19-22-30-06'
RETRIEVAL_DIR = '/home2/lishengrui/TCGA_experiment/result_all_tcga/retrieval'

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_feature_database(feature_coordinate_dir, database_dir, model_dir, model_id):
    dataset = RandomHyperGraph(feature_coordinate_dir, 10)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=0)

    model_path = os.path.join(model_dir, model_id, 'model_best.pth')
    model = torch.load(model_path)
    model.evaluate()
    encoder_q = model.encoder_q

    result_dict = {}
    with torch.no_grad():
        for feature_0, H0, feature_1, H1, path in tqdm(dataloader):
            hidden = encoder_q((feature_0.to(device), H0.to(device)))
            hidden = hidden.cpu().detach().numpy()
            for i in range(128):
                result_dict[path[i]] = hidden[i]
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
    x = np.array(feature_list)
    dis_matrix = get_distance_matrix(x)
    _, top_idx = torch.topk(torch.from_numpy(dis_matrix), 11, dim=1, largest=False)
    return key_list, top_idx


def collect_image(key_list, idxs, save_to):
    result_id = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    os.mkdir(os.path.join(save_to, result_id))
    for i, idx in enumerate(idxs):
        npy_file = key_list[idx]
        png_file = npy_file.replace('.npy', '.jpg')
        shutil.copy(png_file, os.path.join(save_to, result_id, '{}.jpg'.format(i)))


def accuracy(key_list, top_idx):
    sum_num = top_idx.shape[0]
    top_1_3_5_10 = [0, 0, 0, 0]
    for top in top_idx:
        class_truth = key_list[top[0]].split("/")[-2]
        print()
        print(class_truth, end=' ')
        top_k = -1
        for k in range(1, top.shape[0]):
            class_pred = key_list[top[k]].split("/")[-2]
            print(class_pred, end=' ')
            if class_pred == class_truth:
                top_k = k
                break
        if top_k == -1:
            pass
        elif top_k == 1:
            for i in range(4):
                top_1_3_5_10[i] += 1
        elif top_k <= 3:
            for i in range(1, 4):
                top_1_3_5_10[i] += 1
        elif top_k <= 5:
            for i in range(2, 4):
                top_1_3_5_10[i] += 1
        elif top_k <= 10:
            top_1_3_5_10[3] += 1

    return top_1_3_5_10[0] / sum_num, top_1_3_5_10[1] / sum_num, top_1_3_5_10[2] / sum_num, top_1_3_5_10[3] / sum_num


def val_evaluate(model, dataloader_with_name):
    encoder_q = model.encoder_q
    result_dict = {}
    with torch.no_grad():
        for feature_0, H0, feature_1, H1, path in tqdm(dataloader_with_name):
            hidden = encoder_q((feature_0.to(device), H0.to(device)))
            hidden = hidden.cpu().detach().numpy()
            for i in range(128):
                result_dict[path[i]] = hidden[i]
    key_list, top_idx = retrieval(result_dict)
    top1, top3, top5, top10 = accuracy(key_list, top_idx)
    return top1, top3, top5, top10


class Evaluator:
    def __init__(self, batch_size):
        self.encoder_q = None
        self.batch_size = batch_size
        self.result_dict = None

    def reset(self, model):
        self.encoder_q = model.encoder_q
        self.result_dict = {}

    def add_data(self, feature_0, H0, path):
        with torch.no_grad():
            hidden = self.encoder_q((feature_0, H0))
            hidden = hidden.cpu().detach().numpy()
            for i in range(self.batch_size):
                self.result_dict[path[i]] = hidden[i]

    def report(self):
        key_list, top_idx = retrieval(self.result_dict)
        top1, top3, top5, top10 = accuracy(key_list, top_idx)
        return top1, top3, top5, top10


if __name__ == '__main__':
    database_path = os.path.join(DATABASE_DIR, 'database.npy')
    if not os.path.exists(database_path):
        save_feature_database(FEATURE_COORDINATE_DIR, DATABASE_DIR, MODEL_DIR, MODEL_ID)

    npy_dict = np.load(database_path, allow_pickle=True)
    key_list, top_idx = retrieval(npy_dict)

    # optional
    top1, top3, top5, top10 = accuracy(key_list, top_idx)
    print('acc: top1:{.4f} top3:{.4f} top5:{.4f} top10:{.4f}'.format(top1, top3, top5, top10))
    exit()
    collect_image(key_list, top_idx[1], RETRIEVAL_DIR)

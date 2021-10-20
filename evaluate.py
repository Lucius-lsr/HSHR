# -*- coding: utf-8 -*-
"""
@Time    : 2021/7/9 10:55
@Author  : Lucius
@FileName: evaluate.py
@Software: PyCharm
"""
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances



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
    # dis_matrix = get_distance_matrix(x)
    # dis_matrix = pairwise_distances(x)
    dis_matrix = -cosine_similarity(x)  # better

    _, top_idx = torch.topk(torch.from_numpy(dis_matrix), 11, dim=1, largest=False)
    return key_list, top_idx


def pair_match(feature_list_0, feature_list_1):
    assert len(feature_list_0) == len(feature_list_1)
    list_size = len(feature_list_0)
    x = np.array(feature_list_0 + feature_list_1)
    dis_matrix = get_distance_matrix(x)
    _, top_idx = torch.topk(torch.from_numpy(dis_matrix), 11, dim=1, largest=False)
    sum_num = top_idx.shape[0]
    top_1_3_5_10 = [0, 0, 0, 0]
    for top in top_idx:
        top_k = -1
        for k in range(1, 11):
            if top[0] == top[k] + list_size or top[0] == top[k] - list_size:
                top_k = k
                break
        if top_k == -1:
            pass
        elif top_k == 1:
            top_1_3_5_10[0] += 1
        elif top_k <= 3:
            top_1_3_5_10[1] += 1
        elif top_k <= 5:
            top_1_3_5_10[2] += 1
        elif top_k <= 10:
            top_1_3_5_10[3] += 1
    top_sum = [top_1_3_5_10[0] / sum_num, 0, 0, 0]
    for i in range(3):
        top_sum[i + 1] = top_1_3_5_10[i + 1] / sum_num + top_sum[i]

    return top_sum[0], top_sum[1], top_sum[2], top_sum[3]


def accuracy(key_list, top_idx):
    sum_num = top_idx.shape[0]
    top_1_3_5_10 = [0, 0, 0, 0]
    for top in top_idx:
        class_truth = key_list[top[0]].split("/")[-2]
        top_k = -1
        for k in range(1, top.shape[0]):
            class_pred = key_list[top[k]].split("/")[-2]
            if class_pred == class_truth:
                top_k = k
                break
        if top_k == -1:
            pass
        elif top_k == 1:
            top_1_3_5_10[0] += 1
        elif top_k <= 3:
            top_1_3_5_10[1] += 1
        elif top_k <= 5:
            top_1_3_5_10[2] += 1
        elif top_k <= 10:
            top_1_3_5_10[3] += 1
    top_sum = [top_1_3_5_10[0] / sum_num, 0, 0, 0]
    for i in range(3):
        top_sum[i + 1] = top_1_3_5_10[i + 1] / sum_num + top_sum[i]

    return top_sum[0], top_sum[1], top_sum[2], top_sum[3]


class Evaluator:
    def __init__(self):
        self.encoder = None
        self.result_dict = {}
        self.feature_list_0 = []
        self.feature_list_1 = []

    def reset(self, model_encoder):
        self.encoder = model_encoder
        self.result_dict = {}
        self.feature_list_0 = []
        self.feature_list_1 = []

    def add_data(self, feature_0, H0, feature_1, H1, path):
        with torch.no_grad():
            hidden0 = self.encoder((feature_0, H0))
            hidden0 = hidden0.cpu().detach().numpy()
            hidden1 = self.encoder((feature_1, H1))
            hidden1 = hidden1.cpu().detach().numpy()
            for i in range(feature_0.shape[0]):
                self.result_dict[path[i]] = hidden0[i]
                self.feature_list_0.append(hidden0[i])
                self.feature_list_1.append(hidden1[i])

    def add_data_without_H(self, feature_0, feature_1, path):
        with torch.no_grad():
            hidden0 = self.encoder(feature_0)
            hidden0 = hidden0.cpu().detach().numpy()
            hidden1 = self.encoder(feature_1)
            hidden1 = hidden1.cpu().detach().numpy()
            for i in range(feature_0.shape[0]):
                self.result_dict[path[i]] = hidden0[i]
                self.feature_list_0.append(hidden0[i])
                self.feature_list_1.append(hidden1[i])

    def add_result(self, feature_0, feature_1, path):
        self.result_dict[path] = feature_0
        self.feature_list_0.append(feature_0)
        self.feature_list_1.append(feature_1)

    def report(self):
        key_list, top_idx = retrieval(self.result_dict)
        top1, top3, top5, top10 = accuracy(key_list, top_idx)
        top1_, top3_, top5_, top10_ = pair_match(self.feature_list_0, self.feature_list_1)
        return top1, top3, top5, top10, top1_, top3_, top5_, top10_


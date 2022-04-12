# -*- coding: utf-8 -*-
"""
@Time    : 2021/7/9 10:55
@Author  : Lucius
@FileName: evaluate.py
@Software: PyCharm
"""
from collections import Counter

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from retrieval_utils import hyedge_similarity, generate_incidence


class ConMat():
    def __init__(self):
        self.dict = {}

    def record(self, truth, pred):
        if truth not in self.dict.keys():
            self.dict[truth] = {}
        if pred not in self.dict[truth].keys():
            self.dict[truth][pred] = 0
        self.dict[truth][pred] += 1

    def report(self):
        print(sorted(self.dict.keys()))
        for truth in sorted((self.dict.keys())):
            for pred in sorted((self.dict.keys())):
                if pred not in self.dict[truth].keys():
                    print(0, end=' ')
                else:
                    print(self.dict[truth][pred], end=' ')
            print()
        print()


def hamming_retrieval(database_dict):
    key_list = []
    feature_list = []
    for key in database_dict.keys():
        key_list.append(key)
        feature_list.append(database_dict[key][0])
    x = np.array(feature_list)
    hash_arr = np.sign(x)
    dis_matrix = -cosine_similarity(hash_arr)
    _, top_idx = torch.topk(torch.from_numpy(dis_matrix), x.shape[0], dim=1, largest=False)
    return key_list, top_idx, dis_matrix


def mmv_accuracy(at_k, key_list, top_idx):
    result = {}
    cm = ConMat()
    for idx, top in enumerate(top_idx):
        # key_list[idx] has the form of SUBTYPE/SLIDE_NAME
        class_truth = key_list[idx].split("/")[-2]
        preds = []
        check = 0
        for k in range(top.shape[0]):
            if idx == top[k] or key_list[idx].split("/")[-1] == key_list[top[k]].split("/")[-1]:
                continue
            check += 1
            class_pred = key_list[top[k]].split("/")[-2]
            preds.append(class_pred)
            if check == at_k:
                break
        if Counter(preds).most_common(1)[0][0] == class_truth:
            hit = 1
        else:
            hit = 0
        cm.record(class_truth, Counter(preds).most_common(1)[0][0])
        if class_truth not in result.keys():
            result[class_truth] = list()
        result[class_truth].append(hit)

    for key in result:
        li = result[key]
        result[key] = np.mean(li)

    return {k: round(result[k], 4) for k in sorted(result)}, cm


def map_accuracy(at_k, key_list, top_idx):
    result = {}

    for idx, top in enumerate(top_idx):
        # key_list[idx] has the form of SUBTYPE/SLIDE_NAME
        class_truth = key_list[idx].split("/")[-2]
        check = 0
        corr_index = []
        for k in range(top.shape[0]):
            if idx == top[k] or key_list[idx].split("/")[-1] == key_list[top[k]].split("/")[-1]:
                continue
            check += 1
            class_pred = key_list[top[k]].split("/")[-2]
            if class_pred == class_truth:
                corr_index.append(check - 1)
            if check == at_k:
                break
        if class_truth not in result.keys():
            result[class_truth] = list()
        if len(corr_index) == 0:
            result[class_truth].append(0)
        else:
            ap_at_k = 0
            for idx, i_corr in enumerate(corr_index):
                tmp = idx + 1
                tmp /= (i_corr + 1)
                ap_at_k += tmp
            ap_at_k /= len(corr_index)
            result[class_truth].append(ap_at_k)

    for key in result:
        li = result[key]
        result[key] = np.mean(li)

    return {k: round(result[k], 4) for k in sorted(result)}


class Evaluator:
    def __init__(self):
        self.result_dict = {}

    def reset(self):
        self.result_dict = {}

    def add_patches(self, patches, paths):
        assert len(patches.shape) == 3
        assert patches.shape[0] == len(paths)
        for i in range(patches.shape[0]):
            self.add_patch(patches[i], paths[i])

    def add_patch(self, patch, path):
        assert len(patch.shape) == 2
        for j in range(patch.shape[0]):
            self.result_dict[path + '@' + str(j)] = (patch[j], 1)

    def report_patch(self, k, num_cluster=20):
        key_list, top_idx, dis_mat = hamming_retrieval(self.result_dict)
        inc, list_slide_id = generate_incidence(key_list, top_idx, num_cluster, k)
        for alpha in [0, 0.5, 1, 2, 1000]:
            for beta in [0, 0.5, 1, 2, 1000]:
                slide_top_idx = hyedge_similarity(inc, alpha, beta)
                mMV, cm = mmv_accuracy(5, list_slide_id, slide_top_idx)
                mAP = map_accuracy(5, list_slide_id, slide_top_idx)
                print(k, alpha, beta)
                print(mMV, mAP)
                # cm.report()

    def fixed_report_patch(self, k=10, alpha=1, beta=1, num_cluster=20):
        key_list, top_idx, dis_mat = hamming_retrieval(self.result_dict)
        inc, list_slide_id = generate_incidence(key_list, top_idx, num_cluster, k)
        slide_top_idx = hyedge_similarity(inc, alpha, beta)
        mMV, cm = mmv_accuracy(5, list_slide_id, slide_top_idx)
        return mMV


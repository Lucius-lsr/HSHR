# -*- coding: utf-8 -*-
"""
@Time    : 2021/7/9 10:55
@Author  : Lucius
@FileName: evaluate.py
@Software: PyCharm
"""
import time
from collections import Counter

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import os
import pickle

from accuracy.const import mean_acc
from accuracy.hyedge_retrieval import hyedge_similarity, generate_incidence
from accuracy.hyedge_retrieval_patch import hyedge_similarity_patch
from accuracy.patch_acc import patch_accuracy_emd, patch_accuracy_v3


# from trans_hyper_learing import transfer_from_hash

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
            print()
            for pred in sorted((self.dict.keys())):
                if pred not in self.dict[truth].keys():
                    print(0, end=' ')
                else:
                    print(self.dict[truth][pred], end=' ')
        print()
        print()


def get_distance_matrix(x):
    G = np.dot(x, x.T)
    # 把G对角线元素拎出来，列不变，行复制n遍。
    H = np.tile(np.diag(G), (x.shape[0], 1))
    D = H + H.T - G * 2
    return D


def min_max_binarized(feats):
    all_output = []
    for feat in feats:
        prev = float('inf')
        output_binarized = []
        for ele in feat:
            if ele < prev:
                code = -1
                output_binarized.append(code)
            elif ele >= prev:
                code = 1
                output_binarized.append(code)
            prev = ele
        all_output.append(output_binarized)

    return np.array(all_output)


def hamming_retrieval(database_dict, hash_pre, types: list = None):
    key_list = []
    feature_list = []
    # weight_list = []
    for key in database_dict.keys():
        if types is None or key.split('/')[-2] in types:
            key_list.append(key)
            feature_list.append(database_dict[key][0])
            # weight_list.append(database_dict[key][1])
    x = np.array(feature_list)
    if hash_pre:
        hash_arr = np.sign(x)
    else:
        hash_arr = min_max_binarized(x)

    dis_matrix = -cosine_similarity(hash_arr)
    _, top_idx = torch.topk(torch.from_numpy(dis_matrix), x.shape[0], dim=1, largest=False)
    return key_list, top_idx, dis_matrix  # np.array(weight_list)


def mmv_accuracy(at_k, key_list, top_idx):
    TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp'
    with open(os.path.join(TMP, 'path2pid'), 'rb') as f:
        path2pid = pickle.load(f)
    result = {}
    cm = ConMat()
    for idx, top in enumerate(top_idx):
        class_truth = key_list[idx].split("/")[-2]
        preds = []
        check = 0
        for k in range(top.shape[0]):
            if idx == top[k] or path2pid[key_list[idx]] == path2pid[key_list[top[k]]]:
                continue
            check += 1
            class_pred = key_list[top[k]].split("/")[-2]
            preds.append(class_pred)
            # if class_pred == class_truth:
            #     hit += 1
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
    TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp'
    with open(os.path.join(TMP, 'path2pid'), 'rb') as f:
        path2pid = pickle.load(f)
    result = {}

    for idx, top in enumerate(top_idx):
        class_truth = key_list[idx].split("/")[-2]
        check = 0
        corr_index = []
        for k in range(top.shape[0]):
            if idx == top[k] or path2pid[key_list[idx]] == path2pid[key_list[top[k]]]:
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
        self.result_dicts = [{} for i in range(100)]

    def filter_slide(self, permit_key, path):
        uuid = path.split('/')[-1]
        if uuid in permit_key.keys():
            return False
        return True

    def reset(self):
        self.result_dict = {}
        self.result_dicts = [{} for i in range(100)]

    def add_result(self, hidden, path):
        if len(hidden.shape) == 1:
            self.result_dict[path] = hidden
        else:
            for i in range(hidden.shape[0]):
                self.result_dict[path[i]] = hidden[i]

    def add_patches(self, patches, paths):
        # permit_key_path = '/home2/lishengrui/retrieval/script/svs/permit_key'
        # with open(permit_key_path, 'rb') as f:
        #     permit_key = pickle.load(f)
        assert len(patches.shape) == 3
        assert patches.shape[0] == len(paths)
        for i in range(patches.shape[0]):
            self.add_patch(patches[i], paths[i])
            # if self.filter_slide(permit_key, paths[i]):
            #     self.add_patch(patches[i], paths[i])
            # else:
            #     print(paths[i], 'ignored')

    def add_patch(self, patch, path):
        assert len(patch.shape) == 2
        for j in range(patch.shape[0]):
            self.result_dict[path + '@' + str(j)] = (patch[j], 1)

    def report_mmv(self, types=None):
        key_list, top_idx, _ = hamming_retrieval(self.result_dict, False, types)
        # key_list, top_idx = retrieval(self.result_dict, types)
        result, cm = mmv_accuracy(5, key_list, top_idx)
        return {k: round(result[k], 4) for k in sorted(result)}

    def report_patch(self, k, types=None):
        #### min-median
        # key_list, top_idx, dis_mat = hamming_retrieval(self.result_dict, hash_pre=False, types=types)
        # list_slide_id, slide_top_idx = patch_accuracy_v3(key_list, top_idx, dis_mat)
        # mMV, cm = mmv_accuracy(5, list_slide_id, slide_top_idx)
        # mAP = map_accuracy(5, list_slide_id, slide_top_idx)
        # print(mean_acc(mMV), mMV, mean_acc(mAP), mAP)

        # demo
        # slide_top_idx = hyedge_similarity(inc, 1, 1)
        # for i, top in enumerate(slide_top_idx):
        #     p1 = list_slide_id[i]
        #     p2 = list_slide_id[top[1]]
        #     print(p1, p2)
        #     # if p1 == 'paad_tcga/a53d4cb8-15ce-4c35-83a6-dd865da991b9':
        #     #     for r in range(10):
        #     #         print(list_slide_id[top[r]])
        # exit()

        #### hypergraph-guided
        key_list, top_idx, dis_mat = hamming_retrieval(self.result_dict, hash_pre=True, types=types)
        inc, list_slide_id = generate_incidence(key_list, top_idx, 20, k)
        # main
        tmp_best = 0
        tmp_cm = None
        b_a, b_b, b1, b2, b3, b4 = None, None, None, None, None, None
        for alpha in [0, 0.5, 1, 2, 1000]:
            for beta in [0, 0.5, 1, 2, 1000]:
                slide_top_idx = hyedge_similarity(inc, alpha, beta)
                mMV, cm = mmv_accuracy(5, list_slide_id, slide_top_idx)
                mAP = map_accuracy(5, list_slide_id, slide_top_idx)
                if mean_acc(mMV) > tmp_best:
                    tmp_best = mean_acc(mMV)
                    tmp_cm = cm
                    b_a, b_b, b1, b2, b3, b4 = alpha, beta, mean_acc(mMV), mMV, mean_acc(mAP), mAP
        print(k)
        print(b_a, b_b, b1, b2, b3, b4)
        tmp_cm.report()

        #### fixed a,b
        # key_list, top_idx, dis_mat = hamming_retrieval(self.result_dict, hash_pre=True, types=types)
        # inc, list_slide_id = generate_incidence(key_list, top_idx, 20, k)
        # slide_top_idx = hyedge_similarity(inc, 1, 1)
        # mMV, cm = mmv_accuracy(5, list_slide_id, slide_top_idx)
        # mAP = map_accuracy(5, list_slide_id, slide_top_idx)
        # print(k, mean_acc(mMV))

    def fixed_report_patch(self, k=10, alpha=1, beta=1, types=None, inc=None, list_slide_id=None):
        if inc is None or list_slide_id is None:
            key_list, top_idx, dis_mat = hamming_retrieval(self.result_dict, hash_pre=True, types=types)
            inc, list_slide_id = generate_incidence(key_list, top_idx, 20, k)
        slide_top_idx = hyedge_similarity(inc, alpha, beta)
        mMV, cm = mmv_accuracy(5, list_slide_id, slide_top_idx)
        return mean_acc(mMV), inc, list_slide_id

# if __name__ == '__main__':
#     from data.utils import get_files_type
#     import pickle
#     TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp'
#     SVS_DIR = '/lishengrui/TCGA'
#     SVS_DIR_2 = '/home2/lishengrui/tcga_result'
#     svs_list = get_files_type(SVS_DIR, 'svs')
#     svs_list_2 = get_files_type(SVS_DIR_2, 'svs')
#     path_pid = {}
#     for svs in svs_list+svs_list_2:
#         base_name = os.path.basename(svs)
#         dir_name = os.path.dirname(svs)
#         p_id = base_name.split('-')[2]
#         path_pid[dir_name] = p_id
#     # for f in os.listdir(os.path.join(SVS_DIR, p)):
#     #     if f.endswith('.svs'):
#     #         return f.split('-')[2]
#     # return 'not find'
#     with open(os.path.join(TMP, 'path2pid'), 'wb') as fp:
#         pickle.dump(path_pid, fp)

# -*- coding: utf-8 -*-
"""
@Time    : 2021/12/29 15:39
@Author  : Lucius
@FileName: yottixel.py
@Software: PyCharm
"""
import os
import pickle
import random
import sys
import heapq
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from numpy import mean
import numpy as np

sys.path.append("..")

from data.utils import get_files_type


class WSI:
    def __init__(self, sub_type: str, bs: list, p_id: str):
        self.sub_type = sub_type
        self.bs = bs
        self.p_id = p_id


def wsi_distance(w1: WSI, w2: WSI):
    b1 = np.frombuffer(bytes(''.join(w1.bs), 'utf-8'), np.uint8) - 48
    b1 = b1.reshape([-1, 1024])
    b2 = np.frombuffer(bytes(''.join(w2.bs), 'utf-8'), np.uint8) - 48
    b2 = b2.reshape([-1, 1024])

    dis_matrix = -cosine_similarity(b1, b2)
    dis = np.min(dis_matrix, axis=1)
    dis = np.median(dis, axis=0)
    return dis


def WSI_retrieval(wsis: list, top=5):
    mmv_result = {}
    map_result = {}
    for i, q in enumerate(tqdm(wsis)):
        return_slides = []
        for j, w in enumerate(wsis):
            if q.p_id == w.p_id:
                continue
            dis = float(wsi_distance(q, w))
            heapq.heappush(return_slides, (dis, w.sub_type))

        preds = []
        corr_index = []
        for i in range(top):
            dis, sub_type = heapq.heappop(return_slides)
            preds.append(sub_type)
            if sub_type == q.sub_type:
                corr_index.append(i)
        if Counter(preds).most_common(1)[0][0] == q.sub_type:
            acc = 1
        else:
            acc = 0

        if q.sub_type not in mmv_result.keys():
            mmv_result[q.sub_type] = [acc]
        if q.sub_type not in map_result.keys():
            map_result[q.sub_type] = []

        mmv_result[q.sub_type].append(acc)

        if len(corr_index) == 0:
            map_result[q.sub_type].append(0)
        else:
            ap_at_k = 0
            for idx, i_corr in enumerate(corr_index):
                tmp = idx + 1
                tmp /= (i_corr + 1)
                ap_at_k += tmp
            ap_at_k /= len(corr_index)
            map_result[q.sub_type].append(ap_at_k)

    print('mMV:')
    for k in mmv_result.keys():
        print(k, mean(mmv_result[k]))
    print('mAP:')
    for k in map_result.keys():
        print(k, mean(map_result[k]))


if __name__ == '__main__':
    DATASETs = ['brain', 'gastrointestinal', 'gynecologic', 'hematopoietic', 'melanocytic', 'pulmonary', 'urinary',
                'prostate_testis', 'endocrine', 'liver']

    for DATASET in DATASETs:
        dataset = '/home2/lishengrui/FISH_experiment/{}/DATA/LATENT'.format(DATASET)
        subs_ = os.listdir(os.path.join(dataset, 'SITE'))
        for s in subs_:
            assert s[-5:] == '_tcga'
        subs = [s[:-5] for s in subs_]
        subtype_dirs = [os.path.join(dataset, 'SITE/{}_tcga/20x/densenet'.format(sub)) for sub in subs]
        wsi_lists = [get_files_type(d, '.pkl') for d in subtype_dirs]
        loaded_wsis = []
        for wsi_subtype, b_dir, b_files in zip(subs, subtype_dirs, wsi_lists):
            for wsi_name in b_files:
                full_path = os.path.join(b_dir, wsi_name)
                p_id = wsi_name.split('-')[2]
                with open(full_path, 'rb') as f:
                    b = pickle.load(f)
                wsi = WSI(wsi_subtype, b, p_id)
                loaded_wsis.append(wsi)
        WSI_retrieval(loaded_wsis, 5)

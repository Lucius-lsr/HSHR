# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/19 16:19
@Author  : Lucius
@FileName: demo.py
@Software: PyCharm
"""
import os
import numpy as np
import openslide
import pickle

import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

SVS_DIR = '/home2/lishengrui/tcga_result'
SVS_DIR_2 = '/lishengrui/TCGA'
FEATURE_DIR = '/home2/lishengrui/all_tcga'
TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp'


def feat_cor_slide(path):
    svs_path = os.path.join(SVS_DIR, path)
    data_path = os.path.join(FEATURE_DIR, path)
    svs, npy, pkl = None, None, None
    for f in os.listdir(svs_path):
        if f.endswith('.svs'):
            svs = os.path.join(svs_path, f)
    for f in os.listdir(data_path):
        if f == '0.npy':
            npy = os.path.join(data_path, f)
        if f == '0.pkl':
            pkl = os.path.join(data_path, f)
    assert svs is not None and npy is not None and pkl is not None
    svs = openslide.open_slide(svs)
    npy = np.load(npy)
    with open(pkl, 'rb') as handle:
        pkl = pickle.load(handle)
    return svs, npy, pkl


def cluster_center(feature):
    cluster = KMeans(n_clusters=20)
    predict_labels = cluster.fit_predict(feature)
    c_list = []
    for i in range(20):
        l = np.where(predict_labels == i)[0]
        f = feature[l]
        mf = np.mean(f, axis=0)
        c_list.append(mf)
    clusters = np.array(c_list)
    return clusters, predict_labels


def getInfo(sample):
    svs, npy, pkl = feat_cor_slide(sample)
    clusters, predict_labels = cluster_center(npy)
    return svs, npy, pkl, clusters, predict_labels


def sampleImage(sample, cor, slide, num, ordered_label, c_idx):
    selected = np.where(ordered_label == num)[0]
    for idx, s in enumerate(selected):
        if idx >= 20:
            break
        coor = cor[s]
        img = slide.read_region((coor[0], coor[1]), 0, (coor[2], coor[3])).convert('RGB')
        save_to = os.path.join(TMP, 'demoImages', sample, str(c_idx))
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        img.save(os.path.join(save_to, str(idx)) + '.jpg')


def pair_match(sample1, sample2):
    svs1, npy1, pkl1, clusters1, predict_labels1 = getInfo(sample1)
    svs2, npy2, pkl2, clusters2, predict_labels2 = getInfo(sample2)

    dis_matrix = -cosine_similarity(clusters1, clusters2)
    _, top_idx = torch.topk(torch.from_numpy(dis_matrix), 3, dim=1, largest=False)
    for idx, top in enumerate(top_idx):
        l1 = idx
        l2 = top[0].item()
        sampleImage(sample1, pkl1, svs1, l1, predict_labels1, idx)
        sampleImage(sample2, pkl2, svs2, l2, predict_labels2, idx)


def drawWSI(svs_path):
    def get_just_gt_level(slide: openslide, size):
        level = slide.level_count - 1
        while level >= 0 and slide.level_dimensions[level][0] < size[0] and \
                slide.level_dimensions[level][1] < size[1]:
            level -= 1
        return level

    slide_name = None

    for f in os.listdir(svs_path):
        if f.endswith('.svs'):
            slide_name = os.path.join(svs_path, f)
    assert slide_name is not None
    slide = openslide.open_slide(slide_name)
    mini_frac = 32
    mini_size = np.ceil(np.array(slide.level_dimensions[0]) / mini_frac).astype(np.int)
    mini_level = get_just_gt_level(slide, mini_size)
    img = slide.read_region((0, 0), mini_level, slide.level_dimensions[mini_level]).convert('RGB')
    img = img.resize(mini_size)
    save_to = os.path.join(TMP, 'demoImages')
    if not os.path.exists(save_to):
        os.makedirs(save_to, svs_path.split('/')[:-1])
    img.save(os.path.join(save_to, svs_path.split('/')[-1]) + '.jpg')


if __name__ == '__main__':
    sample1 = 'paad_tcga/a53d4cb8-15ce-4c35-83a6-dd865da991b9'
    sample2 = 'paad_tcga/02d41edb-26b7-4c55-958f-e0a3e4e85876'

    s = ['paad_tcga/a53d4cb8-15ce-4c35-83a6-dd865da991b9',
         'paad_tcga/02d41edb-26b7-4c55-958f-e0a3e4e85876',
         'paad_tcga/48546e3a-3c76-4e68-a360-121b87933a06',
         'paad_tcga/4fb0f334-ca7d-4cb9-aad6-924c50eabe76',
         'lihc_tcga/cc09d591-b75f-41f3-a10d-c55c014cec36',
         'lihc_tcga/92ebe42b-a166-4775-ab81-51cc335e523e',
         'paad_tcga/addfc7fa-bdd5-4c93-8645-d99e42317db9',
         'lihc_tcga/b2146eba-1022-4947-bc20-20598f36f527',
         'paad_tcga/0c0ec588-1030-44da-9985-bf4917f7fa02',
         'paad_tcga/16de414e-6cdc-4ab7-b13b-6585af4f7a40']
    for i in s:
        try:
            drawWSI(os.path.join(SVS_DIR, i))
        except FileNotFoundError:
            drawWSI(os.path.join(SVS_DIR_2, i))

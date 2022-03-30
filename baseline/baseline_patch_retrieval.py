# -*- coding: utf-8 -*-
"""
@Time    : 2021/12/17 14:38
@Author  : Lucius
@FileName: baseline_patch_retrieval.py
@Software: PyCharm
"""

import os
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans

import sys
from base_model import HashLayer

sys.path.append("..")
from data.utils import get_files_type
from evaluate import Evaluator


def cluster_feature(data_from, data_to, types):
    TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp'
    feature_and_coordinate_dir = '/home2/lishengrui/all_tcga'
    p = os.path.join(TMP, "&".join(types))
    # feature_and_coordinate_dir = '/home2/lishengrui/fish_feature'
    # p = os.path.join(TMP, "&".join(types)+'|fish_feature')
    if os.path.exists(p + '.npy'):
        clusters = np.load(p + '.npy')
        with open(p + '.pkl', 'rb') as f:
            paths = pickle.load(f)
        return clusters, paths
    else:
        clusters = list()
        paths = list()
        feature_list = get_files_type(feature_and_coordinate_dir, '0.npy')
        feature_list.sort()
        # shuffle
        r = random.random
        random.seed(6)
        random.shuffle(feature_list, r)
        size = len(feature_list)
        for feature_path in tqdm(feature_list[int(data_from * size):int(data_to * size)]):
            base_name = os.path.basename(feature_path)
            dir_name = os.path.join(feature_and_coordinate_dir, os.path.dirname(feature_path))
            if base_name == '0.npy':
                files = os.listdir(dir_name)
                if '0.pkl' in files:
                    if dir_name.split('/')[-2] in types:
                        paths.append(os.path.dirname(feature_path))
                        npy_file_0 = os.path.join(dir_name, '0.npy')
                        x0 = np.load(npy_file_0)
                        while len(x0.shape) > 2:
                            x0 = x0.squeeze()
                        km = KMeans(n_clusters=20)
                        km.fit(x0)
                        clusters.append(km.cluster_centers_)
        clusters = np.array(clusters)

        np.save(p, clusters)
        with open(p + '.pkl', 'wb') as fp:
            pickle.dump(paths, fp)
        return clusters, paths


# def weighted_cluster_feature(data_from, data_to, types):
#     TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp/weighted_feature'
#     feature_and_coordinate_dir = '/home2/lishengrui/all_tcga'
#     p = os.path.join(TMP, "&".join(types))
#     if os.path.exists(p + '.npy'):
#         clusters = np.load(p + '.npy')
#         weights = np.load(p + 'weight.npy')
#         with open(p + '.pkl', 'rb') as f:
#             paths = pickle.load(f)
#         return clusters, paths, weights
#     else:
#         clusters = list()
#         paths = list()
#         weight = list()
#         feature_list = get_files_type(feature_and_coordinate_dir, '0.npy')
#         feature_list.sort()
#         # shuffle
#         r = random.random
#         random.seed(6)
#         random.shuffle(feature_list, r)
#         size = len(feature_list)
#         for feature_path in tqdm(feature_list[int(data_from * size):int(data_to * size)]):
#             base_name = os.path.basename(feature_path)
#             dir_name = os.path.join(feature_and_coordinate_dir, os.path.dirname(feature_path))
#             if base_name == '0.npy':
#                 files = os.listdir(dir_name)
#                 if '0.pkl' in files:
#                     if dir_name.split('/')[-2] in types:
#                         num_clusters = 20
#                         paths.append(os.path.dirname(feature_path))
#                         npy_file_0 = os.path.join(dir_name, '0.npy')
#                         x0 = np.load(npy_file_0)
#                         while len(x0.shape) > 2:
#                             x0 = x0.squeeze()
#                         km = KMeans(n_clusters=num_clusters)
#                         km.fit(x0)
#
#                         predict_labels = km.predict(x0)
#                         c_list = []
#                         w = []
#                         for i in range(num_clusters):
#                             l = np.where(predict_labels == i)[0]
#                             f = x0[l]
#                             mf = np.mean(f, axis=0)
#                             num_f = f.shape[0]
#                             w.append(num_f)
#                             c_list.append(mf)
#                         cs = np.array(c_list)
#                         clusters.append(cs)
#                         weight.append([i / sum(w) for i in w])
#
#         clusters = np.array(clusters)
#         weights = np.array(weight)
#         np.save(p, clusters)
#         np.save(p + 'weight', weight)
#         with open(p + '.pkl', 'wb') as fp:
#             pickle.dump(paths, fp)
#         return clusters, paths, weights


def fine_tune(raw):
    feature_in = 512
    feature_out = 1024
    depth = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = HashLayer(feature_in, feature_out, depth)
    MODEL_DIR = '/home2/lishengrui/TCGA_experiment/result_all_tcga/models'
    model_path = os.path.join(MODEL_DIR, 'ssl', 'model_best.pth')
    encoder.load_state_dict(torch.load(model_path))
    encoder = encoder.to(device)

    with torch.no_grad():
        raw = torch.from_numpy(raw).to(device)
        output = encoder(raw).cpu().detach().numpy()
    return output


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    # exp = ['dlbc_tcga', 'thym_tcga']
    # k = 5
    # exp = ['acc_tcga', 'pcpg_tcga', 'thca_tcga']
    # k = 50
    exp = ['ucec_tcga', 'cesc_tcga', 'ucs_tcga', 'ov_tcga']
    k = 20

    print('*' * 20)
    clusters, paths = cluster_feature(0, 1, exp)  # clusters: num_slide x num_cluster x feature_dim

    clusters = fine_tune(clusters)
    evaluator = Evaluator()
    evaluator.add_patches(clusters, paths)
    evaluator.report_patch(k)
    # inc, list_slide_id = None, None
    # for alpha in [0.1*i for i in range(20)]:
    #     for beta in [0.1*i for i in range(20)]:
    #         t1, t2 = evaluator.fixed_report_patch(alpha=alpha, beta=beta, k=k, inc=inc, list_slide_id=list_slide_id)
    #         inc, list_slide_id = t1, t2

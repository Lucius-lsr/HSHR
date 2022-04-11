# -*- coding: utf-8 -*-
"""
@Time    : 2021/10/19 13:35
@Author  : Lucius
@FileName: baseline_wsisa.py
@Software: PyCharm
"""
import pickle

import copy
import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans, SpectralClustering
from tqdm import tqdm

from AHGAE.ahgae_processor import preprocess_hypergraph
from baseline.base_model import HashLayer
from data.utils import get_files_type
from evaluate import Evaluator
from self_supervision.call import get_moco
from scipy.sparse import coo_matrix

feature_and_coordinate_dir = '/home2/lishengrui/all_tcga'
TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp'


def hg_feature(data_from, data_to, types, n_clusters=30, g_weight=2 / 3, g_layer=1):
    def nnidx2H(nn_idx):
        self_idx = np.arange(nn_idx.shape[0]).reshape(-1, 1)
        nn_idx = np.concatenate((self_idx, nn_idx), axis=1).reshape(-1)
        hyedge_idx = np.expand_dims(np.arange(2000), axis=0).repeat(20, 1).transpose(1, 0).reshape(-1)
        row, col = nn_idx, hyedge_idx
        data = np.ones_like(row)
        return coo_matrix((data, (row, col)), shape=(2000, 2000), dtype=float)

    def sm_feature(raw_f, inc, weight=2 / 3, layer=3):
        adj_norm_s = preprocess_hypergraph(inc, layer, weight=weight)
        sm_fea_s = raw_f
        for i in range(layer):
            sm_fea_s = adj_norm_s[i].dot(sm_fea_s)
        return sm_fea_s

    def clustering(feature, Cluster, n_clusters):
        f_adj = np.matmul(feature, np.transpose(feature))
        predict_labels = Cluster.fit_predict(f_adj)
        c_list = []
        for i in range(n_clusters):
            l = np.where(predict_labels == i)[0]
            f = feature[l]
            mf = np.mean(f, axis=0)
            c_list.append(mf)
        clusters = np.array(c_list)
        return clusters

    # p = os.path.join(TMP, 'hg_cluster_ssl_feature')
    # p0 = p + '0'
    # p1 = p + '1'
    # if os.path.exists(p0 + '.npy'):
    #     means_0 = np.load(p0 + '.npy')
    #     means_1 = np.load(p1 + '.npy')
    #     with open(p + '.pkl', 'rb') as f:
    #         paths = pickle.load(f)
    #     print('load cache')
    #     return means_0, means_1, paths
    # else:
    print(n_clusters, g_weight, g_layer)
    paths = list()
    feature_list = get_files_type(feature_and_coordinate_dir, '0.npy')
    feature_list.sort()
    # shuffle
    r = random.random
    random.seed(6)
    random.shuffle(feature_list, r)
    size = len(feature_list)
    cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    all_clusters_0 = list()
    all_clusters_1 = list()
    for feature_path in tqdm(feature_list[int(data_from * size):int(data_to * size)]):
        base_name = os.path.basename(feature_path)
        dir_name = os.path.join(feature_and_coordinate_dir, os.path.dirname(feature_path))
        if base_name == '0.npy' and (types is None or dir_name.split('/')[-2] in types):
            files = os.listdir(dir_name)
            if '1.npy' in files and '0.pkl' in files and '1.pkl' in files:
                paths.append(os.path.dirname(feature_path))

                npy_file_0 = os.path.join(dir_name, '0.npy')
                x0 = np.load(npy_file_0)
                idx_file_0 = os.path.join(dir_name, '0_nearest_idx.npy')
                nearest_idx_0 = np.load(idx_file_0)
                H0 = nnidx2H(nearest_idx_0)
                sm_fea_0 = sm_feature(x0, H0, weight=g_weight, layer=g_layer)
                clusters_0 = clustering(sm_fea_0, cluster, n_clusters)

                # npy_file_1 = os.path.join(dir_name, '1.npy')
                # x1 = np.load(npy_file_1)
                # idx_file_1 = os.path.join(dir_name, '1_nearest_idx.npy')
                # nearest_idx_1 = np.load(idx_file_1)
                # H1 = nnidx2H(nearest_idx_1)
                # sm_fea_1 = sm_feature(x1, H1, weight=2 / 3, layer=2)
                # clusters_1 = clustering(sm_fea_1, cluster, n_clusters)

                all_clusters_0.append(clusters_0)
                # all_clusters_1.append(clusters_1)

    return np.array(all_clusters_0), paths


# class HGDataset(Dataset):
#
#     def __init__(self, data_from, data_to) -> None:
#         super().__init__()
#         cluster_means_0, cluster_means_1, paths = hg_feature(data_from, data_to)
#         self.data_0 = cluster_means_0
#         self.data_1 = cluster_means_1
#         self.paths = paths
#
#     def __getitem__(self, item):
#         return self.data_0[item], self.data_1[item], self.paths[item]
#
#     def __len__(self) -> int:
#         return len(self.data_0)
def load_hg_feature(exp, n_clusters, g_weight, g_layer):
    TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp/ahgae_cluster'
    p = os.path.join(TMP, "&".join(exp) + '|' + "&".join([str(n_clusters), str(g_weight), str(g_layer)]))
    if os.path.exists(p + '.npy'):
        clusters = np.load(p + '.npy')
        with open(p + '.pkl', 'rb') as f:
            paths = pickle.load(f)
        print('load cache')
        return clusters, paths
    else:
        print('no cache file')




if __name__ == '__main__':
    exps = [['gbm_tcga', 'lgg_tcga'],
            ['coad_tcga', 'esca_tcga', 'read_tcga', 'stad_tcga'],
            ['ucec_tcga', 'cesc_tcga', 'ucs_tcga', 'ov_tcga'],
            ['dlbc_tcga', 'thym_tcga'],
            ['uvm_tcga', 'skcm_tcga'],
            ['lusc_tcga', 'luad_tcga', 'meso_tcga'],
            ['blca_tcga', 'kirc_tcga', 'kich_tcga', 'kirp_tcga'],
            ['tgct_tcga', 'prad_tcga']]

    TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp/ahgae_cluster'
    n_clusters = 20
    g_weight = 0.2
    g_layer = 1
    for exp in exps:
        clusters0, paths = hg_feature(0, 1, exp, n_clusters=n_clusters, g_weight=g_weight, g_layer=g_layer)

        p = os.path.join(TMP, "&".join(exp) + '|' + "&".join([str(n_clusters), str(g_weight), str(g_layer)]))
        np.save(p, clusters0)
        with open(p + '.pkl', 'wb') as fp:
            pickle.dump(paths, fp)
        # return clusters, paths

    # feature_in = 512
    # feature_out = 1024
    # lr = 0.003
    # momentum = 0.9
    # weight_decay = 1e-4
    # batch_size = 128
    # criterion = nn.CrossEntropyLoss().cuda(True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # depth = 1
    #
    # train_dataset = HGDataset(0, 1)
    # exit()
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    #
    # model = get_moco(HashLayer(feature_in, feature_out, depth), HashLayer(feature_in, feature_out, depth), device,
    #                  feature_out)
    # model = model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    #
    # evaluator = Evaluator()
    # evaluator_patch = Evaluator()
    # # cfs, cf_paths = cluster_feature(0, 1, ['ucec_tcga', 'cesc_tcga', 'ucs_tcga', 'ov_tcga'])
    # # cfs = torch.from_numpy(cfs).to(device)
    # for epoch in range(500):
    #     print('*' * 5, 'epoch: ', epoch, '*' * 5)
    #     loss_sum = 0
    #     loss_count = 0
    #     evaluator.reset()
    #     pre_model = copy.deepcopy(model)
    #     for x0, x1, path in train_dataloader:
    #         x0, x1 = x0.to(device), x1.to(device)
    #         output, target = model(x0, x1)
    #         loss = criterion(output, target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         loss_sum += loss.item()
    #         loss_count += 1
    #         with torch.no_grad():
    #             hidden = pre_model.encoder_q(x0)
    #             # print(abs(hidden.cpu().detach().numpy()).mean())
    #             evaluator.add_result(hidden.cpu().detach().numpy(), path)
    #
    #     loss_ave = loss_sum / loss_count
    #     print("loss: ", loss_ave)
    #
    #     # patch
    #     if epoch % 50 == 0:
    #         # evaluator_patch.reset()
    #         # with torch.no_grad():
    #         #     for slide in range(cfs.shape[0]):
    #         #         raw = cfs[slide]
    #         #         h = pre_model.encoder_q(raw)
    #         #         evaluator_patch.add_patches(h.cpu().detach().numpy(), cf_paths[slide])
    #         # print(evaluator_patch.report_patch())
    #
    #         print(evaluator.report_mmv(['gbm_tcga', 'lgg_tcga']))
    #         print(evaluator.report_mmv(['coad_tcga', 'esca_tcga', 'read_tcga', 'stad_tcga']))
    #         print(evaluator.report_mmv(['ucec_tcga', 'cesc_tcga', 'ucs_tcga', 'ov_tcga']))
    #         print(evaluator.report_mmv(['dlbc_tcga', 'thym_tcga']))
    #         print(evaluator.report_mmv(['uvm_tcga', 'skcm_tcga']))
    #         print(evaluator.report_mmv(['lusc_tcga', 'luad_tcga', 'meso_tcga']))
    #         print(evaluator.report_mmv(['blca_tcga', 'kirc_tcga', 'kich_tcga', 'kirp_tcga']))
    #         print(evaluator.report_mmv(['tgct_tcga', 'prad_tcga']))

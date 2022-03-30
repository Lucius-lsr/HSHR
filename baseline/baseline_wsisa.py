# -*- coding: utf-8 -*-
"""
@Time    : 2021/10/19 13:35
@Author  : Lucius
@FileName: baseline_wsisa.py
@Software: PyCharm
"""
import pickle
import sys

from base_model import HashLayer
from baseline_patch_retrieval import cluster_feature

sys.path.append("..")
import copy
import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans
from tqdm import tqdm

from data.utils import get_files_type, check_dir
from evaluate import Evaluator
from self_supervision.call import get_moco

feature_and_coordinate_dir = '/home2/lishengrui/all_tcga'
TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp'


def mean_feature(data_from, data_to):
    p = os.path.join(TMP, 'ssl_mean_feature')
    p0 = p + '0'
    p1 = p + '1'
    if os.path.exists(p0 + '.npy'):
        means_0 = np.load(p0 + '.npy')
        means_1 = np.load(p1 + '.npy')
        with open(p + '.pkl', 'rb') as f:
            paths = pickle.load(f)
        print('load cache')
        return means_0, means_1, paths
    else:
        means_0 = list()
        means_1 = list()
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
                if '1.npy' in files and '0.pkl' in files and '1.pkl' in files:
                    paths.append(os.path.dirname(feature_path))

                    npy_file_0 = os.path.join(dir_name, '0.npy')
                    x0 = np.load(npy_file_0)
                    mean_0 = np.mean(x0, axis=0)
                    means_0.append(mean_0)

                    npy_file_1 = os.path.join(dir_name, '1.npy')
                    x1 = np.load(npy_file_1)
                    mean_1 = np.mean(x1, axis=0)
                    means_1.append(mean_1)

        means_0 = np.array(means_0)
        means_1 = np.array(means_1)
        np.save(p0, means_0)
        np.save(p1, means_1)
        with open(p + '.pkl', 'wb') as fp:
            pickle.dump(paths, fp)
        return means_0, means_1, paths


class WSISADataset(Dataset):

    def __init__(self, data_from, data_to) -> None:
        super().__init__()
        cluster_means_0, cluster_means_1, paths = mean_feature(data_from, data_to)
        self.data_0 = cluster_means_0
        self.data_1 = cluster_means_1
        self.paths = paths

    def __getitem__(self, item):
        return self.data_0[item], self.data_1[item], self.paths[item]

    def __len__(self) -> int:
        return len(self.data_0)


def one_it():
    feature_in = 512
    feature_out = 1024
    depth = 1

    lr = 0.003
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = 128
    criterion = nn.CrossEntropyLoss().cuda(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = WSISADataset(0, 1)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = get_moco(HashLayer(feature_in, feature_out, depth), HashLayer(feature_in, feature_out, depth), device,
                     feature_out)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    exps = [
        # ['dlbc_tcga', 'thym_tcga'],
        ['acc_tcga', 'pcpg_tcga', 'thca_tcga'],
        # ['chol_tcga', 'lihc_tcga', 'paad_tcga']
    ]
    record = []
    evaluator = Evaluator()
    all_cfs, all_cf_paths = [], []
    for exp in exps:
        cfs, cf_paths = cluster_feature(0, 1, exp)
        cfs = torch.from_numpy(cfs).to(device)
        all_cfs.append(cfs)
        all_cf_paths.append(cf_paths)
    for epoch in range(10):
        # print('*' * 5, 'epoch: ', epoch, '*' * 5)
        loss_sum = 0
        loss_count = 0
        pre_model = copy.deepcopy(model)
        for x0, x1, path in train_dataloader:
            x0, x1 = x0.to(device), x1.to(device)
            output, target = model(x0, x1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            loss_count += 1
            # with torch.no_grad():
            #     hidden = pre_model.encoder_q(x0)
            #     # print(abs(hidden.cpu().detach().numpy()).mean())
            #     evaluator.add_result(hidden.cpu().detach().numpy(), path)

        loss_ave = loss_sum / loss_count
        # print("loss: ", loss_ave)

        # patch
        # if epoch == 200:
        #     MODEL_DIR = '/home2/lishengrui/TCGA_experiment/result_all_tcga/models'
        #     torch.save(model.encoder_q.state_dict(), check_dir(os.path.join(MODEL_DIR, 'ssl', 'model_best.pth')))

        # if epoch % 5 == 0:
        # print('*' * 5, 'epoch: ', epoch, '*' * 5)
        print('#')
        for cfs, cf_paths in zip(all_cfs, all_cf_paths):
            evaluator.reset()
            with torch.no_grad():
                raw = cfs
                h = pre_model.encoder_q(raw)
                evaluator.add_patches(h.cpu().detach().numpy(), cf_paths)
                acc = evaluator.fixed_report_patch()
                record.append(acc)
    return record


if __name__ == '__main__':
    rr1, rr2, rr3 = [], [], []
    for _ in range(10):
        r = one_it()
        r1, r2, r3 = [], [], []
        for i, t in enumerate(r):
            # if i % 3 == 0:
            #     r1.append(t)
            # elif i % 3 == 1:
            #     r2.append(t)
            # else:
            r3.append(t)
        # rr1.append(r1)
        # rr2.append(r2)
        rr3.append(r3)
    # rr1 = np.array(rr1)
    # rr2 = np.array(rr2)
    rr3 = np.array(rr3)

    # mean1 = rr1.mean(axis=0)
    # std1 = rr1.std(axis=0)
    # mean2 = rr2.mean(axis=0)
    # std2 = rr2.std(axis=0)
    mean3 = rr3.mean(axis=0)
    std3 = rr3.std(axis=0)

    # print(mean1)
    # print(std1)
    # print(mean2)
    # print(std2)
    print(mean3)
    print(std3)

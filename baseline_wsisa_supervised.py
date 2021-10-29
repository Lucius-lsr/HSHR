# -*- coding: utf-8 -*-
"""
@Time    : 2021/10/19 13:35
@Author  : Lucius
@FileName: baseline_wsisa.py
@Software: PyCharm
"""
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

from baseline_supervised import ClassifyLayer, Labeler
from data.utils import get_files_type
from evaluate import Evaluator
from self_supervision.call import get_moco

feature_and_coordinate_dir = '/home2/lishengrui/all_tcga'
TMP = '/home2/lishengrui/TCGA_experiment/result_all_tcga/tmp'


def cluster_feature(data_from, data_to):
    cluster_means_0 = list()
    cluster_means_1 = list()
    loaded = False
    if os.path.exists(TMP + '/cluster_means_0_{}_{}.npy'.format(data_from, data_to)) and os.path.exists(
            TMP + '/cluster_means_1_{}_{}.npy'.format(data_from, data_to)):
        cluster_means_0 = np.load(TMP + '/cluster_means_0_{}_{}.npy'.format(data_from, data_to))
        cluster_means_1 = np.load(TMP + '/cluster_means_1_{}_{}.npy'.format(data_from, data_to))
        loaded = True
    print(loaded)
    paths = list()
    feature_list = get_files_type(feature_and_coordinate_dir, 'npy')
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
                paths.append(dir_name)
                if not loaded:
                    npy_file_0 = os.path.join(dir_name, '0.npy')
                    x0 = np.load(npy_file_0)
                    km = KMeans(n_clusters=5)
                    km.fit(x0)
                    mean_0 = np.mean(km.cluster_centers_, axis=0)

                    npy_file_1 = os.path.join(dir_name, '1.npy')
                    x1 = np.load(npy_file_1)
                    km = KMeans(n_clusters=5)
                    km.fit(x1)
                    mean_1 = np.mean(km.cluster_centers_, axis=0)

                    cluster_means_0.append(mean_0)
                    cluster_means_1.append(mean_1)

    cluster_means_0 = np.array(cluster_means_0)
    cluster_means_1 = np.array(cluster_means_1)
    np.save(TMP + '/cluster_means_0_{}_{}'.format(data_from, data_to), cluster_means_0)
    np.save(TMP + '/cluster_means_1_{}_{}'.format(data_from, data_to), cluster_means_1)
    return cluster_means_0, cluster_means_1, paths


def mean_feature(data_from, data_to):
    means_0 = list()
    means_1 = list()
    paths = list()
    feature_list = get_files_type(feature_and_coordinate_dir, 'npy')
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
                paths.append(dir_name)

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
    return means_0, means_1, paths


class WSISADataset(Dataset):

    def __init__(self, data_from, data_to, cluster=True) -> None:
        super().__init__()
        if cluster:
            cluster_means_0, cluster_means_1, paths = cluster_feature(data_from, data_to)
        else:
            cluster_means_0, cluster_means_1, paths = mean_feature(data_from, data_to)
        self.data_0 = cluster_means_0
        self.data_1 = cluster_means_1
        self.paths = paths

    def __getitem__(self, item):
        return self.data_0[item], self.data_1[item], self.paths[item]

    def __len__(self) -> int:
        return len(self.data_0)


class Layer(nn.Module):
    def __init__(self, feature_in, feature_out, depth) -> None:
        super().__init__()
        if depth == 1:
            self.fc = nn.Linear(feature_in, feature_out)
        elif depth == 2:
            self.fc = nn.Sequential(
                nn.Linear(feature_in, 2 * feature_out),
                nn.Linear(2*feature_out, feature_out),
            )

    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    feature_in = 512
    feature_out = 128
    lr = 0.03
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = 128
    criterion = nn.CrossEntropyLoss().cuda(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CLUSTER = False
    depth = 2
    class_num = 30

    train_dataset = WSISADataset(0, 0.8, cluster=CLUSTER)
    test_dataset = WSISADataset(0.8, 1, cluster=CLUSTER)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    model = Layer(feature_in, feature_out, depth)
    model = model.to(device)
    last_layer = ClassifyLayer(feature_out, class_num)
    last_layer = last_layer.to(device)
    labeler = Labeler(class_num)

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    evaluator = Evaluator()

    for epoch in range(500):
        print('*' * 5, 'epoch: ', epoch, '*' * 5)
        loss_sum = 0
        loss_count = 0
        for x0, x1, path in train_dataloader:
            x0 = x0.to(device)
            feature = model(x0)
            output = last_layer(feature)
            label = labeler.get_label(path).to(device)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            loss_count += 1
        loss_ave = loss_sum / loss_count
        print("loss: ", loss_ave)

        evaluator.reset(copy.deepcopy(model))
        for x0, x1, path in test_dataloader:
            x0, x1 = x0.to(device), x1.to(device)
            evaluator.add_data_without_H(x0, x1, path)
        top1, top3, top5, top10, top1_, top3_, top5_, top10_ = evaluator.report()
        print('class acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(top1, top3, top5, top10))

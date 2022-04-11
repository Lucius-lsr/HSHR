# -*- coding: utf-8 -*-
"""
@Time    : 2021/10/6 15:52
@Author  : Lucius
@FileName: baseline_supervised.py
@Software: PyCharm
"""

import copy
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from pooling.hgrnet_pooling import get_pooling_model
from data.dataloader import RandomHyperGraph
from evaluate import Evaluator
import sys

sys.path.append('../pooling')

FEATURE_DIR = '/home2/lishengrui/all_tcga'
COORDINATE_DIR = '/home2/lishengrui/all_tcga'
MODEL_DIR = '/home2/lishengrui/TCGA_experiment/result_all_tcga/models'

hidden_dim = 128  # 128
n_target = 64  # 64

lr = 0.03
momentum = 0.9
weight_decay = 1e-4
batch_size = 128

g_class_num = 30
criterion = nn.CrossEntropyLoss()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_pooling_model(hidden_dim, n_target)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    assert FEATURE_DIR == COORDINATE_DIR
    train_dataset = RandomHyperGraph(FEATURE_DIR, 10, 0, 0.8)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_dataset = RandomHyperGraph(FEATURE_DIR, 10, 0.8, 1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

    # switch to train mode
    model.train()
    evaluator = Evaluator()
    last_layer = ClassifyLayer(n_target, g_class_num)
    last_layer = last_layer.to(device)
    labeler = Labeler(g_class_num)

    for epoch in range(500):
        print('*' * 5, 'epoch: ', epoch, '*' * 5)
        # ----------------train-----------------
        loss_sum = 0
        loss_count = 0

        for feature_0, H0, feature_1, H1, path in train_dataloader:
            feature_0, H0, feature_1, H1 = feature_0.to(device), H0.to(device), feature_1.to(device), H1.to(device)

            # compute output
            feature = model((feature_0, H0), False)
            output = last_layer(feature)
            label = labeler.get_label(path).to(device)
            loss = criterion(output, label)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            loss_count += 1

        loss_ave = loss_sum / loss_count
        print('loss: ', loss_ave)

        # ----------------val-----------------
        evaluator.reset()
        with torch.no_grad():
            for feature_0, H0, feature_1, H1, path in test_dataloader:
                feature_0, H0, feature_1, H1 = feature_0.to(device), H0.to(device), feature_1.to(device), H1.to(device)
                hidden0 = model((feature_0, H0), False).cpu().detach().numpy()
                hidden1 = model((feature_1, H1), False).cpu().detach().numpy()
                evaluator.add_result(hidden0, hidden1, path)
        top1, top3, top5, top10, top1_, top3_, top5_, top10_ = evaluator.report()
        print('class acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(top1, top3, top5, top10))


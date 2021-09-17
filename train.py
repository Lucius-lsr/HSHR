# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/29 10:36
@Author  : Lucius
@FileName: train.py
@Software: PyCharm
"""
import copy
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pooling.hgrnet_pooling import get_pooling_model
from data.dataloader import RandomHyperGraph

from evaluate import Evaluator

import sys

sys.path.append('pooling')

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FEATURE_DIR = '/home2/lishengrui/tcga_result/all_tcga'
COORDINATE_DIR = '/home2/lishengrui/tcga_result/all_tcga'
MODEL_DIR = '/home2/lishengrui/TCGA_experiment/result_all_tcga/models'
hidden_dim = 128
n_target = 64
K = 65536
m = 0.999
T = 0.07
lr = 0.03
momentum = 0.9
weight_decay = 1e-4
batch_size = 128

if __name__ == '__main__':
    if len(sys.argv) == 3:
        model_path = os.path.join(MODEL_DIR, sys.argv[2], 'model_best.pth')
        model = torch.load(model_path)
    else:
        model = self_supervision.moco.builder.MoCo(
            get_pooling_model(hidden_dim, n_target).to(device),
            get_pooling_model(hidden_dim, n_target).to(device),
            device,
            n_target, K, m, T, False)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().cuda(True)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    assert FEATURE_DIR == COORDINATE_DIR
    dataset = RandomHyperGraph(FEATURE_DIR, 10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # switch to train mode
    model.train()

    best_loss = 1e5
    model_id = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    evaluator = Evaluator(batch_size)

    for epoch in range(500):
        print('*' * 5, 'epoch: ', epoch, '*' * 5)
        # ----------------train-----------------
        loss_sum = 0
        loss_count = 0
        evaluator.reset(copy.deepcopy(model))
        for feature_0, H0, feature_1, H1, path in tqdm(dataloader):
            feature_0, H0, feature_1, H1 = feature_0.to(device), H0.to(device), feature_1.to(device), H1.to(device)

            # evaluate last epoch
            evaluator.add_data(feature_0, H0, path)

        #     # compute output
        #     output, target = model(im_q=(feature_0, H0), im_k=(feature_1, H1))
        #     loss = criterion(output, target)
        #
        #     # compute gradient and do SGD step
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #
        #     loss_sum += loss.item()
        #     loss_count += 1
        # loss_ave = loss_sum / loss_count
        # print('loss: ', loss_ave)
        # if loss_ave < best_loss:
        #     best_loss = loss_ave
        #     torch.save(model, check_dir(os.path.join(MODEL_DIR, model_id, 'model_best.pth')))

        # ----------------val-----------------
        top1, top3, top5, top10 = evaluator.report()
        print('acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(top1, top3, top5, top10))

# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/29 10:36
@Author  : Lucius
@FileName: train.py
@Software: PyCharm
"""
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

import moco.builder
from data.utils import check_dir
from pooling.hgrnet_pooling import get_pooling_model
from data.dataloader import get_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FEATURE_DIR = '/home/lishengrui/TCGA_experiment/all_tcga'
COORDINATE_DIR = '/home/lishengrui/TCGA_experiment/all_tcga'
MODEL_DIR = '/home/lishengrui/TCGA_experiment/result_lusc_tcga/models'
dim = 128
K = 65536
m = 0.999
T = 0.07
lr = 0.03
momentum = 0.9
weight_decay = 1e-4


if __name__ == '__main__':

    torch.cuda.device(1)

    model = moco.builder.MoCo(
        get_pooling_model(dim).to(device),
        get_pooling_model(dim).to(device),
        device,
        dim, K, m, T, False)
    print(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().cuda(True)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    assert FEATURE_DIR == COORDINATE_DIR
    dataset = get_dataset(FEATURE_DIR, 10)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=0)

    # switch to train mode
    model.train()

    best_loss = 9999999
    model_id = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    for epoch in range(500):
        print('*'*5, 'epoch: ', epoch, '*'*5)
        loss_sum = 0
        loss_count = 0
        for feature_0, H0, feature_1, H1 in dataloader:
            feature_0, H0, feature_1, H1 = feature_0.to(device), H0.to(device), feature_1.to(device), H1.to(device)

            # compute output
            output, target = model(im_q=(feature_0, H0), im_k=(feature_1, H1))
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            loss_count += 1
        loss_ave = loss_sum/loss_count
        print('loss: ', loss_ave)
        if epoch > 70:
            if loss_ave < best_loss:
                best_loss = loss_ave
                torch.save(model, check_dir(os.path.join(MODEL_DIR, model_id, 'model_best.pth')))


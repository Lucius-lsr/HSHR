# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/29 10:36
@Author  : Lucius
@FileName: train.py
@Software: PyCharm
"""
import torch
from torch import nn
from torch.utils.data import DataLoader

import moco.builder
from pooling import get_pooling_model
from data.dataloader import get_dataset

FEATURE_DIR = '/Users/lishengrui/client/tmp'
COORDINATE_DIR = '/Users/lishengrui/client/tmp'
dim = 128
K = 65536
m = 0.999
T = 0.07
lr = 0.0003
momentum = 0.9
weight_decay = 1e-4


if __name__ == '__main__':
    model = moco.builder.MoCo(
        get_pooling_model(dim),
        get_pooling_model(dim),
        dim, K, m, T, False)
    print(model)

    criterion = nn.CrossEntropyLoss().cuda(True)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    dataset = get_dataset(FEATURE_DIR, COORDINATE_DIR, 10)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # switch to train mode
    model.train()

    for epoch in range(100):
        print(epoch)
        for i, (feature, H1, H2) in enumerate(dataloader):
            # feature, H1, H2 = feature[0], H1[0], H2[0]

            # compute output
            output, target = model(im_q=(feature, H1), im_k=(feature, H2))
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.item())

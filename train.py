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

from data.utils import check_dir
from pooling.hgrnet_pooling import get_pooling_model
from data.dataloader import RandomHyperGraph
from evaluate import Evaluator
import sys

from self_supervision.call import get_moco, get_simsiam

sys.path.append('pooling')

FEATURE_DIR = '/home2/lishengrui/all_tcga'
COORDINATE_DIR = '/home2/lishengrui/all_tcga'
MODEL_DIR = '/home2/lishengrui/TCGA_experiment/result_all_tcga/models'

hidden_dim = 128  # 128
n_target = 64  # 64

lr = 0.003  # 0.003
momentum = 0.9
weight_decay = 1e-4
batch_size = 128

if __name__ == '__main__':
    print(hidden_dim, n_target, lr)
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = None
    if len(sys.argv) == 3:
        model_path = os.path.join(MODEL_DIR, sys.argv[2], 'model_best.pth')
        model = torch.load(model_path)
    else:
        model = get_moco(get_pooling_model(hidden_dim, n_target), get_pooling_model(hidden_dim, n_target), device, n_target)
        criterion = nn.CrossEntropyLoss().cuda(True)

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    assert FEATURE_DIR == COORDINATE_DIR

    train_dataset = RandomHyperGraph(FEATURE_DIR, 10, 0, 0.8)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_dataset = RandomHyperGraph(FEATURE_DIR, 10, 0.8, 1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

    best_top1 = 0
    model_id = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))+'{}_{}_{}'.format(hidden_dim, n_target, lr)
    evaluator = Evaluator()
    model.train()
    for epoch in range(500):
        print('*' * 5, 'epoch: ', epoch, '*' * 5)
        # ----------------train-----------------
        loss_sum = 0
        loss_count = 0

        for feature_0, ni_0, feature_1, ni_1, path in train_dataloader:
            feature_0, ni_0, feature_1, ni_1 = feature_0.to(device), ni_0.to(device), feature_1.to(device), ni_1.to(device)

            # compute output
            output, target = model(im_q=(feature_0, ni_0), im_k=(feature_1, ni_1))
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            loss_count += 1

        loss_ave = loss_sum / loss_count
        print("loss: ", loss_ave)

        # ----------------val-----------------
        evaluator.reset()
        for feature_0, H0, feature_1, H1, path in test_dataloader:
            feature_0, H0, feature_1, H1 = feature_0.to(device), H0.to(device), feature_1.to(device), H1.to(device)
            with torch.no_grad():
                hidden_0 = model.encoder_q((feature_0, H0)).cpu().detach().numpy()
                hidden_1 = model.encoder_q((feature_1, H1)).cpu().detach().numpy()
            evaluator.add_result(hidden_0, hidden_1, path)
        top1, top3, top5, top10, top1_, top3_, top5_, top10_ = evaluator.report()
        print('class acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(top1, top3, top5, top10))
        if top1 > best_top1:
            best_top1 = top1
            torch.save(model, check_dir(os.path.join(MODEL_DIR, model_id, 'model_best.pth')))

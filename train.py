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
from data.utils import check_dir
from pooling.hgrnet_pooling import get_pooling_model
from data.dataloader import RandomHyperGraph
from evaluate import Evaluator
import sys

from self_supervision.call import get_moco, get_simsiam

sys.path.append('pooling')

FEATURE_DIR = '/home2/lishengrui/tcga_result/all_tcga'
COORDINATE_DIR = '/home2/lishengrui/tcga_result/all_tcga'
MODEL_DIR = '/home2/lishengrui/TCGA_experiment/result_all_tcga/models'

hidden_dim = 128  # 128
n_target = 64  # 64

lr = 0.003  # 0.03
momentum = 0.9
weight_decay = 1e-4
batch_size = 128

self_super = "moco"

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = None
    if len(sys.argv) == 3:
        model_path = os.path.join(MODEL_DIR, sys.argv[2], 'model_best.pth')
        model = torch.load(model_path)
    elif self_super == 'moco':
        model = get_moco(get_pooling_model(hidden_dim, n_target), get_pooling_model(hidden_dim, n_target), device, n_target)
        criterion = nn.CrossEntropyLoss().cuda(True)
    elif self_super == 'simsiam':
        model = get_simsiam(get_pooling_model(hidden_dim, n_target))
        criterion = nn.CosineSimilarity(dim=1).cuda(True)
    else:
        raise Exception('No valid self-supervision framework')

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    assert FEATURE_DIR == COORDINATE_DIR

    train_dataset = RandomHyperGraph(FEATURE_DIR, 10, 0, 0.8)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_dataset = RandomHyperGraph(FEATURE_DIR, 10, 0.8, 1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    best_top1 = 0
    model_id = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    evaluator = Evaluator()
    model.train()
    for epoch in range(500):
        print('*' * 5, 'epoch: ', epoch, '*' * 5)
        # ----------------train-----------------
        loss_sum = 0
        loss_count = 0

        for feature_0, H0, feature_1, H1, path in train_dataloader:
            feature_0, H0, feature_1, H1 = feature_0.to(device), H0.to(device), feature_1.to(device), H1.to(device)

            # compute output
            if self_super == 'moco':
                output, target = model(im_q=(feature_0, H0), im_k=(feature_1, H1))
                loss = criterion(output, target)
            elif self_super == 'simsiam':
                p1, p2, z1, z2 = model(x1=(feature_0, H0), x2=(feature_1, H1))
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            else:
                loss = None

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            loss_count += 1

        loss_ave = loss_sum / loss_count
        print("loss: ", loss_ave)

        # ----------------val-----------------
        if self_super == 'moco':
            evaluator.reset(copy.deepcopy(model.encoder_q))
        elif self_super == 'simsiam':
            evaluator.reset(copy.deepcopy(model.encoder))

        for feature_0, H0, feature_1, H1, path in test_dataloader:
            feature_0, H0, feature_1, H1 = feature_0.to(device), H0.to(device), feature_1.to(device), H1.to(device)
            evaluator.add_data(feature_0, H0, feature_1, H1, path)
        top1, top3, top5, top10, top1_, top3_, top5_, top10_ = evaluator.report()
        print('class acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(top1, top3, top5, top10))
        print('pair acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(top1_, top3_, top5_, top10_))
        if top1 > best_top1:
            best_top1 = top1
            torch.save(model, check_dir(os.path.join(MODEL_DIR, model_id, 'model_best.pth')))

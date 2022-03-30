# -*- coding: utf-8 -*-
"""
@Time    : 2021/11/3 14:33
@Author  : Lucius
@FileName: train_swin_hgnn.py
@Software: PyCharm
"""
import copy
import os
import sys
import time

import torch
from torch import nn
from tqdm import tqdm

from data.swin_dataset import PatchesInLevels
from data.utils import check_dir
from evaluate import Evaluator
from pooling.swin_hgnn import NaiveSWinHGNNet, NaNaiveSWinHGNNet
from torch.utils.data import DataLoader

from self_supervision.call import get_moco

FEATURE_DIR = '/home2/lishengrui/all_tcga'
COORDINATE_DIR = '/home2/lishengrui/all_tcga'
MODEL_DIR = '/home2/lishengrui/TCGA_experiment/result_all_tcga/models'

num_level = 3
level_dim = 512
hidden_dim = 256
k = 10
dropout = 0.9

lr = 0.003
momentum = 0.9
weight_decay = 1e-4
batch_size = 128

criterion = nn.CrossEntropyLoss()

os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = get_moco(
    NaNaiveSWinHGNNet(num_level=num_level, level_dim=level_dim, hidden_dim=hidden_dim, k=k, dropout=dropout),
    NaNaiveSWinHGNNet(num_level=num_level, level_dim=level_dim, hidden_dim=hidden_dim, k=k, dropout=dropout),
    device,
    3*hidden_dim)
model = nn.DataParallel(model)

if len(sys.argv) == 2:
    model_path = os.path.join(MODEL_DIR, sys.argv[1], 'model_best.pth')
    model.module.load_state_dict(torch.load(model_path))

model = model.to(device)

optimizer = torch.optim.SGD(model.module.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
assert FEATURE_DIR == COORDINATE_DIR

train_dataset = PatchesInLevels(FEATURE_DIR, 0, 1)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

best_top1 = 0
model_id = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
evaluator = Evaluator()
model.train()
for epoch in range(500):
    print('*' * 5, 'epoch: ', epoch, '*' * 5)
    # ----------------train-----------------
    loss_sum = 0
    loss_count = 0
    evaluator.reset()
    pre_model = copy.deepcopy(model)
    for f_0, f_1, f_2x, f_4x, c_0, c_1, c_2x, c_4x, path in tqdm(train_dataloader):
        f_0, f_1, f_2x, f_4x, c_0, c_1, c_2x, c_4x = f_0.to(device), f_1.to(device), f_2x.to(device), f_4x.to(
            device), c_0.to(device), c_1.to(device), c_2x.to(device), c_4x.to(device)

        output, target = model(
            im_q=([f_0, f_2x[:, :250, :], f_4x[:, :62, :]], [c_0, c_2x[:, :250, :], c_4x[:, :62, :]]),
            im_k=([f_1, f_2x[:, 250:, :], f_4x[:, 62:, :]], [c_1, c_2x[:, 250:, :], c_4x[:, 62:, :]]))
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        loss_count += 1

        with torch.no_grad():
            hidden = pre_model.module.encoder_q(([f_0, f_2x, f_4x], [c_0, c_2x, c_4x]))
            evaluator.add_result(hidden.cpu().detach().numpy(), path)

    loss_ave = loss_sum / loss_count
    print("loss: ", loss_ave)
    top1, top3, top5, top10 = evaluator.report()
    print('class acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(top1, top3, top5, top10))
    if top1 > best_top1:
        best_top1 = top1
        torch.save(model.module.state_dict(), check_dir(os.path.join(MODEL_DIR, model_id, 'model_best.pth')))

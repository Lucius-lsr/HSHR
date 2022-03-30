# -*- coding: utf-8 -*-
"""
@Time    : 2021/11/3 14:33
@Author  : Lucius
@FileName: train_swin_hgnn.py
@Software: PyCharm
"""
import os
import sys
import time

import torch
from torch import nn
from tqdm import tqdm

from data.utils import ClassifyLayer, Labeler
from data.swin_dataset import PatchesInLevels
from data.utils import check_dir
from evaluate import Evaluator
from pooling.swin_hgnn import SWinHGNNet
from torch.utils.data import DataLoader

FEATURE_DIR = '/home2/lishengrui/all_tcga'
COORDINATE_DIR = '/home2/lishengrui/all_tcga'
MODEL_DIR = '/home2/lishengrui/TCGA_experiment/result_all_tcga/models'

num_level = 3
level_dims = [512, 512, 512]
hidden_upper_dims = [128, 64]
hgnn_dims = [256, 256, 256]
target_dim = 128
k = 10
dropout = 0.9

lr = 0.03
momentum = 0.9
weight_decay = 1e-4
batch_size = 32

g_class_num = 30
criterion = nn.CrossEntropyLoss()

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SWinHGNNet(num_level=num_level, level_dims=level_dims, hidden_upper_dims=hidden_upper_dims, hgnn_dims=hgnn_dims,
                   target_dim=target_dim, k=k, dropout=dropout)

# last_layer = ClassifyLayer(target_dim, g_class_num)
last_layer = ClassifyLayer(896, g_class_num)

if len(sys.argv) == 2:
    model_path = os.path.join(MODEL_DIR, sys.argv[1], 'model_best.pth')
    model.load_state_dict(torch.load(model_path))
    layer_path = os.path.join(MODEL_DIR, sys.argv[1], 'last_layer_best.pth')
    last_layer.load_state_dict(torch.load(layer_path))
model = model.to(device)
last_layer = last_layer.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
assert FEATURE_DIR == COORDINATE_DIR

train_dataset = PatchesInLevels(FEATURE_DIR, 0, 0.8)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
test_dataset = PatchesInLevels(FEATURE_DIR, 0.8, 1)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

labeler = Labeler()

best_top1 = 0
model_id = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
evaluator = Evaluator()
model.train()
for epoch in range(500):
    print('*' * 5, 'epoch: ', epoch, '*' * 5)
    # ----------------train-----------------
    loss_sum = 0
    loss_count = 0

    for f_0, f_1, f_2x, f_4x, c_0, c_1, c_2x, c_4x, path in tqdm(train_dataloader):
        f_0, f_1, f_2x, f_4x, c_0, c_1, c_2x, c_4x = f_0.to(device), f_1.to(device), f_2x.to(device), f_4x.to(
            device), c_0.to(device), c_1.to(device), c_2x.to(device), c_4x.to(device)

        # feature = model([f_0, f_2x, f_4x], [c_0, c_2x, c_4x])
        model.set_complex_output(True)
        feature = torch.cat(model([f_0, f_2x, f_4x], [c_0, c_2x, c_4x]), dim=-1)
        model.set_complex_output(False)

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
    print("loss: ", loss_ave)

    # ----------------val-----------------
    evaluator.reset()
    with torch.no_grad():
        for f_0, f_1, f_2x, f_4x, c_0, c_1, c_2x, c_4x, path in test_dataloader:
            f_0, f_1, f_2x, f_4x, c_0, c_1, c_2x, c_4x = f_0.to(device), f_1.to(device), f_2x.to(device), f_4x.to(
                device), c_0.to(device), c_1.to(device), c_2x.to(device), c_4x.to(device)
            model.set_complex_output(True)
            hidden_list = []
            tmp = model([f_0, f_2x, f_4x], [c_0, c_2x, c_4x])
            for h in tmp[:-1]:
                hidden_list.append(h.cpu().detach().numpy())
            evaluator.add_multiple_results(hidden_list, path)
            hidden = tmp[-1].cpu().detach().numpy()
            evaluator.add_result(hidden, path)
            model.set_complex_output(False)
    top1, top3, top5, top10 = evaluator.report()
    print('class acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(top1, top3, top5, top10))
    results = evaluator.multiple_report(3)
    for r in results:
        print('complex acc: top1:{:.4} top3:{:.4f} top5:{:.4f} top10:{:.4f}'.format(r[0], r[1], r[2], r[3]))
    if top1 > best_top1:
        best_top1 = top1
        torch.save(model.state_dict(), check_dir(os.path.join(MODEL_DIR, model_id, 'model_best.pth')))
        torch.save(last_layer.state_dict(), check_dir(os.path.join(MODEL_DIR, model_id, 'last_layer_best.pth')))

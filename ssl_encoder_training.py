# -*- coding: utf-8 -*-
"""
@Time    : 2021/10/19 13:35
@Author  : Lucius
@FileName: baseline_wsisa.py
@Software: PyCharm
"""
import argparse
import copy
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_utils import check_dir
from utils.evaluate import Evaluator
from self_supervision.call import get_moco
from utils.model.base_model import HashLayer
from utils.feature import cluster_feature, mean_feature


class WSISADataset(Dataset):

    def __init__(self, result_dir, tmp, data_from=0, data_to=1) -> None:
        super().__init__()
        cluster_means_0, cluster_means_1, paths = mean_feature(result_dir, tmp, data_from, data_to)
        self.data_0 = cluster_means_0
        self.data_1 = cluster_means_1
        self.paths = paths

    def __getitem__(self, item):
        return self.data_0[item], self.data_1[item], self.paths[item]

    def __len__(self) -> int:
        return len(self.data_0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Preprocess raw WSIs")
    parser.add_argument("--RESULT_DIR", type=str, required=True, help="A path to save your preprocessed results.")
    parser.add_argument("--TMP", type=str, required=True, help="The path to save some necessary tmp files.")
    parser.add_argument("--MODEL_DIR", type=str, required=True, help="The path of ssl hash encoder model.")
    parser.add_argument("--DATASETS", type=list, nargs='+', required=True, help="A list of datasets.")
    args = parser.parse_args()

    feature_in = 512
    feature_out = 1024
    depth = 1
    lr = 0.003
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = 128
    num_cluster = 20

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().cuda(True)
    train_dataset = WSISADataset(args.RESULT_DIR, args.TMP)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    model = get_moco(HashLayer(feature_in, feature_out, depth), HashLayer(feature_in, feature_out, depth), device,
                     feature_out)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    evaluator = Evaluator()
    cfs, cf_paths = cluster_feature(args.RESULT_DIR, args.TMP, args.DATASETS, num_cluster)
    cfs = torch.from_numpy(cfs).to(device)

    for epoch in range(30):
        print('*' * 5, 'epoch: ', epoch, '*' * 5)
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

        loss_ave = loss_sum / loss_count
        print("loss: ", loss_ave)

        evaluator.reset()
        with torch.no_grad():
            raw = cfs
            h = pre_model.encoder_q(raw)
            evaluator.add_patches(h.cpu().detach().numpy(), cf_paths)
            acc = evaluator.fixed_report_patch()
            print(acc)

    torch.save(model.encoder_q.state_dict(), check_dir(os.path.join(args.MODEL_DIR, 'ssl', 'model_best.pth')))


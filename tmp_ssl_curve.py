import numpy as np
import argparse
import copy
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from CONST import EXPERIMENTS
from ssl_encoder_training import PairCenterDataset
from utils.evaluate import Evaluator
from self_supervision.call import get_moco
from utils.model.base_model import HashEncoder, AttenHashEncoder


class Recorder:
    def __init__(self, names):
        self.exp_name = names
        self.exp_cur = {}
        self.exp_all = {}
        for name in names:
            self.exp_all[name] = []

    def begin_train(self):
        self.exp_cur = {}
        for name in self.exp_name:
            self.exp_cur[name] = []

    def finish_train(self):
        for name in self.exp_name:
            self.exp_all[name].append(self.exp_cur[name])

    def record(self, name, val):
        self.exp_cur[name].append(val)

    def report(self):
        for name in self.exp_name:
            arr = np.array(self.exp_all[name])
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            print(name)
            print(mean)
            print(std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Preprocess raw WSIs")
    parser.add_argument("--RESULT_DIR", type=str, required=True, help="A path to save your preprocessed results.")
    parser.add_argument("--TMP", type=str, required=True, help="The path to save some necessary tmp files.")
    args = parser.parse_args()
    # python tmp_ssl_curve.py --RESULT_DIR /home2/lishengrui/new_exp/HSHR/PREPROCESSED_SSL_DENSE --TMP /home2/lishengrui/new_exp/HSHR/TMP

    feature_in = 512
    feature_out = 1024
    depth = 1
    lr = 0.003
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = 128  # 128
    num_cluster = 20
    gamma = 0.99

    repeat = 10
    num_epoch = 20

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().cuda(True)
    train_dataset = PairCenterDataset(args.RESULT_DIR, args.TMP)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    base_model = lambda: AttenHashEncoder(feature_in, feature_out, depth)

    evaluator = Evaluator()
    exps = []
    names = ['Hematopoietic', 'Melanocytic', 'Prostate/Testis', 'Endocrine']
    for name in names:
        valid_dataset = PairCenterDataset(args.RESULT_DIR, args.TMP, False, EXPERIMENTS[name])
        cfs = []
        cf_paths = []
        for c1, _, path in valid_dataset.centers:
            cfs.append(c1)
            cf_paths.append(path)
        cfs = np.array(cfs)
        exps.append((cfs, cf_paths, name))

    recorder = Recorder(names)
    for r in range(repeat):
        model = get_moco(base_model(), base_model(), device, feature_out)
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        recorder.begin_train()
        for epoch in range(num_epoch):
            print('*' * 5, 'epoch: ', epoch, '*' * 5)
            loss_sum = 0
            loss_count = 0
            pre_model = copy.deepcopy(model)
            for x0, x1, _ in train_dataloader:
                x0, x1 = x0.to(device), x1.to(device)
                output, target = model(x0, x1)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                loss_count += 1
            scheduler.step()
            loss_ave = loss_sum / loss_count
            print("loss: ", loss_ave)

            with torch.no_grad():
                for cfs, cf_paths, name in exps:
                    evaluator.reset()
                    cfs = torch.from_numpy(np.array(cfs)).to(device)
                    raw = cfs
                    h, w = pre_model.encoder_q(raw, no_pooling=True, weight=True)
                    evaluator.add_patches(h.cpu().detach().numpy(), cf_paths)
                    evaluator.add_weight(w.cpu().detach().numpy())
                    acc, ave = evaluator.eval()
                    recorder.record(name, ave)
        recorder.finish_train()
    recorder.report()

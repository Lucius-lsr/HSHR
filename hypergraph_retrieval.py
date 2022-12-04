# -*- coding: utf-8 -*-
"""
@Time    : 2021/12/17 14:38
@Author  : Lucius
@FileName: hypergraph_retrieval.py
@Software: PyCharm
"""
import argparse
import os
import torch

from CONST import EXPERIMENTS
from ssl_encoder_training import PairCenterDataset
from utils.model.base_model import HashLayer, HashEncoder, AttenHashEncoder
from utils.evaluate import Evaluator
from utils.feature import cluster_feature, min_max_binarized
import numpy as np

num_cluster = 20
feature_in = 512
feature_out = 1024
depth = 1


def fine_tune(raw, model_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = AttenHashEncoder(feature_in, feature_out, depth)
    model_path = os.path.join(model_dir, 'model_best.pth')
    encoder.load_state_dict(torch.load(model_path))
    encoder = encoder.to(device)

    with torch.no_grad():
        raw = torch.from_numpy(raw).to(device)
        h, w = encoder(raw, no_pooling=True, weight=True)
    return h, w


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hypergraph-guided retrive")
    parser.add_argument("--MODEL_DIR", type=str, required=True, help="The path of ssl hash encoder model.")
    parser.add_argument("--RESULT_DIR", type=str, required=True, help="A path to save your preprocessed results.")
    parser.add_argument("--TMP", type=str, required=True, help="The path to save some necessary tmp files.")
    parser.add_argument("--DATASETS", type=list, nargs='+', required=False, help="A list of datasets.")
    args = parser.parse_args()
    # python hypergraph_retrieval.py --RESULT_DIR /home2/lishengrui/new_exp/HSHR/PREPROCESSED_SSL_DENSE --TMP /home2/lishengrui/new_exp/HSHR/TMP --MODEL_DIR /home2/lishengrui/new_exp/HSHR/ENCODER/double_ssl_att
    # python hypergraph_retrieval.py --RESULT_DIR /home2/lishengrui/new_exp/HSHR/PREPROCESSED --TMP /home2/lishengrui/new_exp/HSHR/TMP --MODEL_DIR /home2/lishengrui/new_exp/HSHR/ENCODER/ssl+att

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    evaluator = Evaluator()
    exps = []
    for name in EXPERIMENTS.keys(): # , 'Endocrine', 'Liver/PB']:
        valid_dataset = PairCenterDataset(args.RESULT_DIR, args.TMP, False, EXPERIMENTS[name])
        cfs = []
        cf_paths = []
        for c1, _, path in valid_dataset.centers:
            cfs.append(c1)
            cf_paths.append(path)
        cfs = np.array(cfs)
        exps.append((cfs, cf_paths, name))
    for cfs, cf_paths, name in exps:
        evaluator.reset()

        # min-max binarization
        # raw = min_max_binarized(cfs)
        # evaluator.add_patches(raw, cf_paths)

        # CaEncoder
        h, w = fine_tune(cfs, args.MODEL_DIR)
        evaluator.add_patches(h.cpu().detach().numpy(), cf_paths)
        evaluator.add_weight(w.cpu().detach().numpy())

        acc, ave = evaluator.eval()
        print(name, ave, acc)

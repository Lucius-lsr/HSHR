# -*- coding: utf-8 -*-
"""
@Time    : 2021/12/17 14:38
@Author  : Lucius
@FileName: hypergraph_retrieval.py
@Software: PyCharm
"""

import os
import torch
from utils.model.base_model import HashLayer
from utils.evaluate import Evaluator
from utils.feature import cluster_feature

DATASETS = 'A LIST OF DATASETS'
MODEL_DIR = 'THE/PATH/OF/SSL/HASH/ENCODER/MODEL'

num_cluster = 20
feature_in = 512
feature_out = 1024
depth = 1


def fine_tune(raw):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = HashLayer(feature_in, feature_out, depth)
    model_path = os.path.join(MODEL_DIR, 'ssl', 'model_best.pth')
    encoder.load_state_dict(torch.load(model_path))
    encoder = encoder.to(device)

    with torch.no_grad():
        raw = torch.from_numpy(raw).to(device)
        output = encoder(raw).cpu().detach().numpy()
    return output


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    clusters, paths = cluster_feature(0, 1, DATASETS)

    clusters = fine_tune(clusters)
    evaluator = Evaluator()
    evaluator.add_patches(clusters, paths)
    for k in [5, 10, 15, 20, 25, 30]:
        evaluator.report_patch(k, num_cluster)


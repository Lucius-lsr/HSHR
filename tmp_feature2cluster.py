import os
import numpy as np
from utils.feature import cluster_reduce
from tqdm import tqdm

ROOT = '/home2/lishengrui/new_exp/HSHR/PREPROCESSED'
TARGET = '/home2/lishengrui/new_exp/HSHR/PREPROCESSED'
c0, c1 = 0, 0
for root, dirs, files in tqdm(os.walk(ROOT)):
    for file in files:
        if file == '0.npy':
            features = np.load(os.path.join(root, file))
            clu_centers = cluster_reduce(features, 20)
            np.save(os.path.join(root, 'clu_0.npy'), clu_centers)
        if file == '1.npy':
            features = np.load(os.path.join(root, file))
            clu_centers = cluster_reduce(features, 20)
            np.save(os.path.join(root, 'clu_1.npy'), clu_centers)


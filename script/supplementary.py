# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/19 13:24
@Author  : Lucius
@FileName: supplementary.py
@Software: PyCharm
"""
import csv
import pickle as pkl
import os

# with open('path2pid', 'rb') as f:
#     data = pkl.load(f)
#     print(data)

SVS_DIR = '/lishengrui/TCGA'
SVS_DIR_2 = '/home2/lishengrui/tcga_result'

col0 = []
col1 = []

data = os.listdir('sup')
for d in data:
    if d == '.DS_Store':
        continue
    d = os.path.join('sup', d)
    with open(d, 'rb') as f:
        dic = pkl.load(f)
        for slide in dic:
            full_slide = os.path.join(SVS_DIR, slide)
            try:
                file = os.listdir(full_slide)
            except FileNotFoundError:
                full_slide = os.path.join(SVS_DIR_2, slide)
                file = os.listdir(full_slide)
            for f in file:
                if f.endswith('.svs'):
                    col0.append(full_slide.split('/')[-2])
                    col1.append(f)

with open('supData.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    for i in range(len(col0)):
        spamwriter.writerow((col0[i], col1[i]))

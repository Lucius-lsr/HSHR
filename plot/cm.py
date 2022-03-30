# -*- coding: utf-8 -*-
"""
@Time    : 2022/2/27 14:58
@Author  : Lucius
@FileName: cm.py
@Software: PyCharm
"""
import re
from math import sqrt
import numpy as np

import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# s = '''
# 1951 97
# 120 1446
# '''
# labels = ['GBM', 'LGG']
# title = 'Brain'

# s = '''
# 1148 12 171 108
# 11 292 1 91
# 377 2 112 39
# 125 81 23 939
# '''
# title = 'Gastrointestinal'
# labels = ['COAD', 'ESCA', 'READ', 'STAD']

# s = '''
# 482 11 105 5
# 3 1457 19 1
# 45 32 1271 10
# 12 4 46 91
# '''
# labels = ['CESC', 'OV','UCEC','UCS']
# title = 'Gynecologic'

# s = '''
# 94 7
# 4 313
# '''
# labels = ['DLBC', 'THYM']
# title = 'Hematopoietic'

# s = '''
# 940 8
# 11 136
# '''
# labels = ['SKCM', 'UVM']
# title = 'Melanocytic'

# s = '''
# 1258 339 8
# 386 1216 8
# 14 11 149
# '''
# labels = ['LUAD', 'LUSC', 'MESO']
# title = 'Pulmonary'

# s = '''
# 891 2 7 18
# 1 308 7 9
# 5 8 2111 44
# 42 8 96 623
# '''
# labels = ['BLCA', 'KICH', 'KIRC', 'KIRP']
# title = 'Urinary'

# s = '''
# 1160 10
# 8 393
# '''
# labels = ['PRAD', 'TGCT']
# title = 'Prostate/Testis'

# s = '''
# 289 21 13
# 5 368 12
# 7 7 1054
# '''
# labels = ['ACC', 'PCPG', 'THCA']
# title = 'Endocrine'

s = '''
47 40 19
9 842 18
6 14 446
'''
labels = ['CHOL', 'LIHC', 'PAAD']
title = 'Liver/PB'



numbers = re.findall(r"\d+\.?\d*", s)
n = int(sqrt(len(numbers)))
C = [[int(i) for i in numbers[n * j:n * (j + 1)]] for j in range(n)]
C = np.array(C)
Csum = C.sum(axis=1)
Cp = (C.transpose() / Csum).transpose()
print(Csum)
sns.set()
plt.rc('font', family="Arial")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
f, ax = plt.subplots()
sns.heatmap(Cp, annot=C, ax=ax, fmt='g', xticklabels=labels, yticklabels=labels, cmap='YlGnBu', vmax=1, vmin=0)  # 画热力图

ax.set_title(title, fontsize=20, font="Arial")  # 标题
ax.set_xlabel('Predict', fontsize=20, font="Arial")  # x轴
ax.set_ylabel('True', fontsize=20, font="Arial")  # y轴
plt.tight_layout()
# plt.show()
plt.savefig('/Users/lishengrui/Desktop/prepared paper/fig3/{}.pdf'.format(title.replace('/','_')), format='pdf', dpi=300, pad_inches=0)

# -*- coding: utf-8 -*-
"""
@Time    : 2022/2/26 14:49
@Author  : Lucius
@FileName: barchart.py
@Software: PyCharm
"""
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 构建数据


# s = '''
# 0.9339	0.9159	0.9399
# 0.9381	0.8886	0.9526
# 0.9283	0.9516	0.9234
# '''
# labels = ['GBM', 'LGG']
# title = 'Brain'

# s = '''
# 0.6692	0.5696	0.7053
# 0.8727	0.6228	0.7978
# 0.7212	0.9026	0.7392
# 0.0702	0.1618	0.2113
# 0.6728	0.5764	0.8039
# '''
# labels = ['COAD', 'ESCA','READ','STAD']
# title = 'Gastrointestinal'

# s = '''
# 0.8858	0.8536	0.9185
# 0.6766	0.6988	0.7993
# 0.9793	0.9491	0.9845
# 0.9334	0.8467	0.9359
# 0.3831	0.6013	0.5948
# '''
# labels = ['CESC', 'OV','UCEC','UCS']
# title = 'Gynecologic'

# s = '''
# 0.904	0.9052	0.9737
# 0.6832	0.79	0.9307
# 0.9743	0.9419	0.9874
# '''
# labels = ['DLBC', 'THYM']
# title = 'Hematopoietic'

# s = '''
# 0.9659	0.951	0.9827
# 0.9916	0.9757	0.9916
# 0.8	0.7917	0.9252
# '''
# labels = ['SKCM', 'UVM']
# title = 'Melanocytic'

# s = '''
# 0.7074	0.6836	0.774
# 0.7634	0.673	0.7838
# 0.6902	0.6962	0.7553
# 0.3505	0.6647	0.8563
# '''
# labels = ['LUAD', 'LUSC', 'MESO']
# title = 'Pulmonary'

# s = '''
# 0.9034	0.881	0.9409
# 0.9597	0.9618	0.9706
# 0.8497	0.8985	0.9477
# 0.9624	0.9387	0.9737
# 0.6925	0.6145	0.8101
# '''
# labels = ['BLCA', 'KICH', 'KIRC', 'KIRP']
# title = 'Urinary'

# s = '''
# 0.9738	0.9744	0.9886
# 0.9845	0.975	0.9915
# 0.9426	0.9725	0.98
# '''
# labels = ['PRAD', 'TGCT']
# title = 'Prostate/Testis'

# s = '''
# 0.9249	0.898	0.9634
# 0.8785	0.8813	0.8947
# 0.853	0.8632	0.9558
# 0.9649	0.9156	0.9869
# '''
# labels = ['ACC', 'PCPG', 'THCA']
# title = 'Endocrine'

s = '''
0.8902	0.8624	0.9264
0.3429	0.4808	 0.4434
0.9608	0.9331	0.9689
0.8829	0.8174	 0.9571
'''
labels = ['CHOL', 'LIHC', 'PAAD']
title = 'Liver/PB'


labels.insert(0, 'Average')
d1 = []
d2 = []
d3 = []
numbers = re.findall(r"\d+\.?\d*", s)
for i, n in enumerate(numbers):
    n = round(100 * float(n), 1)
    if i % 3 == 0:
        d2.append(n)
    elif i % 3 == 1:
        d3.append(n)
    else:
        d1.append(n)

plt.rc('font', family="Arial")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.grid(linestyle='-.', linewidth=1)

bar_width = 0.25
# y轴范围
plt.ylim([0, 100])

# 绘图
plt.bar(np.arange(len(d1)), d1, align='center', color='limegreen', alpha=1, width=bar_width)
plt.bar(np.arange(len(d2)) + bar_width, d2, align='center', color='indianred', alpha=1, width=bar_width)
plt.bar(np.arange(len(d2)) + 2 * bar_width, d3, align='center', color='steelblue', alpha=1, width=bar_width)
plt.title(title, fontsize=20, pad=10)
plt.legend(labels=['HSHR', 'Yottixel', 'FISH'], fontsize=10, loc='lower left')
# 轴标签
# plt.xlabel('top citys', fontsize=20)
plt.ylabel('mMV@5(%)', fontsize=20)

# 刻度标签
plt.xticks(np.arange(len(d1)) + bar_width, labels, fontsize=15)
plt.yticks(fontsize=15)

for x, y in enumerate(d1):
    plt.text(x, y + 1, y, ha='center', fontsize=12)
for x, y in enumerate(d2):
    plt.text(x + bar_width, y + 1, y, ha='center', fontsize=12)
for x, y in enumerate(d3):
    plt.text(x + 2 * bar_width, y + 1, y, ha='center', fontsize=12)
plt.tight_layout()
# plt.show()
plt.savefig('/Users/lishengrui/Desktop/prepared paper/fig2/{}.pdf'.format(title.replace('/','_')), format='pdf', dpi=300, pad_inches=0)

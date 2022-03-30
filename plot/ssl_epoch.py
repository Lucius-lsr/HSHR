# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/1 16:16
@Author  : Lucius
@FileName: ssl_epoch.py
@Software: PyCharm
"""

import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

loss_s = '''
loss:  5.859713112067313
loss:  5.517270569348227
loss:  4.904524596028738
loss:  4.304250058005838
loss:  3.731888767820678
loss:  3.2035287472996776
loss:  2.7332331012277042
loss:  2.327064487729137
loss:  1.993317985966195
'''
loss = re.findall(r"\d+\.?\d*", loss_s)
loss = [float(l) for l in loss]

y1 = [0.95694689, 0.9543067, 0.95622514, 0.95814577, 0.96030995, 0.9641278,
      0.96316402, 0.96342122, 0.96436289, 0.96219895]
y1_s = [0.00576388, 0.00629302, 0.0047946, 0.00579154, 0.0067761, 0.00535816,
        0.00480509, 0.00303792, 0.00249698, 0.00382419]
y1 = [100 * y for y in y1]
y1_s = [100 * y for y in y1_s]

y2 = [0.90624728, 0.90858906, 0.90976286, 0.91307751, 0.91354589, 0.91291097,
      0.91446098, 0.91487531, 0.91543501, 0.91463667]
y2_s = [0.00313884, 0.00231232, 0.0019974, 0.00211698, 0.00279318, 0.00399131,
        0.00278829, 0.00297647, 0.00288914, 0.0044442]
y2 = [100 * y for y in y2]
y2_s = [100 * y for y in y2_s]

y3 = [0.93420507, 0.93843796, 0.93954583, 0.94239996, 0.94406942, 0.9424729,
      0.94321093, 0.94292478, 0.94450199, 0.94466287]
y3_s = [0.00367771, 0.00310782, 0.0026861, 0.00239653, 0.00226967, 0.0020385,
        0.00188587, 0.00209769, 0.00265517, 0.00368703]
y3 = [100 * y for y in y3]
y3_s = [100 * y for y in y3_s]

r = 9
x = [i for i in range(r)]

title = 'Performance of HSHR During SSL Procedure'

# paint
# set style
plt.rc('font', family="Arial")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.figure(figsize=(15, 6))

# plt.xlim(xmax=x_list[-1]*1.2, xmin=0)
plt.ylim(ymax=98, ymin=90)

fontsize = 25
plt.grid(linestyle='-.', linewidth=1)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.plot(x, y1[:r], '-o', ms=10, label='Hematopoietic', linewidth=4)
plt.plot(x, y2[:r], '-o', ms=10, label='Liver/PB', linewidth=4)
plt.plot(x, y3[:r], '-o', ms=10, label='Endocrine', linewidth=4)

plt.fill_between(range(r),
                 [m - s for m, s in zip(y1, y1_s)][:r],
                 [m + s for m, s in zip(y1, y1_s)][:r],
                 color='b',
                 alpha=0.2)
plt.fill_between(range(r),
                 [m - s for m, s in zip(y2, y2_s)][:r],
                 [m + s for m, s in zip(y2, y2_s)][:r],
                 color='orange',
                 alpha=0.4)
plt.fill_between(range(r),
                 [m - s for m, s in zip(y3, y3_s)][:r],
                 [m + s for m, s in zip(y3, y3_s)][:r],
                 color='green',
                 alpha=0.2)

plt.xlabel('Epoch', fontsize=30)
plt.xticks(x, fontsize=fontsize)
plt.ylabel('mMV@5(%)', fontsize=30)
plt.legend(fontsize=15, loc="upper left")
plt.yticks(fontsize=fontsize)

ax2 = plt.twinx()  # this is the important function
ax2.plot(loss, color='grey', linewidth=3)
ax2.set_ylabel('Loss', fontsize=fontsize)

plt.yticks(fontsize=fontsize)
plt.title(title, fontsize=30)

plt.tight_layout()
# plt.show()
plt.savefig('/Users/lishengrui/Desktop/prepared paper/fig5/{}.pdf'.format(title.replace('/','_')), format='pdf', dpi=300, pad_inches=0)

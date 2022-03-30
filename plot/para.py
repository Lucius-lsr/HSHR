# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/2 14:19
@Author  : Lucius
@FileName: para.py
@Software: PyCharm
"""

import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

y1 = [0.9689090909090909,
     0.9593600478468899,
     0.9689220095693779,
     0.9593253588516748,
     0.9569744019138756,
     0.9569332535885168,
     0.9593600478468899]
subtype1 = 'Hematopoietic'

y2 = [
    0.9126053941908714,
    0.918096265560166,
    0.9257914246196405,
    0.9216367911479945,
    0.9181594744121715,
    0.9174383125864453,
    0.9166773167358228
]
subtype2 = 'Liver/PB'

y3 = [
    0.9434240464344942,
    0.950065008291874,
    0.9555507462686568,
    0.9627391929242676,
    0.9588096738529573,
    0.957155223880597,
    0.9544058595909342
]
subtype3 = 'Endocrine'



title = 'Performance under different K'
x = [i for i in range(7)]

y1 = [100 * t for t in y1]
y2 = [100 * t for t in y2]
y3 = [100 * t for t in y3]
# paint
# set style
plt.rc('font', family="Arial")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# plt.xlim(xmax=x_list[-1]*1.2, xmin=0)
plt.ylim(ymax=100, ymin=90)

fontsize = 25
plt.grid(linestyle='-.', linewidth=1)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.plot(x, y1, '-o', ms=10, label=subtype1, linewidth=4)
plt.plot(x, y2, '-o', ms=10, label=subtype2, linewidth=4)
plt.plot(x, y3, '-o', ms=10, label=subtype3, linewidth=4)

plt.xlabel('K', fontsize=20)
plt.xticks(x, ['5', '10', '20', '50', '100', '150', '200'], fontsize=fontsize)
plt.ylabel('mMV@5(%)', fontsize=20)
plt.legend(fontsize=12, loc="upper left")
plt.yticks(fontsize=fontsize)

plt.yticks(fontsize=fontsize)
plt.title(title, fontsize=20)

plt.tight_layout()
# plt.show()
plt.savefig('/Users/lishengrui/Desktop/prepared paper/fig5/{}.pdf'.format(title.replace('/', '_')), format='pdf', dpi=300, pad_inches=0)
# plt.close()

# -*- coding: utf-8 -*-
"""
@Time    : 2021/7/13 13:28
@Author  : Lucius
@FileName: topk.py
@Software: PyCharm
"""

import matplotlib
import matplotlib.image as imgplt
import matplotlib.pyplot as plt

x = imgplt.imread('/Users/lishengrui/client/tmp/88e4e428-865f-4dbc-9e16-f6dbb15023cb/TCGA-85-6560-01A-01-BS1.cfe87f00-0766-40b4-ba88-90821adff0da.png')
plt.figure()
for i in range(1, 6):
    plt.subplot(1, 6, i)
    plt.imshow(x)
    plt.xticks([])
    plt.yticks([])
plt.show()

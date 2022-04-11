# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/29 16:49
@Author  : Lucius
@FileName: paint.py
@Software: PyCharm
"""

import matplotlib
import matplotlib.pyplot as plt


def paint(x, y):
    plt.xlim(xmax=1, xmin=0)
    plt.ylim(ymax=1, ymin=0)

    for i, num in enumerate(y):
        num = num.item()
        color = plt.cm.Set1(num % 9)
        plt.scatter(x[i][0].item(), x[i][1].item(), color=color)
    plt.show()
    plt.close()



# -*- coding: utf-8 -*-
"""
@Time    : 2022/1/17 15:17
@Author  : Lucius
@FileName: const.py
@Software: PyCharm
"""

NumDict = {
    'gbm': 2048,
    'lgg': 1566,

    'coad': 1439,
    'esca': 395,
    'read': 530,
    'stad': 1168,

    'cesc': 603,
    'ov': 1453,
    'ucec': 1480,
    'ucs': 153,

    'dlbc': 101,
    'thym': 317,

    'skcm': 948,
    'uvm': 147,

    'luad': 1605,
    'lusc': 1610,
    'meso': 174,

    'blca': 918,
    'kich': 325,
    'kirc': 2168,
    'kirp': 769,

    'prad': 1170,
    'tgct': 401,

    'acc':	323,
    'pcpg':	385,
    'thca':	1101,

    'chol':	110,
    'lihc':	870,
    'paad':	466

}


def mean_acc(result):
    sum = 0
    sum_acc = 0
    for k in result.keys():
        num = NumDict[k.split('_')[0]]
        sum_acc += num * result[k]
        sum += num
    return sum_acc / sum
# -*- coding: utf-8 -*-
"""
@Time    : 2021/12/27 16:30
@Author  : Lucius
@FileName: ahgae_processor.py
@Software: PyCharm
"""

import numpy as np
import scipy.sparse as sp


def preprocess_hypergraph(inc, layer, norm='sym', weight=2/3):
    inc = sp.coo_matrix(inc)
    ident = sp.eye(inc.shape[0])

    H = inc

    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = sp.coo_matrix(np.ones(n_edge))
    # the degree of the node
    DV = np.array((H * W.T).sum(1))
    # the degree of the hyperedge
    DE = np.array(H.sum(0))

    invDE = sp.diags(np.power(DE, -1).flatten())
    DV2 = sp.diags(np.power(DV, -0.5).flatten())
    W = sp.diags(W.toarray().flatten())
    HT = H.T
    DV = sp.diags(DV.flatten())

    if norm == 'sym':
        G = DV2 * H * W * invDE * HT * DV2
        laplacian = ident - G
        # laplacian = G

    elif norm == 'left':
        G = DV * H * W * invDE * HT
        laplacian = ident - G
        laplacian = G

    reg = [weight] * (layer)  #

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))
    return adjs

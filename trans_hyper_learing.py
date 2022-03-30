# -*- coding: utf-8 -*-
"""
@Time    : 2022/1/3 13:27
@Author  : Lucius
@FileName: trans_hyper_learing.py
@Software: PyCharm
"""
import numpy as np
from models.HyperG.hyperg.utils import init_label_matrix
from sklearn.metrics import pairwise_distances
import scipy.sparse as sparse
from numpy.linalg import inv

from models.HyperG.hyperg.base import HyperG


def gen_knn_hg(X, n_neighbors, prob=1, is_prob=True, with_feature=False, dis_metric="euclidean"):
    """
    :param X: numpy array, shape = (n_samples, n_features)
    :param n_neighbors: int,
    :param is_prob: bool, optional(default=True)
    :param with_feature: bool, optional(default=False)
    :param dis_metric:
    :return: instance of HyperG

    """

    assert isinstance(X, (np.ndarray, list))
    assert n_neighbors > 0

    X = np.array(X)
    n_nodes = X.shape[0]
    n_edges = n_nodes

    m_dist = pairwise_distances(X, metric=dis_metric)

    # top n_neighbors+1
    m_neighbors = np.argpartition(m_dist, kth=n_neighbors+1, axis=1)
    m_neighbors_val = np.take_along_axis(m_dist, m_neighbors, axis=1)

    m_neighbors = m_neighbors[:, :n_neighbors+1]
    m_neighbors_val = m_neighbors_val[:, :n_neighbors+1]

    # check
    for i in range(n_nodes):
        if not np.any(m_neighbors[i, :] == i):
            m_neighbors[i, -1] = i
            m_neighbors_val[i, -1] = 0.

    node_idx = m_neighbors.reshape(-1)
    edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors+1)).reshape(-1)

    if not is_prob:
        values = np.ones(node_idx.shape[0])
    else:
        avg_dist = np.mean(m_dist)
        m_neighbors_val = m_neighbors_val.reshape(-1)
        values = np.exp(-np.power(m_neighbors_val, 2.) / np.power(prob*avg_dist, 2.))

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))

    w = np.ones(n_edges)

    if with_feature:
        return HyperG(H, w=w, X=X)

    return HyperG(H, w=w)


def trans_infer(hg, y, lbd):
    """transductive inference from the "Learning with Hypergraphs:
     Clustering, Classification, and Embedding" paper

    :param hg: instance of HyperG
    :param y: numpy array, shape = (n_nodes,)
    :param lbd: float, the positive tradeoff parameter of empirical loss
    :return: F
    """
    assert isinstance(hg, HyperG)

    Y = init_label_matrix(y)
    n_nodes = Y.shape[0]
    THETA = hg.theta_matrix()

    L2 = sparse.eye(n_nodes) - (1 / (1 + lbd)) * THETA
    F = ((lbd + 1) / lbd) * inv(L2.toarray()).dot(Y)

    return F


def transfer_from_hash(hash_arr, y, edge_d):
    """

    Args:
        hash_arr: 0 or 1, num_feature x hash_len

    Returns:

    """
    hg = gen_knn_hg(hash_arr, n_neighbors=edge_d)
    pred = trans_infer(hg, y, lbd=100)
    return pred

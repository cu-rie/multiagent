import dgl
import torch

import matplotlib.pyplot as plt

from numpy import linalg as LA
from ..utils.general_util import dn

key_list = ['init_node_feature', 'pos', 'node_type']
VERY_LARGE_NUM = 999


def distance(u: torch.Tensor, v: torch.Tensor):
    dist = torch.pow(u - v, 2).sum()
    return dist


def spectral_cluster(g: dgl.DGLGraph, key=None):
    if key is None:
        key = 'init_node_feature'

    nf = g.ndata[key]

    n = g.number_of_nodes()
    dist_matrix = torch.ones(size=(n, n)) * VERY_LARGE_NUM
    similarity_matrix = torch.zeros(size=(n, n))

    for i in range(n):
        for j in range(n):
            if i <= j:
                similarity_matrix[i, j] = (nf[i] * nf[j]).sum()
                continue
            dist = distance(nf[i], nf[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

            similarity_matrix[i, j] = (nf[i0] * nf[j]).sum()

    min_dist = dist_matrix.min()
    for i in range(n):
        dist_matrix[i, i] = min_dist

    W = 1 / dist_matrix

    D = torch.diag(W.sum(1))
    L = D - W
    # L = D - similarity_matrix
    L_numpy = dn(L)
    eigenvalue, eigenvector = LA.eig(L_numpy)

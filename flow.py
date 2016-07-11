from __future__ import absolute_import

import networkx as nx
import numpy as np
import scipy as sp, scipy.sparse

from .array import densify, spdiag

def PTDF(G, susceptance='Y', nodelist=None):
    if nodelist is None:
        nodelist = G.nodes()
    lap = densify(nx.laplacian_matrix(nx.Graph(G), nodelist=nodelist,
                                      weight=susceptance)) # shallow undirected copy
    K = nx.incidence_matrix(G, nodelist=nodelist, weight=susceptance, oriented=True)
    return np.asarray(- K.T.dot(np.linalg.pinv(lap)))

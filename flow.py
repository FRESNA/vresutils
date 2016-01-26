from __future__ import absolute_import

import networkx as nx
import numpy as np
import scipy as sp, scipy.sparse

from .array import densify, spdiag

def PTDF(G, susceptance='Y'):
    lap = densify(nx.laplacian_matrix(nx.Graph(G), nodelist=G.nodes(),
                                      weight=susceptance)) # shallow undirected copy
    K = nx.incidence_matrix(G, weight=susceptance, oriented=True)
    return np.asarray(- K.T.dot(np.linalg.pinv(lap)))

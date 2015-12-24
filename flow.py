from __future__ import absolute_import

import networkx as nx
import numpy as np
import scipy as sp, scipy.sparse

from .array import densify

def PTDF(G, susceptance='Y'):
    lap = densify(nx.laplacian_matrix(nx.Graph(G), nodelist=G.nodes(),
                                      weight=susceptance)) # shallow undirected copy
    K = nx.incidence_matrix(G, oriented=True)
    if susceptance is not None:
        Y = np.fromiter((d[susceptance] for i,o,d in G.edges_iter(data=True)),
                        dtype=np.float, count=G.number_of_edges())
        K = K.dot(spdiag(Y) if sp.sparse.isspmatrix(K) else np.diag(Y))

    return np.asarray(- K.T.dot(np.linalg.pinv(lap)))

# -*- coding: utf-8 -*-

## Copyright 2015-2017 Frankfurt Institute for Advanced Studies

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
"""

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

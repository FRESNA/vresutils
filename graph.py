#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from itertools import izip
from scipy.spatial import Voronoi

def to_directed(G):
    """ Returns directed version of graph G, with randomly assigned directions.
    """

    G2 = G.to_directed()

    # remove symmetric edges
    for n, nbrdict in G2.adjacency_iter():
        for n2 in nbrdict.iterkeys():
            if G2.has_edge(n2, n):
                G2.remove_edge(n2, n)

    return G2

def subgraph(G, lat, lon, size):
    """ Returns cutout of G with node positions inside a rectangle described by 
    lat, lon with tolerance (e.g. grid cell width) size.
    """

    g = G.copy()
    pos = np.array((lon, lat))
    g.remove_nodes_from(n
                        for n, p in nx.get_node_attributes(G, 'pos').iteritems()
                        if np.abs(p - pos).max() > size/2)
    return next(nx.connected_component_subgraphs(g))

def BreadthFirstLevels(G, root):
    """ Generator of sets of 1-neighbours, 2-neighbours, ... k-neighbours
    of node `root` in topology of `G`.
    """

    visited = set()
    currentLevel = set((root,))
    while currentLevel:
        yield currentLevel
        visited.update(currentLevel)
        currentLevel = set(w
                           for v in currentLevel
                           for w in G[v]
                           if w not in visited)

def NodeListBFS(G, root, depth=10):
    """ Return array of 1, 2, ..., <depth>th neighbours of root node <root> 
    in graph <G>.
    """

    levels = []
    nodes = []
    if G.nodes() != range(G.number_of_nodes()):
        root = G.nodes().index(root)
        G = nx.convert_node_labels_to_integers(G)

    for level, nodesoflevel in izip(xrange(depth), BreadthFirstLevels(G, root)):
        levels += [level] * len(nodesoflevel)
        nodes  += nodesoflevel

    return np.array(levels), np.array(nodes)

def voronoi_partition(G):
    """ For 2D-embedded graph <G>, returns the shapes of the Voronoi cells 
    corresponding to each node.
    """

    if G.nodes() != range(G.number_of_nodes()):
        G = nx.convert_node_labels_to_integers(G)

    pointdict = nx.get_node_attributes(G, 'pos')
    points = [pointdict[i] for i in range(G.number_of_nodes())]
    vor = Voronoi(points)

    return vor, G

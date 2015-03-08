#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from itertools import izip

def to_directed(G):
    G2 = G.to_directed()

    # remove symmetric edges
    for n, nbrdict in G2.adjacency_iter():
        for n2 in nbrdict.iterkeys():
            if G2.has_edge(n2, n):
                G2.remove_edge(n2, n)

    return G2

def subgraph(G, lat, lon, size):
    g = G.copy()
    pos = np.array((lon, lat))
    g.remove_nodes_from(n
                        for n, p in nx.get_node_attributes(G, 'pos').iteritems()
                        if np.abs(p - pos).max() > size/2)
    return next(nx.connected_component_subgraphs(g))

def BreadthFirstLevels(G, root):
    """Generator of sets of 1-neighbours, 2-neighbours, ... k-neighbours
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
    levels = []
    nodes = []
    if G.nodes() != range(G.number_of_nodes()):
        root = G.nodes().index(root)
        G = nx.convert_node_labels_to_integers(G)

    for level, nodesoflevel in izip(xrange(depth), BreadthFirstLevels(G, root)):
        levels += [level] * len(nodesoflevel)
        nodes  += nodesoflevel

    return np.array(levels), np.array(nodes)

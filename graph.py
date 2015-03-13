#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from itertools import izip
from shapely.geometry import Point, Polygon
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

def largest_connected_component(G, copy=True):
    g = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    if copy:
        return g.copy()
    else:
        return g

def cell_subgraph(G, lat, lon, size):
    """ Returns cutout of G with node positions around a point described by 
    lat, lon with tolerance (e.g. grid cell width) size.
    """

    g = G.copy()
    pos = np.array((lon, lat))
    g.remove_nodes_from(n
                        for n, p in nx.get_node_attributes(G, 'pos').iteritems()
                        if np.abs(p - pos).max() > size/2)
    return largest_connected_component(g)

def polygon_subgraph(G, polygon, nneighbours=0):
    """ Cut out portion of graph <G>'s nodes contained in shapely.geometry.Polygon <polygon>
    and their <nneighbours>th neighbours (which are partly outside of polygon).
    """
    
    # if that's not done here, the BFS will do it later and mess the node labels up
    if G.nodes() != range(G.number_of_nodes()):
        G = nx.convert_node_labels_to_integers(G)

    g = G.copy()
    # core graph inside polygon
    g.remove_nodes_from(n
                        for n, p in nx.get_node_attributes(G, 'pos').iteritems()
                        if not polygon.contains(Point(p)) )

    # core graph + nneighbours neighbours
    nodelist = g.nodes()
    for n in g.nodes():
        levels, neighbours = NodeListBFS(G, n, depth=nneighbours)
        nodelist.extend(neighbours)

    nodelist = set(nodelist)

    g = G.copy()
    g.remove_nodes_from(n
                        for n in G.nodes()
                        if not n in nodelist)

    return largest_connected_component(g)

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
    corresponding to each node. Strips the Graph off nodes that are not interior
    to any Voronoi cell and returns the remainder of <G> with the Voronoi cell 
    region as an additional node attribute.
    """

    if G.nodes() != range(G.number_of_nodes()):
        G = nx.convert_node_labels_to_integers(G)

    pointdict = nx.get_node_attributes(G, 'pos')
    points = [pointdict[i] for i in range(G.number_of_nodes())]
    vor = Voronoi(points)

    # convert Voronoi output to Polygons as additional node attribute
    # exclude the nodes which are not interior to any Voronoi region
    G.remove_nodes_from(n 
                        for n in G.nodes()
                        if vor.point_region[n] == -1)

    for n in G.nodes():
        region = Polygon(vor.vertices[vor.regions[vor.point_region[n]]])
        G.node[n]['region'] = region

    return G

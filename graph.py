#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from itertools import izip, islice
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
from scipy.spatial import distance
from scipy import sparse

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

def giant_component(G, copy=True):
    g = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    if copy:
        return g.copy()
    else:
        return g

def minimum_spanning_tree(G):
    """
    Given a spatially embedded graph `G`, find the minimum spanning
    tree using the node distances as link weights. Returns the minimum
    spanning tree graph g with node distances as link weights. If
    returndist=True, also returns distance matrix (as dense array)
    between all nodes.
    """

    if G.nodes() != range(G.number_of_nodes()):
        G = nx.convert_node_labels_to_integers(G)

    pos = nx.get_node_attribute(G, 'pos')
    pos = [pos[i] for i in range(G.number_of_nodes())]

    # calculate distances
    # TODO these matrix formats are probably inefficient
    distance_matrix = distance.pdist(pos, metric='euclidean')
    distance_matrix = distance.squareform(distance_matrix)

    # calculate minimum spanning tree
    span_tree = sparse.csgraph(distance_matrix, overwrite=True)

    # translate back to graph edges and add them to g, store distance in edge weight
    span_tree = sparse.coo_matrix(span_tree)

    g = nx.Graph()
    g.add_nodes_from(G.nodes())
    g.add_weighted_edges_from((n, m, weight)
                              for n, m, weight in zip(span_tree.row, span_tree.col, span_tree.data))

    return g

def cell_subgraph(G, lat, lon, size, copy=True):
    """
    Returns cutout of G with node positions around a point described
    by lat, lon with tolerance (e.g. grid cell width) size.
    """

    pos = np.array((lon, lat))
    nodes = (n
             for n, p in nx.get_node_attributes(G, 'pos').iteritems()
             if np.abs(p - pos).max() <= size/2)
    return giant_component(G.subgraph(nodes), copy=copy)

def polygon_subgraph(G, polygon, nneighbours=0, copy=True):
    """
    Cut out portion of graph `G`s nodes contained in
    shapely.geometry.Polygon `polygon` and their `nneighbours`th
    neighbours (which are partly outside of polygon).
    """

    nodes = set(n
                for n, p in nx.get_node_attributes(G, 'pos').iteritems()
                if polygon.contains(Point(p)))

    if nneighbours > 0:
        # extend by nneighbours layers of neighbours
        nodes.update(reduce(set.union, islice(BreadthFirstLevels(G, nodes),
                                              1, 1+nneighbours)))

    return giant_component(G.subgraph(nodes), copy=copy)

def BreadthFirstLevels(G, root):
    """
    Generator of sets of 1-neighbours, 2-neighbours, ... k-neighbours
    of node or list of nodes `root` in topology of `G`.
    """

    visited = set()
    if type(root) in (set, list):
        currentLevel = set(root)
    else:
        currentLevel = set((root,))
    while currentLevel:
        yield currentLevel
        visited.update(currentLevel)
        currentLevel = set(w
                           for v in currentLevel
                           for w in G[v]
                           if w not in visited)

def voronoi_partition(G):
    """
    For 2D-embedded graph `G`, returns the shapes of the Voronoi cells
    corresponding to each node. Strips the Graph off nodes that are not interior
    to any Voronoi cell and returns the remainder of `G` with the Voronoi cell
    region as an additional node attribute.
    """

    points = nx.get_node_attributes(G, 'pos').values()
    vor = Voronoi(points)

    # convert Voronoi output to Polygons as additional node attribute
    # exclude the nodes which are not interior to any Voronoi region
    for i,n in enumerate(G.nodes()):
        if vor.point_region[i] == -1:
            G.remove_node(n)
        else:
            region = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])
            G.node[n]['region'] = region

    return G

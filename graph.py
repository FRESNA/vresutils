#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from itertools import izip, islice, chain
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
from scipy.spatial import distance
from scipy import sparse
from collections import OrderedDict
from scipy.linalg import norm

def to_directed(G):
    """
    Returns directed version of graph G, with randomly assigned directions.
    """

    G2 = G.to_directed()

    # remove symmetric edges
    for n, nbrdict in G2.adjacency_iter():
        for n2 in nbrdict.iterkeys():
            if G2.has_edge(n2, n):
                G2.remove_edge(n2, n)

    return G2


def giant_component(G, copy=True):
    H = G.subgraph(max(nx.connected_components(G), key=len))
    if copy:
        return H.copy()
    else:
        return H


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

def get_distance_matrix(G):
    """ Given a spatially embedded graph <G>, get the node positions and
    calculate the pairwise euclidean distance of all nodes.
    """

    if G.nodes() != range(G.number_of_nodes()):
        G = nx.convert_node_labels_to_integers(G)

    pos = nx.get_node_attributes(G, 'pos')
    pos = [pos[i]
           for i in range(G.number_of_nodes())]
    pos = np.array(pos)

    # TODO these matrix formats are probably inefficient
    # (but at least understood across different packages!)
    distance_matrix = distance.pdist(pos, metric='euclidean')
    distance_matrix = distance.squareform(distance_matrix)

    distance_matrix[np.diag_indices_from(distance_matrix)] = 1.e-12

    return distance_matrix


def get_hop_distance(G):
    """ Given a graph <G>, find the hop distance between all pairs of
    nodes and return it as a matrix. Will only work properly if matrix
    nodes are labelled with integer range, otherwise, the matrix entries
    end up in random places.
    """

    if G.nodes() != range(G.number_of_nodes()):
        G = nx.convert_node_labels_to_integers(G)

    Nnodes = G.number_of_nodes()
    hop_distance = np.zeros((Nnodes, Nnodes))

    hop_distance_dict = nx.shortest_path_length(G, weight=None)
    for key1, inner_dict in hop_distance_dict.iteritems():
        for key2, dist in inner_dict.iteritems():
            hop_distance[key1,key2] = dist
            hop_distance[key2,key1] = dist

    return hop_distance


def minimum_spanning_tree(G, distance_matrix=None):
    """ Given a graph <G> and some underlying metric encoded in
    <distance_matrix> (defaults to euclidean distance, which requires nodes
    to have a position), find the minimum spanning tree using the node distances
    as link weights. Returns the minimum spanning tree graph g with node
    distances as link weights. If returndist=True, also returns distance matrix
    (as dense array) between all nodes.
    """

    if G.nodes() != range(G.number_of_nodes()):
        G = nx.convert_node_labels_to_integers(G)

    # calculate distances
    if distance_matrix == None:
        distance_matrix = get_distance_matrix(G)

    # calculate minimum spanning tree
    span_tree = sparse.csgraph.minimum_spanning_tree(distance_matrix, overwrite=True)

    # translate back to graph edges and add them to H, store distance in edge weight
    span_tree = sparse.coo_matrix(span_tree)

    H = nx.Graph()

    H.add_nodes_from(sorted(G.nodes(data=True), key=str))
    H.add_weighted_edges_from((n, m, weight)
                              for n, m, weight in zip(span_tree.row, span_tree.col, span_tree.data))

    return H


def find_N_minus_one_critical_links(G, edges=None):
    """ Given a graph <G>, find the critical links whose removal leads to
    network decomposition into multiple components.
    Optionally, restrict the search to edge list <edges>.

    Brute force for now, better ideas welcome.
    """

    if nx.number_connected_components(G) > 1:
        return G.edges()

    if edges == None:
        edges = G.edges()

    critical_links = []
    for edge in edges:
        G.remove_edge(*edge)
        if nx.number_connected_components(G) > 1:
            critical_links.append(edge)
        G.add_edge(*edge)

    return critical_links


def cell_subgraph(G, lat, lon, size, copy=True):
    """
    Returns cutout of `G` with node positions around a point described
    by `lat`, `lon` with tolerance (e.g. grid cell width) `size`.
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


def polygon_subgraph_environment(G, polygon, environment_polygons):
    """
    Return the subgraph induced on nodes within shapely Polygon. Nodes
    covered by environment_polygons will be consolidated to one node
    each and combine all the previous links to its bunch. The node
    attributes length, X and Y are filled in again. All the nodes,
    which have been subsumed into one supernode are accessible through
    its node attribute 'nodes'.

    Parameters
    ----------
    G : Graph
        A spatially embedded graph with the node attribute pos and
        edge attributes length and X.

    polygon : shapely.geometry.Polygon
        The region from where to keep nodes

    environment_polygons : dict(region_name=Polygon)
        For each Polygon all nodes within are collectively replaced by
        one node, which unifies also all the previous links.

    Returns
    -------
    G : Graph
    """

    H = G.__class__()
    queue = OrderedDict()

    def add_link(n, m, d):
        if n != m:
            H.adj[n][m] = d
            H.adj[m][n] = d

    def add_node(n):
        if n in queue:
            return queue[n]

        attr_dict = G.node[n]
        pos = Point(attr_dict['pos'])
        if polygon.contains(pos):
            # in Polygon: keep it
            H.adj[n] = {}
            H.node[n] = attr_dict
            return n
        else:
            # this two-step procedure checks the last successful
            # polygon (env_poly) first, as we are link-hopping
            # through the graph
            if not add_node.env_poly.contains(pos):
                for name, p in environment_polygons.iteritems():
                    if p.contains(pos):
                        add_node.env_name = name
                        add_node.env_poly = p
                        break
                else:
                    # The point is not in any polygon => throw it
                    return None
            # n is in env_poly
            if add_node.env_name not in H:
                H.adj[add_node.env_name] = {}
                H.node[add_node.env_name] = {'pos': np.asarray(add_node.env_poly.centroid),
                                             'nodes': {n: attr_dict}}
            else:
                H.node[add_node.env_name]['nodes'][n] = attr_dict
            return add_node.env_name
    add_node.env_name, add_node.env_poly = next(environment_polygons.iteritems())

    # nodes are added to done, as soon as all its adjoining neighbours
    # and links have been added.
    done = set()
    for n in G:
        if n in done: continue

        a = add_node(n)
        if a is None: continue
        queue[n] = a

        while queue:
            n, a = queue.popitem()
            for m, d in G.adj[n].iteritems():
                if m in done: continue

                b = add_node(m)
                if b is None: continue
                add_link(a, b, d)
                queue[m] = b
            done.add(n)

    for env in environment_polygons:
        # estimate X / length ratio
        x = np.mean([m['X'] / m['length']
                     for m in chain(H.adj[env].itervalues(),
                                    (a for n in H.node[env]['nodes']
                                     for a in G.adj[n].itervalues()))])
        pos = H.node[env]['pos']
        for n, attr in H.adj[env].iteritems():
            length = norm(pos - H.node[n]['pos'])
            X = x * length
            attr.update(length=length, X=X, Y=1./X)

    # graph attributes
    H.graph = G.graph
    return H

def get_voronoi_regions(G, outline=None):
    if 'region' not in next(G.node.itervalues()):
        if callable(outline):
            outline = outline()
        assert outline is not None
        voronoi_partition(G, Polygon(outline))
    return nx.get_node_attributes(G, 'region').values()

def voronoi_partition(G, outline):
    """
    For 2D-embedded graph `G`, within the boundary given by the shapely polygon
    `outline`, returns `G` with the Voronoi cell region as an additional node
    attribute.
    """

    G = polygon_subgraph(G, outline, copy=False)

    # this loop is necessary to get the points into the right order to match
    # the nodes with the correct Voronoi regions later on
    points = [n[1]['pos'] for n in G.nodes(data=True)]

    # to avoid any network positions outside all Voronoi cells, append
    # the corners of a rectangle framing these points
    xmin, xmax = np.amin(np.array(points)[:,0]), np.amax(np.array(points)[:,0])
    xspan = xmax-xmin

    ymin, ymax = np.amin(np.array(points)[:,1]), np.amax(np.array(points)[:,1])
    yspan = ymax-ymin

    points.extend([[xmin-3.*xspan, ymin-3.*yspan],
                   [xmin-3.*xspan, ymax+3.*yspan],
                   [xmax+3.*xspan, ymin-3.*yspan],
                   [xmax+3.*xspan, ymax+3.*yspan]])

    vor = Voronoi(points)

    # convert Voronoi output to Polygons as additional node attribute
    for i,n in enumerate(G.nodes()):

        region = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

        if not region.is_valid:
            region = region.buffer(0)

        region = region.intersection(outline)

        try:
            polygons = region.geoms # if that works, we have a MultiPolygon
            # pick the part with the largest area
            region = max(polygons, key=lambda pg: pg.area)
        except:
            pass

        G.node[n]['region'] = region

    return G

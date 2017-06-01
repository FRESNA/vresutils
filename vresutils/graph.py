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

import numpy as np
import networkx as nx
import warnings
from distutils.version import StrictVersion, LooseVersion
from itertools import islice, chain, count
from operator import itemgetter
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
from scipy.spatial import distance
from scipy import sparse
from collections import OrderedDict
from scipy.linalg import norm
from six import iterkeys, iteritems, itervalues
from six.moves import map, range, zip
from functools import reduce

def to_directed(G):
    """
    Returns directed version of graph G, with randomly assigned directions.
    """

    G2 = G.to_directed()

    # remove symmetric edges
    for n, nbrdict in G2.adjacency_iter():
        for n2 in iterkeys(nbrdict):
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

    if G.nodes() != list(range(G.number_of_nodes())):
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

    if G.nodes() != list(range(G.number_of_nodes())):
        G = nx.convert_node_labels_to_integers(G)

    Nnodes = G.number_of_nodes()
    hop_distance = np.zeros((Nnodes, Nnodes))

    hop_distance_dict = nx.shortest_path_length(G, weight=None)
    for key1, inner_dict in iteritems(hop_distance_dict):
        for key2, dist in iteritems(inner_dict):
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

    if G.nodes() != list(range(G.number_of_nodes())):
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
             for n, p in iteritems(G.node)
             if np.abs(p['pos'] - pos).max() <= size/2)
    return giant_component(G.subgraph(nodes), copy=copy)


def polygon_subgraph(G, polygon, nneighbours=0, copy=True):
    """
    Cut out portion of graph `G`s nodes contained in
    shapely.geometry.Polygon `polygon` and their `nneighbours`th
    neighbours (which are partly outside of polygon).
    """

    nodes = set(n
                for n, p in iteritems(G.node)
                if polygon.contains(Point(p['pos'])))

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
                for name, p in iteritems(environment_polygons):
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
    add_node.env_name, add_node.env_poly = next(iteritems(environment_polygons))

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
            for m, d in iteritems(G.adj[n]):
                if m in done: continue

                b = add_node(m)
                if b is None: continue
                add_link(a, b, d)
                queue[m] = b
            done.add(n)

    for env in environment_polygons:
        # estimate X / length ratio
        x = np.mean([m['X'] / m['length']
                     for m in chain(itervalues(H.adj[env]),
                                    (a for n in H.node[env]['nodes']
                                     for a in itervalues(G.adj[n])))])
        pos = H.node[env]['pos']
        for n, attr in iteritems(H.adj[env]):
            length = norm(pos - H.node[n]['pos'])
            X = x * length
            attr.update(length=length, X=X, Y=1./X)

    # graph attributes
    H.graph = G.graph
    return H

def coarsify_graph(G, shapes, lost_nodes=None):
    """
    Generate a graph with a node for each shape and edges that agree
    with the finer grid `G`.

    Parameters
    ----------
    G : nx.Graph
        A spatially embedded finely-grained graph with the node
        attribute pos.

    shapes : OrderedDict of (name, shapely.Polygon) pairs

    Returns
    -------
    H : OrderedGraph
    """

    H = OrderedGraph()
    H.add_nodes_from((n, dict(region=sh, pos=np.asarray(sh.centroid)))
                     for n, sh in iteritems(shapes))

    queue = OrderedDict()

    def add_link(n, m, capacity=None):
        if n != m:
            H.add_edge(n, m)
            attrs = H.adj[n][m]
            attrs['lines'] = attrs.get('lines', 0) + 1
            if capacity is not None:
                attrs['capacity'] = attrs.get('capacity', 0) + capacity

    def do_node(n):
        if n in queue:
            return queue[n]

        pos = Point(G.node[n]['pos'])
        # this two-step procedure checks the last successful shape
        # first, as we are link-hopping through the graph
        if do_node.shape.contains(pos):
            return do_node.node
        else:
            for node, shape in iteritems(shapes):
                if shape.contains(pos):
                    do_node.node = node
                    do_node.shape = shape
                    return node
            else:
                # The point is not in any polygon => we assume that's
                # a problem of not matching shapefiles and just claim
                # its still in the same shape. This might drop links
                # if a node with more than two links falls through
                if len(G.adj[n]) > 2:
                    warnings.warn("The algorithm had to skip over at least one node with more than two links lying outside of any shape. Consider checking the damage done by examining the status of these lost nodes by supplying an extra list to the lost_nodes argument of this function.")
                    if isinstance(lost_nodes, list):
                        lost_nodes.append(n)
                return do_node.node
    do_node.node, do_node.shape = next(iteritems(shapes))

    # nodes are added to done, as soon as all its adjoining neighbours
    # and links have been added.
    done = set()
    for n in G:
        if n in done: continue

        queue[n] = do_node(n)

        while queue:
            n, a = queue.popitem()
            for m, d in iteritems(G.adj[n]):
                if m in done: continue
                b = do_node(m)
                add_link(a, b, d.get('capacity', None))
                queue[m] = b
            done.add(n)

    from .grid import penalize
    for n1, n2, d in H.edges_iter(data=True):
        if 'lines' in d and 'capacity' in d:
            d['capacity'] = penalize(d['capacity'], d['lines'])

    return H

def stitch_graphs(G, G2, nodes, region=None):
    """
    Stitch a subgraph of the finer `G` into the coarse `G2` graph by
    replacing the node `label`.

    Parameters
    ----------
    G : Graph
        A spatially embedded finely-grained graph with the node
        attribute pos.

    G2 : Graph
        Spatially embedded coarse-grained graph with the node
        attribute region holding the shapely polygons for each node.

    nodes : Node labels
        Labels of the `G2` graph, where `G` should be stitched into.

    region : shapely geometry
        The geographic region on which `G` should be stitched into `G2`.
        Defaults to the union of regions of `nodes` in `G2`.

    Returns
    -------
    H : Graph
    """

    regions = nx.get_node_attributes(G2, 'region')

    if region is None:
        from shapely.ops import cascaded_union
        region = cascaded_union([regions[n] for n in nodes])

    H = polygon_subgraph(G, region, copy=False)
    neigh_nodes = reduce(set.union, islice(BreadthFirstLevels(G, H.nodes()), 1, 2))

    assert len(set(H.nodes()).intersection(G2.nodes())) == 0, \
        "The node labels between G and G2 may not overlap"

    H.add_nodes_from(G2.nodes_iter(data=True))
    H.add_edges_from(G2.edges_iter(data=True))
    H.remove_nodes_from(nodes)

    from .grid import node_distance, specific_susceptance
    def edge_attrs(H, n1, n2, d):
        a = d.copy()
        a['length'] = node_distance(H, n1, n2)
        a['X'] = specific_susceptance * a['length']
        a['Y'] = 1./a['X']
        return a

    for n in neigh_nodes:
        pos = Point(G.node[n]['pos'])
        for n2, reg in iteritems(regions):
            if reg.contains(pos):
                H.add_edges_from((n2, n3, edge_attrs(H, n2, n3, d))
                                 for n3, d in iteritems(G.adj[n])
                                 if n3 in H)

    return H

def get_voronoi_regions(G, outline=None):
    if 'region' not in next(itervalues(G.node)):
        if callable(outline):
            outline = outline()
        assert outline is not None
        G = voronoi_partition(G, Polygon(outline))
    return list(get_node_attributes(G, 'region').values())

def voronoi_partition_pts(points, outline, no_multipolygons=False):
    """
    Compute the polygons of a voronoi partition of `points` within the
    polygon `outline`

    Attributes
    ----------
    points : Nx2 - ndarray[dtype=float]
    outline : Polygon
    no_multipolygons : bool (default: False)
        If true, replace each MultiPolygon by its largest component

    Returns
    -------
    polygons : N - ndarray[dtype=Polygon|MultiPolygon]
    """

    points = np.asarray(points)

    if len(points) == 1:
        polygons = [outline]
    else:
        xmin, ymin = np.amin(points, axis=0)
        xmax, ymax = np.amax(points, axis=0)
        xspan = xmax - xmin
        yspan = ymax - ymin

        # to avoid any network positions outside all Voronoi cells, append
        # the corners of a rectangle framing these points
        vor = Voronoi(np.vstack((points,
                                 [[xmin-3.*xspan, ymin-3.*yspan],
                                  [xmin-3.*xspan, ymax+3.*yspan],
                                  [xmax+3.*xspan, ymin-3.*yspan],
                                  [xmax+3.*xspan, ymax+3.*yspan]])))

        polygons = []
        for i in range(len(points)):
            poly = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

            if not poly.is_valid:
                poly = poly.buffer(0)

            poly = poly.intersection(outline)

            polygons.append(poly)

    if no_multipolygons:
        def demultipolygon(poly):
            try:
                # for a MultiPolygon pick the part with the largest area
                poly = max(poly.geoms, key=lambda pg: pg.area)
            except:
                pass
            return poly
        polygons = [demultipolygon(poly) for poly in polygons]

    polygons_arr = np.empty((len(polygons),), 'object')
    polygons_arr[:] = polygons
    return polygons_arr

def voronoi_partition(G, outline):
    """
    For 2D-embedded graph `G`, within the boundary given by the shapely polygon
    `outline`, returns `G` with the Voronoi cell region as an additional node
    attribute.
    """

    G = polygon_subgraph(G, outline, copy=False)
    points = list(get_node_attributes(G, 'pos').values())
    regions = voronoi_partition_pts(points, outline, no_multipolygons=True)
    nx.set_node_attributes(G, 'region', dict(zip(G.nodes(), regions)))

    return G

def derive_edgemap(G, nodemap, shapes=None):
    import pandas as pd
    if shapes is None:
        return pd.Series({(n1,n2): nodemap[n1]
                          for n1, n2 in G.edges()
                          if nodemap[n1] == nodemap[n2]})
    else:
        from shapely.geometry import LineString

        def edge_to_shape(e):
            n1, n2 = e
            nm1 = nodemap.get(n1)
            nm2 = nodemap.get(n2)
            if nm1 == nm2:
                return nm1
            else:
                ls = LineString([G.node[n1]['pos'], G.node[n2]['pos']])
                def length(nm):
                    return ls.intersection(shapes[nm]).length if nm is not None else 0
                l1 = length(nm1)
                l2 = length(nm2)
                if l1 >= l2 and l1 > 0:
                    return nm1
                elif l2 > l1 and l2 > 0:
                    return nm2
                else:
                    return np.nan
        edges = G.edges()
        return pd.Series(list(map(edge_to_shape, edges)), index=pd.MultiIndex.from_tuples(edges))

try:
    _nx_version = StrictVersion(nx.__version__)
except ValueError:
    _nx_version = LooseVersion(nx.__version__)

if _nx_version >= '1.12':
    class OrderedGraph(nx.Graph):
        node_dict_factory = OrderedDict
        adjlist_dict_factory = OrderedDict
elif _nx_version >= '1.10':
    class OrderedGraph(nx.Graph):
        node_dict_factory = OrderedDict
        adjlist_dict_factory = OrderedDict

        def __init__(self, data=None, **attr):
            self.node_dict_factory = ndf = self.node_dict_factory
            self.adjlist_dict_factory = self.adjlist_dict_factory
            self.edge_attr_dict_factory = self.edge_attr_dict_factory

            self.graph = {}   # dictionary for graph attributes
            self.node = ndf()  # empty node attribute dict
            self.adj = ndf()  # empty adjacency dict
            # attempt to load graph with data
            if data is not None:
                if isinstance(data, OrderedGraph):
                    try:
                        nx.convert.from_dict_of_dicts(
                            data.adj,
                            create_using=self,
                            multigraph_input=data.is_multigraph()
                        )
                        self.graph = data.graph.copy()
                        self.node.update((n,d.copy()) for n,d in data.node.items())
                    except:
                        raise nx.NetworkXError("Input is not a correct NetworkX graph.")
                else:
                    nx.convert.to_networkx_graph(data, create_using=self)
else:
    class OrderedGraph(nx.Graph):
        """
        This OrderedGraph is intended to be the simplest NetworkX-
        compatible Graph, which preserves node and edge order. The
        functions have been taken from the NX 1.9.1 code only replacing
        all instances where the node or any of the stacked adj
        dictionaries are created by an instance of OrderedDict.

        There are no guarantees everything will indeed be preserved, but
        it is much more likely to work than with the current normal
        nx.Graph. :)
        """
        def __init__(self, data=None, **attr):
            self.graph = {}   # dictionary for graph attributes
            self.node = OrderedDict()    # empty node dict (created before convert)
            self.adj = OrderedDict()     # empty adjacency dict
            # attempt to load graph with data
            if data is not None:
                if isinstance(data, OrderedGraph):
                    try:
                        nx.convert.from_dict_of_dicts(
                            data.adj,
                            create_using=self,
                            multigraph_input=data.is_multigraph()
                        )
                        self.graph = data.graph.copy()
                        self.node.update((n,d.copy()) for n,d in data.node.items())
                    except:
                        raise nx.NetworkXError("Input is not a correct NetworkX graph.")
                else:
                    nx.convert.to_networkx_graph(data, create_using=self)

            # load graph attributes (must be after convert)
            self.graph.update(attr)
            self.edge = self.adj

        def add_node(self, n, attr_dict=None, **attr):
            # set up attribute dict
            if attr_dict is None:
                attr_dict=attr
            else:
                try:
                    attr_dict.update(attr)
                except AttributeError:
                    raise nx.NetworkXError(\
                        "The attr_dict argument must be a dictionary.")
            if n not in self.node:
                self.adj[n] = OrderedDict()
                self.node[n] = attr_dict
            else: # update attr even if node already exists
                self.node[n].update(attr_dict)

        def add_nodes_from(self, nodes, **attr):
            for n in nodes:
                try:
                    newnode=n not in self.node
                except TypeError:
                    nn,ndict = n
                    if nn not in self.node:
                        self.adj[nn] = OrderedDict()
                        newdict = attr.copy()
                        newdict.update(ndict)
                        self.node[nn] = newdict
                    else:
                        olddict = self.node[nn]
                        olddict.update(attr)
                        olddict.update(ndict)
                    continue
                if newnode:
                    self.adj[n] = OrderedDict()
                    self.node[n] = attr.copy()
                else:
                    self.node[n].update(attr)

        def add_edge(self, u, v, attr_dict=None, **attr):
            # set up attribute dictionary
            if attr_dict is None:
                attr_dict=attr
            else:
                try:
                    attr_dict.update(attr)
                except AttributeError:
                    raise nx.NetworkXError(\
                        "The attr_dict argument must be a dictionary.")
            # add nodes
            if u not in self.node:
                self.adj[u] = OrderedDict()
                self.node[u] = {}
            if v not in self.node:
                self.adj[v] = OrderedDict()
                self.node[v] = {}
            # add the edge
            datadict=self.adj[u].get(v,{})
            datadict.update(attr_dict)
            self.adj[u][v] = datadict
            self.adj[v][u] = datadict

        def add_edges_from(self, ebunch, attr_dict=None, **attr):
            # set up attribute dict
            if attr_dict is None:
                attr_dict=attr
            else:
                try:
                    attr_dict.update(attr)
                except AttributeError:
                    raise nx.NetworkXError(\
                        "The attr_dict argument must be a dictionary.")
            # process ebunch
            for e in ebunch:
                ne=len(e)
                if ne==3:
                    u,v,dd = e
                elif ne==2:
                    u,v = e
                    dd = {}
                else:
                    raise nx.NetworkXError(\
                        "Edge tuple %s must be a 2-tuple or 3-tuple."%(e,))
                if u not in self.node:
                    self.adj[u] = OrderedDict()
                    self.node[u] = {}
                if v not in self.node:
                    self.adj[v] = OrderedDict()
                    self.node[v] = {}
                datadict=self.adj[u].get(v,{})
                datadict.update(attr_dict)
                datadict.update(dd)
                self.adj[u][v] = datadict
                self.adj[v][u] = datadict

        def subgraph(self, nbunch):
            bunch =self.nbunch_iter(nbunch)
            # create new graph and copy subgraph into it
            H = self.__class__()
            # copy node and attribute dictionaries
            for n in bunch:
                H.node[n]=self.node[n]
            # namespace shortcuts for speed
            H_adj=H.adj
            self_adj=self.adj
            # add nodes and edges (undirected method)
            for n in H.node:
                Hnbrs=OrderedDict()
                H_adj[n]=Hnbrs
                for nbr,d in self_adj[n].items():
                    if nbr in H_adj:
                        # add both representations of edge: n-nbr and nbr-n
                        Hnbrs[nbr]=d
                        H_adj[nbr][n]=d
            H.graph=self.graph
            return H

def set_node_positions_from_nodelabels(G):
    from . import shapes as vshapes

    nodes = G.nodes()
    if all(type(n) is str and len(n) == 2 for n in nodes):
        region = vshapes.countries(subset=nodes)
        pos = dict((n, np.array((p.centroid.x, p.centroid.y)))
                   for n, p in iteritems(region))
    else:
        region = dict()
        pos = nx.spring_layout(G)

    nx.set_node_attributes(G, 'region', region)
    nx.set_node_attributes(G, 'pos', pos)

def relabel_nodes(G, mapping):
    """
    Order preserving relabel_nodes for !disjunct! relabeling.
    """
    H = G.__class__()
    H.name = G.name
    H.add_nodes_from((mapping.get(n, n), d) for n, d in G.nodes(data=True))
    if G.is_multigraph():
        H.add_edges_from( (mapping.get(n1, n1),mapping.get(n2, n2),k,d)
                          for (n1,n2,k,d) in G.edges_iter(keys=True, data=True))
    else:
        H.add_edges_from( (mapping.get(n1, n1),mapping.get(n2, n2),d)
                          for (n1, n2, d) in G.edges_iter(data=True))
    H.graph.update(G.graph)

    return H

def convert_node_labels_to_integers(G):
    return relabel_nodes(G, dict(zip(G.nodes(), count())))

def get_node_attributes(G, attr):
    return OrderedDict((n, d[attr]) for n, d in iteritems(G.node))

def get_edge_attributes(G, attr):
    return OrderedDict(((u, v), d[attr]) for u, v, d in iteritems(G.edges_iter(data=True)))

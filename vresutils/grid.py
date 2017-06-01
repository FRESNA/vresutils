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

from operator import itemgetter
from six.moves import zip, map, cPickle as pickle
import six

from .graph import OrderedGraph

from . import make_toDataDir
toDataDir = make_toDataDir(__file__)

def penalize(x, n):
    """
    Thumb-rule for the aggregation of cross-border power lines.

    Parameters
    ---------
    x : float
        total line capacity
    n : int
        number of power lines

    Returns
    -------
    c : float
        resulting capacity
    """

    if n == 1:
        return x
    elif n == 2:
        return 5./6. * x
    elif n == 3:
        return 4./6. * x
    else:
        return .5 * x

##
# Functions which provide access to special network data
#

def bialek_data():
    import pandas as pd
    from operator import itemgetter

    buses, lines, coords = itemgetter('Bus Records', 'Line Records', 'DisplayBus') \
        (pd.read_excel(toDataDir('bialek.xlsx'),
                       sheetname=None, skiprows=1, header=0))

    buses.set_index('Number', inplace=True)
    coords.set_index('Number', inplace=True)
    coords.rename(columns={'X/Longitude Location': u'lon',
                           'Y/Latitude Location': u'lat'},
                  inplace=True)
    lines.set_index(['From Number', 'To Number'], inplace=True)

    # trafo from Xp, Yp to Xg, Yg according to diss of Qiang Zhou
    # (p. 139-140)
    XYpul = np.array((-500, 1500))
    XYplr = np.array((1500, -500))
    XYgul = np.array((-10.386207, 57.980912))
    XYglr = np.array((25.954773, 35.416979))
    lonlat = (coords.loc[:,['lon', 'lat']] - XYpul) * (XYglr - XYgul)/(XYplr - XYpul) + XYgul
    # lonlat = lonlat.assign(**{u'pos':list(np.asarray(lonlat))})
    buses = pd.concat((buses, lonlat), axis=1)

    return buses, lines

def bialek():
    G = OrderedGraph()
    buses, lines = bialek_data()
    nodes = buses.loc[:,['Name', 'Area Name', 'PU Volt']] \
                 .rename(columns={'Name': 'name', 'Area Name': 'area name',
                                  'PU Volt': 'voltage'}) \
                 .assign(pos=list(np.asarray(buses.loc[:,['lon','lat']])))
    links = lines.loc[:,['From Name', 'To Name', 'X', 'Lim A MVA', 'Circuit']] \
                 .rename(columns={'From Name': 'from name',
                                  'To Name': 'to name',
                                  'Lim A MVA': 'limit',
                                  'Circuit': 'circuit'}) \
                 .assign(Y=1./lines['X'])

    G.add_nodes_from(nodes.iterrows())
    G.add_edges_from((u,v,d) for (u,v),d in links.iterrows())

    return G

def entsoe_tue():
    with open(toDataDir("entsoe_2009_final.gpickle"), 'rb') as f:
        if six.PY2:
            G = pickle.load(f)
        else:
            G = pickle.load(f, encoding='latin-1')
    return OrderedGraph(G)

def entsoe_tue_linecaps(with_manual_link=True):
    G = entsoe_tue()

    # Add linecapacities by assuming:
    # - number of circuits is always 2
    # - 380kV if any of the connected nodes are 380kV and 220kV else
    # - 380kV lines have a capacity of 1500MW per circuit
    # - 220kV lines have a capacity of 500MW per circuit.

    voltages = nx.get_node_attributes(G, 'voltage')
    for n1, n2, attr in G.edges_iter(data=True):
        voltage = max(voltages.get(n1, 380), voltages.get(n2, 380))
        capacity = 2. * (1.5 if voltage == 380 else 0.5)
        attr.update(voltage=voltage, capacity=capacity)

    # Add missing link
    if with_manual_link:
        length = node_distance(G, '782', '788')
        X = specific_susceptance * length
        G.add_edge('788', '782',
                   capacity=3.0, X=X, Y=1/X,
                   length=length, limit=0.0, voltage=380)

    return G

# Given by bialek's network
# TODO : replace this by a well founded value from oeding or similar
# as soon as we can switch to a scigrid based network
specific_susceptance = 0.00068768296005101493  # mean of X / L

def node_distance(G, n1, n2):
    """
    A distance measure between two nodes in graph `G` which correlates
    well with what is already present in the length edge attribute in
    the bialek grid.

    Arguments
    ---------
    G : nx.Graph
    n1 : node label 1
    n2 : node label 2

    Returns
    -------
    d : float
        distance
    """
    return 110. * np.sqrt(np.sum((G.node[n1]['pos'] - G.node[n2]['pos'])**2))

def heuristically_extend_edge_attributes(G, it=None):
    if it is None:
        it = G.edges_iter(data=True)
    for n1, n2, d in it:
        d.setdefault('length', node_distance(G, n1, n2))
        d.setdefault('voltage', 380)
        d.setdefault('X', specific_susceptance * d['length'])
        d.setdefault('Y', 1./d['X'])

    return G

def eu():
    with open(toDataDir("EU.gpickle"), 'rb') as f:
        if six.PY2:
            return pickle.load(f)
        else:
            return pickle.load(f, encoding='latin-1')

def read_scigrid(nodes_csv="vertices_de_power_150601.csv",
                 links_csv="links_de_power_150601.csv"):
    """
    Read SCIGrid output csv files into a NX Graph.
    """
    import pandas as pd

    G = OrderedGraph()

    N = pd.read_csv(toDataDir(nodes_csv), delimiter=';', index_col=0)
    N['pos'] = [np.asarray(v) for k,v in N.ix[:,['lon', 'lat']].iterrows()]
    G.add_nodes_from(N.iterrows())

    L = pd.read_csv(toDataDir(links_csv), delimiter=';', index_col=0)
    # TODO: for some reason the SCIGrid data is missing impedance
    # values for a third of its lines, although L.x / L.length_m is a
    # constant for each voltage level. For now we just extend these to
    # cover the missing ones as well, but we should rather find out
    # the reason for the NaNs.
    L['X'] = (L.x / L.length_m).groupby(L.voltage).fillna(method='ffill') * L.length_m
    L['Y'] = 1./L['X']
    L['length'] = L['length_m']
    L['voltage'] /= 1000  # voltage in kV for readability
    G.add_edges_from(zip(L.v_id_1, L.v_id_2,
                         map(itemgetter(1),
                             L.loc[:,['voltage', 'cables', 'wires', 'frequency',
                                      'length', 'geom', 'r', 'x', 'c',
                                      'i_th_max', 'X', 'Y']].iterrows())))

    return G

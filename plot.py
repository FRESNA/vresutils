import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cbook as cb
from matplotlib.colors import colorConverter
from matplotlib.collections import LineCollection, PolyCollection

from . import shapes
from . import make_toModDir
toModDir = make_toModDir(__file__)

def set_style():
    mpl.style.use(toModDir('mplstyle'))

def germany2(with_laender=False, ax=None, linewidth=10, **kwargs):
    if ax is None:
        ax = plt.gca()

    if with_laender:
        laender = LineCollection(shapes.laender().itervalues(), colors="gray", zorder=0)
        ax.add_collection(laender)
    line, = plt.plot(*shapes.germany().T, color='k')
    line.set_zorder(1)

def landkreise(data, colorbar=True, cmap=None, ax=None):
    """
    Plot data on german Landkreis level. Needs a pandas Series with
    the corresponding regionalschluessel as index.

    Parameters
    ----------
    data : pd.Series
        Float valued data to be plotted.

    Returns
    -------
    collection : PolyCollection
    """

    if ax is None:
        ax = plt.gca()

    lk = shapes.Landkreise()
    coll = PolyCollection(lk.getPoints(i)
                          for i in lk.series().reindex(data.index))
    coll.set_array(data)
    coll.set_cmap(cmap)
    ax.add_collection(coll)
    ax.autoscale_view()

    if colorbar:
        plt.colorbar(mappable=coll, ax=ax)

    return coll

try:
    from mpl_toolkits.basemap import Basemap

    def germany(resolution='l', ax=None, meta=None):

        if meta is None:
            llcrnrlat = 47
            urcrnrlat = 56
            llcrnrlon = 5.5
            urcrnrlon = 15.5
        else:
            llcrnrlat = meta['latitudes'][-1,0]
            urcrnrlat = meta['latitudes'][0,-1]
            llcrnrlon = meta['longitudes'][-1,0]
            urcrnrlon = meta['longitudes'][0,-1]
        m = Basemap(projection='cyl', resolution=resolution,
                    llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                    llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                    ax=ax)
        m.drawcoastlines()
        m.drawcountries()
        return m
except ImportError:
    pass

def draw_edges(G, segments, pos=None, edgelist=None, width=1.0, color='k',
               style='solid', alpha=None, ax=None, **kwds):
    """Draw the edges of the graph G.

    This draws the edge segments given by a separation of the links in
    `data` of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    segments : L x M array
       The segmentation of each link. (segments.sum(axis=1) == 1).all()

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.
       (default=nx.get_node_attributes(G, 'pos'))

    edgelist : collection of edge tuples
       Draw only specified edges(default=G.edges())

    width : float or array of floats
       Line width of edges (default =1.0)

    color : tuple of color strings
       Edge Segments color. Can be a single color format string (default='r'),
       or a sequence of colors with the same length as data.shape[1].

    style : string
       Edge line style (default='solid') (solid|dashed|dotted,dashdot)

    alpha : float
       The edge transparency (default=1.0)

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    Returns
    -------
    matplotlib.collection.LineCollection
        `LineCollection` of the edge segments

    """
    if not np.allclose(segments.sum(axis=1), 1):
        segments = segments / segments.sum(axis=1)[:,np.newaxis]

    if ax is None:
        ax = plt.gca()

    if pos is None:
        pos = nx.get_node_attributes(G, 'pos')

    if edgelist is None:
        edgelist = G.edges()

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None

    if not cb.iterable(width):
        lw = (width,)
    else:
        lw = width

    if cb.iterable(color) \
           and len(color) == segments.shape[1]:
        if np.alltrue([cb.is_string_like(c) for c in color]):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([colorConverter.to_rgba(c, alpha) for c in color])
        elif (np.alltrue([not cb.is_string_like(c) for c in color])
              and np.alltrue([cb.iterable(c) and len(c) in (3, 4) for c in color])):
            edge_colors = tuple(color)
        else:
            raise ValueError('color must consist of either color names or numbers')
    else:
        if cb.is_string_like(color) or len(color) == 1:
            edge_colors = (colorConverter.to_rgba(edge_color, alpha), )
        else:
            raise ValueError('color must be a single color or list of exactly m colors where m is the number of segments')

    assert len(edgelist) == segments.shape[0], "Number edges and segments have to line up"

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    src = edge_pos[:,0]
    dest = edge_pos[:,1]

    positions = src[:,np.newaxis] + np.cumsum(np.hstack((np.zeros((len(segments), 1)), segments)), axis=1)[:,:,np.newaxis]*(dest - src)[:,np.newaxis]

    linecolls = []
    for s in range(segments.shape[1]):
        coll = LineCollection(positions[:,s:s+2],
                              colors=edge_colors[s:s+1],
                              linewidths=lw,
                              antialiaseds=(1,),
                              linestyle=style,
                              transOffset = ax.transData)

        coll.set_zorder(1)  # edges go behind nodes
        # coll.set_label(label)

        if cb.is_numlike(alpha):
            coll.set_alpha(alpha)

        ax.add_collection(coll)
        linecolls.append(coll)

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx-minx
    h = maxy-miny
    padx,  pady = 0.05*w, 0.05*h
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    return linecolls

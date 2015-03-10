import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

from . import make_toModDir
toModDir = make_toModDir(__file__)

def set_style():
    mpl.style.use(toModDir('mplstyle'))

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

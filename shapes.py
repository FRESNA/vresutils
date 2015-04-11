# -*- coding: utf-8 -*-

from pyproj import Proj
import shapefile
from shapely.geometry import Polygon
from itertools import izip, chain, count
from collections import OrderedDict
import numpy as np
import pandas as pd
import warnings

from vresutils import make_toModDir, Singleton, staticvars, cachable
toModDir = make_toModDir(__file__)

def simplify_poly(poly, tolerance):
    if tolerance is None:
        return poly
    else:
        return poly.simplify(tolerance)

def simplify_pts(pts, tolerance=0.03):
    return points(simplify_poly(Polygon(pts), tolerance))

def points(poly):
    return np.asarray(poly.boundary.coords)

class Dict(dict): pass

@cachable(keepweakref=True)
def germany(tolerance=0.03):
    return simplify_poly(Polygon(np.load(toModDir('data/germany.npy'))), tolerance)

def _shape2poly(sh, tolerance=0.03):
    pts = np.asarray(sh.points[:sh.parts[1] if len(sh.parts) > 1 else None])
    poly = Polygon(np.asarray(_shape2poly.p(*pts.T, inverse=True)).T)
    return simplify_poly(poly, tolerance)
_shape2poly.p = Proj('+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

@cachable(keepweakref=True, version=2)
def laender(tolerance=0.03, shortnames=True):
    if shortnames:
        name = {u'Baden-Württemberg': u'BW', u'Bayern': u'BY',
                u'Berlin': u'BE', u'Brandenburg': u'BB', u'Bremen':
                u'HB', u'Hamburg': u'HH', u'Hessen': u'HE',
                u'Mecklenburg-Vorpommern': u'MV', u'Niedersachsen':
                u'NI', u'Nordrhein-Westfalen': u'NW',
                u'Rheinland-Pfalz': u'RP', u'Saarland': u'SL',
                u'Sachsen': u'SN', u'Sachsen-Anhalt': u'ST',
                u'Schleswig-Holstein': u'SH', u'Thüringen':
                u'TH'}.__getitem__
    else:
        name = lambda x: x

    sf = shapefile.Reader(toModDir('data/vg250/VG250_LAN'))
    return OrderedDict(sorted([(name(rec[6].decode('utf-8')), _shape2poly(sh, tolerance))
                               for rec, sh in izip(sf.iterRecords(), sf.iterShapes())
                               if rec[1] == 4],
                              key=lambda x: x[0]))

@cachable(keepweakref=True)
def landkreise(tolerance=0.03):
    sf_kreise = shapefile.Reader(toModDir('data/vg250/VG250_KRS'))
    # for special casing hamburg and berlin
    sf_land = shapefile.Reader(toModDir('data/vg250/VG250_LAN'))

    fields = [f[0] for f in sf_kreise.fields[1:]]
    fields = {n:fields.index(n) for n in ('GF', 'RS')}

    kreise = ((int(sr[fields['RS']]),
               _shape2poly(sf_kreise.shape(ind), tolerance))
              for ind, sr in izip(count(), sf_kreise.iterRecords())
              if sr[fields['GF']] == 4)

    berlinhamburg = ((int(sr[fields['RS']]),
                      _shape2poly(sf_land.shape(ind), tolerance))
                     for ind, sr in izip(count(), sf_land.iterRecords())
                     if (sr[fields['RS']] in ('11', '02')
                         and sr[fields['GF']] == 4))

    return Dict(chain(berlinhamburg, kreise))

class Landkreise(Singleton):
    def __init__(self):
        warnings.warn("The Landkreise Singleton is deprecated. Use the landkreise function instead!", DeprecationWarning)

        self.sf_kreise = shapefile.Reader(toModDir('data/vg250/VG250_KRS'))
        # for special casing hamburg and berlin
        self.sf_land = shapefile.Reader(toModDir('data/vg250/VG250_LAN'))
        self.projection = Proj('+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

        fields = [f[0] for f in self.sf_kreise.fields[1:]]
        self.fields = {n:fields.index(n) for n in ('GF', 'RS')}

    def getShape(self, index):
        if index >= 0:
            return self.sf_kreise.shape(index)
        else:
            return self.sf_land.shape(-index-1)

    def getPoints(self, index, tolerance=0.03):
        return np.asarray(self.getPolygon(index, tolerance).boundary.coords)

    def getPolygon(self, index, tolerance=0.03):
        return Polygon(self._shape2points(self.getShape(index))).simplify_pts(tolerance)

    def _shape2points(self, sh):
        pts = np.array(sh.points[:sh.parts[1]] if len(sh.parts) > 1 else sh.points)
        return np.array(self.projection(*pts.T, inverse=True)).T

    def series(self):
        """
        Return pandas Series between shapes and indices.
        """
        return pd.Series(dict(iter(self)))

    def __iter__(self):
        """
        Iterate over the first shape of all kreis shapes and then
        berlin and hamburg. For each shape we return a tuple with
        (regionalschlüssel, index)
        """
        kreise = ((int(sr[self.fields['RS']]), ind)
                  for ind, sr in izip(count(), self.sf_kreise.iterRecords())
                  if sr[self.fields['GF']] == 4)

        inds = [ind for ind, rec in enumerate(self.sf_land.records())
                if rec[self.fields['RS']] in ('11', '02')
                and rec[self.fields['GF']] == 4]

        berlinhamburg = ((int(self.sf_land.record(ind)[self.fields['RS']]),
                          -ind-1)
                         for ind in inds)

        return chain(kreise, berlinhamburg)

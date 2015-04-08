# -*- coding: utf-8 -*-

from pyproj import Proj
import shapefile
from shapely.geometry import Polygon
from itertools import izip, chain, count
import numpy as np
import pandas as pd

from vresutils import make_toModDir, Singleton, staticvars
toModDir = make_toModDir(__file__)

def germany():
    return np.load(toModDir('data/germany.npy'))

def simplify(pts, tolerance=0.03):
    return np.asarray(maybe_simplify(Polygon(pts), tolerance).boundary.coords)

def points(poly):
    return np.asarray(poly.boundary.coords)

class Dict(dict): pass

@cachable(keepweakref=True)
def germany(tolerance=0.03):
    return maybe_simplify(Polygon(np.load(toModDir('data/germany.npy'))), tolerance)

def _shape2poly(sh, tolerance=0.03):
    pts = np.asarray(sh.points[:sh.parts[1] if len(sh.parts) > 1 else None])
    poly = Polygon(np.asarray(_shape2poly.p(*pts.T, inverse=True)).T)
    return maybe_simplify(poly, tolerance)
_shape2poly.p = Proj('+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

@cachable(keepweakref=True)
def laender(tolerance=0.03):
    sf = shapefile.Reader(toModDir('data/vg250/VG250_LAN'))
    p = Proj('+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    def _shape2points(sh):
        return np.array(p(*np.array(sh.points[:sh.parts[1] if len(sh.parts) > 1 else None]).T,
                          inverse=True)).T
    return dict((rec[6].decode('utf-8'), _shape2points(sh))
                for rec, sh in izip(sf.iterRecords(), sf.iterShapes())
                if rec[1] == 4)

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

    def getPoints(self, index):
        return self._shape2points(self.getShape(index))

    def getPolygon(self, index):
        return Polygon(self.getPoints(index))

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

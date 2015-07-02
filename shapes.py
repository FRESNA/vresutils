# -*- coding: utf-8 -*-

from pyproj import Proj
import shapefile
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from itertools import izip, chain, count, imap
from operator import itemgetter, attrgetter
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
        return poly.simplify(tolerance, preserve_topology=True)

def simplify_pts(pts, tolerance=0.03):
    return points(simplify_poly(Polygon(pts), tolerance))

def points(poly):
    return np.asarray(poly.boundary.coords)

class Dict(dict): pass

@cachable(keepweakref=True)
def germany(tolerance=0.03):
    return simplify_poly(Polygon(np.load(toModDir('data/germany.npy'))), tolerance)

def _shape2poly_wgs(sh, tolerance=0.03, minarea=0.03):
    def invproj(pts):
        return np.asarray(_shape2poly_wgs.p(*pts.T, inverse=True)).T
    polys = map(Polygon, imap(invproj, np.split(sh.points, sh.parts[1:])))
    mpoly = reduce(lambda p1,p2: p1 if p1.intersects(p2) else p1.union(p2),
                   sorted((p for p in polys if p.area >= minarea),
                          key=attrgetter('area'), reverse=True),
                   GeometryCollection())
    if isinstance(mpoly, GeometryCollection):
        mpoly = max(polys, key=attrgetter('area'))
    return simplify_poly(mpoly, tolerance)
_shape2poly_wgs.p = Proj('+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

def _shape2poly(sh, tolerance=0.03, minarea=0.03):
    if len(sh.points) == 0:
        return None
    polys = map(Polygon, np.split(sh.points, sh.parts[1:]))
    mpoly = reduce(lambda p1,p2: p1 if p1.intersects(p2) else p1.union(p2),
                   sorted((p for p in polys if p.area >= minarea),
                          key=attrgetter('area'), reverse=True),
                   GeometryCollection())
    if isinstance(mpoly, GeometryCollection):
        mpoly = max(polys, key=attrgetter('area'))
    return simplify_poly(mpoly, tolerance)

@cachable(keepweakref=True)
def nuts_countries(tolerance=0.03):
    sf = shapefile.Reader(toModDir('data/NUTS_2010_60M_SH/data/NUTS_RG_60M_2010'))
    return OrderedDict(sorted([(rec[0].decode('utf-8'), _shape2poly(sh, tolerance))
                               for rec, sh in izip(sf.iterRecords(), sf.iterShapes())
                               if rec[1] == 0],
                              key=itemgetter(0)))

@cachable(keepweakref=True)
def nuts1_regions(tolerance=0.03, minarea=1.):
    sf = shapefile.Reader(toModDir('data/NUTS_2010_60M_SH/data/NUTS_RG_60M_2010'))
    return OrderedDict(sorted([(rec[0].decode('utf-8'), _shape2poly(sh, tolerance, minarea))
                               for rec, sh in izip(sf.iterRecords(), sf.iterShapes())
                               if rec[1] == 1],
                              key=itemgetter(0)))

@cachable(keepweakref=True)
def nuts1_regions_ext(tolerance=0.03, minarea=1.):
    """
    Add the countries in Europe missing to nuts1_regions using countries
    """
    cmap = {'BA': u'BA1', 'RS': u'RS1', 'AL': u'AL1', 'KV': u'KV1'}
    nutsext = nuts1_regions(tolerance, minarea).copy()
    nutsext.update((cmap[k], v) for k,v in countries(cmap.keys(), tolerance).iteritems())
    return nutsext


@cachable(keepweakref=True, version=3)
def countries(subset=None, tolerance=0.03):
    sf = shapefile.Reader(toModDir('data/ne_10m_admin_0_countries/ne_10m_admin_0_countries'))
    fields = dict(izip(map(itemgetter(0), sf.fields[1:]), count()))
    if subset is not None:
        subset = frozenset(subset)
        include = lambda x: x in subset
    else:
        # '-99' means 'not available' in this dataset
        include = lambda x: True
    def name(rec):
        if rec[fields['ISO_A2']] != '-99':
            return rec[fields['ISO_A2']]
        elif rec[fields['WB_A2']] != '-99':
            return rec[fields['WB_A2']]
        else:
            return rec[fields['ADM0_A3']][:-1]
    return OrderedDict(sorted([(n, _shape2poly(sf.shape(i), tolerance))
                               for i, rec in enumerate(sf.iterRecords())
                               for n in (name(rec),)
                               if include(n) and rec[fields['scalerank']] == 0],
                              key=itemgetter(0)))

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
    return OrderedDict(sorted([(name(rec[6].decode('utf-8')), _shape2poly_wgs(sh, tolerance))
                               for rec, sh in izip(sf.iterRecords(), sf.iterShapes())
                               if rec[1] == 4],
                              key=itemgetter(0)))

@cachable(keepweakref=True)
def landkreise(tolerance=0.03):
    sf_kreise = shapefile.Reader(toModDir('data/vg250/VG250_KRS'))
    # for special casing hamburg and berlin
    sf_land = shapefile.Reader(toModDir('data/vg250/VG250_LAN'))

    fields = [f[0] for f in sf_kreise.fields[1:]]
    fields = {n:fields.index(n) for n in ('GF', 'RS')}

    kreise = ((int(sr[fields['RS']]),
               _shape2poly_wgs(sf_kreise.shape(ind), tolerance))
              for ind, sr in izip(count(), sf_kreise.iterRecords())
              if sr[fields['GF']] == 4)

    berlinhamburg = ((int(sr[fields['RS']]),
                      _shape2poly_wgs(sf_land.shape(ind), tolerance))
                     for ind, sr in izip(count(), sf_land.iterRecords())
                     if (sr[fields['RS']] in ('11', '02')
                         and sr[fields['GF']] == 4))

    return Dict(chain(berlinhamburg, kreise))

@cachable(keepweakref=True, version=2)
def postcodeareas(tolerance=0.03):
    sf = shapefile.Reader(toModDir('data/plz-gebiete/plz-gebiete.shp'))
    return Dict((float(rec[0]), _shape2poly(sh))
                for rec, sh in izip(sf.iterRecords(), sf.iterShapes()))

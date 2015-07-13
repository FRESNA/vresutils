# -*- coding: utf-8 -*-

from pyproj import Proj
import shapefile
from shapely.geometry import LinearRing, Polygon, MultiPolygon, GeometryCollection
from itertools import izip, chain, count, imap, takewhile
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

def _shape2poly(sh, tolerance=0.03, minarea=0.03, projection=None):
    if len(sh.points) == 0:
        return None

    if projection is None:
        pts = sh.points
    elif projection == 'invwgs':
        pts = np.asarray(_shape2poly.wgs(*np.asarray(sh.points).T, inverse=True)).T
    else:
        raise TypeError("Unknown projection {}".format(projection))

    minlength = 2*np.pi*np.sqrt(minarea / np.pi)
    def parts2polys(parts):
        rings = map(LinearRing, parts)
        while(rings):
            exterior = rings.pop(0)
            interiors = list(takewhile(attrgetter('is_ccw'), rings))
            rings = rings[len(interiors):]
            yield Polygon(exterior, [x for x in interiors if x.length > minlength])

    polys = sorted(parts2polys(np.split(pts, sh.parts[1:])),
                   key=attrgetter('area'), reverse=True)
    if polys[0].area > minarea:
        mpoly = MultiPolygon(list(takewhile(lambda p: p.area > minarea, polys)))
    else:
        mpoly = polys[0]
    return simplify_poly(mpoly, tolerance)
_shape2poly.wgs = Proj('+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

@cachable(keepweakref=True)
def nuts0(tolerance=0.03, minarea=1.):
    sf = shapefile.Reader(toModDir('data/NUTS_2010_60M_SH/data/NUTS_RG_60M_2010'))
    return OrderedDict(sorted([(rec[0].decode('utf-8'), _shape2poly(sh, tolerance, minarea))
                               for rec, sh in izip(sf.iterRecords(), sf.iterShapes())
                               if rec[1] == 0],
                              key=itemgetter(0)))

@cachable(keepweakref=True)
def nuts1(tolerance=0.03, minarea=1., extended=True):
    sf = shapefile.Reader(toModDir('data/NUTS_2010_60M_SH/data/NUTS_RG_60M_2010'))
    nuts = OrderedDict(sorted([(rec[0].decode('utf-8'), _shape2poly(sh, tolerance, minarea))
                               for rec, sh in izip(sf.iterRecords(), sf.iterShapes())
                               if rec[1] == 1],
                              key=itemgetter(0)))
    if extended:
        cntry_map = {'BA': u'BA1', 'RS': u'RS1', 'AL': u'AL1', 'KV': u'KV1'}
        cntries = countries(cntry_map.keys(), tolerance, minarea)
        nuts.update((cntry_map[k], v) for k,v in cntries.iteritems())

    return nuts

@cachable(keepweakref=True, version=3)
def countries(subset=None, tolerance=0.03, minarea=1.):
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
    return OrderedDict(sorted([(n, _shape2poly(sf.shape(i), tolerance, minarea))
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
    return OrderedDict(sorted([(name(rec[6].decode('utf-8')), _shape2poly(sh, tolerance, projection='invwgs'))
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
               _shape2poly(sf_kreise.shape(ind), tolerance, projection='invwgs'))
              for ind, sr in izip(count(), sf_kreise.iterRecords())
              if sr[fields['GF']] == 4)

    berlinhamburg = ((int(sr[fields['RS']]),
                      _shape2poly(sf_land.shape(ind), tolerance, projection='invwgs'))
                     for ind, sr in izip(count(), sf_land.iterRecords())
                     if (sr[fields['RS']] in ('11', '02')
                         and sr[fields['GF']] == 4))

    return Dict(chain(berlinhamburg, kreise))

@cachable(keepweakref=True, version=2)
def postcodeareas(tolerance=0.03):
    sf = shapefile.Reader(toModDir('data/plz-gebiete/plz-gebiete.shp'))
    return Dict((float(rec[0]), _shape2poly(sh))
                for rec, sh in izip(sf.iterRecords(), sf.iterShapes()))

# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pyproj
import shapefile
import fiona
from shapely.ops import transform
from shapely.geometry import (LinearRing, Polygon, MultiPolygon,
                              GeometryCollection, shape)
from countrycode.countrycode import countrycode

from functools import partial
from itertools import chain, count, takewhile
from operator import itemgetter, attrgetter
from collections import OrderedDict
import numpy as np
import pandas as pd
import warnings
from six import iteritems
from six.moves import map, zip

from .decorators import staticvars, cachable
from . import make_toModDir, Singleton
toModDir = make_toModDir(__file__)

def haversine(*coords):
    if len(coords) == 1:
        coords = coords[0]
    lon, lat = np.deg2rad(np.asarray(coords)).T
    a = np.sin((lat[1]-lat[0])/2.)**2 + np.cos(lat[0]) * np.cos(lat[1]) * np.sin((lon[0] - lon[1])/2.)**2
    return 6371.000 * 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )

def simplify_poly(poly, tolerance):
    if tolerance is None:
        return poly
    else:
        return poly.simplify(tolerance, preserve_topology=True)

def simplify_pts(pts, tolerance=0.03):
    return points(simplify_poly(Polygon(pts), tolerance))

def points(poly):
    return np.asarray(poly.exterior)

def area(geom):
    return reproject(geom).area

def reproject(geom, fr=pyproj.Proj(proj='longlat'), to=pyproj.Proj(proj='aea')):
    reproject_pts = partial(pyproj.transform, fr, to)
    return transform(reproject_pts, geom)

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
        rings = list(map(LinearRing, parts))
        while(rings):
            exterior = rings.pop(0)
            interiors = list(takewhile(attrgetter('is_ccw'), rings))
            rings = rings[len(interiors):]
            yield Polygon(exterior, [x for x in interiors if x.length > minlength])

    polys = sorted(parts2polys(np.split(pts, sh.parts[1:])),
                   key=attrgetter('area'), reverse=True)
    mainpoly = polys[0]
    mainlength = np.sqrt(mainpoly.area/(2.*np.pi))
    if mainpoly.area > minarea:
        mpoly = MultiPolygon([p
                              for p in takewhile(lambda p: p.area > minarea, polys)
                              if mainpoly.distance(p) < mainlength])
    else:
        mpoly = mainpoly
    return simplify_poly(mpoly, tolerance)
_shape2poly.wgs = pyproj.Proj('+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

@cachable(keepweakref=True)
def nuts0(tolerance=0.03, minarea=1.):
    sf = shapefile.Reader(toModDir('data/NUTS_2010_60M_SH/data/NUTS_RG_60M_2010'))
    return OrderedDict(sorted([(rec[0].decode('utf-8'), _shape2poly(sh, tolerance, minarea))
                               for rec, sh in zip(sf.iterRecords(), sf.iterShapes())
                               if rec[1] == 0],
                              key=itemgetter(0)))

@cachable(keepweakref=True)
def nuts1(tolerance=0.03, minarea=1., extended=True):
    sf = shapefile.Reader(toModDir('data/NUTS_2010_60M_SH/data/NUTS_RG_60M_2010'))
    nuts = OrderedDict(sorted([(rec[0].decode('utf-8'), _shape2poly(sh, tolerance, minarea))
                               for rec, sh in zip(sf.iterRecords(), sf.iterShapes())
                               if rec[1] == 1],
                              key=itemgetter(0)))
    if extended:
        cntry_map = {'BA': u'BA1', 'RS': u'RS1', 'AL': u'AL1', 'KV': u'KV1'}
        cntries = countries(list(cntry_map.keys()), tolerance=tolerance, minarea=minarea)
        nuts.update((cntry_map[k], v) for k,v in iteritems(cntries))

    return nuts

@cachable(keepweakref=True)
def eez(subset=None, filter_remote=True, tolerance=0.03):
    names = []
    shapes = []
    countries3 = frozenset(countrycode(subset, origin='iso2c', target='iso3c'))
    with fiona.drivers(), fiona.open(toModDir('data/World_EEZ/World_EEZ_v8_2014.shp')) as f:
        for sh in f:
            name = sh['properties']['ISO_3digit']
            if name in countries3:
                names.append(sh['properties']['ISO_3digit'])
                shapes.append(simplify_poly(shape(sh['geometry']), tolerance=tolerance))

    names = countrycode(names, origin='iso3c', target='iso2c')
    if filter_remote:
        country_shapes = countries(subset)
        return pd.Series(dict((name, shape)
                              for name, shape in zip(names, shapes)
                              if shape.distance(country_shapes[name]) < 1e-3)).sort_index()
    else:
        return pd.Series(shapes, index=names)

    sf = shapefile.Reader(toModDir('data/World_EEZ/World_EEZ_v8_2014'))
    fields = dict(zip(map(itemgetter(0), sf.fields[1:]), count()))
    if subset is not None:
        subset = frozenset(subset)
        include = lambda x: x in subset
    else:
        # '-99' means 'not available' in this dataset
        include = lambda x: True
    def name(rec):
        return countrycode(rec[fields['ISO_3digit']], origin='iso3c', target='iso2c')
    return OrderedDict(sorted([(n, _shape2poly(sf.shape(i), tolerance, minarea))
                               for i, rec in enumerate(sf.iterRecords())
                               for n in (name(rec),)
                               if include(n)],
                              key=itemgetter(0)))

@cachable(keepweakref=True, version=3)
def countries(subset=None, name_field=None, tolerance=0.03, minarea=1.):
    sf = shapefile.Reader(toModDir('data/ne_10m_admin_0_countries/ne_10m_admin_0_countries'))
    fields = dict(zip(map(itemgetter(0), sf.fields[1:]), count()))
    if subset is not None:
        subset = frozenset(subset)
        include = lambda x: x in subset
    else:
        # '-99' means 'not available' in this dataset
        include = lambda x: True
    if name_field is None:
        def name(rec):
            if rec[fields['ISO_A2']] != '-99':
                return rec[fields['ISO_A2']]
            elif rec[fields['WB_A2']] != '-99':
                return rec[fields['WB_A2']]
            else:
                return rec[fields['ADM0_A3']][:-1]
    else:
        name = itemgetter(fields[name_field])
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
                               for rec, sh in zip(sf.iterRecords(), sf.iterShapes())
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
              for ind, sr in zip(count(), sf_kreise.iterRecords())
              if sr[fields['GF']] == 4)

    berlinhamburg = ((int(sr[fields['RS']]),
                      _shape2poly(sf_land.shape(ind), tolerance, projection='invwgs'))
                     for ind, sr in zip(count(), sf_land.iterRecords())
                     if (sr[fields['RS']] in ('11', '02')
                         and sr[fields['GF']] == 4))

    return Dict(chain(berlinhamburg, kreise))

@cachable(keepweakref=True, version=2)
def postcodeareas(tolerance=0.03):
    sf = shapefile.Reader(toModDir('data/plz-gebiete/plz-gebiete.shp'))
    return Dict((float(rec[0]), _shape2poly(sh, tolerance=tolerance))
                for rec, sh in zip(sf.iterRecords(), sf.iterShapes()))

def save_graph_as_shapes(G, nodes_fn, links_fn):
    import networkx as nx

    sf_nodes = shapefile.Writer(shapefile.POINT)

    # Prepare fields
    sf_nodes.field('label')
    for f in next(G.nodes_iter(data=True))[1]: sf_nodes.field(f)
    extractor = itemgetter(*map(itemgetter(0), sf_nodes.fields[1:]))

    for n, d in G.nodes_iter(data=True):
        sf_nodes.point(*d['pos'])
        sf_nodes.record(n, *extractor(d))

    sf_nodes.save(nodes_fn)


    sf_links = shapefile.Writer(shapefile.POLYLINE)
    for f in next(G.edges_iter(data=True))[2]: sf_links.field(f)
    extractor = itemgetter(*map(itemgetter(0), sf_links.fields))

    for n1, n2, d in G.edges_iter(data=True):
        sf_links.line(parts=[[list(G.node[n]['pos']) for n in (n1, n2)]])
        sf_links.record(*extractor(d))

    sf_links.save(links_fn)

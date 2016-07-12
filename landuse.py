from __future__ import absolute_import

import numpy as np
import pandas as pd
import tempfile, os.path
import rasterio
import subprocess
from shutil import rmtree
from six import iteritems
from six.moves import map

from .decorators import cachable

from . import make_toModDir
toModDir = make_toModDir(__file__)

@cachable(ignore=set(('tmpdir',)))
def corine_label1(cutout, tmpdir=None):
    """
    Fractional landuse of LABEL1 categories of corine land cover 2006
    data for the REatlas grid of a `cutout`.

    Parameters
    ----------
    cutout : vreatlas.Cutout
        Used by corine_by_groups to determine the grid geometry
    tmpdir : None|str (default: None)
        If not None, where the intermediate files are kept

    Returns
    -------
    groups, landuse : [str], ndarray shape=(len(groups), LATS, LONS)
    """
    legend = pd.read_excel(toModDir('data/corine/clc_legend.xls')) \
               .set_index('GRID_CODE').loc[:47] # above 47, there is only NAN data

    def simplify_name(x):
        try:
            x = x[:x.index(' ')]
        except ValueError:
            pass
        return x.lower()

    groups = {simplify_name(k):v
              for k,v in iteritems(legend.groupby(legend.LABEL1).groups)}

    return list(groups), corine_by_groups(cutout, groups, tmpdir=tmpdir)

@cachable(ignore=set(('tmpdir',)))
def corine_renewable(cutout, tmpdir=None):
    """
    Corine classification by hand :)

    wind is preferred from PhD thesis by ed sharp p. 177
    (
    preferred : Scrub, Herbaceous Vegetation, Forest, Pasture, Arable
           land and Inland Wetland, Marine waters
    ifneeded : Industrial, Commercial, Transport, Urban Fabric,
               heterogeneous agricultural and sparsely vegetated land
    )

    solar comes from everything but forests and waters

    """

    # ifneeded might be added later
    groups = {'wind': [12, 13, 14, 18, 23, 24, 25, 26, 27, 28, 29, 35, 36, 42, 43, 44],
              'solar': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]}

    return list(groups), corine_by_groups(cutout, groups, tmpdir=tmpdir)


def corine_by_groups(cutout, groups, fn=toModDir('data/corine/g100_06.tif'), tmpdir=None):
    own_tmpdir = tmpdir is None
    if own_tmpdir:
        tmpdir = tempfile.mkdtemp()

    landuse = np.asarray([corine_for_cutout(cutout, indices, label=group, fn=fn, tmpdir=tmpdir)
                          for group, indices in iteritems(groups)])

    if own_tmpdir:
        rmtree(tmpdir, ignore_errors=True)

    return landuse

def corine_for_cutout(cutout, grid_codes, label=None, natura=False,
                      fn=toModDir('data/corine/g250_clc06_V18_5.tif'),
                      natura_fn=toModDir('data/Natura2000/Natura2000_end2015.shp'),
                      tmpdir=None):
    own_tmpdir = tmpdir is None
    if own_tmpdir:
        tmpdir = tempfile.mkdtemp()

    if label is None:
        # Unsafe, but alternative is unwieldy
        label = os.path.basename(tempfile.mktemp(suffix=".tif", dir=tmpdir)[:-4])

    # Write matching grid_codes out into a file in tmpdir, and convert
    # this file using gdalwarp

    with rasterio.drivers():
        with rasterio.open(fn) as src:
            meta = src.meta.copy()
            meta.update(transform=src.meta['affine'],
                        compress='lzw')

            windows = src.block_windows(1)

            with rasterio.open(os.path.join(tmpdir, '{}.tif'.format(label)), 'w', **meta) as dst:
                for idx, window in windows:
                    src_data, = src.read(window=window)
                    dst_data = np.in1d(src_data.ravel(), grid_codes).astype("uint8").reshape(src_data.shape)
                    dst.write_band(1, dst_data, window=window)

        cornersc = cutout.grid_coordinates()[[0,-1]]
        minc = np.minimum(*cornersc)
        maxc = np.maximum(*cornersc)
        span = (maxc - minc)/(np.asarray(cutout.shape)[[1,0]]-1)
        minx, miny = minc - span/2.
        maxx, maxy = maxc + span/2.

        if natura:
            # rasterio does not include the coordinate reference
            # system in a proper manner, so we add it manually
            ret = subprocess.call(['gdal_edit.py', '-a_srs', 'EPSG:3035',
                                   os.path.join(tmpdir, '{}.tif'.format(label))])
            assert ret == 0, "gdal_edit for group '{}' did not return successfully.".format(label)

            ret = subprocess.call(['gdal_rasterize', '-burn', '0',
                                   natura_fn, os.path.join(tmpdir, '{}.tif'.format(label))])
            assert ret == 0, "gdal_rasterize for group '{}' did not return successfully.".format(label)

        ret = subprocess.call(['gdalwarp', '-overwrite',
                               '-s_srs', 'EPSG:3035',
                               '-t_srs', 'EPSG:4326',
                               '-te', str(minx), str(miny), str(maxx), str(maxy),
                               '-ts', str(cutout.shape[1]), str(cutout.shape[0]),
                               '-r', "average",
                               '-wt', 'Float64',
                               '-ot', 'Float64',
                               '-srcnodata', 'None',
                               '-dstnodata', 'None',
                               os.path.join(tmpdir, '{}.tif'.format(label)),
                               os.path.join(tmpdir, '{}_avg.tif'.format(label))])
        assert ret == 0, "gdalwarp for group '{}' did not return successfully.".format(label)


        with rasterio.open(os.path.join(tmpdir, '{}_avg.tif'.format(label))) as avg:
            landuse = avg.read()[0]

    if own_tmpdir:
        rmtree(tmpdir, ignore_errors=True)

    # reversed_cutout_dims = np.r_[False,cornersc[0] > cornersc[1]]
    # if reversed_cutout_dims.any():
    #     landuse = landuse[tuple(slice(None, None, -1) if r else slice(None)
    #                             for r in reversed_cutout_dims)]

    return landuse

# Mappings between CORINE Label1 and fractional use
wind = pd.Series(dict(agricultural=1.0, forest=1.0, wetlands=1.0))
solar = pd.Series(dict(artificial=1.0, agricultural=1.0, wetlands=1.0))

def potential(mapping, cutout, func=corine_label1):
    """
    Returns the landuse potential, the fraction of the usable area

    Parameters
    ----------
    mapping : pd.Series
        Map between group and fraction
    cutout : vreatlas.Cutout
        Used by func to determine the cell geometry
    func : function : vreatlas.Cutout -> (groups, landuse)
        Default is corine_label1

    Example
    -------
    cutout = vreatlas.Cutout(cutoutname='Europe_2011_2014', username='becker')
    potential = vlanduse.potential(vlanduse.wind, cutout)
    """
    if callable(func):
        groups, landuse = func(cutout)
    else:
        groups, landuse = func
    return np.dot(landuse.transpose((1,2,0)), mapping.reindex(groups).fillna(0.))

@cachable
def solarpotentials(cutout, natura=True):
    from vresutils import shapes as vshapes
    reatlas_cell_areas = np.asarray(list(map(vshapes.area, cutout.grid_cells()))).reshape(cutout.shape)*1e-6
    return 17. * corine_for_cutout(cutout, [1, 2, 3, 4, 5, 6, 12, 13, 14, 18, 26, 27, 28, 29], natura=natura) * reatlas_cell_areas

@cachable
def windonshorepotentials(cutout, natura=True):
    from vresutils import shapes as vshapes
    reatlas_cell_areas = np.asarray(list(map(vshapes.area, cutout.grid_cells()))).reshape(cutout.shape)*1e-6
    return 5. * corine_for_cutout(cutout, [12, 13, 14, 18, 23, 24, 25, 26, 27, 28, 29, 35, 36], natura=True) * reatlas_cell_areas

@cachable
def windoffshorepotentials(cutout, natura=True):
    from vresutils import shapes as vshapes
    reatlas_cell_areas = np.asarray(list(map(vshapes.area, cutout.grid_cells()))).reshape(cutout.shape)*1e-6
    return 5.* corine_for_cutout(cutout, [44, 255], natura=True) * reatlas_cell_areas

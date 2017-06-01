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

from tempfile import TemporaryFile
from operator import itemgetter
import numpy as np
import pandas as pd
import os

from . import mapping as vmapping
from . import make_toDataDir
toDataDir = make_toDataDir(__file__)

# Import REatlas
try:
    # Rename or link REatlas-client directory to REatlas_client
    from REatlas_client import reatlas_client
    clientdir = os.path.dirname(reatlas_client.__file__)
except ImportError:
    # Let's try to be clever
    import sys
    _toplevel = os.path.dirname(os.path.dirname(__file__))
    clientdir = os.path.join(_toplevel, 'REatlas-client')
    sys.path.insert(sys.path.index(_toplevel), clientdir)
    import reatlas_client


from . import array as varray

from . import get_config
from .decorators import timer, CachedAttribute

def partition_from_shapes(shapes, cutout):
    if not isinstance(shapes, pd.Series):
        shapes = pd.Series(shapes)

    raise NotImplemented

def partition_from_emil(cutout, path=toDataDir("Europe_2011_2014")):
    if str(cutout) != '<Cutout becker/Europe_2011_2014>':
        raise "Partition from emil does probably not correspond to cutout {}".format(cutout)

    mapping = vmapping.countries_to_nuts1(series=True)
    countries = mapping.unique()
    iso2toiso3 = vmapping.iso2_to_iso3()

    return pd.Series(dict((iso2, np.load(os.path.join(path, "masks/{}.npy".format(iso2toiso3[iso2])))[::-1])
                           for iso2 in countries))

def turbineconf_to_powercurve_object(fn):
    if isinstance(fn, dict):
        # Already a config object!?
        return fn

    if '/' not in fn:
        fn = os.path.join(clientdir, 'TurbineConfig', fn + '.cfg')
    return reatlas_client.turbineconf_to_powercurve_object(fn)

def solarpanelconf_to_solar_panel_config_object(fn):
    if isinstance(fn, dict):
        # Already a config object!?
        return fn

    if '/' not in fn:
        fn = os.path.join(clientdir, 'SolarPanelData', fn + '.cfg')
    return reatlas_client.solarpanelconf_to_solar_panel_config_object(fn)

def solarpanel_rated_capacity_per_m2(panel):
    # one unit in the capacity layout is interpreted as one panel of a
    # capacity (A + 1000 * B + log(1000) * C) * 1000W/m^2 * (k / 1000)

    panelconf = solarpanelconf_to_solar_panel_config_object(panel)
    A, B, C = itemgetter('A', 'B', 'C')(panelconf)
    return (A + B * 1000. + C * np.log(1000.))*1e3

def windturbine_rated_capacity_per_unit(turbine):
    powercurve = turbineconf_to_powercurve_object(turbine)
    return max(powercurve['POW'])

class Cutout(object):
    def __init__(self, cutoutname, username, reatlas=None):
        self.cutoutname = cutoutname
        self.username = username

        if reatlas is None:
            reatlas = REatlas()
        self._reatlas = reatlas

    @property
    def reatlas(self):
        self._reatlas.select_cutout_from_obj(self)
        return self._reatlas

    @property
    def shape(self):
        return self.meta['latitudes'].shape

    @CachedAttribute
    def meta(self):
        self._reatlas.reconnect_if_disconnected()
        self._reatlas.prepare_cutout_metadata(cutoutname=self.cutoutname,
                                              username=self.username)

        fn = 'meta_{}.npz'.format(self.cutoutname)
        return self._reatlas.download_delete_and_load(remote_file=fn, name='metadata')

    def grid_coordinates(self, latlon=False):
        meta = self.meta
        if latlon:
            return np.asarray((np.ravel(meta['latitudes']),
                               np.ravel(meta['longitudes']))).T
        else:
            return np.asarray((np.ravel(meta['longitudes']),
                               np.ravel(meta['latitudes']))).T

    def grid_cells(self):
        from shapely.geometry import box

        coords = self.grid_coordinates()
        span = (coords[self.shape[1]+1] - coords[0]) / 2
        return [box(*c) for c in np.hstack((coords - span, coords + span))]

    @property
    def extent(self):
        return (list(self.meta['longitudes'].ravel()[[0, -1]]) +
                list(self.meta['latitudes'].ravel()[[-1, 0]]))

    def __repr__(self):
        return '<Cutout {}/{}>'.format(self.username, self.cutoutname)

class REatlas(reatlas_client.REatlas):
    def __init__(self, **kwds):
        self.config = get_config('.reatlas.config', defaults=dict(notify=False), overwrites=kwds)
        self.cutout = self.config.get('cutout', None)

        super(REatlas, self).__init__(self.config['hostname'])

        self.reconnect_if_disconnected(check_first=False)

    def reconnect_if_disconnected(self, check_first=True):
        if check_first:
            try:
                self.echo()
            except reatlas_client.ConnectionError:
                pass
            else:
                return

        if not self.connect_and_login(username=self.config['username'],
                                      password=self.config['password']):
            raise reatlas_client.ConnectionError("Could not log into REatlas")

        self.notify_by_mail(notify=self.config['notify']);

        if self.cutout is not None:
            self.select_cutout(**self.cutout)

    def select_cutout_from_obj(self, cutout=None, **kwargs):
        self.reconnect_if_disconnected()

        if isinstance(cutout, Cutout):
            kwargs.update(cutoutname=cutout.cutoutname,
                          username=cutout.username)
        # store the cutout so we are able to re-init the connection later
        self.cutout = kwargs

        return self.select_cutout(**kwargs)

    def add_pv_orientations_by_config_file(self, fn):
        self.reconnect_if_disconnected()

        if '/' not in fn:
            fn = os.path.join(clientdir, 'orientation_examples', fn + '.cfg')
        return super(REatlas, self).add_pv_orientations_by_config_file(fn)

    def convert_and_aggregate(self, resource, capacity_layouts=[], save_sum=False, **kwargs):
        self.reconnect_if_disconnected()

        if save_sum:
            assert len(capacity_layouts) == 0, "Only save_sum or capacity_layouts supported"
            capacity_layouts_fn = []
        elif self._protocol_version < 3:
            capacity_layouts_fn = [self.upload_from_data_and_rename(l)
                                   for l in capacity_layouts]
        else:
            capacity_layouts_fn = [self.upload_from_data_and_rename(capacity_layouts)]

        job_fn = self._get_unique_npy_file()

        if set(('onshore', 'offshore')).issubset(resource):
            onshorepowercurve = turbineconf_to_powercurve_object(resource['onshore'])
            offshorepowercurve = turbineconf_to_powercurve_object(resource['offshore'])

            job_id = self.convert_and_aggregate_wind(
                result_name=job_fn[:-4],
                onshorepowercurve=onshorepowercurve,
                offshorepowercurve=offshorepowercurve,
                capacitylayouts=capacity_layouts_fn,
                save_sum=save_sum,
                **kwargs
            )

            solar = False
        elif set(('panel', 'orientation')).issubset(resource):
            self.add_pv_orientations_by_config_file(resource['orientation'])
            panel = solarpanelconf_to_solar_panel_config_object(resource['panel']);
            job_id = self.convert_and_aggregate_pv(
                result_name=job_fn[:-4],
                solar_panel_config=panel,
                capacitylayouts=capacity_layouts_fn,
                save_sum=save_sum,
                **kwargs
            )
            solar = True
        else:
            raise TypeError('`resource` must either contain onshore and offshore or panel and orientation')

        self.wait_for_job(job_id=job_id)

        if save_sum:
            return self.download_delete_and_load(remote_file=job_fn[:-4]+"_sum.npy",
                                                 name='save_sum')
        else:
            for fn in capacity_layouts_fn:
                self.delete_file(filename=fn)

            timeseries = self.download_delete_and_load(remote_file=job_fn, name='job')

            if solar:
                timeseries = varray.interpolate(timeseries)

            return timeseries

    def download_delete_and_load(self, remote_file, name=''):
        f = TemporaryFile()
        self.download_file_and_rename(remote_file=remote_file, local_file=f)
        self.delete_file(filename=remote_file)
        f.seek(0)
        try:
            return np.load(f)
        except IOError:
            raise RuntimeError("Couldn't read downloaded {} data".format(name))

    def upload_from_data_and_rename(self, data, remote_file=None):
        if remote_file is None:
            remote_file = self._get_unique_npy_file()

        f = TemporaryFile()
        np.save(f, data)
        f.seek(0)

        self.upload_from_file_and_rename(f, remote_file)
        return remote_file

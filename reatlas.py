from __future__ import absolute_import

# Import REatlas
import sys, os
_toplevel = os.path.dirname(os.path.dirname(__file__))
clientdir = os.path.join(_toplevel, 'REatlas-client')
sys.path.insert(sys.path.index(_toplevel), clientdir)
import reatlas_client

from tempfile import TemporaryFile
import numpy as np

from . import timer, CachedAttribute
from . import array as varray

def turbineconf_to_powercurve_object(fn):
    if '/' not in fn:
        fn = os.path.join(clientdir, 'TurbineConfig', fn + '.cfg')
    return reatlas_client.turbineconf_to_powercurve_object(fn)

def solarpanelconf_to_solar_panel_config_object(fn):
    if '/' not in fn:
        fn = os.path.join(clientdir, 'SolarPanelData', fn + '.cfg')
    return reatlas_client.solarpanelconf_to_solar_panel_config_object(fn)

class Cutout(object):
    def __init__(self, cutoutname, username, reatlas=None):
        self.cutoutname = cutoutname
        self.username = username

        if reatlas is None:
            reatlas = REatlas()
        self.reatlas = reatlas

    @property
    def shape(self):
        return self.meta['latitudes'].shape

    @CachedAttribute
    def meta(self):
        self.reatlas.prepare_cutout_metadata(cutoutname=self.cutoutname,
                                             username=self.username)

        fn = 'meta_{}.npz'.format(self.cutoutname)
        f = TemporaryFile()
        self.reatlas.download_file_and_rename(remote_file=fn, local_file=f)
        self.reatlas.delete_file(filename=fn)
        f.seek(0)
        try:
            meta = np.load(f)
        except IOError:
            raise RuntimeError("Couldn't read downloaded metadata")

        return meta

    def grid_coordinates(self, latlon=False):
        meta = self.meta
        if latlon:
            return np.array((np.ravel(meta['latitudes']),
                             np.ravel(meta['longitudes']))).T
        else:
            return np.array((np.ravel(meta['longitudes']),
                             np.ravel(meta['latitudes']))).T

    def grid_cells(self):
        from shapely.geometry import box

        coords = self.grid_coordinates()
        span = (coords[self.shape[1]+1] - coords[0]) / 2
        return [box(*c) for c in np.hstack((coords - span, coords + span))]

    def __repr__(self):
        return '<Cutout {}/{}>'.format(self.username, self.cutoutname)

class REatlas(reatlas_client.REatlas):
    def __init__(self, **kwds):
        config = dict(
            notify=False
        )

        config_fn = os.path.expanduser('~/.reatlas.config')
        if os.path.exists(config_fn):
            exec(compile(open(config_fn).read(), config_fn, 'exec'), dict(), config)

        config.update(kwds)

        super(REatlas, self).__init__(config['hostname'])
        if not self.connect_and_login(username=config['username'],
                                      password=config['password']):
            raise Exception("Could not log into REatlas")

        self.notify_by_mail(notify=config['notify']);

        if 'cutout' in config:
            self.select_cutout(**config['cutout'])

    def select_cutout_from_obj(self, cutout=None, **kwargs):
        if isinstance(cutout, Cutout):
            kwargs.update(cutoutname=cutout.cutoutname,
                          username=cutout.username)

        return self.select_cutout(**kwargs)

    def add_pv_orientations_by_config_file(self, fn):
        if '/' not in fn:
            fn = os.path.join(clientdir, 'orientation_examples', fn + '.cfg')
        return super(REatlas, self).add_pv_orientations_by_config_file(fn)

    def convert_and_aggregate(self, resource, capacity_layouts):
        def upload_capacity_layout(layout):
            f = TemporaryFile()
            np.save(f, layout)
            f.seek(0)
            fn = self._get_unique_npy_file()
            self.upload_from_file_and_rename(f, fn)
            return fn
        if self._protocol_version < 3:
            capacity_layouts_fn = [upload_capacity_layout(l)
                                   for l in capacity_layouts]
        else:
            capacity_layouts_fn = [upload_capacity_layout(capacity_layouts)]

        job_fn = self._get_unique_npy_file()

        if set(('onshore', 'offshore')).issubset(resource):
            onshorepowercurve = turbineconf_to_powercurve_object(resource['onshore'])
            offshorepowercurve = turbineconf_to_powercurve_object(resource['offshore'])

            job_id = self.convert_and_aggregate_wind(
                result_name=job_fn[:-4],
                onshorepowercurve=onshorepowercurve,
                offshorepowercurve=offshorepowercurve,
                capacitylayouts=capacity_layouts_fn
            )

            solar = False
        elif set(('panel', 'orientation')).issubset(resource):
            self.add_pv_orientations_by_config_file(resource['orientation'])
            panel = solarpanelconf_to_solar_panel_config_object(resource['panel']);
            job_id = self.convert_and_aggregate_pv(
                result_name=job_fn[:-4],
                solar_panel_config=panel,
                capacitylayouts=capacity_layouts_fn
            )
            solar = True
        else:
            raise TypeError('`resource` must either contain onshore and offshore or panel and orientation')

        self.wait_for_job(job_id=job_id)

        for fn in capacity_layouts_fn:
            self.delete_file(filename=fn)

        f = TemporaryFile()
        self.download_file_and_rename(remote_file=job_fn, local_file=f)
        self.delete_file(filename=job_fn)
        f.seek(0)
        try:
            timeseries = np.load(f)
        except IOError:
            raise RuntimeError("Couldn't read downloaded job data")

        if solar:
            timeseries = varray.interpolate(timeseries)

        return timeseries

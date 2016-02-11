from __future__ import absolute_import

from tempfile import TemporaryFile
import numpy as np
import os

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
        self.reatlas.reconnect_if_disconnected()
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

    def convert_and_aggregate(self, resource, capacity_layouts=[], save_sum=False):
        def upload_capacity_layout(layout):
            f = TemporaryFile()
            np.save(f, layout)
            f.seek(0)
            fn = self._get_unique_npy_file()
            self.upload_from_file_and_rename(f, fn)
            return fn
        self.reconnect_if_disconnected()

        if save_sum:
            assert len(capacity_layouts) == 0, "Only save_sum or capacity_layouts supported"
            capacity_layouts_fn = []
        elif self._protocol_version < 3:
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
                capacitylayouts=capacity_layouts_fn,
                save_sum=save_sum
            )

            solar = False
        elif set(('panel', 'orientation')).issubset(resource):
            self.add_pv_orientations_by_config_file(resource['orientation'])
            panel = solarpanelconf_to_solar_panel_config_object(resource['panel']);
            job_id = self.convert_and_aggregate_pv(
                result_name=job_fn[:-4],
                solar_panel_config=panel,
                capacitylayouts=capacity_layouts_fn,
                save_sum=save_sum
            )
            solar = True
        else:
            raise TypeError('`resource` must either contain onshore and offshore or panel and orientation')

        self.wait_for_job(job_id=job_id)

        if save_sum:
            f = TemporaryFile()
            self.download_file_and_rename(remote_file=job_fn[:-4]+"_sum.npy",
                                          local_file=f)
            self.delete_file(filename=job_fn)
            f.seek(0)
            try:
                return np.load(f)
            except IOError:
                raise RuntimeError("Couldn't read save_sum data")
        else:
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

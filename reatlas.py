# Import REatlas
import sys, os
_toplevel = os.path.dirname(os.path.dirname(__file__))
clientdir = os.path.join(_toplevel, 'REatlas-client')
sys.path.insert(sys.path.index(_toplevel), clientdir)
import reatlas_client

from tempfile import TemporaryFile
import numpy as np

from . import timer
import array as varray

def turbineconf_to_powercurve_object(fn):
    if '/' not in fn:
        fn = os.path.join(clientdir, 'TurbineConfig', fn + '.cfg')
    return reatlas_client.turbineconf_to_powercurve_object(fn)

def solarpanelconf_to_solar_panel_config_object(fn):
    if '/' not in fn:
        fn = os.path.join(clientdir, 'SolarPanelData', fn + '.cfg')
    return reatlas_client.solarpanelconf_to_solar_panel_config_object(fn)

class REatlas(reatlas_client.REatlas):
    def __init__(self, **kwds):
        config = dict(
            notify=False
        )

        config_fn = os.path.expanduser('~/.reatlas.config')
        if os.path.exists(config_fn):
            execfile(config_fn, dict(), config)

        config.update(kwds)

        super(REatlas, self).__init__(config['hostname'])
        if not self.connect_and_login(username=config['username'],
                                      password=config['password']):
            raise Exception("Could not log into REatlas")

        self.notify_by_mail(notify=config['notify']);

        if 'cutout' in config:
            self.select_cutout(**config['cutout'])

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
        with timer("Uploading capacity layouts"):
            capacity_layouts_fn = [upload_capacity_layout(capacity_layouts)]

        job_fn = self._get_unique_npy_file()

        with timer("Running convert and aggregate"):
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

        with timer("Deleting uploaded capacity layouts"):
            for fn in capacity_layouts_fn:
                self.delete_file(filename=fn)

        with timer("Downloading result"):
            f = TemporaryFile()
            self.download_file_and_rename(remote_file=job_fn, local_file=f)
            self.delete_file(filename=job_fn)
            f.seek(0)
            try:
                timeseries = np.load(f)
            except IOError:
                raise RuntimeError("Couldn't read downloaded job data")

        if solar:
            with timer("Interpolating nan values"):
                timeseries = varray.interpolate(timeseries)

        return timeseries

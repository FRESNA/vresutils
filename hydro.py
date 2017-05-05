import pandas as pd
import cPickle as pickle
import os

from .decorators import cachable, timer
from . import make_toDataDir
toDataDir = make_toDataDir(__file__)

def get_hydro_capas(fn=None):
    if fn is None:
        fn = toDataDir('Hydro_Inflow/emil_hydro_capas.csv')
    return pd.read_csv(fn, index_col=0)#,names=[''])

def get_ror_shares(fn=None):
    if fn is None:
        fn = toDataDir('Hydro_Inflow/run_of_shares/ror_ENTSOe_Restore2050.csv')
    return pd.read_csv(fn, index_col=0, squeeze=True)

@cachable
def get_hydro_inflow(inflow_dir=None):
    """Return hydro inflow data for europe. [GWh]"""

    if inflow_dir is None:
        inflow_dir = toDataDir('Hydro_Inflow')

    def read_inflow(country):
        return (pd.read_csv(os.path.join(inflow_dir,
                                         'Hydro_Inflow_{}.csv'.format(country)),
                            parse_dates={'date': [0,1,2]})
                .set_index('date')['Inflow [GWh]'])

    europe = ['AT','BA','BE','BG','CH','CZ','DE','GR',
              'ES','FI','FR','HR','HU','IE','IT','KV',
              'LT','LV','ME','MK','NL','NO','PL','PT',
              'RO','RS','SE','SI','SK','GB']

    hyd = pd.DataFrame({cname: read_inflow(cname) for cname in europe})
    #hyd.rename(columns={'EL':'GR','UK':'GB'}, inplace=True)

    with timer('resampling hydro data with cubic interpolation'):
        hydro = hyd.resample('D').interpolate('cubic')

    if True: #default norm
        normalization_factor = (hydro.index.size/float(hyd.index.size)) #normalize to new sampling frequency
    else:
        normalization_factor = hydro.sum() / hyd.sum() #conserve total inflow for each country separately
    hydro /= normalization_factor
    return hydro

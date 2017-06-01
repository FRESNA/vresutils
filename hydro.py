import pandas as pd
import os
import six
if six.PY3:
    from countrycode.countrycode import countrycode
else:
    from countrycode import countrycode

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

def get_eia_annual_hydro_generation(fn=None):
    if fn is None:
        fn = toDataDir('Hydro_Inflow/EIA_hydro_generation_2000_2014.csv')

    # in billion KWh/a = TWh/a
    eia_hydro_gen = pd.read_csv(fn, skiprows=4, index_col=1, na_values=[u' ','--']).drop(['Unnamed: 0','Unnamed: 2'],axis=1).dropna(how='all')

    countries_iso2c = countrycode(eia_hydro_gen.index.values, origin='country_name', target='iso2c')

    eia_hydro_gen.index = pd.Index(countries_iso2c, name='countries')
    eia_hydro_gen.rename(index={'Kosovo':'KV'}, inplace=True)

    eia_hydro_gen = eia_hydro_gen.T

    return eia_hydro_gen * 1e6 #in MWh/a


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
        hydro = hyd.resample('H').interpolate('cubic')

    if True: #default norm
        normalization_factor = (hydro.index.size/float(hyd.index.size)) #normalize to new sampling frequency
    else:
        normalization_factor = hydro.sum() / hyd.sum() #conserve total inflow for each country separately
    hydro /= normalization_factor
    return hydro



def inflow_timeseries(cutout, country_shapes, rolling_mean_period='24h', clip_quantile=0.01):
    '''Return hydro inflow timeseries for countries in `country_shapes.index` in
    units of MWh.

    They are normalized such that the average inflow energy over several years
    equals the national average hydro energy generation reported by EIA. The 
    longer the calibration period the better, as inflow in a given year might 
    not be used for generation in the same year, i.e., might be stored.

    The surface runoff data from atlite/NCEP fluctuates strongly, probably
    due to relatively discrete precipitation events. Assuming that the 
    aggregated surface runoff does not immediately reach the hydro power
    facility, and to reduce nummerical issues, the runoff is smoothed via a
    rolling mean window of 24h (`rolling_mean_period`).

    The NCEP runoff contains unrealistic negative values. They are removed by
    clipping the runoff to a minimum value defined by the 1% (`clip_quantile`)
    quantile in each country. This increases the inflow energy by no more than
    0.03%.
    '''
    


    ## Calculate runoff with atlite, smooth it, and clip it:

    #trans_matrix_onshore = vtransfer.Shapes2Shapes(country_shapes.values,cutout.grid_cells(),normed=False).T
    #runoff = cutout.runoff(matrix=trans_matrix_onshore,index=country_shapes.index.rename('countries'))
    # same as above but without reusable indicator/transfer matrix (and different normalization):
    runoff = cutout.runoff(shapes=country_shapes,index=country_shapes.index.rename('countries'))

    runoff.name = 'runoff' #need name to convert to pd.Dataframe
    runoff=runoff.to_dataframe().unstack(0)[runoff.name].rolling(rolling_mean_period).mean()

    # fixing non-sensical negative inflow values: set minimum inflow to 0.5% quantile;
    # this also helps avoiding +1e-15 values
    # this increases the inflow energy by no more than 0.03%
    runoff=runoff.clip(lower=runoff.quantile(clip_quantile),axis=1)


    ## Normalize the runoff with annual generation data from EIA:
    eia_hydro_gen = get_eia_annual_hydro_generation() #in MWh/a
    eia_hydro_gen.index = pd.to_datetime(eia_hydro_gen.index).rename('time')


    # find the time overlap between yearly EIA generation data and the hourly runoff data; 
    # this assumes that if Jan 1 0:00 AM is present in the runoff data, the full year is available
    time_overlap = pd.DatetimeIndex(set(eia_hydro_gen.index.values) & set(runoff.index.values))

    norm_eia = eia_hydro_gen.loc[time_overlap].sum()

    norm_runoff = runoff.loc[slice(str(time_overlap.year.min()),str(time_overlap.year.max()))].sum()


    inflow = runoff / norm_runoff * norm_eia.loc[runoff.columns]

    return inflow


@cachable
def get_inflow_NCEP_EIA(cutoutname='europe-2011-2016'):
    import atlite
    from . import mapping as vmapping, shapes as vshapes
    cutout = atlite.Cutout(cutoutname)

    mapping = vmapping.countries_to_nuts3()
    countries = mapping.value_counts().index.sort_values()


    country_shapes = pd.Series(vshapes.countries(countries, minarea=0.1, tolerance=0.01, add_KV_to_RS=True))

    inflow = inflow_timeseries(cutout,country_shapes)

    return inflow

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

    return eia_hydro_gen * 1e6

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

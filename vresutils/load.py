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

import numpy as np
import pandas as pd

import os
import glob
import pytz, datetime
from scipy.optimize import leastsq
from six.moves import range, zip

from . import shapes as vshapes, mapping as vmapping, transfer as vtransfer
from .decorators import cachable

from . import make_toDataDir
toDataDir = make_toDataDir(__file__)


@cachable(keepweakref=True)
def timeseries_entsoe(years=list(range(2011, 2015+1)), countries=None, directory=None):
    """
    Read consumption data from ENTSO-E country packages

    Parameters
    ----------
    years : list of int
        Years for which to read consumption data (defaults to
        2011-2015)
    countries : list or None
        Country names in the encoding of ENTSO-E as full names
        (refer to the data/entsoe_country_packages directory).
        If None, read data for all countries (default).

    Returns
    -------
    load : pd.DataFrame
        Load time-series with UTC timestamps x ISO-2 countries
    """

    # Only take into account years from 2006 to 2015
    years = [y for y in years if y >= 2006 and y <= 2015]

    if directory is None:
        directory = toDataDir('entsoe_country_packages')
    fns = sum((glob.glob(os.path.join(directory, '{}_{}.xls'.format(c, y)))
               for y in years
               for c in (('*',) if countries is None else countries)), [])
    def read_all_excel(fns):
        for fn in fns:
            try:
                yield pd.read_excel(fn, skiprows=6, header=0, sheetname='hourly_load_values', na_values=[u' '])
            except StopIteration:
                pass
    tz = pytz.timezone('Europe/Berlin')
    data = pd.concat(read_all_excel(fns))
    transdata = data.ix[:,['Country','Date/Time', '3B:00:00']] \
            .set_index(['Country', 'Date/Time']).stack().unstack(0)
    del data['3B:00:00']

    data = data \
        .set_index(['Country', 'Date/Time']) \
        .stack().unstack(0)

    transitions = [t for t in tz._utc_transition_times if t.year in years]
    since = datetime.datetime(years[0], 1, 1)
    for forward, backward in zip(*[iter(transitions)]*2):
        forward_ind = 24*(forward - since).days + 2
        backward_ind = 24*(backward - since).days + 2
        data.iloc[forward_ind:backward_ind+1] = data.iloc[forward_ind:backward_ind+1].shift(-1)
        try:
            data.iloc[backward_ind] = transdata.loc[backward.strftime("%Y-%m-%d"), "3B:00:00"]
        except KeyError:
            data.iloc[backward_ind] = data.iloc[backward_ind - 1]

    data = data \
        .set_index(pd.date_range('{}-01-01'.format(years[0]),
                                 '{}-01-01'.format(int(years[-1]) + 1),
                                 closed='left', freq='1h', tz=tz)) \
        .tz_convert(pytz.utc)

    if countries is None or set(('Kosovo', 'Albania')).issubset(countries):
        # manual alterations:
        # Kosovo gets the same load curve as Serbia
        # scaled by energy consumption ratio from IEA 2012
        data['KV'] = data['RS'] * (4.8 / 27.)
        # Albania gets the same load curve as Macedonia
        data['AL'] = data['MK'] * (4.1 / 7.4)

    return data


def timeseries_opsd(years=slice("2011", "2015"), fn=None):
    """
    Read load data from OPSD time-series package.

    Parameters
    ----------
    years : None or slice()
        Years for which to read load data (defaults to
        slice("2011","2015"))

    Returns
    -------
    load : pd.DataFrame
        Load time-series with UTC timestamps x ISO-2 countries
    """

    if fn is None:
        fn = toDataDir('time_series_60min_singleindex_filtered.csv')

    load = (pd.read_csv(fn, index_col=0, parse_dates=True)
            .loc[:, lambda df: df.columns.to_series().str.endswith('_load_old')]
            .rename(columns=lambda s: s[:-len('_load_old')])
            .dropna(how="all", axis=0))

    if years is not None:
        load = load.loc[years]

    # manual alterations:
    # Kosovo gets the same load curve as Serbia
    # scaled by energy consumption ratio from IEA 2012
    load['KV'] = load['RS'] * (4.8 / 27.)
    # Albania gets the same load curve as Macedonia
    load['AL'] = load['MK'] * (4.1 / 7.4)

    # To fill the half week gap in Greece from start to stop,
    # we copy the week before into it
    start = pd.Timestamp('2015-08-11 21:00')
    stop = pd.Timestamp('2015-08-15 20:00')
    w = pd.Timedelta(weeks=1)

    if start in load.index and stop in load.index:
        load.loc[start:stop, 'GR'] = load.loc[start-w:stop-w, 'GR'].values

    # There are three missing hours in 2014 and four in 2015
    # we interpolate linearly (copying from the previous week
    # might be better)
    load['EE'] = load['EE'].interpolate()

    return load

def _upsampling_fitfunc(weights, gdp, pop):
    return weights[0] * gdp + weights[1] * pop

def _upsampling_weights(load):
    """
    Fit the weights for gdp and pop using leastsq from
    the some load data for each country in europe.

    Parameters
    ----------
    load : pd.DataFrame (index=times, columns=ISO2 country codes)

    Returns
    -------
    weights : np.array((gdp, pop), dtype=np.float)
    """

    load = load.resample('AS').sum()
    if (load.iloc[0] < 0.1 * load.iloc[1]).all():
        # Year is not complete
        load = load.iloc[1:]

    def read_eurostat(fn, extradims=[]):
        data = pd.read_csv(toDataDir(fn), thousands=' ', na_values=':')
        data = data.set_index(['TIME', 'GEO'] + extradims).unstack()['Value']
        data = data.unstack(list(range(-len(extradims), 0)))
        data.set_index(pd.to_datetime(data.index, format="%Y"), inplace=True)
        return data

    def reindex_like_load(data, load):
        data = data.stack().reindex(load.columns, level=1).unstack()
        data = data.reindex(load.index)
        data.interpolate('time', inplace=True)
        data.bfill(inplace=True)

        return data

    gdp = reindex_like_load(read_eurostat('nama_10_gdp_1_Data.csv', ['NA_ITEM']), load)
    pop = reindex_like_load(read_eurostat('demo_gind_1_Data.csv'), load)

    def normed(x):
        return x.divide(x.sum(axis=1), axis=0)

    data = pd.Panel(dict(gdp=normed(gdp['Gross domestic product at market prices']),
                         # gdpva=normed(gdp['Value added, gross']),
                         pop=normed(pop),
                         load=normed(load)))

    data.dropna(axis=2, inplace=True)

    gdp_n = np.ravel(data["gdp"])
    pop_n = np.ravel(data["pop"])
    y = np.ravel(data["load"])

    Jerr = - np.hstack((gdp_n[:,np.newaxis],
                        pop_n[:,np.newaxis]))
    def errfunc(weights, gdp, pop):
        return y - _upsampling_fitfunc(weights, gdp, pop)

    weights, cov_x, infodict, mesg, ier = \
    leastsq(errfunc, np.array((0.5, 0.5)), Dfun=lambda x,_,__: Jerr,
            args=(gdp_n, pop_n), full_output=True)

    return weights / weights.sum()

def gdppop_nuts3():
    pop = pd.read_table(toDataDir('nama_10r_3popgdp.tsv.gz'), na_values=[':'], delimiter=' ?\t', engine='python')
    pop = (pop
           .set_index(pd.MultiIndex.from_tuples(pop.pop('unit,geo\\time').str.split(','))).loc['THS']
           .applymap(lambda x: pd.to_numeric(x, errors='coerce'))
           .fillna(method='bfill', axis=1))['2014']

    gdp = pd.read_table(toDataDir('nama_10r_3gdp.tsv.gz'), na_values=[':'], delimiter=' ?\t', engine='python')
    gdp = (gdp
           .set_index(pd.MultiIndex.from_tuples(gdp.pop('unit,geo\\time').str.split(','))).loc['EUR_HAB']
           .applymap(lambda x: pd.to_numeric(x, errors='coerce'))
           .fillna(method='bfill', axis=1))['2014']

    # Swiss data
    cantons = pd.read_csv(toDataDir('ch_cantons.csv'))
    cantons = cantons.set_index(cantons['HASC'].str[3:])['NUTS']
    swiss = pd.read_excel(toDataDir('je-e-21.03.02.xls'), skiprows=3, index_col=0)
    swiss.columns = swiss.columns.to_series().map(cantons)

    pop = pop.append(pd.to_numeric(swiss.loc['Residents in 1000', 'CH04':]))
    gdp = gdp.append(pd.to_numeric(swiss.loc['Gross domestic product per capita in Swiss francs', 'CH04':]))

    return gdp, pop


def timeseries_shapes(shapes, countries, years=slice("2011", "2015"), weights=None, load=None):
    if load is None:
        load = timeseries_opsd(years)

    if weights is None:
        weights = _upsampling_weights(load=load)

    gdp, pop = gdppop_nuts3()
    nuts3 = pd.Series(vshapes.nuts3(tolerance=None, minarea=0.))
    mapping = vmapping.countries_to_nuts3()

    def normed(x): return x.divide(x.sum())

    def upsample(cntry, group):
        l = load[cntry]
        if len(group) == 1:
            return pd.DataFrame({group.index[0]: l})
        else:
            nuts3_inds = mapping.index[mapping == cntry]
            transfer = vtransfer.Shapes2Shapes(group, nuts3.reindex(nuts3_inds), normed=False).T.tocsr()
            gdp_n = pd.Series(transfer.dot(gdp.reindex(nuts3_inds, fill_value=1.).values), index=group.index)
            pop_n = pd.Series(transfer.dot(pop.reindex(nuts3_inds, fill_value=1.).values), index=group.index)
            factors = normed(_upsampling_fitfunc(weights, normed(gdp_n), normed(pop_n)))
            return pd.DataFrame(factors.values * l.values[:,np.newaxis], index=l.index, columns=factors.index)
    return pd.concat([upsample(cntry, group)
                      for cntry, group in shapes.groupby(countries)], axis=1)

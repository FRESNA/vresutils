from __future__ import division
import pandas as pd

from . import make_toModDir
toModDir = make_toModDir(__file__)

discountrate = 0.07

USD2013_to_EUR2013 = 0.7532
# [EUR/USD] ECB: https://www.ecb.europa.eu/stats/exchange/eurofxref/html/eurofxref-graph-usd.en.html # noqa: E501


def annualize(rate, lifetime):
    return rate/(1. - 1. / (1. + rate)**lifetime)


def get_full_cost_CO2(ref, CO2cost=0.,
                      filename=toModDir('data/costdata.xls'),
                      discountrate=discountrate):
    '''Return cost dataframe for different fuel types,
    including annualized capital costs 'captial', and CO2 price dependent
    variable costs 'marginal'.
    Power costs in [Eur/MW/year], Energy costs in [Eur/MWh]'''
    # costs = pd.read_csv('diw2030.csv',index_col=0, comment='#')
    costs = pd.read_excel(filename, sheetname=ref,
                          index_col=0, skiprows=2, header=0)

    costs['annualization'] = annualize(rate=discountrate,
                                       lifetime=costs['lifetime'])

    costs['capital'] = costs['annualization'] * costs['investment'] + costs['FOM']  # noqa: E501
    costs['marginal'] = (costs['variable'] +
                         (CO2cost*costs['CO2intensity']/costs['efficiency']))

    return costs

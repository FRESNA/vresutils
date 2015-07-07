import pandas as pd
import shapefile
from itertools import izip, count, chain
from operator import itemgetter
from collections import OrderedDict

from shapes import nuts1

from . import make_toModDir
toModDir = make_toModDir(__file__)

def countries_to_nuts1(series=False):
    """
    Returns a mapping from european countries to lists of their nuts1
    regions.  Some countries like Kosovo or Turkey are omitted, as
    well as a couple of islands.
    """

    excludenuts = set(('EL4', 'PT2', 'PT3', 'ES7', 'FI2', 'FR9'))
    excludecountry = set(('MT', 'TR', 'LI', 'IS', 'CY', 'KV'))
    mapcountry = dict(UK=u'GB', EL=u'GR')

    mapping = pd.Series({x: mapcountry.get(x[:2], x[:2])
                         for x in nuts1()
                         if x not in excludenuts and
                            x[:2] not in excludecountry})

    if not series:
        od = OrderedDict()
        for nuts, country in mapping.iteritems():
            od.setdefault(country, []).append(nuts)
        mapping = od

    return mapping

def iso2_to_iso3():
    """
    Extract a mapping from iso2 country codes to iso3 country codes
    from the countries dataset.
    """
    sf = shapefile.Reader(toModDir('data/ne_10m_admin_0_countries/ne_10m_admin_0_countries'))
    fields = dict(izip(map(itemgetter(0), sf.fields[1:]), count()))
    def name(rec):
        if rec[fields['ISO_A2']] != '-99':
            return rec[fields['ISO_A2']]
        elif rec[fields['WB_A2']] != '-99':
            return rec[fields['WB_A2']]
        else:
            return rec[fields['ADM0_A3']][:-1]
    return dict(chain(
        ((name(r),r[fields['ISO_A3']])
         for r in sf.iterRecords()
         if r[fields['ISO_A3']] != '-99'),
        dict(FR="FRA",
             NO="NOR").iteritems()
    ))

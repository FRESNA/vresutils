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
import networkx as nx
import random
from operator import attrgetter
from six import iteritems
from six.moves import map, zip

from . import shapes as vshapes, mapping as vmapping
from .graph import get_node_attributes
from .array import positive, negative
from .decorators import cachable

from . import make_toDataDir
toDataDir = make_toDataDir(__file__)

@cachable(version=2)
def read_kraftwerksliste(with_latlon=True):
    import pandas as pd

    kraftwerke = pd.read_csv(toDataDir('Kraftwerksliste_CSV_deCP850ed.csv'),
                             delimiter=';', encoding='utf-8', thousands='.', decimal=',')
    def sanitize_names(x):
        try:
            x = x[:x.index('(')]
        except ValueError:
            pass
        return x.replace(u'\n', u' ').strip()
    kraftwerke.columns = kraftwerke.columns.map(sanitize_names)
    def sanitize_plz(x):
        try:
            x = x.strip()
            if len(x) > 5:
                x = x[:5]
            return float(x)
        except (ValueError, AttributeError):
            return np.NAN
    kraftwerke.PLZ = kraftwerke.PLZ.apply(sanitize_plz)

    if with_latlon:
        postcodes = {pc: sh.centroid
                     for pc, sh in iteritems(vshapes.postcodeareas())
                     if sh is not None}
        kraftwerke['lon'] = kraftwerke.PLZ.map({pc: c.x for pc, c in iteritems(postcodes)})
        kraftwerke['lat'] = kraftwerke.PLZ.map({pc: c.y for pc, c in iteritems(postcodes)})
        #kraftwerke.dropna(subset=('lon','lat'), inplace=True)

    kraftwerke[u'Type'] = kraftwerke[u"Auswertung Energieträger"].map({
        u'Erdgas': u'Gas',
        u'Grubengas': u'Gas',
        u'Laufwasser': u'Hydro',
        u'Pumpspeicher': u'Hydro',
        u'Speicherwasser (ohne Pumpspeicher)': u'Hydro',
        u'Mineralölprodukte': u'Oil',
        u'Steinkohle': u'Coal',
        u'Braunkohle': u'Coal',
        u'Abfall': u'Waste',
        u'Kernenergie': u'Nuclear',
        u'Sonstige Energieträger\n(nicht erneuerbar) ': u'Other',
        u'Mehrere Energieträger\n(nicht erneuerbar)': u'Multiple'
    })

    return kraftwerke

def read_globalenergyobservatory():
    import pandas as pd
    import sqlite3

    db = sqlite3.connect(toDataDir('global_energy_observatory_power_plants.sqlite'))

    cur = db.execute(
        "select"
        "   name, type, country, design_capacity_mwe_nbr,"
        "   CAST(longitude_start AS REAL) as lon,"
        "   CAST(latitude_start AS REAL) as lat "
        "from"
        "   powerplants "
        "where"
        "   lat between 33 and 71 and"
        "   lon between -12 and 41 and"
        "   status_of_plant_itf=='Operating Fully' and"
        "   design_capacity_mwe_nbr > 0"
    )

    return pd.DataFrame(cur.fetchall(), columns=["Name", "Type", "Country", "Capacity", "lon", "lat"])

def read_eurostat_nrg113a():
    import pandas as pd

    fn = toDataDir('nrg_113a.xls')

    # Data2 is the 2013 data in which NOrway is missing, while the
    # 2012 data in Data doesn't contain Romania
    sheets = pd.read_excel(fn, sheetname=['Data', 'Data2'], thousands=',', skiprows=10, header=0, skipfooter=3, na_values=':')

    data = sheets['Data2']
    data['NO'] = sheets['Data']['NO']
    data['INDIC_NRG/GEO'] = data['INDIC_NRG/GEO'].map(lambda x: x[len('Electrical capacity, main activity producers -'):])

    data.set_index('INDIC_NRG/GEO', inplace=True)

    return data

@cachable
def read_enipedia():
    import pandas as pd
    import sparql
    import datetime

    res = sparql.query('http://enipedia.tudelft.nl/wiki/Special:SparqlExtension', """
        BASE <http://enipedia.tudelft.nl/wiki/>
        PREFIX a: <http://enipedia.tudelft.nl/wiki/>
        PREFIX prop: <http://enipedia.tudelft.nl/wiki/Property:>
        PREFIX cat: <http://enipedia.tudelft.nl/wiki/Category:>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        select ?plant_name ?fuel_used ?country ?elec_capacity_MW ?longitude ?latitude ?year_built ?status
        where {
             ?plant rdf:type cat:Powerplant .
             ?plant rdfs:label ?plant_name .
             ?plant prop:Latitude ?latitude .
             ?plant prop:Longitude ?longitude .
             ?plant prop:Primary_fuel_type ?fuel_type .
             ?fuel_type rdfs:label ?fuel_used .
             ?plant prop:Annual_Energyoutput_MWh ?OutputMWh .
             OPTIONAL{?plant prop:Generation_capacity_electrical_MW ?elec_capacity_MW }.
             OPTIONAL{?plant prop:Country ?country_link .
                      ?country_link rdfs:label ?country }.
             OPTIONAL{?plant prop:Year_built ?year_built }.
             OPTIONAL{?plant prop:Status ?status }.
        }
     """)

    def literal_to_python(l):
        if isinstance(l, tuple):
            return list(map(literal_to_python, l))
        elif l is None:
            return None
        elif l.datatype is None:
            return l.value
        else:
            parse_datetime = lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ" )
            return \
                {'decimal': float, 'double': float, 'integer': int,
                 'dateTime': parse_datetime, 'gYearMonth': parse_datetime} \
                [l.datatype[len('http://www.w3.org/2001/XMLSchema#'):]](l.value)

    df = pd.DataFrame(list(map(literal_to_python, res.fetchone())),
                      columns=["Name", "Type", "Country", "Capacity",
                               "lon", "lat", "Built", "Status"])

    return df

def backup_capacity_nuts_grid(G, plants=None):
    from shapely.geometry import Point

    if plants is None:
        plants = read_globalenergyobservatory()
    name_to_iso2 = dict((v,k) for k,v in iteritems(vmapping.iso2_to_name()))
    iso2_to_nuts = vmapping.countries_to_nuts1(series=False)

    def nutsidofaplant(x):
        p = Point(x["lon"], x["lat"])
        try:
            nregs = iso2_to_nuts[name_to_iso2[x['Country']]]
            for n in nregs:
                if G.node[n]['region'].contains(p):
                    return n
            else:
                return min(nregs, key=lambda n: G.node[n]['region'].distance(p))
        except KeyError:
            return np.NaN

    nutsids = plants.apply(nutsidofaplant, axis=1)

    return plants['Capacity'].groupby((nutsids, plants['Type'])).sum() / 1e3

def backup_capacity_nuts_from_eurostat(G, mapping=None):
    import pandas as pd

    capacities = read_eurostat_nrg113a().sum()
    areas = pd.Series(get_node_attributes(G, 'region')).map(attrgetter('area'))

    plants = read_globalenergyobservatory().groupby('Country')['Capacity'].sum()
    capacities['BA'] = plants['Bosnia and Herzegovina']
    capacities['CH'] = plants['Switzerland']
    capacities.rename(dict({'UK': 'GB', 'EL': 'GR'}), inplace=True)

    if mapping is None:
        mapping = vmapping.countries_to_nuts1()

    # rescale = lambda x: x*(capacities[x.name]/x.sum())
    rescale = lambda x: x + max(0, capacities[x.name] - x.sum())/x.count()
    return areas.groupby(mapping).apply(rescale) / 1e3

def backup_capacity_german_grid(G):
    from shapely.geometry import Point

    plants = read_kraftwerksliste()
    plants = plants[plants["Kraftwerksstatus"] == u"in Betrieb"]
    cells = {n: d["region"]
             for n, d in G.nodes_iter(data=True)
             if type(n) is int or n.isdigit()}

    def nodeofaplant(x):
        if np.isnan(x["lon"]) or np.isnan(x["lat"]):
            return random.choice(list(cells.keys()))
        p = Point(x["lon"], x["lat"])
        for n, cell in iteritems(cells):
            if cell.contains(p):
                return n
        else:
            return min(cells, key=lambda n: cells[n].distance(p))
    nodes = plants.apply(nodeofaplant, axis=1)

    capacity = plants['Netto-Nennleistung'].groupby((nodes, plants[u'Type'])).sum() / 1e3
    capacity.name = 'Capacity'

    return capacity

class CapacityClasses(object):
    def __init__(self, KB=None, G=None, classes=['Coal', 'Nuclear', 'Gas', 'Oil', 'Hydro', 'Waste'], unit='GW'):
        if KB is None:
            assert isinstance(G, nx.Graph), "Either KB or G must be specified"
            KB = vdispatch.backup_capacity_german_grid(G).reindex_axis(G.nodes(), level=0)

        conv = dict(GW=1e-3, MW=1., kW=1e3)[unit]

        self.classes = classes
        self.capacity = np.asarray(
            KB.unstack()
            .reindex_axis(KB.index.levels[0], axis=0)
            .reindex_axis(classes, axis=1).fillna(0.)
            .T
        )
        self.cumcapacity = np.cumsum(self.capacity.sum(axis=1))

    def __call__(self, Delta):
        global_Delta = Delta.sum(axis=1)

        B = np.zeros((len(self.classes),) + Delta.shape, dtype=Delta.dtype)
        C = np.zeros_like(Delta)

        deficit = global_Delta < 0
        if deficit.any():
            global_deficit = - global_Delta[deficit]

            deficit_times = np.where(deficit)[0]

            prevcum = 0
            was_sm = np.ones(global_deficit.shape, dtype=bool)
            for cum, cap, cB in zip(self.cumcapacity, self.capacity, B):
                is_sm = cum <= global_deficit

                was_and_is_sm = was_sm & is_sm
                if was_and_is_sm.any():
                    cB[deficit_times[was_and_is_sm]] += cap

                was_but_isnot_sm = was_sm & (~ is_sm)
                if was_but_isnot_sm.any():
                    cB[deficit_times[was_but_isnot_sm]] += (global_deficit[was_but_isnot_sm] - prevcum)[:,np.newaxis] * (cap / (cum - prevcum))

                was_sm = is_sm
                prevcum = cum

        surplus = ~ deficit
        if surplus.any():
            weight = positive(Delta[surplus])
            C[surplus] = (global_Delta[surplus]
                          / weight.sum(axis=1))[:,np.newaxis] * weight

        P = Delta + B.sum(axis=0) - C

        return P, B, C

def synchronized(N, mean_load=True, optimized_curtailment=True, calc_flows=True, susceptance='Y'):
    """
    Derive balancing and curtailment for a Nodes instance `N`
    according to synchronized balancing.

    Arguments
    ---------
    N : Nodes class

    Returns
    -------
    M : Nodes class
      A new Nodes instance, with synchronized balancing

    Examples
    --------
    import regions.generate
    N = synchronized(regions.generate.embedded_germany())

    Gives you a fully solved model with unconstrained synchronized balancing.
    """

    M = N.__class__(N)
    global_Delta = M.mismatch.sum(axis=1)

    deficit = global_Delta < 0
    if deficit.any():
        M.balancing = np.zeros(M.balancing.shape)
        weight = M.mean if mean_load else M.load[deficit]
        M.balancing[deficit] = (- global_Delta[deficit]
                                / weight.sum(axis=-1))[:,np.newaxis] * weight

    surplus = ~ deficit
    if surplus.any():
        M.curtailment = np.zeros(M.balancing.shape)
        #weight = M.solar[surplus] + M.wind[surplus] if optimized_curtailment else M.mean
        weight = positive(M.mismatch[surplus]) if optimized_curtailment else M.mean
        M.curtailment[surplus] = (global_Delta[surplus]
                                  / weight.sum(axis=-1))[:,np.newaxis] * weight

    M.region = "{}({})".format(M.region, "synchronized+" if optimized_curtailment else "synchronized")

    if calc_flows:
        from .flow import PTDF
        M.flows = M.injection_pattern.dot(PTDF(M.graph, susceptance=susceptance).T)

    return M

# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

from vresutils import shapes as vshapes, mapping as vmapping
from vresutils.graph import get_node_attributes
from operator import attrgetter


from vresutils import make_toModDir, cachable
toModDir = make_toModDir(__file__)

@cachable
def read_kraftwerksliste(with_latlon=True):
    import pandas as pd

    kraftwerke = pd.read_csv(toModDir('data/Kraftwerksliste_CSV_deCP850ed.csv'),
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
                     for pc, sh in vshapes.postcodeareas().iteritems()
                     if sh is not None}
        kraftwerke['lon'] = kraftwerke.PLZ.map({pc: c.x for pc, c in postcodes.iteritems()})
        kraftwerke['lat'] = kraftwerke.PLZ.map({pc: c.y for pc, c in postcodes.iteritems()})
        kraftwerke.dropna(subset=('lon','lat'), inplace=True)

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
        u'Sonstige Energieträger\n(nicht erneuerbar) ': np.NaN,
        u'Mehrere Energieträger\n(nicht erneuerbar)': np.NaN
    })

    return kraftwerke

def read_globalenergyobservatory():
    import pandas as pd
    import sqlite3

    db = sqlite3.connect(toModDir('data/global_energy_observatory_power_plants.sqlite'))

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

    fn = toModDir('data/nrg_113a.xls')

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
             ?plant prop:Status ?status .
        }
     """)

    def literal_to_python(l):
        if isinstance(l, tuple):
            return map(literal_to_python, l)
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

    df = pd.DataFrame(map(literal_to_python, res.fetchone()),
                      columns=["Name", "Type", "Country", "Capacity",
                               "lon", "lat", "Built", "Status"])

    return df[df.Status == 'Operational']

def backup_capacity_nuts_grid(G, plants=None):
    from shapely.geometry import Point

    if plants is None:
        plants = read_globalenergyobservatory()
    name_to_iso2 = dict((v,k) for k,v in vmapping.iso2_to_name().iteritems())
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
        p = Point(x["lon"], x["lat"])
        for n, cell in cells.iteritems():
            if cell.contains(p):
                return n
        else:
            return min(cells, key=lambda n: cells[n].distance(p))
    nodes = plants.apply(nodeofaplant, axis=1)

    capacity = plants['Netto-Nennleistung'].groupby((nodes, plants[u'Type'])).sum() / 1e3
    capacity.name = 'Capacity'

    return capacity

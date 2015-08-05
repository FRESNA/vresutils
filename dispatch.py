# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

from vresutils import shapes as vshapes, mapping as vmapping

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

def backup_capacity_nuts_grid(G):
    from shapely.geometry import Point

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

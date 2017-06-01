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

import numpy as np
import pandas as pd

#af : annualization factor
#ICi: Overnight investment cost / kW
#FCi: Fix O&M costs /kW/year
#VCi: Variable O&M costs /kWh
#efi: efficiency

NhoursPerYear=8760.
USD2013_to_EUR2013 = 0.7532 #[EUR/USD] # ECB: https://www.ecb.europa.eu/stats/exchange/eurofxref/html/eurofxref-graph-usd.en.html

discountrate=0.07
CO2intens=np.array([0.27,0.32,0.27,0.,0.45])/1000. # [t/kWht] # Hirth+13

#0.55,0.69 ,0.33, 0, 0.85

def annuity(n,r=discountrate):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if isinstance(n, pd.Series):
        return (1/n).where(r == 0, r/(1. - 1./(1.+r)**n))
    else:
        return np.where(r == 0, 1/n, r/(1. - 1./(1.+r)**n))

def get_cost(ref, CO2cost=0.):
    '''Return cost dictionary for different fuel types.
    Power costs in [Eur/MW/year], Energy costs in [Eur/MWh]'''
    #Gas-GT,Coal-new,Gas-CCGT,Nuclear,Lignite
    if ref=='Sch' or ref=='Schaber+12':
        # Schaber+12
        cost={'ref':'Schaber+12'}
        cost['fueltype']=np.char.array(['OCGT','Coal','CCGT','Nuclear','Lignite'])
        cost['lifetime']=np.array([25.,30,25,40,30]) #Schaber+12b
        cost['ICi']=np.array([400.,1400,650,3000,2300])  #[Euro/kW] # Schaber+12
        cost['FCi']=np.array([18.,35,18,65,40])          #[Euro/kW/year] # Schaber+12
        cost['VCi']=np.array([68.,21,44,12,13])/1000.    #[Euro/kWhel] # Schaber+12
        cost['efi']=np.array([0.38,0.46,0.60,0.33,0.43]) #[1] # Schaber+12
    elif ref=='NREL':
        # NREL+15
        cost={'ref':'NREL 15'}
        cost['fueltype']=np.char.array(['OCGT','Coal','CCGT','Nuclear'])
        cost['lifetime']=30
        cost['ICi']=np.array([866.79,3446.5,1020.74,6481.77])*USD2013_to_EUR2013 #[Euro/kW] #NREL+15
        cost['FCi']=np.array([7.3,31.65,14.48,94.68])*USD2013_to_EUR2013         #[Euro/kW/year] #NREL+15
        cost['VCi']=np.array([58.14,25.74,33.02,10.09])/1000.*USD2013_to_EUR2013 #[Euro/kWhel] #NREL+15
        cost['efi']=np.array([0.38,0.46,0.60,0.33]) # GUESS # (Schaber+Hirth)/2
        #efi_NREL_guess=np.array([0.38,0.46,0.60,0.33]) # (Schaber+Hirth)/2
    elif ref=='Hir' or ref=='Hirth+13':
        # Hirth+13
        cost={'ref':'Hirth+13'}
        cost['fueltype']=np.char.array(['OCGT','Coal','CCGT','Nuclear','Lignite'])
        cost['lifetime']=np.array([25.,25,25,50,25])
        cost['ICi']=np.array([600.,1500,1000,4000,2200]) #[Euro/kW] # Hirth+13
        cost['FCi']=np.array([7.,25,12,40,30])           #[Euro/kW/year] # Hirth+13
        cost['vai']=np.array([2.,1,2,2,1])/1000.         #[Euro/kWhel] #variable cost# Hirth+13
        cost['fui']=np.array([50.,11.5,25,3,3])/1000.    #[Euro/kWht] #fuel cost# Hirth+13
        cost['efi']=np.array([0.30,0.39,0.48,0.33,0.38]) #[1] #efficiency# Hirth+13

    elif ref=='diw' or ref=='Schroeder+12':
    #Gas-GT,Coal-new,Gas-CCGT,Nuclear,Lignite
        #diw
        cost={'ref':'diw'}
        cost['fueltype']=np.char.array(['OCGT','Coal','CCGT','Nuclear','Lignite'])
        cost['lifetime']=np.array([30.,40,30,50,40])
        cost['ICi']=np.array([400.,1300,800,4000,2000])  #[Euro/kW] # diw  #proposed values except for higher price for lignite
        cost['FCi']=np.array([15.,25,20,0,30])           #[Euro/kW/year] # diw
        cost['vai']=np.array([3.,6,4,8,7])/1000.         #[Euro/kWh] #variable cost# diw
        cost['fui']=np.array([21.6,8.4,21.6,3,2.9])/1000.   #[Euro/kWht] #fuel cost# diw
        cost['efi']=np.array([0.3,0.46,0.6,0.33,0.39]) #[1] #efficiency# diw
    #Gas-GT,Coal-new,Gas-CCGT,Nuclear,Lignite
        #diw http://hdl.handle.net/10419/80348
    elif ref=='diw2030':
        cost={'ref':'diw2030'}
        cost['fueltype']=np.char.array(['windon','windoff','solar','PHS','hydro','ror','OCGT','Coal','CCGT','Nuclear','Lignite'])
        cost['lifetime']=np.array([20.,20,20,100,100,100,30,40,30,50,40])
        cost['ICi']=np.array([1182,2506,600,2000,2000,3000,400,1300,800,4000,2000 ])  #[Euro/kW] # diw  #proposed values 2030
        cost['FCi']=np.array([35,80,25,20,20,60,15,25,20,0,30 ])              #[Euro/kW/year] # diw #proposed values 2030
        cost['vai']=np.array([1.5e-2,2e-2,1e-2,0,0,0,3,6,4,8,7 ])/1000.    #[Euro/kWh] #variable cost #diw #own assumptions for wind/solar
        cost['fui']=np.array([0.,0,0,0,0,0,21.6,8.4,21.6,3,2.9])/1000.             #[Euro/kWht] #fuel cost# diw
        cost['efi']=np.array([1,1,1,0.75,0.90,0.90,0.39,0.46,0.6,0.33,0.39 ])         #[1] #efficiency# diw
        cost['CO2int']=np.array([0.,0,0,0,0,0,0.1872,0.32,0.1872,0.,0.45 ])/1000.  # Hirth 2013

    elif ref=='iea':
        #http://www.worldenergyoutlook.org/media/weowebsite/2014/weio/WEIO2014PGAssumptions.xlsx
        #http://www.iea.org/publications/freepublications/publication/WEIO2014.pdf
        #Gas-GT,Gas-CCGT,Nuclear,Steam Coal {SUB/SUPER}critical
        # is sub/super crit coal = lignite/coal?
        # DONT provide variable costs!
        raise NotImplemented
    elif ref=='Spiecker,Weber 13':
        #http://www.sciencedirect.com/science/article/pii/S0301421513010549#bib17
        #give (large) price ranges
        #no direct fuel costs
        #but startup variable and fuel costs
        raise NotImplemented

    else:
        raise KeyError('No data available for ref={0}'.format(ref))

    cost = pd.DataFrame(cost).set_index('fueltype')

    ## Define fallbacks if the chosen source does not include all of
    ## the data

    # CO2
    if not 'CO2int' in cost:
        cost['CO2int'] = pd.Series(CO2intens, index=['OCGT','Coal','CCGT','Nuclear','Lignite'])
        # [t/kWht] from Hirth+13

    # Renewables
    # Onshore wind
    if 'windon' not in cost.index:
        cost.loc['windon',:] = pd.Series(
            {'lifetime': 20.,
             'ICi': 1e3,  #[Euro/kW]
             'FCi': 0.,   #[Euro/kW/year]
             'vai': 1.5e-5,   #[Euro/kWh] #variable cost#
             'fui': 0.,   #[Euro/kWht] #fuel cost#
             'efi': 1.,   #efficiency#
             'CO2int': 0.})

    if 'windoff' not in cost.index:
        cost.loc['windoff',:] = pd.Series(
            {'lifetime': 20.,
             'ICi': 2e3,  #[Euro/kW]
             'FCi': 0.,   #[Euro/kW/year]
             'vai': 2e-5, #[Euro/kWh] #variable cost#
             'fui': 0.,   #[Euro/kWht] #fuel cost#
             'efi': 1.,   #efficiency#
             'CO2int': 0.})

    if 'solar' not in cost.index:
        cost.loc['solar',:] = pd.Series(
            {'lifetime': 20.,
             'ICi': 1e3,  #[Euro/kW]
             'FCi': 0.,   #[Euro/kW/year]
             'vai': 1e-5, #[Euro/kWh] #variable cost#
             'fui': 0.,   #[Euro/kWht] #fuel cost#
             'efi': 1.,   #efficiency#
             'CO2int': 0.})

    # Globals
    cost['rate'] = discountrate

    # Switch units to refer to MW
    cost.loc[:,['ICi', 'FCi', 'vai', 'fui', 'CO2int']] *= 1e3

    # Derived columns
    cost['afi'] = annuity(cost['lifetime'], cost['rate'])
    cost['VCi'] = cost['vai'] + cost['fui'] / cost['efi']  #[Euro/MWhel] #total variable cost (w/o CO2 costs)
    cost['wki'] = (cost['afi'] * cost['ICi'] + cost['FCi'])
    #[Euro/MW/year] annualized fix costs
    cost['wbi'] = (cost['VCi'] + (CO2cost * cost['CO2int']/cost['efi']))
    #[Euro/MWh] variable energy cost per kWh including CO2 cost

    return cost

def get_full_cost_CO2(ref,CO2cost):
    '''Return cost dictionary for different fuel types,
    including total fix costs 'wki', and CO2 price dep. variable costs 'wbi'.
    Power costs in [Eur/MW/year], Energy costs in [Eur/MWh]'''
    cost=get_cost(ref, CO2cost=CO2cost)
    return cost



def _get_intersect(costCO2,typ1,typ2):
    '''Return the point of intersection in the (util factor, cost)-plane
    between two linear cost functions.'''
    #y=V1*x+I1; y=V2*x+I2
#   I1,I2=costCO2['wki'][np.array([typ1,typ2])]
#   V1,V2=costCO2['wbi'][np.array([typ1,typ2])]*8760.

    I1=costCO2['wki'][typ1]
    I2=costCO2['wki'][typ2]
    V1=costCO2['wbi'][typ1]*8760.
    V2=costCO2['wbi'][typ2]*8760.

    x=(I2-I1)/(V1-V2)
    y=I1+V1*x
    return np.array([x,y]).T


def get_cheapest(cost):
    '''Calculate which technology is cheapest in which utilization factor range.
    Return the (index) of the technology and its maximum (util factor, cost) positon.'''
    sortI=np.argsort(cost['wki'])
    typ=[]
    pos=[]
    ufold=0
    while sortI.size >1:# and ufold<1:
        sects=_get_intersect(cost,sortI[0],sortI[1:])
        nskip=sects[:,1].argmin()
        ufnew=sects[nskip][0]
        if not ufold<ufnew<1:
            #no further intersection within range, follow current type to uf=1
            break
        else: ufold=ufnew

        typ.append(sortI[0])
        pos.append(sects[nskip])

        sortI=sortI[1+nskip:]

    typ.append(sortI[0])
    pos.append([1,1*cost['wbi'][sortI[0]]*8760.+cost['wki'][sortI[0]]])
    return np.array(typ),np.array(pos)

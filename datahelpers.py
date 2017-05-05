import pandas as pd
import cPickle as pickle
import os

defaulthydrocapa='/home/vres/data/david/Hydro_Inflow/emil_hydro_capas.csv'
def get_hydro_capas(savefile=defaulthydrocapa):
    return pd.read_csv(savefile,index_col=0)#,names=[''])

defaultrorsharefile='/home/vres/data/david/Hydro_Inflow/run_of_shares/ror_ENTSOe_Restore2050.csv'
def get_ror_shares(savefile=defaultrorsharefile):
    return pd.read_csv(savefile,index_col=0,squeeze=True)

defaultinflowfile='/home/vres/data/david/Hydro_Inflow/hydro_all_hourly_intercubic.pickle'
def get_hydro_inflow(savefile=defaultinflowfile,recalc=False):
    """Return hydro inflow data for europe. [GWh]"""
    if recalc:
        def read_inflow(country,inflow_dir='~vres/data/david/Hydro_Inflow/'):
            return pd.read_csv(os.path.join(inflow_dir,'Hydro_Inflow_{country}.csv'.format(country=country)),parse_dates={'date':
            [0,1,2]}).set_index('date')['Inflow [GWh]']

        europe=['AT','BA','BE','BG','CH','CZ','DE','GR','ES','FI','FR','HR','HU','IE','IT','KV','LT','LV','ME','MK','NL','NO','PL','PT','RO','RS','SE','SI','SK','GB']

        hyd=pd.DataFrame({cname: read_inflow(cname) for cname in europe})
        #hyd.rename(columns={'EL':'GR','UK':'GB'}, inplace=True)

        print('resampling hydro data with cubic interpolation..')
        hydro = hyd.resample('D').interpolate('cubic')

        if True: #default norm
            normalization_factor = (hydro.index.size/float(hyd.index.size)) #normalize to new sampling frequency
        else:
            normalization_factor = hydro.sum() / hyd.sum() #conserve total inflow for each country separately
        hydro /= normalization_factor
        with open(savefile,'wb') as _file: pickle.dump(hydro,_file)

    with open(savefile,'rb') as _file:
        hydro = pickle.load(_file)

    return hydro

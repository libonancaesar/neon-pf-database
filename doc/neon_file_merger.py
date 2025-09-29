import os 
import pandas as pd 
import numpy as np
from functools import reduce
from path_finder import file_path_finder
from concurrent.futures import ProcessPoolExecutor


class file_merger(file_path_finder):
    
    def __init__(self, folder, site, var_type):
        super().__init__(folder)
        self._site = site
        self._var_type = var_type
        if self._var_type == "soil moisture":
            self._dtype = "SWS_1_minute"
        elif self._var_type == "precipitation":
            self._dtype = None
        else:
            raise ValueError("only works on soil moisture and precpitation data.")
             
    def _file_routes_finder(self):
        routes = self.files_by_site()
        routes['VER_INT'] = routes['VER'].apply(lambda x: int(x))
        if self._dtype == "SWS_1_minute":
            return routes[(routes['DESC'] == self._dtype) & (routes['SITE'] == self._site)]
        else:
            return routes[routes['SITE'] == self._site]
    
    def _convert_datetime(self, dataframe):
        dataframe['startDateTime'] = pd.to_datetime(dataframe['startDateTime'], format = "%Y-%m-%dT%H:%M:%SZ")
        dataframe['endDateTime'] = pd.to_datetime(dataframe['endDateTime'], format = "%Y-%m-%dT%H:%M:%SZ")
        dataframe = dataframe.sort_values('startDateTime', ignore_index = True)    
        return dataframe 
    
    
    def _csv_concat_helper(self, rows):
        dtypes = rows["DESC"].unique()
        print(dtypes)
        if dtypes.shape[0] == 2:
            all_month_sec = pd.concat(map(pd.read_csv,rows.loc[rows["DESC"] == "SECPRE_1min", "PATH"].tolist()))
            all_month_pri =  pd.concat(map(pd.read_csv,rows.loc[rows["DESC"] == "PRIPRE_5min", "PATH"].tolist()))
            results = all_month_sec
            
        elif (dtypes.shape[0]) and (dtypes[0] == "SECPRE_1min"):
            all_month_sec = pd.concat(map(pd.read_csv,rows["PATH"].tolist()), ignore_index=True)
            results = all_month_sec
            
        elif (dtypes.shape[0]) and (dtypes[0] == "PRIPRE_5min"):
            all_month_pri = pd.concat(map(pd.read_csv,rows["PATH"].tolist()), ignore_index=True)
            results = all_month_pri
            
        else:
            raise ValueError("Double check.")
        
        return results
    
    
    def _soil_moisture_getter(self):
        df = self._file_routes_finder()
        
        micro_sites = sorted(df['HOR'].unique()) 
        soil_moisture = {}
        ##Looping each microsite
        for i in micro_sites:
            micro_path = df.loc[df['HOR'] == i, ['HOR','VER','VER_INT','PATH']]
            ##Sort the vertical profile for each soil plot.Note: profile may differ from site to site. 
            micro_path = micro_path.sort_values('VER_INT', ignore_index = True)
            specific_depth = []     
            ##Get the vertical index
            v_position = sorted(micro_path["VER_INT"].unique())
            for v in v_position:
                ##This block returns all year dataset
                v_all_month = micro_path[micro_path['VER_INT'] == v].copy()
                #=============This block concat all soil moisture at a given depth=====
                all_month_sm = pd.concat(map(pd.read_csv,v_all_month['PATH'].tolist()))
                #======================================================================
                all_month_sm['startDateTime'] = pd.to_datetime(all_month_sm['startDateTime'], format = "%Y-%m-%dT%H:%M:%SZ")
                all_month_sm['endDateTime'] = pd.to_datetime(all_month_sm['endDateTime'], format = "%Y-%m-%dT%H:%M:%SZ")
                all_month_sm = all_month_sm.sort_values('startDateTime', ignore_index = True)
                all_month_sm = all_month_sm.rename(columns = {'VSWCMean': f'VSWCMean_{v}',
                                                              'VSWCFinalQF':f'VSWCFinalQF_{v}'})
                specific_depth.append(all_month_sm[['startDateTime', 'endDateTime',
                                                    f'VSWCMean_{v}',f'VSWCFinalQF_{v}']])
            
            specific_depth = reduce(lambda d1,d2: pd.merge(d1,d2,on=['startDateTime','endDateTime'], how = 'outer'), specific_depth)
            ##sort the merged dataframe
            specific_depth = specific_depth.sort_values(by='startDateTime') ##sort the time series  
            soil_moisture[i] = specific_depth
        return soil_moisture
    
    
    
    def _precip_getter(self): 
        routes = self._file_routes_finder()
        row = routes[(routes['DESC'] == "SECPRE_1min")|(routes["DESC"] == "PRIPRE_5min")]
        data = self._csv_concat_helper(row)
        return self._convert_datetime(data)
        
            
    def get_neon_data(self, path = None, if_save = False):
        if self._var_type == "soil moisture":
            data = self._soil_moisture_getter()
            if if_save:
                for k in list(data.keys()):
                    data[k].to_csv(os.path.join(path, f"{self._site}_sm_{k}.csv"), index = False)
                
        elif self._var_type == "precipitation":
            data = self._precip_getter()
            data.to_csv(os.path.join(path, f"{self._site}_precip.csv"), index = False)
       
        return data
 
def para_func(site):
    values_p= file_merger("E:\\AI4PF\\data\\NEON_precipitation",site,"precipitation").get_neon_data(path="E:/AI4PF/extracted_data/precip",if_save=True)
    values_sm= file_merger("E:\\AI4PF\\data\\NEON_soil_moisture",site,"soil moisture").get_neon_data(path="E:/AI4PF/extracted_data/sm",if_save=True)

def main_function():
    soil_moisture = file_path_finder("E:\\AI4PF\\data\\NEON_soil_moisture")
    precip = file_path_finder("E:\\AI4PF\\data\\NEON_precipitation")
    sm_sites = soil_moisture.files_by_site()["SITE"].unique()
    pools = ProcessPoolExecutor(max_workers = 8)
    results = list(pools.map(para_func, sorted(sm_sites)))           

if __name__ == "__main__":
    main_function()


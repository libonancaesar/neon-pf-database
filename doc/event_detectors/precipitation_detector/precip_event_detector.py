# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:54:45 2023

@author: libon
"""

import pandas as pd 
import numpy as np
import copy 
import os

from matplotlib import pyplot as plt

# This script includes all steps to process and delineate precipitation events 


def precip_loader(site: str)->pd.DataFrame:
    """Load precipitation dataset.
    
    Args:
        site (str): Four letter abbreviation of NEON sites.

    Returns:
        precip (pd.DataFrame): Load the precipitation in a dataframe.
    """
    
    # The folder where NEON precipitation data is located.
    precip_path = "E:/AI4PF/extracted_data/precip/"
    site_path = f"{site}_precip.csv"
    file_path = precip_path + site_path 
    precip = pd.read_csv(file_path, parse_dates=["startDateTime"])
    precip = precip.set_index("startDateTime").sort_index().drop(columns="endDateTime")
    return precip



def precip_flagger(data: pd.DataFrame) -> pd.DataFrame:
    """Filter precipitation data with quality flag.

    Args:
        data (pd.DataFrame): precip data.
    Raises:
        ValueError: Placeholder.

    Returns:
        pd.DataFrame: Data after quality flag.
    """
    
    data = data.copy()
    # check precip resolution 
    resolution = (data.index[1] - data.index[0]).total_seconds() / 60  
    if int(resolution) == 5:
        data_field = "priPrecipBulk"
        # Flag the time stamp when the quality flag is raised i.e., finalQF == 1 NEON.DOC.00898, p.18, eq.(26)
        flag = (data["priPrecipFinalQF"] == 1)
        
    elif int(resolution) == 1:
        data_field = "secPrecipBulk"
        # Flags the time stamp where the range quality flag is raised 
        # i.e., RangeQF == 1[fail] OR RangeQF == -1[can't run] 
        flag_range = (data["secPrecipRangeQF"].isin([1, -1]))
        # Flags the science review flag i.e., sciRvwQF == 2 OR sciRvwQF == 1
        flag_scirvw = (data["secPrecipSciRvwQF"].isin([1, 2])) # NEON.DOC.001113 p.11
        # uncertainties of the data should be smaller that threshold specified
        flag = (flag_range | flag_scirvw) 
    else:
        raise ValueError("Wrong resolution!")
    
    data.loc[flag, data_field] = np.nan
    data = data.rename(columns={data_field: "precip"})
    
    return data



def precip_resampler(data: pd.DataFrame, sample_period:str, max_nan_allowed: int)->pd.DataFrame:
    """Down sample precipitation data to desired resolution. 

    Args:
        data (pd.DataFrame): Precipitation data.
        sample_period (str): desired resampling timestamp.
        max_nan_allowed (int): The number of NaN allowed during the sampling.

    Returns:
        pd.DataFrame: Down sampled precipitation data.
    """
    data = copy.deepcopy(data["precip"])
    
    # Fill missing values or bad data points with 0 before resampling
    print(f"Precipitation BEFORE resampling at {sample_period} has {data.shape[0]} observations.")
    resample_precip = data.resample(sample_period).apply(lambda x: x.sum() if x.isnull().sum() <= max_nan_allowed else np.nan)
    print(f"Precipitation AFTER resampling at {sample_period} has {resample_precip.shape[0]} observations.")
    
    return resample_precip.sort_index()


def precip_event_separator(dt:pd.DataFrame, min_termination: float, min_duration:float = None, 
                           max_end = 5,
                           sum_amount = None, plot_storm = False)->pd.DataFrame:
    """Delineate precipitation to different events

    Args:
        dt (pd.DataFrame): precipitation data.
        min_termination (float): Minimum separation (in hours) between events or 
                                 gaps that is smaller than this threshold will not be considered as gaps.
        max_end (float, optional): Maximum days after if the precip is too long. Defaults to 5.
        min_duration (float, optional): Minimum precipitation duration to be considered as an event. Defaults to None.
        sum_amount (None, optional): Minimum precipitation amount to be considered as an event.Defaults to None.
        plot_storm (bool, optional): Event visulizations. Defaults to False.

    Returns:
        pd.DataFrame: Delineated precipitation events.
    """
    # This is function is
    # adapted from https://github.com/RY4GIT/Soil-moisture-signatures-Matlab-ver/blob/main/5_code_sig/util_EventSeparation.m
    # Originally from https://github.com/TOSSHtoolbox/TOSSH/blob/master/TOSSH_code/utility_functions/util_EventSeparation.m
    
    rainfall = copy.deepcopy(dt)
    rainfall = rainfall.reset_index()
    rainfall = rainfall.sort_values(by = "startDateTime", ascending= True)
    # check precipitation resolution 
    rainfall_res = (rainfall["startDateTime"][1] - rainfall["startDateTime"][0]).total_seconds()
    rainfall_res_to_minutes = int(rainfall_res/60) 
    # Fill nans with 0s if there is any.
    rainfall['precip_nafill'] = rainfall["precip"].apply(lambda x: 0 if pd.isna(x) else x)
    # Let's first make a copy of the data for gap detection
    gap_dtc_data = copy.deepcopy(rainfall)
    # Let's make another copy for storm detection 
    # assign 1s to storms where the intensity is smaller than a threshold
    gap_dtc_data['p_lowrain'] = gap_dtc_data['precip_nafill'].apply(lambda x: 1 if x <= 0 else 0)
    gap_dtc_data.loc[0, 'p_lowrain'] = 0    
    
    # Note: This method detects gap using numerical differentiation. i.e., diff = current_value - past_value 
    # at a step of 1, and lable the diff as the current label/index. Therefore, there is no value for the first
    # entry of the dataframe/data series. It's is an important step to ensure the start of the gap is earlier
    # than the end of the gap. This method might detect unclosed gaps i.e., begin gap index = [id1-bg, id2-bg]
    # and end gap index = [id1-ed, ].
    
    gap_dtc_data['p_lowrain_change'] = gap_dtc_data['p_lowrain'].diff(periods = 1)
    # We should remove np.nan i.e., the diff of the first entry.
    p_lowrain_change = copy.deepcopy(gap_dtc_data[['startDateTime','p_lowrain_change']].dropna())
    
    # the begin of a gap is 1 [current value i.e., no rain] - 0 [past value i.e., rain] = 1    
    gap_bg_mask = (p_lowrain_change['p_lowrain_change'] == 1) ## gap begin 
    
    # the end of a gap is 0 - 1 = -1    
    gap_ed_mask = (p_lowrain_change['p_lowrain_change'] == -1)  ## gap end
    
    gap_bg_time = p_lowrain_change.loc[gap_bg_mask, 'startDateTime'].copy() ## begin gap in datetime format
    
    # gap end is one time block ealier
    gap_ed_time = (p_lowrain_change.loc[gap_ed_mask, 'startDateTime'] - pd.Timedelta(rainfall_res_to_minutes, unit="minutes")).copy() 
    
    
    # there should be a 1 - 1 mapping from start to end of the gap index 
    if gap_bg_time.shape[0] > gap_ed_time.shape[0]:
        # in the senario [start_gap_t1, ],  [ ] 
        # i.e., only one event is detected. Then just assign the end
        # of the event being the end of the data series.
        if (gap_bg_time.shape[0] == 1) and (gap_ed_time.shape[0] == 0):  
            gap_ed_time = pd.Series(gap_dtc_data["startDateTime"].iloc[-1])
        # in the senario [start_gap_t1, start_gap_t2, start_gap_t3], [end_gap_t1, end_gap_t2,    ]
        else:
            gap_ed_time = pd.concat([gap_ed_time, pd.Series(gap_dtc_data["startDateTime"].iloc[-1])], 
                                    ignore_index=True, sort = True, axis = 0)          
    gap_info = pd.DataFrame({"start_gap_start":gap_bg_time.to_numpy(), "end_gap_start":gap_ed_time.to_numpy(),
                            "gap_duration": gap_ed_time.to_numpy() +  pd.Timedelta(rainfall_res_to_minutes, unit="minutes") - gap_bg_time.to_numpy()})
    gap_info["gap_duration"] = gap_info["gap_duration"].apply(lambda x: x.total_seconds()/ 3600) # gaps in hours
    # We'd like to throw away gaps, i.e., gaps are to small to be considered as real gaps
    gap_info = gap_info[gap_info['gap_duration'] < min_termination] # throw away these shorter gaps i.e., assign 0s
    filter_gap_dt = copy.deepcopy(gap_dtc_data).set_index("startDateTime")
    ## compute rainfall statistics
    for idx, rws in gap_info.iterrows():
        std_gp = rws['start_gap_start']
        end_gp = rws['end_gap_start']
        filter_gap_dt.loc[std_gp:end_gp, 'p_lowrain'] = 0 ## assign 0 [has rainfall] for shorter gaps
    
    # Now lets work on identifying storm events 

    storm_dt_data = copy.deepcopy(filter_gap_dt.reset_index())
    # p_lowerrain == 0 --> rainfall;  p_lowerrain == 1 --> gap
    # potential_storms == 0 ==> p_lowerrain == 1 --> gap / no rainfall / rainfall amount < threshold
    # potential_storms == 1 ==> p_lowerrain == 0 --> rainfall
    
    storm_dt_data['potential_storms'] = storm_dt_data['p_lowrain'].apply(lambda x: 1 if x <= 0 else 0) # mark stroms 
    storm_dt_data.loc[0, "potential_storms"] = 0 # assign 0 as p_lowrain == 1 ==> p_original <= threshold [no rainfall];
    storm_dt_data['potential_storm_change'] = storm_dt_data['potential_storms'].diff(periods = 1)
    ## identify storms starts from here
    storm_change = copy.deepcopy(storm_dt_data[['startDateTime','potential_storm_change', "p_lowrain"]].dropna()) 
    begin_storm_mask = storm_change["potential_storm_change"] == 1
    end_storm_mask = storm_change["potential_storm_change"] == -1
    begin_storm_time = storm_change.loc[begin_storm_mask, 'startDateTime'] 
    end_storm_time = storm_change.loc[end_storm_mask, 'startDateTime'] - pd.Timedelta(rainfall_res_to_minutes, unit="minutes")
    if begin_storm_time.shape[0] > end_storm_time.shape[0]:
    # in the senario [start_storm_t1, ],  [ ] 
        if (begin_storm_time.shape[0] == 1) and (end_storm_time.shape[0] == 0):
            end_storm_time = pd.Series(gap_dtc_data["startDateTime"].iloc[-1])
    # in the senario [start_gap_t1, start_gap_t2, start_gap_t3], [end_gap_t1, end_gap_t2,    ]
        else:
            end_storm_time = pd.concat([end_storm_time, pd.Series(gap_dtc_data["startDateTime"].iloc[-1])], 
                                ignore_index=True, sort = True, axis = 0)
    end_storm_time = end_storm_time.reset_index(drop = True)
    
    ## Store the storm information i.e., storm duration, storm intensity  
    storm_info = pd.DataFrame({"stormStartTime":begin_storm_time.to_numpy(), 
                               "stormEndTime":end_storm_time.to_numpy()})
    # check the first storm starts
    check_start = copy.deepcopy(dt)
    if (check_start[check_start.index == storm_info.iloc[0, 0]].values[0]) == 0:
        # let's extract the first storm 
        first_storm = check_start.loc[storm_info.iloc[0, 0]:storm_info.iloc[0, 1]]
        localtion = first_storm[first_storm > 0].sort_index()
        storm_info.iloc[0,0] = localtion.index[0]
    
    check_end = copy.deepcopy(dt)
    if (check_end[check_end.index == storm_info.iloc[-1, 1]].values[0]) == 0:
        # let's extract the last storm. 
        last_storm = check_end.loc[storm_info.iloc[-1, 0]:storm_info.iloc[-1, 1]]
        last_storm_location = last_storm[last_storm> 0].sort_index()
        storm_info.iloc[-1, 1] = last_storm_location.index[-1]   
             
    for idx, labels in storm_info.iterrows():
        begin_label = labels["stormStartTime"]
        end_label = labels["stormEndTime"]
        if max_end is not None:
            if (end_label - begin_label).total_seconds()/(60 * 60 * 24) >= max_end:
                end_label =  begin_label + pd.Timedelta(max_end, unit="days")
                storm_info.loc[idx, "stormEndTime"] = end_label
        storm_info.loc[idx, "stormStartValue"] =  filter_gap_dt.loc[begin_label, "precip_nafill"]
        storm_info.loc[idx,"stormEndValue"] = filter_gap_dt.loc[end_label, "precip_nafill"]
        storm_info.loc[idx, "stormSum"] = filter_gap_dt.loc[begin_label:end_label, "precip_nafill"].sum()      # mm
        storm_info.loc[idx, "stormPeakIntensity"] = filter_gap_dt.loc[begin_label:end_label, "precip_nafill"].max() # mm/10min
        storm_info.loc[idx, "stormPeakTime"] = filter_gap_dt.loc[begin_label:end_label, "precip_nafill"].idxmax()  # peak precip intensity   
        
    # Added the resolution to avoid 0 duration problem. 
    # i.e., one can detect a big precip pulse of 16mm/resol 
    # starting at 2020-03-25 14:00:00 and ends at 2020-03-25 14:00:00. The duration is 0 but indeed the duration should be 10min.
    storm_info["stormDuration"] = (storm_info["stormEndTime"] - storm_info["stormStartTime"] + pd.Timedelta(rainfall_res_to_minutes, unit="minutes")).apply(lambda x:x.total_seconds()/3600) # hours
        
    if min_duration is not None:
        storm_info = storm_info[storm_info["stormDuration"] >= min_duration]
    # minimum amount of rainfall amount
    if sum_amount is not None:
        storm_info = storm_info[storm_info["stormSum"] >= sum_amount]
    storm_info = storm_info.reset_index(drop = True)
    if (storm_info.shape[0] > 0) and (plot_storm):    
        fig, ax = plt.subplots(figsize= (15,4))   
        ax.plot(rainfall["startDateTime"], rainfall["precip"])
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Precipitation (mm/{rainfall_res_to_minutes}min)")
        for ip, lb in storm_info.iterrows():
            ax.axvline(x= storm_info.loc[ip,"stormStartTime"], c= "blue", ls = "dashed", alpha = 0.5 )
            ax.axvline(x = storm_info.loc[ip,"stormEndTime"], c= "red", ls = "dashed", alpha = 0.5 )
            ax.axvline(x = storm_info.loc[ip,"stormPeakTime"], c= "orange", ls = "solid", alpha = 0.5 )
    return storm_info

def main_func(args:list)->None:
    """Sensitive main function.

    Args:
        args (list): Parameter information.

    Returns:
        None.

    """
    site_name, min_ter = args
    raw_precip = precip_loader(site_name)
    flagged_precip = precip_flagger(raw_precip, uncertainty_threshold=0.05)
    resampled_precip = precip_resampler(flagged_precip,sample_period="10min", max_nan_allowed=5)
    precip_events = precip_event_separator(resampled_precip, min_termination=min_ter, sum_amount=None, max_end=5)
    precip_events.to_csv(f"E:/AI4PF/doc/sensitivity_results/{site_name}_min_sep_{min_ter}h.csv",index = False)
    return None

def parallel_processing(max_workers:int, list_of_site_names:list, minter:list)->None:
    """Parallel processing wrapper function.

    Args:
        max_workers (int): Number of cores to be used.
        list_of_site_names (list): name abbreviations.
        minter (list): minimum separation between events.

    Returns:
        None.
        
    """
    param_set = []
    for i in list_of_site_names:
        for j in minter:
            param_set.append([i, j])

    with ProcessPoolExecutor(max_workers = 10) as pool:
        results = list(pool.map(main_func, param_set))  
    return None

if __name__ == "__main__":
    SITES = sorted(list(set([i.split("_")[0] for i in os.listdir("E:\AI4PF\extracted_data\sm")])))
    parallel_processing(max_workers = 10, list_of_site_names = SITES, minter=list(range(1,25)))
   
        
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


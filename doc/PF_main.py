# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:53:06 2023

@author: libon
"""

import os
import pandas as pd 
from event_detectors.soil_moisture_detector import soil_moisture_event_detector as smh
from event_detectors.precipitation_detector import precip_event_detector as pcph
from joblib import Parallel, delayed
from typing import Callable, Iterable

# all sites
NEON_SITES = sorted(list(set([i.split("_")[0] for i in os.listdir("E:\AI4PF\extracted_data\sm")])))


# processing precipitation dataset
PRECIP_PARAMS = {
                    "sample_period": "10min",
                    "max_nan_allowed": 5,
                    "min_termination": 6,
                    "sum_amount":0, # change back to 2 if we want to recover 
                    "max_end": 5,
                    }
 
def process_precip_event(site_name:str, params:dict) -> pd.DataFrame:
    """process of precipitation dataset.

    Args:
        site_name (str): site name.
        params (dict): dictionary of precip parameters.

    Returns:
        pd.DataFrame: precipitation events.
    """
    raw_precip = pcph.precip_loader(site_name)
    flagged_precip = pcph.precip_flagger(raw_precip)
    resampled_precip = pcph.precip_resampler(
                                            flagged_precip,
                                            sample_period = params.get("sample_period"), # 10min
                                            max_nan_allowed = params.get("max_nan_allowed") # 5
                                            )
    precip_events = pcph.precip_event_separator(
                                            resampled_precip, 
                                            min_termination = params.get("min_termination"), # 6 hours
                                            sum_amount = params.get("sum_amount"), # 2mm/event
                                            max_end = params.get("max_end") # 5 days
                                            )
    return precip_events

# processing the soil moisture data.
SM_DATA_PARAMS = {
                "sample_period":"10min",
                "min_periods":7,
                "interpolate":True,
                "max_period":36
                }

def process_sm_data(site_name:str, params:dict) -> pd.DataFrame:
    """process soil moisture i.e., interpolation, data curation.

    Args:
        site_name (str): site name.
        params (dict): parameters associated with soil moisture data processing.

    Returns:
        pd.DataFrame: processed soil moisture data.
    """
    raw_sm = smh.sm_loader(site_name)
    flagged_sm = smh.sm_flagger(raw_sm)
    resampled_sm = smh.sm_resampler(
                                flagged_sm,
                                sample_period = params.get("sample_period"), # 10min
                                min_periods = params.get("min_periods"),  # 7
                                interploate = params.get("interpolate"), # True
                                max_period =  params.get("max_period")  #  36
                                ) #fill gaps <= 6 hours with linear interpolation 
                                
    return resampled_sm


# process the soil moisture events
SM_RESPONSE_PARAMS = {
                    "window_size":24 * 6 * 7, "percent_no_nan": 0.5, #hampel filter params
                    "iqr_const":2, "sd_scale":1.5, # outlier remover params
                    "cluster_period": 2, # peak cluster in 2 hour will be removed
                    "min_peak_sep":5, # minimum peak separations
                    "max_peak_separation":1, # searching onset time
                    "hr_pk_no_nan":1, # valid peak hours 1 hour before and 1 hour after
                    "onset_threshold":0.01, # onset threshold
                    "continous_time":24 # examine the functionality of sensor
                    }

def process_sm_events(
                        site_name:str,
                        plot_id:str, 
                        soil_moisture:pd.DataFrame, 
                        precip_events:pd.DataFrame,
                        params:dict
                          ) -> pd.DataFrame:
    """single sensor data processing

    Args:
        site_name (str): NEON name.
        plot_id (str): site name soil plot id.
        soil_moisture (pd.DataFrame): soil moisture time series.
        precip_events (pd.DataFrame): delineated precipitation events.
        params (dict): parameters for event separations

    Returns:
        pd.DataFrame: results contains onsets and peaks.
    """
   
    suffix = plot_id.split("_")[-1]
    derivative = smh.hampel_identifier(
                                    soil_moisture, 
                                    window_size= params.get("window_size"), # 24 * 6 * 7
                                    percent_no_nan = params.get("percent_no_nan") # 0.5
                                       ) # 7 days left and 7 days right
    
    outliers = smh.outlier_remover(
                                derivative,
                                iqr_const = params.get("iqr_const"), # 2
                                sd_scale = params.get("sd_scale") # 1.5
                                )
    
    # select on peak of the clustered within 2 hours, minimum peak separation is 5 hours b
    clusters = smh.cluster_remover(
                                outliers,
                                cluster_period = params.get("cluster_period"), # 2 hour
                                min_peak_sep = params.get("min_peak_sep") # 5 hours
                                )
     
    onsets = smh.onset_identifier(
                                soil_moisture,
                                clusters,
                                max_peak_separation = params.get("max_peak_separation"), # 1 day
                                hr_pk_no_nan = params.get("hr_pk_no_nan"), # 1 hour
                                onset_threshold = params.get("onset_threshold") # 0.01
                                )
    
    final_events = smh.final_event_identifier(
                                precip_events, 
                                onsets, 
                                soil_moisture,
                                continous_time = params.get("continous_time") # 24 hours
                                )
   
    final_events.to_csv(f"E:/AI4PF/doc/results_v2/{site_name}_result_{suffix}.csv", 
                        index=False)
    return None



def parallel_processing(cpu_use:int = 10) -> Callable:
    """decorator factory.

    Args:
        cpu_use (int): number of cup to use to parallelize the task
    """
    def decorator(func: Callable):
        def inner(arg:Iterable):
            if isinstance(arg, Iterable):
                # If arg is an iterable (e.g., list), apply func to each element
                return Parallel(n_jobs=cpu_use)(delayed(func)(*i) for i in arg)
            else:
                # If arg is not an iterable, apply func directly
                return func(arg)
        
        return inner
    
    return decorator


@parallel_processing(cpu_use=10)
def event_by_site(site_name:str, precip_params:dict, sm_params:dict, sm_event_params:dict)->None:
    """Get soil response to precipitation events.

    Args:
        site_name (str): Four letter NEON site abbreviation.
        precip_params (dict): precipitation events parameters. 
        sm_params (dict): soil moisture response parameters.
    Returns:
        None.
    """
    precip = process_precip_event(site_name, params = precip_params)
    soil_moisture = process_sm_data(site_name, params = sm_params)
    for i in soil_moisture.keys():
        sm_plot = soil_moisture[i]
        process_sm_events(site_name, i, sm_plot, precip, params = sm_event_params) 
        
    return None

CREATE_ITERABLE = [(i, PRECIP_PARAMS, SM_DATA_PARAMS, SM_RESPONSE_PARAMS) for i in NEON_SITES]

if __name__ == "__main__":
    event_by_site(CREATE_ITERABLE)
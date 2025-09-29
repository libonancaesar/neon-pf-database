import os
import pandas as pd 
import numpy as np
from event_detectors.soil_moisture_detector import soil_moisture_event_detector as smh
from concurrent.futures import ProcessPoolExecutor

def event_by_site(SITE_NAME:str)->None:
    """Get the precipitation events and soil responses.

    Args:
        site_name (str): Four letter NEON site abbreviation.

    Returns:
        None.

    """
    SM_DATA_PARAMS = {
                "sample_period":"10min",
                "min_periods":7,
                "interpolate":True,
                "max_period":36
                }

    # SOIL MOISTURE PORTION WHILE PARALLELING THESE WE WANT TO LOAD JUST ONECE PERSITE
    raw_sm = smh.sm_loader(SITE_NAME)
    flagged_sm = smh.sm_flagger(raw_sm)
    resampled_sm = smh.sm_resampler(flagged_sm, sample_period=SM_DATA_PARAMS["sample_period"],
                                    min_periods=SM_DATA_PARAMS["min_periods"], 
                                    interploate=SM_DATA_PARAMS["interpolate"], 
                                    max_period = SM_DATA_PARAMS["max_period"]) # fill gaps <= 6 hours with linear interpolation 
    
    pbar = list(resampled_sm.keys())   
    
    
    SM_RESPONSE_PARAMS = {
                    "window_size":24 * 6 * 7, "percent_no_nan": 0.5, #hampel filter params
                    "iqr_const":2, "sd_scale":1.5, # outlier remover params
                    "cluster_period": 2, # peak cluster in 2 hour will be removed
                    "min_peak_sep":5, # minimum peak separations
                    "max_peak_separation":1, # searching onset time
                    "hr_pk_no_nan":1, # valid peak hours 1 hour before and 1 hour after
                    # "onset_threshold":0.01, # onset threshold
                    "continous_time":24 # examine the functionality of sensor
                    }
    
    for SEP_HOUR in [2, 6, 12, 24]:
        precip_name = f"{SITE_NAME}_min_sep_{SEP_HOUR}h.csv"
        # LOAD PRECIP EVENTS DATA
        precip_events = pd.read_csv(f"E:/AI4PF/doc/sensitivity_precip_results/{precip_name}",
                                    parse_dates=["stormStartTime", "stormEndTime", "stormPeakTime"])
        for SM_THRESHOLD in [0.001, 0.01, 0.05, 0.1]:
            for j in pbar:
                
                suffix = j.split("_")[-1]
                hp_derivatives = smh.hampel_identifier(resampled_sm[j], window_size= SM_RESPONSE_PARAMS["window_size"],
                                                       percent_no_nan=SM_RESPONSE_PARAMS["percent_no_nan"]) # 7 days left and 7 days right]
                outlier_j = smh.outlier_remover(hp_derivatives, iqr_const=SM_RESPONSE_PARAMS["iqr_const"], sd_scale=SM_RESPONSE_PARAMS["sd_scale"])
                cluster_j = smh.cluster_remover(outlier_j, cluster_period=SM_RESPONSE_PARAMS["cluster_period"],
                                                min_peak_sep=SM_RESPONSE_PARAMS["min_peak_sep"]) # select on peak of the clustered within 2 hours, minimum peak separation is 5 hours
                onsets_j = smh.onset_identifier(resampled_sm[j], cluster_j,
                                                max_peak_separation=SM_RESPONSE_PARAMS["max_peak_separation"], hr_pk_no_nan=SM_RESPONSE_PARAMS["hr_pk_no_nan"], onset_threshold=SM_THRESHOLD)
                final_events = smh.final_event_identifier(precip_events,onsets_j, resampled_sm[j], continous_time=SM_RESPONSE_PARAMS["continous_time"])
                final_events.to_csv(f"E:/AI4PF/doc/sensitivity_all/{SITE_NAME}_min_sep{SEP_HOUR}h_{SM_THRESHOLD}sm{suffix}.csv", index=False)
                print(f"finished working on {j}.\n")

def parallel_processing(max_workers:int, list_of_site_names:list)->None:
    """Parallel processing wrapper function.

    Args:
        max_workers (int): Number of cores to be used.
        list_of_site_names (TYPE): DESCRIPTION.

    Returns:
        None.

    """
    with ProcessPoolExecutor(max_workers = 10) as pool:
        results = list(pool.map(event_by_site, list_of_site_names))  
    


if __name__ == "__main__":
    neon_sites = sorted(list(set([i.split("_")[0] for i in os.listdir("E:\AI4PF\extracted_data\sm")])))        
    parallel_processing(16, neon_sites)
    event_by_site("ABBY")
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:54:45 2023

@author: libon
"""

import pandas as pd 
import numpy as np
import copy 
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os
from event_detectors.precipitation_detector import precip_event_detector as pcp

def main_func(args:list)->None:
    """Sensitive main function.

    Args:
        args (list): Parameter information.

    Returns:
        None.

    """
    
    PRECIP_PARAMS = {
                    "sample_period": "10min",
                    "max_nan_allowed": 5,
                    "sum_amount":None, 
                    "max_end": 5,
                    }
    
    site_name, min_ter = args
    raw_precip = pcp.precip_loader(site_name)
    flagged_precip = pcp.precip_flagger(raw_precip)
    resampled_precip = pcp.precip_resampler(flagged_precip,sample_period=PRECIP_PARAMS["sample_period"], max_nan_allowed=PRECIP_PARAMS["max_nan_allowed"])
    precip_events = pcp.precip_event_separator(resampled_precip, min_termination=min_ter, sum_amount=PRECIP_PARAMS["sum_amount"], max_end=PRECIP_PARAMS["max_end"])
    precip_events.to_csv(f"E:/AI4PF/doc/sensitivity_precip_results/{site_name}_min_sep_{min_ter}h.csv",index = False)
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
   
        
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


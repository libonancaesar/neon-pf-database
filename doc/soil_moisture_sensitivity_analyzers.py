# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:05:15 2023

@author: libon
"""

import pandas as pd 
import numpy as np
import copy
from scipy.stats import iqr
from typing import Union




def max_porosity_by_site(site_name:str) -> Union[float, int]:
    """Get maxium porosity across all soil pits from NEON.

    Args:
        site_name (str): NEON site name.

    Returns:
        Union[float, int]: maximum porosity.
    """
    NEON_soil_physics = pd.read_excel("E:/AI4PF/doc/NEON soil properties/20220928_NRCS-NEONDataset.xlsx", 
                                  sheet_name="rdata")
    data_fields = ["siteid", "pittype", "midpointcm", "totalporosity"]
    pedon_porosity = NEON_soil_physics.loc[NEON_soil_physics["siteid"] == site_name, data_fields].copy()
    max_porosity = pedon_porosity["totalporosity"].max()
    # If max porosity does not exist or is 0 
    # just return 9999, which is a placeholder. 
    # Soil moisture will never close to 999
    # resulting in the invalidation of porosity filter.
    if np.isnan(max_porosity) or (max_porosity <= 0):
        return 9999
    return max_porosity



def sm_loader(site:str)->dict:
    """Load soil moisture data to a dataframe.
    
    Args:
        site (str): Four letter abbreviate of NEON sites.

    Returns:
        sm_data (dict): Soil moisture data for different soil plots.
    """
    sm_data = {}
    sm_plots =  [f"{site.upper()}_sm_00{i}"for i in np.arange(1,6)]
    for name in sm_plots:
        sm_data[name] = pd.read_csv("E:/AI4PF/extracted_data/sm/" + name + ".csv")
    return sm_data 


def sm_flagger(data:dict)->dict:
    """Apply the quality metrics of NEON soil moisture.

    Args:
        data (dict): Soil moisture data.

    Returns:
        process_sm (dict): Soil moisture after quality control.

    """
    data_copy = copy.deepcopy(data)
    plot_names = list(data_copy.keys())
    process_sm = {}
    site_name = plot_names[0].split("_")[0]
    max_porosity = max_porosity_by_site(site_name)
    for soil_plot in plot_names:
        specific_plot = data_copy[soil_plot].drop(columns = "endDateTime")
        # get the sensor position
        sensor_positions = sorted(list(set([i.split("_")[1] for i in specific_plot.columns[1:]])))
        specific_plot["startDateTime"] = pd.to_datetime(specific_plot["startDateTime"])

        for position in sensor_positions:                           
            # use the quality flag
            specific_plot.loc[(specific_plot[f'VSWCFinalQF_{position}'] == 1), f'VSWCMean_{position}'] = np.nan
            # remove values smaller or equal to 0 
            specific_plot.loc[(specific_plot[f'VSWCMean_{position}'] <= 0), f'VSWCMean_{position}'] = np.nan
            
            specific_plot.loc[(specific_plot[f'VSWCMean_{position}'] > max_porosity), f'VSWCMean_{position}'] = np.nan

        process_sm[soil_plot] = specific_plot.drop(columns = [f'VSWCFinalQF_{k}' for k in sensor_positions])
    return process_sm


def sm_resampler(data:pd.DataFrame, sample_period:str = "10min", min_periods:int = 7, 
                 interploate:bool = True, max_period:int = 6 * 6):
    """Resample soil moisture data to desired temporal resolution

    Args:
        data (pd.DataFrame): DESCRIPTION.
        sample_period (str, optional): Desired soil moisture temporal resolution. Defaults to "10min".
        min_periods (int, optional): Minimum data availability (None Nans) within each sampling block. Defaults to 7.
        interploate (bool, optional): If linear interpolation is applied. Defaults to False.
        max_period (int, optional): In hours. Maximum interpolation period. Defaults to None.
                                    Note: This keyword only compatible with sampl_period of 10min.
                                          Your will have to modify the code in order to using this 
                                          keyword correctly.

    Returns:
        resampled_sm (TYPE): The dataframe that contains the resampled soil moisture.

    """
    data = copy.deepcopy(data)
    # Loop over the dictionary 
    plot_locations = list(data.keys())
    # return new dict
    resampled_sm = dict()
    for k in plot_locations: 
        plot_data = data[k].set_index("startDateTime")
        # compute moving median for every soil profile
        print(f"Soil moisture {k} BEFORE resampling at {sample_period} has {plot_data.shape} obs.")
        medians = plot_data.resample(sample_period).median() #--> Compute the median of the subsampling ignoring np.nan
        counts = plot_data.resample(sample_period).count() #--> The count of group, excluding missing values. 
        resampled = medians.where(counts >= min_periods,np.nan)# --> Replace values where the condition is False.
        print(f"Soil moisture {k} AFTER resampling at {sample_period} has {resampled.shape} obs.")
        if interploate:
            resampled = resampled.interpolate(method='linear', limit = max_period) # hours of interpolation
        resampled_sm[k] = resampled
    return resampled_sm


def hampel_identifier(sm:dict, window_size:int, n_sigmas:float = 3, percent_no_nan:float = 0.5):
    """Intended to take nan derivative values. See https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d for details.
    Args:
        der (pandas.DataFrame): Soil moisture numerical derivatives.
        window_size (int): number of derivatives before and after the current derivate. Total window is (2 * window_size + 1).
        n_sigmas (int, optional): _description_. Defaults to 3.
        
    Returns:
        resampled_sm (dict): Soil moisture derivatives after hampel filter.
        
    """
    sm_copy = copy.deepcopy(sm) # soil moisture 
    der_copy = sm_copy.diff()   # soil moisture derivatives
    der_col_names = der_copy.columns
    # assign derivatives that are smaller than 0 to 0; including nan values
    outlier_index = {}
    
    # Constant scale factor, which depends on the distribution. In this case, we assume normal distribution
    k = 1.4826
    # helper method to compute MAD
    MAD = lambda x: np.nanmedian(np.abs(x - np.nanmedian(x)))
    
    for j in der_col_names:
        # get a soil moisture derivatives from a specific depth. 
        derivative_j = copy.deepcopy(der_copy[j]) 
        # a mask to only consider the derivatives that is positive
        der_id = (derivative_j > 0)  
        # replace derivative values by 0s where conditions are TRUE for hampel filter
        # np.nan <= 0 --> False (do nothing)
        # 1.5 <= 0 ---> False (do nothing)
        # -0.01 <= 0 ---> True (replaced by 0) 
        derivative_j = derivative_j.mask(derivative_j <= 0, 0) # this still contains nan values
        rolling_median = derivative_j.rolling(window = 2 * window_size + 1, center=True,
                                                 min_periods =  int((2 * window_size + 1) * percent_no_nan)).median()
        scaled_mad =  k * derivative_j.rolling(window = 2 * window_size + 1, center=True,
                                                  min_periods = int((2 * window_size + 1) * percent_no_nan)).apply(MAD)        
        diffs = np.abs(derivative_j - rolling_median)
        hp_mask = diffs > n_sigmas * scaled_mad
        
        # get soil moisture where soil moisture is not nan & pass hampel filter & soil moisture derivative is > 0
        sm_hampel = copy.deepcopy(sm_copy.loc[sm_copy[j].notna() & hp_mask & der_id, j])
        out_pd = pd.merge(sm_hampel, der_copy[j], on = 'startDateTime', how="inner", sort=True)
        out_pd = out_pd.rename(columns={j + "_x": j, j + "_y": j + "_dev"})
        outlier_index[j] = out_pd
    return outlier_index


def outlier_remover(data, iqr_const = 2, sd_scale = 1.5) -> pd.DataFrame:
    """Remove outliers based log derivatives. 

    Args:
        data (pd.DataFrame): Data from hampel filter.
        iqr_const (int, optional): _description_. Defaults to 2.
        sd_scale (int, optional): standard deviation scale. Defaults to 1.

    Returns:
        data_out (dict): potential peaks.
    """
    data = copy.deepcopy(data)                 # potential peaks [derivatives] 
    data_out = {}
    for label in data.keys():                  # now each key indicates sensor depth.
        dt  = data[label].copy()               # get the derivative data 
        dt['log'] = np.log(dt[label + "_dev"]) # transform the derivatives to log space.
        IQR = iqr(dt['log'])                   # get the iqr 
        median = dt['log'].median()            # compute the median of the sequence
        lower = median - iqr_const * IQR       # lower bound 
        higher = median + iqr_const * IQR      # the upper bound     
        iqr_filter = dt[(dt['log'] <= higher) & (dt['log'] >= lower)] # keep peaks between IQR
        sd = iqr_filter['log'].std()
        mean = iqr_filter['log'].mean()
        cutoff = np.exp(mean + sd_scale * sd)
        data_out[label] = dt.loc[dt[label + "_dev"] >= cutoff, :].copy()
        
    return data_out




def cluster_remover(potential_peaks:dict, cluster_period:float = 2, min_peak_sep:float = 5) ->dict:
    """Remove clustered peaks.

    Args:
        potential_peaks (dict): Peak dicts. Keys indicates the sensor position
        cluster_period (float, optional): Peak within this value will be removed/selected. Defaults to 1. in hours.
        min_peak_sep (float, optional): Peaks that are not separated by this will be removed. Defaults to 1.in hours.

    Returns:
        dict: left over peak. 
    """
    out_peaks = {}
    for i in potential_peaks.keys():
        pks = potential_peaks[i]
        pks_copy = copy.deepcopy(pks)
        pks_id = list(pks.index)
        indexer = 0
        final_peak = []
        while indexer < len(pks_id):
            pk_time = pks_id[indexer]
            check = True
            num_pk_within = 0
            while check:
                num_pk_within = num_pk_within + 1  
                # Index out of bound indicator 
                if (indexer + num_pk_within) < len(pks_id): 
                    current_time = pks_id[indexer + num_pk_within] 
                    # If separated by cluster_period number of hours
                    if (current_time - pks_id[indexer]).total_seconds() > (cluster_period * 60 * 60):                                                   
                        check = False                                       
                else:
                    check = False
            # No matter what num_pk_within will be added at least 1 time 
            # When check is false num_pk_within is added 1, but in reality i should not be added.         
            if num_pk_within <= 1:
                final_peak.append(pk_time)
                indexer+=1
            else: 
                # address above issues. .loc is inclusive.Therefore, clustered in 
                # number of hours should be from pk_time to the (indexer + num_pk_within - 1)th element 
                # of pk_time. After that, indexer should start from indexer + num_pk_within - 1 + 1
                max_theta_id = pks_copy.loc[pk_time:pks_id[indexer + num_pk_within - 1], i].idxmax()
                indexer = indexer + num_pk_within
        
                final_peak.append(max_theta_id)
        # now working on it is also possible that the peaks after the cluster still not separated by 
        # a fixed number of hours. 
        new_indexer = 1
        
        while new_indexer < len(final_peak):
            cur_step = final_peak[new_indexer]
            pas_step = final_peak[new_indexer - 1]
            # If peaks are separated "min_separation" hours, do nothing
            if (cur_step - pas_step).total_seconds() > (60 * 60 * min_peak_sep):
            # move to the next peak
                new_indexer +=1 
            else: 
            # select the larger peak and then delete the smaller one from the list 
                if (pks_copy.loc[cur_step, i] > pks_copy.loc[pas_step, i]):
                    # delete the current one and do nothing on the indexer
                    del final_peak[new_indexer - 1]
                else:
                    del final_peak[new_indexer]
        out_peaks[i] = pks_copy.loc[final_peak, i].copy()
    return out_peaks



def onset_identifier(soil_moisture:pd.DataFrame, peak_dict:dict,
                     max_peak_separation:int = 1, hr_pk_no_nan:int = 1, onset_threshold = 0.01):
    """Identify soil moisture onsets based on soil moisture peaks.

    Args:
        soil_moisture (pd.DataFrame): Soil moisture values.
        peak_dict (dict): Soil moisture peaks.
        max_peak_separation (int, optional): If two peaks are too far apart, then search onsets from 
                                             "Earlier peak" to "Later peak" - max_peak_separation days.
                                             Defaults to 1 day.
        hr_pk_no_nan (int, optional): Hours of soil moisture values around the peaks contains no na. Defaults to 1 hour.
        onset_threshold (float, optional): changes in soil moisture to be considered as soil moisture onset. 
                                            Defaults to 0.01.

    Returns:
        all_hubs (dict): onsets and peak information for different soil plots.

    """
    soil_moisture = copy.deepcopy(soil_moisture)
    peak_dict = copy.deepcopy(peak_dict)
    temporal_sm_resolution = (soil_moisture.index[1] - soil_moisture.index[0]).total_seconds()/60 
    all_hubs = {}
    for k in peak_dict.keys():
        peaks = copy.deepcopy(peak_dict)
        peaks_layer_k  = peaks[k].reset_index().drop(columns=k)
        # get the peak time
        peak_time = peaks_layer_k.rename(columns={"startDateTime":"peak_time"})
        antecedent_peak_time = peak_time.shift().rename(columns={"peak_time":"antecedent_peak_time"})
        # get the soil moisture data and it's numerical derivatives
        layer_soil_moisture = soil_moisture[k].to_numpy()
        # changes in soil moisture in 10min and 20min
        right_diff = soil_moisture[k].diff().shift(-1)
        right_diff_3points = soil_moisture[k].diff(periods=2).shift(-2)
        sm_and_diffs = pd.DataFrame({
                                        k:layer_soil_moisture,
                                        "right_diff":right_diff.to_numpy(),
                                        "right_diff_3points":right_diff_3points.to_numpy()
                                     }, index = soil_moisture.index)
        
        between_peak = pd.concat([antecedent_peak_time, peak_time], axis = 1)
        onset_hub = []
        # index is sequential 
        for row, value in between_peak.iterrows():
            start = value["antecedent_peak_time"]
            end = value["peak_time"]
            if pd.isna(start):
                # assign to the very beginning of the time series.
                start = sm_and_diffs.index[0]
            if (end - start).total_seconds()/(60 * 60) >= max_peak_separation * 24:
            # only search the start 1 days ago
                start = end - pd.Timedelta(max_peak_separation * 24 * 60,  unit = "minutes")
                
            count_left_right = hr_pk_no_nan * 60  
            # left "hr_pk_no_nan" hour soil moisture data
            left = sm_and_diffs.loc[end - pd.Timedelta(count_left_right,  unit = "minutes"):
                                    end - pd.Timedelta(temporal_sm_resolution, unit="minutes"), k]
            # right "hr_pk_no_nan" hour soil moisture data
            right = sm_and_diffs.loc[end + pd.Timedelta(temporal_sm_resolution,  unit = "minutes"):
                                     end + pd.Timedelta(count_left_right, unit="minutes"), k]
            
            if (left.isna().sum() == 0) and (right.isna().sum() == 0):
                diff_between = sm_and_diffs.loc[start + pd.Timedelta(temporal_sm_resolution, unit = "minutes"):
                                                end - pd.Timedelta(temporal_sm_resolution, unit = "minutes"), :]
                # soil response / onset is defined as the increase of 
                # soil moisture at least by at least onset_threshold
                sm_onsets = diff_between[(diff_between["right_diff"] >= onset_threshold) |
                                         (diff_between["right_diff_3points"] >= onset_threshold)]
                # it is possible that we don't detect the soil response   
                if not sm_onsets.empty:
                    onset_time = sm_onsets[k].idxmin() # if multiple onset is detected
                                                       # we only select the onset with the smallest soil moisture
                    onset_sm = sm_onsets[k].min()      
                          
                    onset_left_ = sm_and_diffs.loc[onset_time - pd.Timedelta(count_left_right,  unit = "minutes"):
                                                   onset_time - pd.Timedelta(temporal_sm_resolution, unit="minutes"), k]
                    
                    onset_right_ = sm_and_diffs.loc[onset_time + pd.Timedelta(temporal_sm_resolution,  unit = "minutes"):
                                                    onset_time + pd.Timedelta(count_left_right, unit="minutes"), k]
                    
                    # soil moisture 1 week ago 
                    sm_7_ahead = sm_and_diffs.loc[onset_time - pd.Timedelta(7, unit = "days"): 
                                                  onset_time - pd.Timedelta(temporal_sm_resolution, unit = "minutes"), k].mean()
                    # soil moisture 2 weeks ago
                    sm_14_ahead = sm_and_diffs.loc[onset_time - pd.Timedelta(14, unit = "days"): 
                                                  onset_time - pd.Timedelta(temporal_sm_resolution, unit = "minutes"), k].mean()
                  
                    
                    if (onset_left_.isna().sum() == 0) and (onset_right_.isna().sum() == 0):
                        onset_hub.append([onset_time, end, onset_sm, sm_and_diffs.loc[end, k], 
                                          sm_7_ahead, sm_14_ahead])
          
        onset_hub = pd.DataFrame(onset_hub, columns = ['smOnsetTime','smPeakTime',
                                                       "smOnset", "smPeak", 
                                                       "sm7DaysMean", "sm14DaysMean"])
        all_hubs[k] = onset_hub
    return all_hubs   



def final_event_identifier(event:pd.DataFrame, peaks:dict) -> pd.DataFrame:
    """Classify different response of precipitation events.

    Args:
        event (pd.DataFrame): Precipitation events.
        peaks (dict): Peaks detected.

    Returns:
        _type_: _description_
    """
    new_event = copy.deepcopy(event)
    new_event["laterStormStartTime"] = event["stormStartTime"].shift(-1)
    sensor_position = list(peaks.keys()) 
    ## pre-allocate some space for each sensor depth 
    for k in sensor_position:
        depth = k.split("_")[1]
        new_event[f"smOnsetTime_{depth}"] = np.nan
        new_event[f"smPeakTime_{depth}"] = np.nan
        new_event[f"smOnset_{depth}"] = np.nan   
        new_event[f"smPeak_{depth}"] = np.nan    
        new_event[f"sm7DaysMean_{depth}"] = np.nan       
        new_event[f"sm14DaysMean_{depth}"] = np.nan 
    
    for j in np.arange(len(sensor_position)):
        good_sm_events = peaks[sensor_position[j]].copy()
        # layer_specific_events = copy.deepcopy(new_event)
        # layer_specific_events["smOnsetTime"] = np.nan
        # find peaks between two rainfall onsets 
        what_depth = sensor_position[j].split("_")[1]
        for index, rows in new_event.iterrows(): 
    
            if pd.isna(rows["laterStormStartTime"]):
                between_ = good_sm_events[(good_sm_events["smOnsetTime"] >= rows["stormStartTime"])]
            else:
                between_ = good_sm_events[(good_sm_events["smOnsetTime"] >= rows["stormStartTime"]) &
                                    (good_sm_events["smOnsetTime"] < rows["laterStormStartTime"]) ]
                
            # To do -> mark the peaks precip events when there is soil moisture response.   
            if between_.shape[0] > 1:
                # select the ealier onset and it's associated peak
                selected_time = between_["smOnsetTime"].min()
                selected_time = pd.to_datetime(selected_time)
                # find information associated with that time 
                useful_info = good_sm_events.loc[good_sm_events["smOnsetTime"] == selected_time, :].values.tolist()
                new_event.loc[index, f"smOnsetTime_{what_depth}": f"sm14DaysMean_{what_depth}"] = useful_info[0]

                
            elif between_.shape[0] == 1:
                selected_time = between_["smOnsetTime"].values[0]
                selected_time = pd.to_datetime(selected_time)
                useful_info = good_sm_events.loc[good_sm_events["smOnsetTime"] == selected_time, :].values.tolist()    
                new_event.loc[index, f"smOnsetTime_{what_depth}": f"sm14DaysMean_{what_depth}"] = useful_info[0]
            else:
                continue
            
    return new_event.drop(columns="laterStormStartTime")


if __name__ == "__main__":
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



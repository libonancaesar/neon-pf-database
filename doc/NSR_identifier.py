import pandas as pd 
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import copy

# Note: warning might show as mid night data is saved as 2019-03-08 and no AM or PM indicator
# currently, i dont think such warning matters a lot on other analysis.

def sort_key(col_name:str) -> int:
    """get the number of sensor depth.

    Args:
        col_name (str): column name.

    Returns:
        int: number of sensor name.
    """
    return int(col_name.split("_")[1])
    

def get_pf_info(df_series:pd.Series) -> list:
    """classify non sequential flow based on a series of response.

    Args:
        df_series (pd.Series): onset time series.

    Returns:
        list: location where pf occurs.
    """
    pf_info = []
    index = df_series.index.tolist()
    # we know that PF occurs 
    for j in np.arange(len(index)):
        for s in np.arange(j + 1, len(index)):
            top_layer = df_series.at[index[j]]
            bottom_layer = df_series.at[index[s]]
            if (bottom_layer - top_layer).total_seconds() < 0:
                top_str = index[j].split("_")[1]
                bottom_str= index[s].split("_")[1]
                ids = f"{top_str}vs{bottom_str}_PF_{bottom_str}"
                pf_info.append(ids)
    return  pf_info





def sort_helper(row:pd.Series) -> tuple:
    """A helper function to classify the events to different flow types.

    Args:
        row (pd.Series): Details see pandas apply methods.

    Returns:
        classification (str): The classified results.
        where_pf_occurs (list): When sensor depth does the flow occurs.

    """
    classification = "notApplicable"
    where_pf_occurs = np.nan
    onset_time_col = sorted([i for i in row.index if "smOnsetTime" in i], key = sort_key)
    onset_time = row.loc[onset_time_col].dropna()
    if onset_time.shape[0] > 1:
        classification = "SequentialFlow"
        cal_diff = onset_time.diff().apply(lambda x: x.total_seconds())
        if (cal_diff < 0).values.any():
            classification = "nonSequentialFlow"
            where_pf_occurs = get_pf_info(onset_time)
    else:
        check_sensor = row[row.isin(["noResponse", "sensorResponded"])]
        # at least two sensors are working
        if check_sensor.shape[0] >= 2:
            classification = "noResponse"
    return classification, where_pf_occurs


def non_sequential_detection(data:pd.DataFrame) -> pd.DataFrame:
    """Detect nonsequential flow.

    Args:
        data (pd.DataFrame): Precip onsets and soil moisture onset info etc.

    Returns:
        data (pd.DataFrame): Detected flow types.

    """
    data_copy = data.copy()
    data_copy = data_copy.replace(['-9999', '-9999.0',
                         -9999, -9999.0], np.nan) # replace values with nan 
    for i in data_copy.columns:
        if "Time" in i:
            data_copy[i] = data_copy[i].apply(pd.to_datetime)
    
    data_copy[["flowTypes", "flowPosition"]] = data_copy.apply(sort_helper, axis=1, result_type = "expand")
    return data_copy



if __name__ == "__main__":
   
    root = "E:/AI4PF/doc/results/"
    list_files = os.listdir(root)
    need_var = ["stormSum", "stormPeakIntensity","stormDuration",
                "flowTypes"]
    hubs = []
    for j in list_files:
        test_case = pd.read_csv(root+j)
        classified = non_sequential_detection(test_case)
        print(classified["flowTypes"])
        
    #     hubs.append(classified[need_var].copy())
    # all_data_hub = pd.concat(hubs,ignore_index=True,axis=0)
    # fig, ax = plt.subplots(1,3,figsize = (15,7))
    # sns.boxplot(data=all_data_hub, x="flowTypes",
    #             y="stormPeakIntensity",ax = ax[0])
    # sns.boxplot(data=all_data_hub, x="flowTypes",
    #             y="stormSum",ax = ax[1])
    # sns.boxplot(data=all_data_hub, x="flowTypes",
    #             y="stormDuration",ax = ax[2])











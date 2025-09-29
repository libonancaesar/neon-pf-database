import pandas as pd 
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def _sort_helper(row, col_names):
    """A helper function to classify the events to different flow types.

    Args:
        row (pd.Series): Details see pandas apply methods.
        col_names (str): The columns name of the input.

    Returns:
        classification (str): The classified results.
        where_pf_occurs (list): When sensor depth does the flow occurs.

    """
    classification = None
    where_pf_occurs = []
    temp_series = row.loc[col_names].dropna()
    if temp_series.shape[0] == 0:
        # precipitation did non induce soil moisture response
        classification = "notApplicable"
        where_pf_occurs = ["noApplicable"]
        # only one sensor responded
    if temp_series.shape[0] == 1:
        classification = "noResponse"
        where_pf_occurs = [temp_series.index[0].split("_")[1]]
    # any paire has non-sequential
    if temp_series.shape[0] > 1:
        # take the differences between each row 
        # if any non-negative values exist
        # then non-sequential flow occurs
        # current - past labeled as current
        cal_diff = temp_series.diff().apply(lambda x: x.total_seconds())
        if (cal_diff < 0).values.any():
            classification = "nonSequentialFlow"
            list_of_series_index = temp_series.index.tolist()
            # we know that PF occurs 
            for j in np.arange(len(list_of_series_index)):
                for s in np.arange(j+1, len(list_of_series_index)):
                    top_layer = temp_series.at[list_of_series_index[j]]
                    bottom_layer = temp_series.at[list_of_series_index[s]]
                    if (bottom_layer - top_layer).total_seconds() < 0:
                        top_str = list_of_series_index[j].split("_")[1]
                        bottom_str= list_of_series_index[s].split("_")[1]
                        ids = f"{top_str}vs{bottom_str}_PF_{bottom_str}"
                        where_pf_occurs.append(ids)
        else:
            classification = "SequentialFlow"

            where_pf_occurs = [i.split("_")[1] for i in temp_series.index]
        
    return classification, where_pf_occurs


def non_sequential_detection(data:pd.DataFrame) -> pd.DataFrame:
    """Detect nonsequential flow.

    Args:
        data (pd.DataFrame): Precip onsets and soil moisture onset info etc.

    Returns:
        data (pd.DataFrame): Detected flow types.

    """
    data = data.copy()
    # sensor numbers 
    assert len(data.columns[8:]) % 6 == 0, "Wrong column number"
    num_sensors = int(len(data.columns[8:])/6)
    sensor_cols = [f"smOnsetTime_50{i}" for i in np.arange(1, num_sensors + 1)]
    for j in sensor_cols:
        data[j] = pd.to_datetime(data[j])
    data[["flowTypes", "flowPosition"]] = data.apply(_sort_helper, args=(sensor_cols,) 
                                                     ,axis=1, result_type = "expand")
    return data



def plots(data:pd.DataFrame) -> None:
    """Plot drawers that can be completed later.

    Args:
        data (pd.DataFrame): DESCRIPTION.

    Returns:
        None: DESCRIPTION.

    """
    pass


if __name__ == "__main__":
    root = "E:/AI4PF/doc/sensitivity_all/"
    list_files = os.listdir(root)
    need_var = ["stormSum", "stormPeakIntensity","stormDuration",
                "flowTypes"]
    hubs = []
    for j in list_files:
        test_case = pd.read_csv(root+j)
        classified = non_sequential_detection(test_case)
        hubs.append(classified[need_var].copy())
    all_data_hub = pd.concat(hubs,ignore_index=True,axis=0)
    fig, ax = plt.subplots(1,3,figsize = (15,7))
    sns.boxplot(data=all_data_hub, x="flowTypes",
                y="stormPeakIntensity",ax = ax[0])
    sns.boxplot(data=all_data_hub, x="flowTypes",
                y="stormSum",ax = ax[1])
    sns.boxplot(data=all_data_hub, x="flowTypes",
                y="stormDuration",ax = ax[2])











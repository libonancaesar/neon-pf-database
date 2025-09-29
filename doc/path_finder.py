# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:55:51 2023

@author: libon
"""

import glob 
import os 
import pandas as pd


class file_path_finder:
    
    def __init__(self, folder):
        """
        Store the NEON soil moisture folder name.
        
        Parameters
        ----------
        folder : str
            folder name that stores soil moisture dataset.
        """
        self._folder = folder
    
    
    def _find_csvs(self):
        """
        Find all csv files
        
        Returns
        -------
        list_of_path : list
            path of csv files.

        """
        list_of_path = list(glob.glob(os.path.join(self._folder,'**/*.csv'),recursive=True))
        return list_of_path

    
    
    def files_by_site(self):
        """
        Get file path and store them in a pandas.DataFrame
        
        Returns:
            A pandas.DataFrame that stores file site name, domain number .etc 
        """
        find_csvs = self._find_csvs()
        file_list = []
        df_names = ['NEON', 'DOM', 'SITE', 'DPL', 'PRNUM', 'REV', 'HOR',
                    'VER', 'TMI', 'DESC', 'YYYY-MM', 'PKGTYPE', 'GENTYPE', 'FILETYPE', 'PATH']
        for i in find_csvs:
            ## last string literal after "\\" should be a .csv file
            split_string = i.split("\\")[-1]
            split_list = split_string.split(".")
            ## refer to https://data.neonscience.org/file-naming-conventions for the abbreviation. 
            if len(split_list) == 14:
                file_list.append(split_list + [i])
        file_list = pd.DataFrame(file_list, columns=df_names)
        return file_list
    
    
    
    
if __name__ == "__main__":
    sm_path = "E:\\AI4PF\\data\\NEON_soil_moisture"
    precip_path = "E:\\AI4PF\\data\\NEON_precipitation"
    sm_finder = file_path_finder(sm_path)
    precip_finder = file_path_finder(precip_path)
    print(sm_finder.files_by_site())
    print(precip_finder.files_by_site())
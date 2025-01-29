"""
"""

__all__ = ["load_sim"]

import pandas as pd
import numpy as np

#######################################################################################

def load_sim(filename, debug=False):
    """
    Load simulation .csv data files
    """
    data = pd.read_pickle(filename)

    if debug:
        for series_name, series in data.items():
            print(series_name + ": ", np.array(series.tolist()).shape)
   
    return data

#######################################################################################
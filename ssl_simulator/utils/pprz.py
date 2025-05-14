"""
"""

__all__ = [
    "pprz_angle",
    "load_pprz_data", 
    "get_pprz_idx",
]

import numpy as np
import pandas as pd

#######################################################################################

def pprz_angle(theta_array):
    """
    Convert an angle from standard mathematical coordinates to 
    Paparazzi UAV convention.

    Parameters
    ----------
    theta_array : np.ndarray
        Input angles in radians.

    Returns
    -------
    np.ndarray
        Converted angles in radians.
    
    Notes
    -----
    - The Paparazzi UAV convention defines 0 radians as pointing north (upward),
      whereas standard mathematical convention defines 0 radians as pointing east (rightward).
    - This function shifts the angle by -theta + π/2 to align with the Paparazzi convention.
    """
    return -theta_array + np.pi / 2

def load_pprz_data(filename: str, t0: float, tf: float = None, sep: str = "\t", 
              time_label: str = "Time") -> pd.DataFrame:
    """
    Load data from a Paparazzi .csv file, filtering it based on time range.

    Parameters
    ----------
    filename : str
        The path to the CSV file.
    t0 : float
        The start time for the data filter.
    tf : float, optional
        The end time for the data filter (default is None, which means no upper 
        time filter).
    sep : str, optional
        The delimiter used in the CSV file (default is tab-separated).
    time_label : str, optional
        The column name that represents time in the dataset (default is "Time").

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the filtered data.
    """
    data = pd.read_csv(filename, sep=sep)
    if tf is None:
        data = data.loc[data[time_label] >= t0]
    else:
        data = data.loc[(data[time_label] >= t0) & (data[time_label] <= tf)]
    return data

def get_pprz_idx(data: pd.DataFrame, t: float, time_label: str = "Time") -> int:
    """
    Get the index of the first row where the time column is greater than or equal to `t`.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the time series data.
    t : float
        The target time value to find in the DataFrame.
    time_label : str, optional
        The column name representing time (default is "Time").

    Returns
    -------
    int
        The index of the first row where `time_label` is greater than or equal to `t`.

    Raises
    ------
    ValueError
        If the DataFrame is empty or if no valid index is found.
    KeyError
        If `time_label` is not found in the DataFrame.

    Example
    -------
    >>> df = pd.DataFrame({"Time": [0, 1, 2, 3, 4, 5]})
    >>> get_idx(df, 2.5)
    3
    """
    if time_label not in data.columns:
        raise KeyError(f"Column '{time_label}' not found in DataFrame.")

    if data.empty:
        raise ValueError("The DataFrame is empty.")

    filtered_data = data[data[time_label] >= t]
    
    if filtered_data.empty:
        raise ValueError(f"No index found where '{time_label}' >= {t}.")

    return filtered_data.index[0]

#######################################################################################
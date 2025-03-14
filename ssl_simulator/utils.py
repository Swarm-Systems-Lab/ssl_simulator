"""
"""

__all__ = [
    "createDir", 
    "create_dir", 
    "load_data",
    "get_idx",
    "parse_kwargs"]

import os
import pandas as pd

#######################################################################################

def createDir(directory: str, verbose: bool = True) -> None:
    create_dir(directory, verbose)

def create_dir(directory: str, verbose: bool = True) -> None:
    """
    Create a new directory if it doesn't already exist.

    Parameters
    ----------
    directory : str
        The path of the directory to create.
    verbose : bool, optional
        Whether to print status messages (default is True).

    Returns
    -------
    None
    """
    try:
        os.mkdir(directory)
        if verbose:
            print(f"Directory '{directory}' created!")
    except FileExistsError:
        if verbose:
            print(f"The directory '{directory}' already exists!")

def load_data(filename: str, t0: float, tf: float = None, sep: str = "\t", 
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

def get_idx(data: pd.DataFrame, t: float, time_label: str = "Time") -> int:
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


def parse_kwargs(kwargs_input, kwargs_default):
    """
    Merge user-provided keyword arguments with default values.

    Parameters
    ----------
    kwargs_input : dict  
        Dictionary containing user-specified keyword arguments.  
    kwargs_default : dict  
        Dictionary containing default keyword arguments.  

    Returns
    -------
    dict  
        A dictionary where user-specified values override the defaults, while 
        preserving unspecified default values.  
    """
    kwargs = kwargs_default.copy()  # Avoid modifying the original default dictionary
    kwargs.update({k: v for k, v in kwargs_input.items() if k in kwargs_default})
    return kwargs

#######################################################################################
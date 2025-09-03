"""
"""

__all__ = [
    "load_sim",
    "load_class",
    "first_larger_index"
]


import os
import json
import numpy as np
import pandas as pd
import importlib

from ssl_simulator import check_file_size, json_to_dict, print_dict
from ssl_simulator.components.scalar_fields import ScalarField

#######################################################################################

def load_sim(filename, debug=False, max_size_mb=100, verbose=False):
    check_file_size(filename, max_size_mb=max_size_mb)

    settings, skiprows = _load_settings_line(filename)
    df = pd.read_csv(filename, skiprows=skiprows)

    data_dict = _parse_dataframe(df)

    if debug:
        _debug_print(settings, data_dict, verbose)

    return (data_dict, settings) if settings else data_dict

def load_class(module_name: str, class_name: str, base_class=None, **init_kwargs):
    """
    Load and instantiate a class given its module and name.
    
    Args:
        module_name (str): The full dotted path to the module (e.g., 'ssl_simulator.components.scalar_field').
        class_name (str): The name of the class to load from the module.
        base_class (type, optional): If provided, the loaded class must inherit from this base class.
        **init_kwargs: Keyword arguments to pass to the class constructor.

    Returns:
        object: An instance of the specified class.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
        TypeError: If the class is not a subclass of the given base_class.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")

    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"Module '{module_name}' does not contain a class named '{class_name}'.")

    if base_class and not issubclass(cls, base_class):
        raise TypeError(f"Class '{class_name}' is not a subclass of '{base_class.__name__}'.")

    return cls(**init_kwargs)

def first_larger_index(times, x, epsilon=1e-8):
    """
    Return the first index where time > x.
    If no such index exists, returns None.
    """
    for i, t in enumerate(times):
        if t >= x-epsilon:
            return i
    return None

#######################################################################################
# --- Helper functions ---

def _load_settings_line(filename):
    with open(filename, 'r') as f:
        first_line = f.readline()
        if first_line.startswith("# SETTINGS:"):
            json_str = first_line[len("# SETTINGS:"):].strip()
            settings = json_to_dict(json_str)
            return settings, 1
    return None, 0

def _parse_dataframe(df):
    labels = df.columns.drop("time").tolist()
    data_dict = {"time": df["time"].to_numpy()}

    for col in labels:
        try:
            data_dict[col] = np.stack(df[col].apply(lambda x: np.array(json.loads(x))).to_numpy())
        except Exception:
            data_dict[col] = df[col].to_numpy()

    return data_dict

def _debug_print(settings, data_dict, verbose=False):
    if settings:
        print("------------------ SETTINGS ------------------")
        print_dict(settings, verbose=verbose)
        print("----------------------------------------------")

    print("------------------- DATA ---------------------")
    print_dict(data_dict)
    print("----------------------------------------------")

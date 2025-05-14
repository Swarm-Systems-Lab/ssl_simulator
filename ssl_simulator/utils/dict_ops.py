"""
"""

__all__ = [
    "parse_kwargs",
    "dict_to_json",
    "json_to_dict",
    "print_dict",
]

import json
import numpy as np

#######################################################################################

def parse_kwargs(kwargs_input, kwargs_default):
    """
    Merge user-provided keyword arguments with default values.

    Args:
        kwargs_input (dict): Dictionary containing user-specified keyword arguments.
        kwargs_default (dict): Dictionary containing default keyword arguments.

    Returns:
        dict: A dictionary where user-specified values override the defaults,
        while preserving unspecified default values.
    """
    kwargs = kwargs_default.copy()  # Avoid modifying the original default dictionary
    kwargs.update({k: v for k, v in kwargs_input.items() if k in kwargs_default})
    return kwargs

def dict_to_json(ditc, dump=False):
    """ 
    Convert a dict to JSON-compatible form, handling arrays, classes, and class configs.
    """

    def is_custom_class(obj):
        cls = obj.__class__
        return cls.__module__ not in ["builtins", "numpy"]
        
    def convert(value):
            if isinstance(value, np.ndarray):
                return json.dumps(value.tolist())
            
            elif isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            
            elif isinstance(value, list):
                return [convert(v) for v in value]
            
            elif is_custom_class(value):
                result = {"__class__": value.__class__.__name__}
                if hasattr(value, "get_config") and callable(value.get_config):
                    result["__params__"] = convert(value.get_config())
                return result
            
            else:
                return value
            
    result = {k: convert(v) for k, v in ditc.items()}
    return json.dumps(result) if dump else result

def json_to_dict(json_str):
    """Recursively convert a JSON string to a Python structure with np.ndarrays where applicable."""
    def convert(value):
        if isinstance(value, list):
            # Check if it's a list of lists (likely a 2D array)
            if all(isinstance(v, (int, float, bool, list)) for v in value):
                try:
                    return np.array(value)
                except Exception:
                    return [convert(v) for v in value]
            else:
                return [convert(v) for v in value]

        elif isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}

        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
                return convert(parsed)
            except Exception:
                return value

        else:
            return value

    raw = json.loads(json_str)
    return convert(raw)

def print_dict(d, indent=0, verbose = False):
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            print_dict(v, indent + 1, verbose=verbose)
        elif isinstance(v, np.ndarray):
            if verbose:
                print(f"{pad}{k}: shape {v.shape} -> {v.tolist()}")
            else:
                print(f"{pad}{k}: shape {v.shape}")
        else:
            print(f"{pad}{k}: {v}")

#######################################################################################
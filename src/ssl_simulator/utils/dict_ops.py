import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

#######################################################################################


def safe_assign(target, source, source_name="dict"):
    """Helper to check for key conflicts during dictionary assignments."""
    for key, value in source.items():
        if key not in target:
            raise KeyError(f"Key '{key}' from {source_name} not found in data.")
        target[key] = value


def safe_update(target, source, source_name="dict"):
    """Helper to check for key conflicts when updating a dictionary."""
    conflict_keys = target.keys() & source.keys()  # intersection of keys
    if conflict_keys:
        raise KeyError(f"Key conflict detected when updating from {source_name}: {conflict_keys}")
    target.update(source.copy())


def validate_dict_attributes(obj, attr_names):
    """
    Validate that specified attributes are dictionaries and that callable items have __call__ method.

    Parameters
    ----------
    obj : object
        The object whose attributes should be validated
    attr_names : list of str
        Names of attributes to validate

    Raises
    ------
    TypeError
        If any attribute is not a dict or if callable items lack __call__ method
    """
    attrs_to_check = {name: getattr(obj, name) for name in attr_names}

    for attr_name, attr_value in attrs_to_check.items():
        if not isinstance(attr_value, dict):
            raise TypeError(
                f"{obj.__class__.__name__}.{attr_name} must be a dict, "
                f"but got {type(attr_value).__name__} instead."
            )

        # Validate that custom class instances have __call__ method
        # (exclude built-in types and numpy arrays/matrices)
        for key, value in attr_value.items():
            value_type = type(value)
            # Check if it's a custom class (not built-in, not numpy)
            is_custom_class = not isinstance(
                value, (int, float, str, bool, list, dict, tuple, set, type(None))
            ) and value_type.__module__ not in ["builtins", "numpy"]

            if is_custom_class and not callable(value):
                raise TypeError(
                    f"\n{'=' * 80}\n"
                    f"VALIDATION ERROR in {obj.__class__.__name__}.{attr_name}\n"
                    f"{'=' * 80}\n"
                    f"Key: '{key}'\n"
                    f"Type: {value_type.__name__} (from module: {value_type.__module__})\n"
                    f"Value: {value}\n\n"
                    f"Problem: Custom class instances in '{attr_name}' must be callable or use a lambda.\n\n"
                    f"Solutions:\n"
                    f"  1. Wrap the value in a lambda:\n"
                    f"     self.{attr_name}['{key}'] = lambda: {value}\n\n"
                    f"  2. Make the class callable by adding a __call__ method:\n"
                    f"     class {value_type.__name__}:\n"
                    f"         def __call__(self):\n"
                    f"             return # return appropriate value\n\n"
                    f"  3. Store it as a direct value (if it's numpy-compatible or built-in type)\n"
                    f"{'=' * 80}\n"
                )


def parse_kwargs(kwargs_input, kwargs_default):
    """
    Merge user-provided keyword arguments with default values.

    Args:
        kwargs_input (dict): Dictionary containing user-specified keyword arguments.
        kwargs_default (dict): Dictionary containing default keyword arguments.

    Returns
    -------
        dict: A dictionary where user-specified values override the defaults,
        while preserving unspecified default values.
    """
    kwargs = kwargs_default.copy()  # Avoid modifying the original default dictionary
    kwargs.update({k: v for k, v in kwargs_input.items() if k in kwargs_default})
    return kwargs


def dict_to_json(ditc, dump=False):
    """Convert a dict to JSON-compatible form, handling arrays, classes, and class configs."""

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
    """
    Recursively convert a JSON string to a Python structure with np.ndarrays
    where applicable.
    """

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


def print_dict(d, indent=0):
    """Print a dictionary. Detailed output at DEBUG level."""
    if not logger.isEnabledFor(logging.INFO):
        return

    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            logger.info(f"{pad}{k}:")
            print_dict(v, indent + 1)
        elif isinstance(v, np.ndarray):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{pad}{k}: shape {v.shape} -> {v.tolist()}")
            else:
                logger.info(f"{pad}{k}: shape {v.shape}")
        else:
            logger.info(f"{pad}{k}: {v}")


#######################################################################################

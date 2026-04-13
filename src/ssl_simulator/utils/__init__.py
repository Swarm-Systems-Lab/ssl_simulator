"""Utility functions for ssl_simulator."""

from .debug import debug_eig
from .dict_ops import (
    dict_to_json,
    json_to_dict,
    parse_kwargs,
    print_dict,
    safe_assign,
    safe_update,
    validate_dict_attributes,
)
from .file_ops import check_file_size, load_class_from_file
from .path_ops import add_src_to_path, create_dir
from .pprz import get_pprz_idx, load_pprz_data, pprz_angle
from .processing import first_larger_index, load_class, load_sim

__all__ = [  # noqa: RUF022
    # Debug utilities
    "debug_eig",
    # Dictionary operations
    "print_dict",
    "safe_assign",
    "safe_update",
    "validate_dict_attributes",
    # File & path operations
    "add_src_to_path",
    "check_file_size",
    "create_dir",
    # Data processing & serialization
    "dict_to_json",
    "first_larger_index",
    "json_to_dict",
    "load_class",
    "load_class_from_file",
    "load_sim",
    "parse_kwargs",
    # Paparazzi utilities
    "get_pprz_idx",
    "load_pprz_data",
    "pprz_angle",
]

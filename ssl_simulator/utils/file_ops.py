"""
"""

__all__ = [
    "check_file_size",
    "load_class_from_file"
]

import importlib.util
import os
from pathlib import Path

#######################################################################################

def check_file_size(filename: str, max_size_mb: int = None):
    file_size = os.path.getsize(filename)
    if max_size_mb:
        if file_size > max_size_mb * 1024 * 1024:
            raise ValueError(
                f"File size is {file_size / (1024 * 1024):.2f} MB, which exceeds the limit of {max_size_mb} MB. "
                f"You can increase this limit by setting the 'max_size_mb' parameter."
            )

def load_class_from_file(module_path: str, class_name: str):
    """Dynamically load a class from a given .py file."""
    module_path = Path(module_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {module_path}")

    return getattr(module, class_name)

#######################################################################################
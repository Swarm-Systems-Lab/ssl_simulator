"""
"""

__all__ = [
    "create_dir",
    "add_src_to_path"
]

import os
import sys
from pathlib import Path

#######################################################################################

def create_dir(directory: str, verbose: bool = True) -> None:
    """
    Create a new directory if it doesn't already exist.

    Args:
        directory (str): The path of the directory to create.
        verbose (bool, optional): Whether to print status messages. Defaults to True.

    Returns:
        None
    """
    try:
        os.mkdir(directory)
        if verbose:
            print(f"Directory '{directory}' created!")
    except FileExistsError:
        if verbose:
            print(f"The directory '{directory}' already exists!")

def add_src_to_path(file=None, relative_path="", deep=0, debug=False):
    """
    Adds the "relative_path" folder to sys.path based on the notebook's location.
    """
    if file:
        root = Path(file).resolve().parents[deep+1]
    else:
        root = Path.cwd().parents[deep]
    target = root / relative_path
    sys.path.append(str(target))

    if debug:
        print("\nroot:", root, "\ntarget:", target, "\n")

#######################################################################################
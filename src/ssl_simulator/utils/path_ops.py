""" """

__all__ = ["add_src_to_path", "create_dir"]

import os
import sys
from pathlib import Path

from ssl_simulator.config import CONFIG

#######################################################################################


def create_dir(directory: str, verbose: bool = True) -> None:
    """
    Create a new directory if it doesn't already exist.

    Args:
        directory (str): The path of the directory to create.
        verbose (bool, optional): Whether to print status messages. Defaults to True.

    Returns
    -------
        None
    """
    try:
        os.mkdir(directory)
        if verbose:
            pass
    except FileExistsError:
        if verbose:
            pass


def add_src_to_path(file=None, relative_path="", deep=0, debug=CONFIG["DEBUG"]):
    """Adds the "relative_path" folder to sys.path based on the notebook's location."""
    root = Path(file).resolve().parents[deep + 1] if file else Path.cwd().parents[deep]
    target = root / relative_path
    sys.path.append(str(target))

    if debug:
        pass


#######################################################################################

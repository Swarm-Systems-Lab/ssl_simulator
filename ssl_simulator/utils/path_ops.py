"""
"""

__all__ = [
    "createDir", 
    "create_dir",
]

import os

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

#######################################################################################
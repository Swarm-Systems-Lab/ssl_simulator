"""
"""

__all__ = ["createDir"]

import os

#######################################################################################

def createDir(dir, verbose=True):
    """
    Create a new directory if it doesn't exist
    """
    try:
        os.mkdir(dir)
        if verbose:
            print("Directory '{}' created!".format(dir))
    except FileExistsError:
        if verbose:
            print("The directory '{}' already exists!".format(dir))

#######################################################################################
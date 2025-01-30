"""
"""

__all__ = ["unit_vec", "R_2D_matrix"]

import numpy as np

#######################################################################################

def unit_vec(v):
    """
    Compute the unit vector of a given vector.

    Parameters
    ----------
    v : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Unit vector in the same direction as `v`. If `v` has zero magnitude, 
        it is returned unchanged to avoid division by zero.
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def R_2D_matrix(angle):
    """
    Generate a 2D rotation matrix for a given angle.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        2x2 rotation matrix that rotates vectors counterclockwise by `angle` radians.
    """
    return np.array([
        [np.cos(angle), -np.sin(angle)], 
        [np.sin(angle), np.cos(angle)]
    ])

#######################################################################################
"""
"""

__all__ = [
    "R_2D_matrix",
    "norm_2", 
]

import numpy as np

#######################################################################################

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

def norm_2(A):
    return np.sqrt(np.max(np.linalg.eigvals(A.T @ A)))

#######################################################################################
"""
"""

__all__ = ["pprz_angle"]

import numpy as np

#######################################################################################

def pprz_angle(theta_array):
    """
    Convert an angle from standard mathematical coordinates to 
    Paparazzi UAV convention.

    Parameters
    ----------
    theta_array : np.ndarray
        Input angles in radians.

    Returns
    -------
    np.ndarray
        Converted angles in radians.
    
    Notes
    -----
    - The Paparazzi UAV convention defines 0 radians as pointing north (upward),
      whereas standard mathematical convention defines 0 radians as pointing east (rightward).
    - This function shifts the angle by -theta + π/2 to align with the Paparazzi convention.
    """
    return -theta_array + np.pi / 2

#######################################################################################
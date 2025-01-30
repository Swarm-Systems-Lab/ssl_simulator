"""
"""

__all__ = ["uniform_distrib"]

import numpy as np
import random

#######################################################################################

def uniform_distrib(N: int, lims: list[float], rc0: list[float] = [0, 0], seed=None):
    """
    Generate a uniform distribution of points within a rectangular region.

    Parameters
    ----------
    N : int
        Number of points to generate.
    lims : list[float]
        Distance limits [lim_x, lim_y] defining the size of the rectangle along each dimension.
    rc0 : list[float], optional (default: [0, 0])
        Coordinates [x, y] of the centroid of the distribution.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing the generated points.
    
    Raises
    ------
    ValueError
        If `rc0` or `lims` do not have exactly two elements.
    """
    if seed is not None:
        random.seed(seed)

    if len(rc0) != 2 or len(lims) != 2:
        raise ValueError("Both 'rc0' and 'lims' must be lists of length 2.")

    X0 = (np.random.rand(N, 2) - 0.5) * 2 * np.array(lims)
    return np.array(rc0) + X0

    # if len(rc0) + len(lims) != 2 * 2:
    #     raise Exception("The dimension of rc0 and lims should be 2")

    # X0 = (np.random.rand(N, 2) - 0.5) * 2
    # for i in range(2):
    #     X0[:, i] = X0[:, i] * lims[i]
    # return rc0 + X0

#######################################################################################
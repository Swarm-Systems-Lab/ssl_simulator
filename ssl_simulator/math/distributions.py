"""
"""

__all__ = [
    "uniform_distrib",
    "regpoly_formation",
    "flower_formation",
    "circular_distrib",
    "XY_distrib",
    "batman_distrib"]

import numpy as np
import random

# Affine transformation
M_scale = lambda s, n=2: np.eye(n)*s

#######################################################################################

# TODO: test
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

# TODO: test
def regpoly_formation(N, r, thetha0=0):
    """
    Function to generate a regular polygon distribution.
    """
    d_theta = 2*np.pi/N
    theta = []

    for i in range(N):
        if N == 2:
            theta.append(d_theta*i + thetha0)
        elif N%2 == 0:
            theta.append(d_theta*i + d_theta/2 + thetha0)
        else:
            theta.append(d_theta*i + d_theta/4 + thetha0)
    
    return np.array([r*np.cos(theta), r*np.sin(theta)]).T

# TODO: test
def flower_formation(N, R, b=3):
    """
    Function to generate a non-uniform (dummy) "flower" distribution of N agents.
    """
    P_form = np.array([[],[]]).T
    while len(P_form) < N:
        P1 = (np.random.rand(int(N/2),2) - 0.5)*2 * R/2 * b/2
        P2 = (np.random.rand(int(N/8),2) - 0.5)*2 * R/5 * b/4
        P = np.vstack([P1,P2])

        p_r = np.sqrt(P[:,0]**2 + P[:,1]**2)
        p_theta = np.arctan2(P[:,0], P[:,1])
        r = R * np.cos(2*p_theta)**2 + b

        P_form = np.vstack([P_form, P[(p_r <= r),:]])

    return P_form[0:N,:]

# TODO: test
# TODO: Adjust the distribution to be uniform in curved space
def circular_distrib(N, rc0, r, h=0, border_noise=0.1):
    # Generate random angles
    rand_ang = 2 * np.pi * np.random.rand(N)
    
    # Compute random directions in 2D (unit vectors)
    rand_dirs = np.array([np.cos(rand_ang), np.sin(rand_ang)]).T
    
    # Generate random radial distances, scaled by r and h, with some border noise
    rand_rads = (r - h) * np.random.rand(N) + h + border_noise * np.random.rand(N)
    
    # Compute the positions in the plane
    X0 = rand_rads[:, None] * rand_dirs / np.linalg.norm(rand_dirs, axis=1)[:, None]
    
    return rc0 + X0

# TODO: test
def XY_distrib(N, n, rc0, lims, scale=1):
    """
    Function to generate uniform rectangular distributions.
    * N: number of points
    * n: dimension of the real space
    * rc0: position in the real space of the central point
    """
    # Generate random points uniformly distributed in [-1, 1] range
    X0 = (np.random.rand(N, n) - 0.5) * 2
    
    # Scale the points to the specified limits for each dimension
    for i in range(n):
        X0[:, i] = X0[:, i] * lims[i]
    
    # Apply scaling matrix
    X0 = X0 @ M_scale(scale, n)
    
    return rc0 + X0

# TODO: test
def batman_distrib(N, rc0, lims, scale=1):
    """
    Batman distribution in 2D.
    * N: number of points
    * rc0: position in the real space of the central point
    * lims = [xlim, ylim]: width and length of the distribution
    """

    x_filt = []
    y_filt = []

    min_eq_value1 = -0.05
    min_eq_value2 = -0.5

    while len(x_filt) < N:
        # Generate random points in the box centered at the origin
        X = (np.random.rand(N, 2) - 0.5) * 2 * 8
        
        x_ = X[:, 0]
        y_ = X[:, 1]

        # Define the different equations to filter points (eq1 to eq6)
        eq1 = lambda x, y: ((x / 7) ** 2 * np.sqrt(abs(abs(x) - 3) / (abs(x) - 3)) + 
                            (y / 3) ** 2 * np.sqrt(abs(y + 3 / 7 * np.sqrt(33)) / 
                            (y + 3 / 7 * np.sqrt(33))) - 1)
        
        eq2 = lambda x, y: (abs(x / 2) - ((3 * np.sqrt(33) - 7) / 112) * x ** 2 - 3 + 
                            np.sqrt(1 - (abs(abs(x) - 2) - 1) ** 2) - y)
        
        eq3 = lambda x, y: (9 * np.sqrt(abs((abs(x) - 1) * (abs(x) - .75)) / 
                            ((1 - abs(x)) * (abs(x) - .75))) - 8 * abs(x) - y)
        
        eq4 = lambda x, y: (3 * abs(x) + .75 * np.sqrt(abs((abs(x) - .75) * (abs(x) - .5)) / 
                            ((.75 - abs(x)) * (abs(x) - .5))) - y)
        
        eq5 = lambda x, y: (2.25 * np.sqrt(abs((x - .5) * (x + .5)) / 
                            ((.5 - x) * (.5 + x))) - y)
        
        eq6 = lambda x, y: (6 * np.sqrt(10) / 7 + (1.5 - .5 * abs(x)) * 
                            np.sqrt(abs(abs(x) - 1) / (abs(x) - 1)) - 
                            (6 * np.sqrt(10) / 14) * np.sqrt(4 - (abs(x) - 1) ** 2) - y)

        # Filtering points based on each equation
        # eq1
        x = x_[np.logical_or(x_ < -4, x_ > 4)]
        y = y_[np.logical_or(x_ < -4, x_ > 4)]
        eq = eq1(x, y)
        z = np.logical_and(eq < 0, eq > min_eq_value1)
        x_filt.extend(x[z])
        y_filt.extend(y[z])

        x = x_[np.logical_and(y_ >= 0, np.logical_and(x_ > -4, x_ < 4))]
        y = y_[np.logical_and(y_ >= 0, np.logical_and(x_ > -4, x_ < 4))]
        eq = eq1(x, y)
        z = np.logical_and(eq < 0, eq > min_eq_value1)
        x_filt.extend(x[z])
        y_filt.extend(y[z])

        # eq2
        x = x_[np.logical_and(y_ < 0, np.logical_and(x_ >= -4, x_ <= 4))]
        y = y_[np.logical_and(y_ < 0, np.logical_and(x_ >= -4, x_ <= 4))]
        eq = eq2(x, y)
        z = np.logical_and(eq < 0, eq > min_eq_value2)
        x_filt.extend(x[z])
        y_filt.extend(y[z])

        # eq3
        x = x_[np.logical_and(y_ > 0, np.logical_and(x_ >= -1, x_ <= -0.75))]
        y = y_[np.logical_and(y_ > 0, np.logical_and(x_ >= -1, x_ <= -0.75))]
        eq = eq3(x, y)
        z = np.logical_and(eq < 0, eq > min_eq_value2)
        x_filt.extend(x[z])
        y_filt.extend(y[z])

        x = x_[np.logical_and(y_ > 0, np.logical_and(x_ >= 0.75, x_ <= 1))]
        y = y_[np.logical_and(y_ > 0, np.logical_and(x_ >= 0.75, x_ <= 1))]
        eq = eq3(x, y)
        z = np.logical_and(eq < 0, eq > min_eq_value2)
        x_filt.extend(x[z])
        y_filt.extend(y[z])

        # eq4
        x = x_[np.logical_and(y_ > 0, np.logical_and(x_ > -0.75, x_ < 0.75))]
        y = y_[np.logical_and(y_ > 0, np.logical_and(x_ > -0.75, x_ < 0.75))]
        eq = eq4(x, y)
        z = np.logical_and(eq < 0, eq > min_eq_value2)
        x_filt.extend(x[z])
        y_filt.extend(y[z])

        # eq5
        x = x_[np.logical_and(y_ > 0, np.logical_and(x_ > -0.5, x_ < 0.5))]
        y = y_[np.logical_and(y_ > 0, np.logical_and(x_ > -0.5, x_ < 0.5))]
        eq = eq5(x, y)
        z = np.logical_and(eq < 0, eq > min_eq_value2)
        x_filt.extend(x[z])
        y_filt.extend(y[z])

        # eq6
        x = x_[np.logical_and(y_ > 0, np.logical_and(x_ > -3, x_ < -1))]
        y = y_[np.logical_and(y_ > 0, np.logical_and(x_ > -3, x_ < -1))]
        eq = eq6(x, y)
        z = np.logical_and(eq < 0, eq > min_eq_value2)
        x_filt.extend(x[z])
        y_filt.extend(y[z])

        x = x_[np.logical_and(y_ > 0, np.logical_and(x_ > 1, x_ < 3))]
        y = y_[np.logical_and(y_ > 0, np.logical_and(x_ > 1, x_ < 3))]
        eq = eq6(x, y)
        z = np.logical_and(eq < 0, eq > min_eq_value2)
        x_filt.extend(x[z])
        y_filt.extend(y[z])

    # Final adjustment of points
    X0 = np.array([x_filt[:N], y_filt[:N]]).T

    # Apply scaling and translation
    for i in range(2):
        X0[:, i] = X0[:, i] / 8 * lims[i]
    X0 = X0 @ M_scale(scale, 2)
    
    return rc0 + X0

#######################################################################################
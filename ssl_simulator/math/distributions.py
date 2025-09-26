"""
"""

__all__ = [
    "uniform_distrib",
    "regpoly_formation",
    "flower_formation",
    "circular_distrib",
    "elliptical_distrib",
    "XY_distrib",
    "batman_distrib"]

import numpy as np
import random

from .algebra import R_2D_matrix

# Affine transformation
M_scale = lambda s, n=2: np.eye(n)*s

#######################################################################################

# TODO: test
import numpy as np

def uniform_distrib(
    N: int,
    lims: list[float],
    rc0: list[float] | None = None,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate a uniform distribution of points within a hyper-rectangular region
    in arbitrary dimensions.

    Parameters
    ----------
    N : int
        Number of points to generate.
    lims : list of float
        Distance limits [lim_1, lim_2, ..., lim_D] defining half-size
        of the box along each dimension. Total side length is 2 * lim_i.
    rc0 : list of float, optional
        Coordinates [c_1, c_2, ..., c_D] of the centroid of the distribution.
        If None, defaults to the origin (0,...,0).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (N, D) containing the generated points.

    Raises
    ------
    ValueError
        If `rc0` or `lims` lengths do not match the dimension D.
    """
    # Fix random seed
    if seed is not None:
        np.random.seed(seed)

    lims = np.array(lims, dtype=float)
    D = lims.shape[0]

    if rc0 is None:
        rc0 = np.zeros(D)
    else:
        rc0 = np.array(rc0, dtype=float)

    if rc0.shape[0] != D:
        raise ValueError("'rc0' and 'lims' must have the same dimension length.")

    # Generate uniform samples in [-lims, lims] per dimension
    X0 = (np.random.rand(N, D) - 0.5) * 2 * lims
    return rc0 + X0


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

def circular_distrib(N, rc0=0, r=1, **kwargs):
    """
    Generate points approximately uniformly distributed in a circular annulus.

    Parameters
    ----------
    N : int
        Number of points to generate.
    rc0 : array-like
        Center of the circular distribution.
    r : float
        Outer radius.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing the 2D coordinates of sampled points.
    """
    return elliptical_distrib(N, rc0, rx=r, ry=r, **kwargs)

def elliptical_distrib(N, rc0=0, rx=1.0, ry=1.0, h=0.0, border_noise=0.0, rot_angle=0.0):
    """
    Generate a uniform distribution in an elliptical (or circular) annulus.

    Parameters
    ----------
    N : int
        Number of points
    rc0 : array-like
        Center of the ellipse
    rx : float
        Horizontal radius (semi-major axis)
    ry : float
        Vertical radius (semi-minor axis)
    h : float
        Inner radius (for annulus); use 0 for full disk
    border_noise : float
        Adds random jitter to the radial position

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with sampled positions
    """
    # Sample random angles uniformly in [0, 2π)
    theta = 2 * np.pi * np.random.rand(N)

    # Sample radius with sqrt to get uniform density in area
    U = np.random.rand(N)
    r_scaled = np.sqrt(U * (1 - h / max(rx, ry)) + h / max(rx, ry))  # normalized

    # Apply elliptical scaling
    x = (r_scaled + border_noise * np.random.rand(N)) * rx * np.cos(theta)
    y = (r_scaled + border_noise * np.random.rand(N)) * ry * np.sin(theta)
    xy = np.stack((x, y), axis=1) @ R_2D_matrix(rot_angle).T

    return rc0 + xy

def XY_distrib(N, rc0, lims, scale=1, n=2):
    """
    Generate a uniform distribution of points in a rectangular region.

    Parameters
    ----------
    N : int
        Number of points to generate.
    rc0 : array-like of shape (n,)
        Central point in real space around which the points are distributed.
    lims : array-like of shape (n,)
        Range limits for each dimension (defines the half-length of the box in each direction).
    scale : float, optional
        Scaling factor applied to the overall distribution (default is 1).
    n : int, optional
        Number of dimensions (default is 2).

    Returns
    -------
    numpy.ndarray of shape (N, n)
        Array of generated points.
    """
    rc0 = np.array(rc0)

    # Generate N points uniformly in [-1, 1]^n
    points = (np.random.rand(N, n) - 0.5) * 2

    # Scale according to specified limits
    for i in range(n):
        points[:, i] *= lims[i]

    # Apply uniform scaling and shift by center point
    points = points @ M_scale(scale, n)

    return rc0 + points

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
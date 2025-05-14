"""
"""

__all__ = [
    "L_sigma",
    "calc_mu_centralized"
]

import numpy as np

#######################################################################################

def L_sigma(X, sigma):
    """
    Cetralised L_sigma calculation function (only for numerical validation).

    Attributes:
        X: numpy array
            (N x 2) matrix of agents position from the centroid
        sigma: numpy array
            (N) vector of simgma_values on each row of X
    """
    N = X.shape[0]
    l_sigma_hat = sigma[:, None].T @ X

    x_norms = np.zeros((N))
    for i in range(N):
        x_norms[i] = X[i, :] @ X[i, :].T
        D_sqr = np.max(x_norms)
    l_sigma_hat = l_sigma_hat / (N * D_sqr)

    return l_sigma_hat.flatten()

def calc_mu_centralized(X, sigma):
    """
    Cetralised mu calculation function (only for numerical validation).

    Attributes:
        X: numpy array
            (N x 2) matrix of agents position from the centroid
        sigma: numpy array
            (N) vector of simgma_values on each row of X
    """
    N = X.shape[0]
    l_sigma_hat = sigma[:, None].T @ X
    l_sigma_hat = l_sigma_hat / N 
    return l_sigma_hat.flatten()

#######################################################################################
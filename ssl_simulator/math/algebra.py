"""
"""

__all__ = [
    "R_2D_matrix",
    "norm_2",
    "cov_matrix"
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

def norm_2(A: np.ndarray) -> float:
    """
    Compute the matrix 2-norm (spectral norm) of a matrix A.

    The 2-norm of a matrix is defined as the largest singular value of A,
    which is equivalent to the square root of the largest eigenvalue of AᵀA.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (m, n).

    Returns
    -------
    float
        The spectral (operator 2-) norm of the matrix A.
    """
    # Compute largest eigenvalue of AᵀA, then take square root
    return np.sqrt(np.max(np.linalg.eigvals(A.T @ A)))

def cov_matrix(X: np.ndarray, sample: bool = False) -> np.ndarray:
    """
    Compute covariance matrix/matrices for D-dimensional vectors.

    Supports both single and batched inputs:
    - Input shape (N, D): a single dataset of N samples in D dimensions.
    - Input shape (K, N, D): K datasets, each with N samples in D dimensions.
      Returns K covariance matrices.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, D) or (K, N, D).
    sample : bool, default=False
        If True, compute the *sample covariance* (divide by N-1).
        If False, compute the *population covariance* (divide by N).

    Returns
    -------
    cov : np.ndarray
        - If input shape is (N, D), returns array of shape (D, D).
        - If input shape is (K, N, D), returns array of shape (K, D, D).
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray")
    if X.ndim not in (2, 3):
        raise ValueError("X must have shape (N, D) or (K, N, D)")

    # Handle single dataset case by promoting to 3D
    if X.ndim == 2:
        X = X[None, ...]  # shape (1, N, D)

    K, N, D = X.shape
    if N < 2:
        raise ValueError("Need at least 2 samples per dataset to compute covariance")

    # Center data along N
    mean = np.mean(X, axis=1, keepdims=True)  # shape (K, 1, D)
    X_centered = X - mean                     # shape (K, N, D)

    # Choose denominator
    denom = N - 1 if sample else N

    # Compute covariances: (K, D, D)
    cov = np.matmul(X_centered.transpose(0, 2, 1), X_centered) / denom

    # Return (D, D) if single dataset
    return cov[0] if cov.shape[0] == 1 else cov

#######################################################################################
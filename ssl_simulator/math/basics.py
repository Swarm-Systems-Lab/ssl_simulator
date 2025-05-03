"""
"""

__all__ = [
    "unit_vec", 
    "adapt_to_nd",
    "Q_prod_xi",
    "exp",
    "angle_of_vectors"
]

import numpy as np
from numpy.linalg import norm

#######################################################################################

def unit_vec(V, delta=0):
    """
    Normalise a bundle of 2D vectors

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
    V = np.asarray(V)

    if V.ndim == 1:
        norm = np.linalg.norm(V)
        return V / norm if norm > delta else V
    else:
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        mask = norms > delta
        V_normalized = np.zeros_like(V)
        V_normalized[mask[:, 0]] = V[mask[:, 0]] / norms[mask,None]
        return V_normalized

def adapt_to_nd(X, target_ndim, dtype=None):
    """
    Adapt the input to the desired number of dimensions.
    
    Parameters
    ----------
    X : any
        Input data to be adapted to a specific number of dimensions. Can be a scalar,
        list, tuple, generator, or array-like object.
    target_ndim : int
        The desired number of dimensions for the output array.
        If the input has fewer dimensions, new axes will be prepended.
        If the input has more dimensions, an error is raised.
    dtype : data-type, optional
        Desired data type for the output array. If None, the data type is inferred.
    
    Returns
    -------
    X_nd : numpy.ndarray
        The input data converted to a numpy array with the specified number of dimensions.
    
    Raises
    ------
    ValueError
        If the input has more than the desired number of dimensions.
    
    Example
    -------
    X = adapt_to_nd([1, 2, 3], target_ndim=2)
    print(X.shape)  # Output: (1, 3)
    
    X = adapt_to_nd(5, target_ndim=3)
    print(X.shape)  # Output: (1, 1, 1)
    """
    # Convert to numpy array
    X = np.array(X, dtype=dtype)

    # Adjust dimensions
    ndim = X.ndim
    if ndim < target_ndim:
        for _ in range(target_ndim - ndim):
            X = np.expand_dims(X, axis=0)
    elif ndim > target_ndim:
        raise ValueError(f"The dimensionality of X is greater than {target_ndim}!")

    return X

def Q_prod_xi(Q, X):
    """
    Apply matrix Q to each row of X.
    
    Parameters
    ----------
    Q : ndarray of shape (D, D)
        Transformation matrix.
    X : ndarray of shape (N, D)
        Input data where each row is a vector to be transformed.
    
    Returns
    -------
    X_transformed : ndarray of shape (N, D)
        Result of applying Q to each row of X.
    
    Example
    -------
    X = np.random.randn(10, 5)
    Q = np.eye(5) * 2
    X2 = Q_prod_xi(Q, X)  # Doubles each row
    """
    return X @ Q.T

def exp(X, Q, x0):
    """
    Compute the exponential of a quadratic form:
    
        exp(X) = exp((X - x0)^T @ Q @ (X - x0))
    
    Parameters
    ----------
    X : array-like of shape (N, D)
        Input points where the function is evaluated.
    Q : ndarray of shape (D, D)
        Quadratic form matrix.
    x0 : array-like of shape (D,)
        Center of the Gaussian (mean).
    
    Returns
    -------
    result : ndarray of shape (N,)
        Result of applying the exponential quadratic form to each point in X.
    """
    X = adapt_to_nd(X, target_ndim=2)
    delta = X - x0                          # (N, D)
    quad_form = Q_prod_xi(Q, delta)         # (N, D)
    exponent = np.sum(delta * quad_form, axis=1)  # (N,)
    return np.exp(exponent)

def angle_of_vectors(A, B):
    """
    Calculate the signed angle between pairs of 2D vectors A and B.

    Parameters
    ----------
    A : numpy.ndarray of shape (N, 2)
        First set of 2D vectors.
    B : numpy.ndarray of shape (N, 2)
        Second set of 2D vectors.

    Returns
    -------
    theta : numpy.ndarray of shape (N,)
        Signed angles between each pair of vectors, in radians.
    """
    dot_product = np.sum(A * B, axis=1)
    cross_product = np.cross(A, B, axis=1)
    theta = np.arctan2(cross_product, dot_product)
    return theta

#######################################################################################
""" """

__all__ = [
    "Q_prod_xi",
    "adapt_to_nd",
    "angle_of_vectors",
    "check_and_parse_dimensions",
    "exp",
    "unit_vec",
]

import logging

import numpy as np

logger = logging.getLogger(__name__)


def unit_vec(V, delta=0, axis=-1):
    """
    Normalize a bundle of 2D vectors.

    Parameters
    ----------
        V : np.ndarray
            Input array of shape (..., 2), e.g. (T, N, 2).
        delta : float
            Threshold below which vectors are considered zero.
        axis : int
            Axis along which to normalize.

    Returns
    -------
        np.ndarray
            Array of unit vectors, same shape as V.
    """
    V = np.asarray(V)
    norms = np.linalg.norm(V, axis=axis, keepdims=True)
    # Avoid division by very small values
    safe_norms = np.where(norms > delta, norms, 1.0)
    unit = V / safe_norms
    unit = np.where(norms > delta, unit, 0.0)  # zero out small vectors
    return unit


def check_and_parse_dimensions(array, expected_shape, name=None, fill_values=None, dtype=float):
    """
    Generic function to check and parse dimensions of an array.

    Args:
        array (np.ndarray): The input array to validate.
        expected_shape (tuple): The expected shape of the array.
            - Use `None` for dimensions that can vary.
            - Use a list/tuple of ints (e.g., [2,3]) for dimensions that can take multiple values.
        name (str, optional): The name of the variable (for error messages). If None, attempts to infer the variable name.
        fill_values (int | list[int], optional): Value(s) to replace `None` dimensions.
            - If an int, all None dimensions are replaced with that value.
            - If a list, it must have as many entries as there are Nones in expected_shape.

    Returns
    -------
        np.ndarray: The reshaped or validated array.

    Raises
    ------
        ValueError: If the array does not match the expected shape.

    Examples
    --------
        >>> arr = np.ones((10, 32, 64))
        >>> check_and_parse_dimensions(arr, (None, 32, 64))
        # passes, first dim is free (10)

        >>> check_and_parse_dimensions(arr, (None, 32, 64), fill_values=10)
        # passes only if first dim == 10

        >>> arr2 = np.ones((5, 32, 7, 64))
        >>> check_and_parse_dimensions(arr2, (None, 32, None, 64), fill_values=[5, 7])
        # passes only if first==5 and third==7

        >>> arr3 = np.ones((5, 2))
        >>> check_and_parse_dimensions(arr3, (None, [2, 3]))
        # passes, since second dim can be 2 or 3

        >>> arr4 = np.ones((1, 3, 3))
        >>> check_and_parse_dimensions(arr4, (5, 3, 3), fill_values=5).shape
        # (5, 3, 3) -> broadcasted from (1, 3, 3)
    """
    array = np.asarray(array, dtype=dtype)  # Ensure the input is a NumPy array

    # Handle special cases for expected shapes (auto-add batch dimension)
    # TODO: generalize this logic
    orig_shape = array.shape
    changed = False

    if len(expected_shape) == 2 and expected_shape[0] is None and array.ndim == 1:
        array = array[np.newaxis, :]
        changed = True
    elif len(expected_shape) == 3 and expected_shape[0] is None and array.ndim == 2:
        array = array[np.newaxis, :, :]
        changed = True
    elif (
        len(expected_shape) == 4
        and expected_shape[0] is None
        and expected_shape[1] is None
        and array.ndim == 3
    ):
        array = array[:, np.newaxis, :, :]
        changed = True
    elif (
        len(expected_shape) == 4
        and expected_shape[0] is None
        and expected_shape[1] is None
        and array.ndim == 2
    ):
        array = array[np.newaxis, np.newaxis, :, :]
        changed = True

    if changed:
        logger.debug(f"Shape changed: {orig_shape} -> {array.shape}")

    # Replace None values in expected_shape with fill_values if provided
    if fill_values is not None:
        if isinstance(fill_values, int):
            expected_shape = tuple(
                dim if dim is not None else fill_values for dim in expected_shape
            )
        else:
            fill_iter = iter(fill_values)
            expected_shape = tuple(
                dim if dim is not None else next(fill_iter) for dim in expected_shape
            )

    # Check shape match with broadcasting for singleton (1) dims
    if array.ndim != len(expected_shape):
        raise ValueError(f"'{name}' must have {len(expected_shape)} dims, got {array.ndim}")

    target_shape = list(array.shape)
    for i, s in enumerate(expected_shape):
        if s is None:
            continue  # free dimension
        elif isinstance(s, (list, tuple)):
            if array.shape[i] not in s:
                raise ValueError(f"'{name}' dim {i} must be one of {s}, got {array.shape[i]}")
        else:  # expected specific size
            if array.shape[i] == s:
                continue
            elif array.shape[i] == 1:
                # allow broadcast
                target_shape[i] = s
            else:
                raise ValueError(f"'{name}' dim {i} must be {s}, got {array.shape[i]}")

    # Broadcast if needed
    if tuple(target_shape) != array.shape:
        array = np.broadcast_to(array, target_shape)

    return array


def adapt_to_nd(X, target_ndim, dtype=None):
    """
    Adapt the input array to the specified number of dimensions.

    Parameters
    ----------
        X : array-like
            Input data to adapt.
        target_ndim : int
            Target number of dimensions.
        dtype : data-type, optional
            Desired data type of the output array.

    Returns
    -------
        np.ndarray
            Array adapted to the specified number of dimensions.
    """
    # Convert to numpy array with the specified dtype
    X = np.array(X, dtype=dtype)

    # Use check_and_parse_dimensions to validate and adapt dimensions
    expected_shape = tuple(None for _ in range(target_ndim))
    return check_and_parse_dimensions(X, expected_shape, name="X")


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
    delta = X - x0  # (N, D)
    quad_form = Q_prod_xi(Q, delta)  # (N, D)
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

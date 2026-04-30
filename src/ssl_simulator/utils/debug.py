import logging

import numpy as np

from ssl_simulator.logging import requires_log_level

_logger = logging.getLogger(__name__)


@requires_log_level(_logger, "DEBUG")
def debug_eig(A: np.ndarray, *, include_eigenvectors: bool = True) -> None:
    """Log eigenvalues (and optionally eigenvectors) of a matrix at DEBUG level.

    Pure diagnostic — no-op when DEBUG is disabled.

    Parameters
    ----------
    A : np.ndarray
        Square matrix to decompose.
    include_eigenvectors : bool, optional
        If True, also include the eigenvector matrix in the log payload.
    """
    eigenvalues, eigenvectors_matrix = np.linalg.eig(A)

    payload = {
        "matrix_shape": list(A.shape),
        "eigenvalues": eigenvalues,
    }
    if include_eigenvectors:
        payload["eigenvectors"] = eigenvectors_matrix

    _logger.debug("eigendecomposition", extra=payload)

""" """

__all__ = [
    "debug_eig",
]

import numpy as np

#######################################################################################


def debug_eig(
    A: np.ndarray, eigenvectors: bool = True, prec_values: int = 8, prec_vectors: int = 3
) -> None:
    """
    Debug function to display eigenvalues and optionally eigenvectors of a matrix.

    Parameters
    ----------
    A : np.ndarray
        The input square matrix for which eigenvalues and eigenvectors are to be computed.
    eigenvectors : bool, optional
        If True, eigenvectors will be displayed along with eigenvalues (default is True).
    prec_values : int, optional
        The number of decimal places to display for the eigenvalues (default is 8).
    prec_vectors : int, optional
        The number of decimal places to display for the eigenvectors (default is 3).

    Returns
    -------
    None
        The function prints the eigenvalues and eigenvectors (if requested) to the console.
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, _eigenvectors_matrix = np.linalg.eig(A)

    # Print eigenvalues
    with np.printoptions(precision=prec_values, suppress=True):
        for _i, _eigval in enumerate(eigenvalues):
            pass

    # Print eigenvectors if requested
    if eigenvectors:
        with np.printoptions(precision=prec_vectors, suppress=True):
            for _i in range(len(eigenvalues)):
                pass


#######################################################################################

# Example usage
if __name__ == "__main__":
    A = np.array([[1, 3], [2, 7]])
    debug_eig(A)

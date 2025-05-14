"""
Non-convex function (two Gaussians + contant * norm)
"""

__all__ = [
    "SigmaNonconvex"
]

import numpy as np
from numpy import linalg as la

from ._scalar_field import ScalarField
from ssl_simulator.math.basics import adapt_to_nd, Q_prod_xi, exp

#######################################################################################

# Helper functions for clearer initialization of Qa and Qb
def _create_Qa():
    """
    Create the quadratic transformation matrix Qa for the first Gaussian.
    """
    S1 = 0.9 * np.array([[1 / np.sqrt(30), 0], [0, 1]])
    return -S1

def _create_Qb():
    """
    Create the quadratic transformation matrix Qb for the second Gaussian.
    """
    S2 = 0.9 * np.array([[1, 0], [0, 1 / np.sqrt(15)]])
    A = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
    return -A.T @ S2 @ A

# Default centers for the Gaussians
default_a = np.array([1, 0])
default_b = np.array([0, -1.5])

class SigmaNonconvex(ScalarField):
    """
    Non-convex scalar field function (two Gaussians + "norm factor" * norm)

    Args:
        k: float
            norm factor
        mu: list
            center of the Gaussian.
        dev: float
            models the scale of the distribution while maintaining its properties
        a: np.ndarray
            center of the first Gaussian
        b: np.ndarray
            center of the second Gaussian
        Qa: numpy array
            2x2 matrix, quadratic transformation of the first Gaussian input
        Qb: numpy array
            2x2 matrix, quadratic transformation of the second Gaussian input
    """

    def __init__(self, k, mu=[0, 0], dev=1, a=default_a, b=default_b, Qa=None, Qb=None):
        self.k = k
        self.mu = mu
        self.dev = dev

        # Set default Qa and Qb if not provided
        self.Qa = Qa if Qa is not None else _create_Qa()
        self.Qb = Qb if Qb is not None else _create_Qb()
    
        # Convert a and b to numpy arrays if they are lists
        self.a = np.array(a) if isinstance(a, list) else a
        self.b = np.array(b) if isinstance(b, list) else b

        self.mu_x0 = 0
        self.mu = self._find_max(mu)
        self.mu_x0 = mu - self.mu

    def eval_value(self, X):
        x0 = np.array(self.mu) + self.mu_x0
        X = adapt_to_nd(X, target_ndim=2)
        X = (X - x0) / self.dev
        sigma = (
            -2
            - exp(X, self.Qa, self.a)
            - exp(X, self.Qb, self.b)
            + self.k * la.norm(X, axis=1)
        )
        return -sigma

    def eval_grad(self, X):
        x0 = np.array(self.mu) + self.mu_x0
        X = adapt_to_nd(X, target_ndim=2)
        X = (X - x0) / self.dev
        alfa = 0.0001  # Trick to avoid x/0
        sigma_grad = (
            -Q_prod_xi(self.Qa, X - self.a) * exp(X, self.Qa, self.a)[:, None]
            - Q_prod_xi(self.Qb, X - self.b) * exp(X, self.Qb, self.b)[:, None]
            + self.k * X / (la.norm(X, axis=1)[:, None] + alfa)
        )
        return -sigma_grad

    def eval_hessian(self, X):
        return None

    def get_config(self):
        return dict(k=self.k, mu=self.mu_x0 + self.mu, dev=self.dev, a=self.a, 
                    b=self.b, Qa=self.Qa, Qb=self.Qb)

#######################################################################################
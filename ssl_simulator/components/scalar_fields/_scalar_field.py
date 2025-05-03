"""
- Scalar field common class -
"""

__all__ = [
    "ScalarField",
    "PlotterScalarField"
]

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize

# Graphic tools
import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FixedLocator, NullFormatter

# Our utils
from ssl_simulator import parse_kwargs
from ssl_simulator.math import unit_vec, Q_prod_xi, R_2D_matrix
from ssl_simulator.visualization import vector2d, alpha_cmap, get_nice_ticks

MY_CMAP = alpha_cmap(plt.cm.jet, 0.3)

#######################################################################################
# -------- SCALAR FIELD ABSTRACT BASE CLASS -------- #

class ScalarField(ABC):
    mu = np.array([0, 0])
    transf_a, transf_w = 1, 0
    A = None

    # ------------------------------------------------------------------------
    # These operations have to be implemented in the subclass ################

    @abstractmethod
    def eval_value(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluation of the scalar field for a vector of values
        """
        pass

    @abstractmethod
    def eval_grad(self, X: np.ndarray) -> np.ndarray:
        """
        Gradient vector of the scalar field for a vector of values
        """
        pass

    @abstractmethod
    def eval_hessian(self, X: np.ndarray) -> np.ndarray:
        """
        Hessian matrix of the scalar field for a vector of values
        """
        pass

    # ------------------------------------------------------------------------
    # Evaluation #############################################################

    def value(self, X: np.ndarray) -> np.ndarray:
        R = R_2D_matrix(self.transf_w)
        A = (np.eye(2) * self.transf_a) @ R
        X = Q_prod_xi(A, X - self.mu) + self.mu
        return self.eval_value(X)

    def grad(self, X: np.ndarray) -> np.ndarray:
        R = R_2D_matrix(self.transf_w)
        A = (np.eye(2) * self.transf_a) @ R
        X = Q_prod_xi(A, X - self.mu) + self.mu
        grad = self.eval_grad(X)
        return Q_prod_xi(A.T, grad)

    def hessian(self, X: np.ndarray) -> np.ndarray: # TODO: fix for affine transformations
        # R = R_2D_matrix(self.transf_w)
        # A = (np.eye(2) * self.transf_a) @ R
        # X = Q_prod_xi(A, X - self.mu) + self.mu
        H = self.eval_hessian(X)
        return H

    def L1(self, pc: np.ndarray, P: np.ndarray):
        """
        Funtion for calculating and drawing L^1

        Attributes
        ----------
        pc: numpy array
            [x,y] position of the centroid
        P: numpy array
            (N x 2) matrix of agents position
        """
        if isinstance(pc, list):
            pc = np.array(pc)

        N = P.shape[0]
        X = P - pc

        grad_pc = self.grad(pc)[0]
        l1_sigma_hat = (grad_pc[:, None].T @ X.T) @ X

        x_norms = np.zeros((N))
        for i in range(N):
            x_norms[i] = (X[i, :]) @ X[i, :].T
            D_sqr = np.max(x_norms)

        l1_sigma_hat = l1_sigma_hat / (N * D_sqr)
        return l1_sigma_hat.flatten()

    # --------------------------------------------------------------------------
    # Internal helper methods
    # --------------------------------------------------------------------------

    def _find_max(self, x0: np.ndarray) -> np.ndarray:
        return minimize(lambda x: -self.value(np.array([x])), x0).x
    
#######################################################################################
# -------- PLOTTER CLASS -------- #

class PlotterScalarField:
    def __init__(self, field: ScalarField):
        self.field = field
        self.fig = None
        self.ax = None

        self.xy_mesh = None
        self.X = self.Y = None
        self.n = None

        self.color_map = None
        self.contour_levels = None
        self.contour_map_min = None
        self.contour_map_maj = None
        self.center_point = None

        self._colorbar = None
        self._cbar_axes = None

    def draw(self, **kwargs):
        self.fig, self.ax = kwargs.get("fig"), kwargs.get("ax")
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(16, 9), dpi=100)

        self._compute_mesh(**kwargs)
        Z = self.field.value(self.xy_mesh).reshape(self.n, self.n)
        self._draw_field(Z, **kwargs)
        self._draw_contours(Z, **kwargs)
        self._draw_center()

    def update(self):
            self._update_center()
            Z = self.field.value(self.xy_mesh).reshape(self.n, self.n)
            self._update_field(Z)
            self._update_contours(Z)
            plt.draw()

    def draw_grad(self, x: np.ndarray, axis: plt.Axes, ret_arr: bool = True, norm_fct=0, fct=1, **kw_arr):
        """Draws the gradient vector at a given point."""
        x = np.array(x) if isinstance(x, list) else x
        grad_x = self.field.grad(np.array([x]))[0]
        if norm_fct:
            grad_x = unit_vec(grad_x) * norm_fct
        quiv = vector2d(axis, x, grad_x * fct, **kw_arr)
        return quiv if ret_arr else grad_x

    def draw_L1(self, x: np.ndarray, axis: plt.Axes, ret_arr: bool = True, norm_fct=0, fct=1, **kw_arr):
        """Draws the always ascending direction L^1."""
        pass # TODO
    
    # --------------------------------------------------------------------------
    # Internal helper methods
    # --------------------------------------------------------------------------

    def _compute_mesh(self, xlim=None, ylim=None, n=256, **kwargs):
        mu = self.field.mu

        # Handle xlim
        if xlim is None:
            xmin, xmax = self.ax.get_xlim()
        elif isinstance(xlim, (list, tuple)) and len(xlim) == 2:
            xmin, xmax = xlim
        else:  # assume scalar
            xmin, xmax = mu[0] - xlim, mu[0] + xlim

        # Handle ylim
        if ylim is None:
            ymin, ymax = self.ax.get_ylim()
        elif isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            ymin, ymax = ylim
        else:  # assume scalar
            ymin, ymax = mu[1] - ylim, mu[1] + ylim
        
        x = np.linspace(xmin, xmax, n)
        y = np.linspace(ymin, ymax, n)
        X, Y = np.meshgrid(x, y)
        self.xy_mesh = np.stack([X.ravel(), Y.ravel()], axis=1)
        self.X, self.Y = X, Y
        self.n = n

    def _draw_field(self, Z, cmap=MY_CMAP, **kwargs):
        self.color_map = self.ax.pcolormesh(self.X, self.Y, Z, cmap=cmap, shading='auto')

    def _draw_contours(self, Z, contour_levels=10, contour_lw=0.3, **kwargs):
        vmin, vmax = np.nanmin(Z), np.nanmax(Z)
        major_levels, minor_levels, _ = get_nice_ticks(vmin, vmax, contour_levels, 4)
        opts = dict(colors='k', linestyles='-', alpha=0.2)

        self.contour_map_min = self.ax.contour(self.X, self.Y, Z, minor_levels, linewidths=contour_lw, **opts)
        self.contour_map_maj = self.ax.contour(self.X, self.Y, Z, major_levels, linewidths=contour_lw*2, **opts)
        
        self.kw_cbar = parse_kwargs(kwargs, dict(cbar_sw=True, cbar_lab=r"$\sigma$ [u]", cbar_minor_ticks=True))
        self._update_colorbar(major_levels, minor_levels)

    def _draw_center(self):
        mu = self.field.mu
        self.center_point, = self.ax.plot(mu[0], mu[1], "+k")

    def _update_center(self):
        mu = self.field.mu
        self.center_point.set_xdata([mu[0]])
        self.center_point.set_ydata([mu[1]])

    def _update_field(self, Z):
        self.color_map.set_array(Z.ravel())
        self.color_map.autoscale()

    def _update_contours(self, Z, contour_levels=10, contour_lw=0.3):
        for coll in getattr(self.contour_map_min, "collections", []):
            coll.remove()
        for coll in getattr(self.contour_map_maj, "collections", []):
            coll.remove()

        vmin, vmax = np.nanmin(Z), np.nanmax(Z)
        major_levels, minor_levels, _ = get_nice_ticks(vmin, vmax, contour_levels, 4)
        opts = dict(colors='k', linestyles='-', alpha=0.2)

        self.contour_map_min = self.ax.contour(self.X, self.Y, Z, minor_levels, linewidths=contour_lw, **opts)
        self.contour_map_maj = self.ax.contour(self.X, self.Y, Z, major_levels, linewidths=contour_lw*2, **opts)
        #self._update_colorbar(major_levels, minor_levels)

    def _update_colorbar(self, major_levels, minor_levels):
        cbar_sw = self.kw_cbar["cbar_sw"]
        cbar_lab = self.kw_cbar["cbar_lab"]
        cbar_minor_ticks = self.kw_cbar["cbar_minor_ticks"]

        if not cbar_sw and self._colorbar is None:
            return

        if self._cbar_axes is not None:
            self._cbar_axes.remove()  # Remove from figure
            self._cbar_axes = None

        # Create new colorbar
        divider = make_axes_locatable(self.ax)
        self._cbar_axes = divider.append_axes("right", size="2%", pad=0.05)
        self._colorbar = self.fig.colorbar(self.color_map, cax=self._cbar_axes, ticks=major_levels)
        self._colorbar.set_label(label=cbar_lab, labelpad=10)

        if cbar_minor_ticks:
            self._colorbar = self.fig.colorbar(self.color_map, cax=self._cbar_axes, ticks=major_levels)
            self._colorbar.ax.yaxis.set_minor_locator(FixedLocator(minor_levels))
            self._colorbar.ax.yaxis.set_minor_formatter(NullFormatter())

        # Hide the color bar if requested
        if not cbar_sw:
            self._colorbar.set_visible(False)
        
#######################################################################################
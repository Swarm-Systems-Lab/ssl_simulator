"""
- Scalar field common class -
"""

__all__ = [
    "ScalarField"
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
from ssl_simulator.math.basics import unit_vec, Q_prod_xi
from ssl_simulator.visualization.utils import vector2d, alpha_cmap, get_nice_ticks

MY_CMAP = alpha_cmap(plt.cm.jet, 0.3)

#######################################################################################

class ScalarField(ABC):
    A = np.eye(2)

    # ------------------------------------------------------------------------
    # These operations have to be implemented in the subclass ################

    @property
    @abstractmethod
    def mu(self) -> np.ndarray:
        pass

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
    # Evaluation

    def value(self, X: np.ndarray) -> np.ndarray:
        X = Q_prod_xi(self.A, X - self.mu) + self.mu
        return self.eval_value(X)

    def grad(self, X: np.ndarray) -> np.ndarray:
        X = Q_prod_xi(self.A, X - self.mu) + self.mu
        grad = self.eval_grad(X)
        return Q_prod_xi(self.A.T, grad)

    def hessian(self, X: np.ndarray) -> np.ndarray: # TODO: fix for affine transformations
        # X = Q_prod_xi(self.A, X - self.mu) + self.mu
        H = self.eval_hessian(X)
        return H

    def find_max(self, x0: np.ndarray) -> np.ndarray:
        return minimize(lambda x: -self.value(np.array([x])), x0).x

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
    
    # ------------------------------------------------------------------------
    # Draw

    def draw(
        self,
        fig: matplotlib.figure.Figure = None,
        ax: plt.Axes = None,
        xlim: float = None,
        ylim: float = None,
        cmap: matplotlib.colors.ListedColormap = MY_CMAP,
        n: int = 256,
        contour_levels: int = 0,
        contour_lw: float = 0.3,
        cbar_sw: bool = True,
        cbar_text: bool = True,
        cbar_lab: str = r"$\sigma$ [u]"
    ):
        """
        Scalar field drawing function.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to plot on. If None, a new figure is created.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, a new axis is created.
        xlim : tuple of float, optional
            (xmin, xmax) limits for the x-axis.
        ylim : tuple of float, optional
            (ymin, ymax) limits for the y-axis.
        cmap : matplotlib.colors.ListedColormap, optional
            Colormap to use.
        n : int, optional
            Number of points per dimension.
        contour_levels : int, optional
            Number of contour levels to plot.
        contour_lw : float, optional
            Line width of the contours.
        cbar_sw : bool, optional
            Whether to display the colorbar.
        cbar_text : bool, optional
            Whether to display text on the colorbar ticks.
        cbar_lab : str, optional
            Label for the colorbar.
        """

        # Create figure and axis if needed
        if fig is None:
            fig = plt.figure(figsize=(16, 9), dpi=100)
            ax = fig.subplots()
        elif ax is None:
            ax = fig.subplots()

        # Extract limits if needed
        is_empty = (len(ax.collections) == 0) and (len(ax.patches) == 0) and (len(ax.lines) == 0)

        if xlim is None:
            if is_empty:
                xlim = 30
                print("Warning: Axis is empty. Default 'xlim' = 30 will be used.")
            else:
                xlims = ax.get_xlim()
                xlim = abs(xlims[0] - xlims[1])
  
        if ylim is None:
            if is_empty:
                ylim = 30
                print("Warning: Axis is empty. Default 'ylim' = 30 will be used.")
            else:
                ylims = ax.get_ylim()
                ylim = abs(ylims[0] - ylims[1])
       
        # Meshgrid over the specified range
        x = np.linspace(self.mu[0] - xlim, self.mu[0] + xlim, n)
        y = np.linspace(self.mu[1] - ylim, self.mu[1] + ylim, n)
        X, Y = np.meshgrid(x, y)

        P = np.array([list(X.flatten()), list(Y.flatten())]).T
        Z = self.value(P).reshape(n, n)

        # Draw color mesh
        ax.plot(self.mu[0], self.mu[1], "+k") # mark the center
        color_map = ax.pcolormesh(X, Y, Z, cmap=cmap)

        # Draw contours if requested
        if contour_levels != 0:            
            vmin, vmax = np.nanmin(Z), np.nanmax(Z)
            major_levels, minor_levels, _ = get_nice_ticks(vmin, vmax, contour_levels, 4)

            kwargs = dict(colors="k", linestyles="-", alpha=0.2)
            ax.contour(X, Y, Z, minor_levels, linewidths=contour_lw, **kwargs)
            ax.contour(X, Y, Z, major_levels, linewidths=contour_lw*2, **kwargs)
        
        # Draw colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)

        if contour_levels != 0:
            cbar = fig.colorbar(color_map, cax=cax, ticks=major_levels)
            # Set minor ticks without labels
            cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_levels))
            cbar.ax.yaxis.set_minor_formatter(NullFormatter())  # no text on minor ticks

        else:
            cbar = fig.colorbar(color_map, cax=cax)
        
        if not cbar_text:
            cbar.ax.yaxis.set_major_formatter(NullFormatter())  # no text on minor ticks

        cbar.set_label(label=cbar_lab, labelpad=10)
        
        # Hide the color bar if requested
        if not cbar_sw:
            cax.set_visible(False)

        return color_map

    def draw_grad(
        self, x: np.ndarray, 
        axis: plt.Axes, ret_arr: bool = True, norm_fct=0, fct=1, **kw_arr
    ):
        """
        Function for drawing the gradient at a given point in space
        """
        # kw_arr: c="k", ls="-", lw = 0.7, hw=0.1, hl=0.2, s=1, alpha=1
        if isinstance(x, list):
            x = np.array(x)

        grad_x = self.grad(x)[0]

        if norm_fct:
            grad_x = unit_vec(grad_x)*norm_fct

        quiv = vector2d(axis, x, grad_x*fct, **kw_arr)

        if ret_arr:
            return quiv
        else:
            return grad_x

    def draw_L1(
        self, pc: np.ndarray, P: np.ndarray, 
        axis: plt.Axes, ret_arr: bool = True, norm_fct=0, fct=1, **kw_arr
    ):
        """
        Funtion for drawing L^1
        """
        pass # TODO

#######################################################################################

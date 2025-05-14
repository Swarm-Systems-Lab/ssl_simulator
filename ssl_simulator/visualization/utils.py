"""
File: utils.py

Description:
    This Python module contains a collection of utility functions designed to 
    facilitate data visualization and Matplotlib customization. The functions in 
    this file simplify common tasks such as plotting 2D vectors, configuring 
    Matplotlib axes, and applying alpha blending to colormaps. These utilities are 
    intended to enhance the visual quality and flexibility of plots, particularly 
    for scientific and engineering applications.

Usage:
    Import this module into your script or Jupyter notebook to leverage the 
    utilities for numerical and graphical tasks. The functions can be directly 
    used to customize your plots and improve the appearance of your visualizations.

"""

__all__ = [
    "set_paper_parameters",
    "smooth_interpolation",
    "vector2d",
    "zoom_range",
    "alpha_cmap",
    "config_data_axis",
    "config_axis",
    "get_nice_ticks"
]

# Standard libraries
import os

# Third-party libraries -------------------

# Algebra
import numpy as np
from numpy import linalg as la

# Visualization
import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

# Interpolation
from scipy.interpolate import interp1d, UnivariateSpline

#######################################################################################

def set_paper_parameters(fontsize=12, fontfamily="serif", uselatex=True):
    """
    Set Matplotlib parameters for consistent, publication-quality plots.

    This function configures font styles, sizes, and LaTeX rendering options for consistent 
    figure appearance across plots.

    Parameters:
        fontsize (int, optional): Global font size. Default is 12.
        fontfamily (str, optional): Font family (e.g., "serif", "Arial"). Default is "serif".
        uselatex (bool, optional): Whether to use LaTeX rendering. Default is True.

    Returns:
        None

    Notes:
        - Requires LaTeX installed if `uselatex=True`.
        - Applies to all future plots in the session.
        - Math rendering uses the AMS math package and Computer Modern fonts.

    Example:
        set_paper_parameters(fontsize=14, fontfamily="Arial", uselatex=False)
    """

    matplotlib.rc("font", **{
        "size": fontsize, 
        "family": fontfamily
        })
    
    matplotlib.rc("text", **{
        "usetex": uselatex, 
        "latex.preamble": r"\usepackage{amsmath}"
        })

    matplotlib.rc("mathtext", **{
        "fontset": "cm"
        })

def smooth_interpolation(x, y, method="cubic", num_points=100):
    """
    Perform smooth interpolation over 1D data.

    This function interpolates data points using a specified method, optionally 
    filling in missing values (NaNs) and returning a smooth curve for plotting.

    Parameters:
        x (array-like): X-coordinates of the input data.
        y (array-like): Y-coordinates of the input data. Can contain NaNs.
        method (str, optional): Interpolation method: "linear", "quadratic", "cubic", 
            or "spline". Defaults to "cubic".
        num_points (int, optional): Number of interpolated points. Default is 100.

    Returns:
        tuple:
            - x_smooth (np.ndarray): Interpolated X values.
            - y_smooth (np.ndarray): Interpolated Y values.

    Example:
        x_smooth, y_smooth = smooth_interpolation(x, y, method="cubic")
        plt.plot(x_smooth, y_smooth)
        plt.scatter(x, y)
    """
    # Convert to numpy arrays
    x, y = np.array(x), np.array(y)

    # Remove duplicates while keeping order
    unique_x, unique_indices = np.unique(x, return_index=True)
    y = y[unique_indices]

    # Remove remaining NaNs (start or end of data)
    mask = ~np.isnan(y)
    x, y = unique_x[mask], y[mask]

    # Generate smooth x values
    x_smooth = np.linspace(x.min(), x.max(), num_points)

    # Choose interpolation method
    if method in ["linear", "quadratic", "cubic"]:
        interp_func = interp1d(x, y, kind=method, fill_value="extrapolate")
    elif method == "spline":
        interp_func = UnivariateSpline(x, y, s=1)  # s=1 controls smoothness
    else:
        raise ValueError("Invalid method. Choose 'linear', 'quadratic', 'cubic', or 'spline'.")

    # Compute smooth y values
    y_smooth = interp_func(x_smooth)

    return x_smooth, y_smooth


def vector2d(ax, P0, Pf, c="k", ls="-", s=1, lw=0.7, hw=0.1, hl=0.2, alpha=1, zorder=1):
    """
    Draw a 2D vector as an arrow on a Matplotlib Axes.

    This function adds a vector (arrow) from point P0 to point Pf on the given axes. 
    The appearance of the vector (color, style, size, etc.) can be customized.

    Parameters:
        ax (matplotlib.axes.Axes): Axes on which the vector is plotted.
        P0 (tuple): Starting point (x, y) of the vector.
        Pf (tuple): Ending point (x, y) of the vector.
        c (str, optional): Arrow color. Default is "k" (black).
        ls (str, optional): Line style. Default is "-" (solid line).
        s (float, optional): Scale factor for vector magnitude. Default is 1.
        lw (float, optional): Line width. Default is 0.7.
        hw (float, optional): Arrowhead width. Default is 0.1.
        hl (float, optional): Arrowhead length. Default is 0.2.
        alpha (float, optional): Transparency (0 to 1). Default is 1.
        zorder (int, optional): Z-order for layering. Default is 1.

    Returns:
        matplotlib.patches.FancyArrowPatch: The drawn vector arrow.

    Notes:
        - The arrow length includes the head by default.
        - This function is useful for visualizing direction fields or forces.

    Example:
        vector2d(ax, P0=(0, 0), Pf=(1, 2), c="red", s=1.5)
    """

    arrow_params = {
        "lw": lw,
        "color": c,
        "ls": ls,
        "head_width": hw,
        "head_length": hl,
        "length_includes_head": True,
        "alpha": alpha,
        "zorder": zorder
    }

    quiv = ax.arrow(P0[0], P0[1], s * Pf[0], s * Pf[1], **arrow_params)
    return quiv


def zoom_range(begin, end, center, scale_factor):
    """
    Calculate a zoomed-in 1D range centered at a given point.

    This function returns a new range that zooms in or out relative to a center point 
    by scaling the distance from the center to the range boundaries.

    Parameters:
        begin (float): Starting value of the original range.
        end (float): Ending value of the original range.
        center (float): The center point around which to zoom.
        scale_factor (float): The factor by which to scale the range.
            Values < 1 zoom in; values > 1 zoom out.

    Returns:
        tuple (float, float): The new (min, max) bounds of the zoomed range.

    Reference:
        Adapted from: https://gist.github.com/dukelec/e8d4171ef4d12f9998295cfcbe3027ce
    """
    if begin < end:
        min_, max_ = begin, end
    else:
        min_, max_ = end, begin

    old_min, old_max = min_, max_

    offset = (center - old_min) / (old_max - old_min)
    range_ = (old_max - old_min) / scale_factor
    new_min = center - offset * range_
    new_max = center + (1.0 - offset) * range_

    if begin < end:
        return new_min, new_max
    else:
        return new_max, new_min


def alpha_cmap(cmap, alpha):
    """
    Apply a fixed alpha (transparency) to an existing colormap.

    This function modifies a given Matplotlib colormap by blending its colors with a white 
    background using a specified alpha value. This is particularly useful when using 
    functions like `pcolormesh`, which can behave unpredictably with transparent overlays.

    Parameters:
        cmap (matplotlib.colors.Colormap): The base colormap to modify.
        alpha (float): The transparency level, ranging from 0 (fully transparent) 
            to 1 (fully opaque).

    Returns:
        matplotlib.colors.ListedColormap: A new colormap with the alpha applied.

    Notes:
        - The alpha is applied uniformly across all colors in the colormap.
        - Blending is done with a white background.
        - This method avoids rendering issues that can arise from applying alpha directly 
          to plots like `pcolormesh`.

    Reference:
        https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
    """
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Define the alphas in the range from 0 to 1
    alphas = np.linspace(alpha, alpha, cmap.N)

    # Define the background as white
    BG = np.asarray(
        [
            1.0,
            1.0,
            1.0,
        ]
    )

    # Mix the colors with the background
    for i in range(cmap.N):
        my_cmap[i, :-1] = my_cmap[i, :-1] * alphas[i] + BG * (1.0 - alphas[i])

    # Create new colormap which mimics the alpha values
    my_cmap = ListedColormap(my_cmap)
    return my_cmap


def config_data_axis(ax, y_right = True, x_tick = True, y_tick = True, **kwargs):
    config_axis(ax, **kwargs)
    if y_right: # y-axis ticks to the right
        ax.yaxis.tick_right()
    if not x_tick: # remove x-axis tick labels
        ax.set_xticklabels([])
    if not y_tick: # remove y-axis tick labels
        ax.set_yticklabels([])

def config_axis(ax, x_step = None, y_step = None, format_float = False, xlims = None, ylims = None,
                max_major_ticks=6, n_minor=4, max_major_ticks_x=None, max_major_ticks_y=None):
    """
    Configure the visual and tick properties of a Matplotlib Axes object.

    This utility function allows fine control over the axis ticks, tick formatting, 
    and limits for a Matplotlib plot. It automatically configures minor ticks and 
    gridlines and adapts to the data being shown.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object to configure.
        x_step (float, optional): Fixed spacing between major ticks on the x-axis.
            Minor ticks will be placed at one-fourth of this interval. If not set,
            spacing is inferred from xlims or current axis content.
        y_step (float, optional): Fixed spacing between major ticks on the y-axis.
            Minor ticks will be placed at one-fourth of this interval. If not set,
            spacing is inferred from ylims or current axis content.
        format_float (bool, optional): If True, format y-axis tick labels as floats 
            with two decimal places.
        xlims (tuple[float, float], optional): If provided, sets (xmin, xmax) limits.
        ylims (tuple[float, float], optional): If provided, sets (ymin, ymax) limits.
        max_major_ticks (int, optional): Max number of major ticks to generate if 
            step size is not specified. Default is 6.
        n_minor (int, optional): Number of minor subdivisions between major ticks.
            Default is 4.

    Returns:
        None

    Notes:
        - Automatically sets gridlines to be visible.
        - If neither steps nor limits are set, uses current axis limits and data.
        - Uses `get_nice_ticks()` to compute evenly spaced "nice" ticks.
    """

    if max_major_ticks_x:
        kwticksx = dict(max_major_ticks=max_major_ticks_x, n_minor=n_minor)
    else:
        kwticksx = dict(max_major_ticks=max_major_ticks, n_minor=n_minor)
    if max_major_ticks_y:
        kwticksy = dict(max_major_ticks=max_major_ticks_y, n_minor=n_minor)
    else:
        kwticksy = dict(max_major_ticks=max_major_ticks, n_minor=n_minor)

    if format_float:
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    is_empty = (len(ax.collections) == 0) and (len(ax.patches) == 0) and (len(ax.lines) == 0)

    if x_step is not None and xlims is None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_step / 4))
    elif xlims is not None:
        major_tiks, minor_tiks, _ = get_nice_ticks(xlims[0], xlims[1], **kwticksx)
        ax.xaxis.set_major_locator(ticker.FixedLocator(major_tiks))
        ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_tiks))
    elif not is_empty:
        xlims = ax.get_xlim()
        major_tiks, minor_tiks, _ = get_nice_ticks(xlims[0], xlims[1], **kwticksx)
        ax.xaxis.set_major_locator(ticker.FixedLocator(major_tiks))
        ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_tiks))

    if y_step is not None and ylims is None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_step / 4))
    elif ylims is not None:
        major_tiks, minor_tiks, _ = get_nice_ticks(ylims[0], ylims[1], **kwticksy)
        ax.yaxis.set_major_locator(ticker.FixedLocator(major_tiks))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_tiks))
    elif not is_empty:
        ylims = ax.get_ylim()
        major_tiks, minor_tiks, _ = get_nice_ticks(ylims[0], ylims[1], **kwticksy)
        ax.yaxis.set_major_locator(ticker.FixedLocator(major_tiks))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_tiks))
        
    ax.grid(True)

def get_nice_ticks(vmin, vmax, max_major_ticks=6, n_minor=4):
    """
    Compute 'nice' evenly spaced major and minor tick locations for a given range.

    This helper function uses matplotlib's `MaxNLocator` to determine a clean and 
    readable spacing between major ticks and then inserts minor ticks uniformly 
    between them.

    Parameters:
        vmin (float): Minimum value of the axis range.
        vmax (float): Maximum value of the axis range.
        max_major_ticks (int, optional): Desired maximum number of major ticks.
        n_minor (int, optional): Number of minor ticks between each major tick.

    Returns:
        tuple:
            - major_levels (np.ndarray): Array of major tick positions.
            - minor_levels (np.ndarray): Array of minor tick positions.
            - major_step (float): Distance between major ticks.
    """
    # Use MaxNLocator to pick a good major step
    locator = ticker.MaxNLocator(nbins=max_major_ticks, steps=[1, 2, 2.5, 4, 5, 10], min_n_ticks=max_major_ticks)
    major_levels = locator.tick_values(vmin, vmax)

    # Compute actual step
    major_step = np.round(major_levels[1] - major_levels[0], 10)

    # Align start and end
    major_start = np.floor(vmin / major_step) * major_step
    major_end   = np.ceil(vmax / major_step) * major_step
    major_levels = np.arange(major_start, major_end + major_step, major_step)

    # Minor levels
    minor_step = major_step / (n_minor + 1)
    minor_start = np.floor(vmin / minor_step) * minor_step
    minor_end   = np.ceil(vmax / minor_step) * minor_step
    all_minor = np.arange(minor_start, minor_end + minor_step, minor_step)

    # Remove any minor levels that are major
    minor_levels = np.setdiff1d(np.round(all_minor, 10), np.round(major_levels, 10))

    return major_levels, minor_levels, major_step

#######################################################################################
    
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
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# Interpolation
from scipy.interpolate import interp1d, UnivariateSpline

# -----------------------------------------

__all__ = [
    "set_paper_parameters",
    "smooth_interpolation",
    "vector2d",
    "zoom_range",
    "alpha_cmap",
    "config_data_axis"
]

#######################################################################################

def set_paper_parameters(fontsize=12, fontfamily="serif", uselatex=True):
    """
    Set Matplotlib parameters for consistent styling.

    This function configures Matplotlib settings to control the appearance of text, 
    fonts, and math formatting in plots. It allows you to set the font size, font 
    family, and whether to use LaTeX for rendering text and mathematical expressions.

    Parameters:
        fontsize (int, optional): The font size for the text in the plot. Default is 12.
        fontfamily (str, optional): The font family to use for the text. Default is "serif".
        uselatex (bool, optional): If True, LaTeX is used to render the text and math. 
            Default is True. If False, Matplotlib's default text rendering is used.

    Returns:
        None

    Notes:
        - LaTeX rendering is enabled by setting the `usetex` parameter, which requires 
        a working LaTeX installation.
        - The function sets the font to be used for both text and math expressions and 
        loads the `amsmath` LaTeX package to handle advanced math formatting.
        - The math font is set to use the Computer Modern font set (a standard in LaTeX).

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
    Perform smooth interpolation to avoid big jumps.
    
    Parameters
    ----------
    x : array-like
        The x-coordinates of the data points.
    y : array-like
        The y-coordinates of the data points (may contain NaNs for missing values).
    method : str, optional
        Interpolation method: "linear", "quadratic", "cubic", or "spline".
        - "linear", "quadratic", "cubic" use `interp1d`
        - "spline" uses `UnivariateSpline` for smooth results
    num_points : int, optional
        Number of interpolated points for smooth plotting (default: 100).
    
    Returns
    -------
    x_smooth : numpy.ndarray
        Interpolated x values for smooth plotting.
    y_smooth : numpy.ndarray
        Interpolated y values.
    
    Example
    -------
    x_smooth, y_smooth = smooth_interpolation(x, y, method="cubic")
    plt.plot(x_smooth, y_smooth, label="Smoothed Curve")
    plt.scatter(x, y, color="red", label="Original Data")
    plt.legend()
    plt.show()
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
    Plot a 2D vector on a Matplotlib Axes object.

    This function simplifies plotting a 2D vector as an arrow on a given `ax` (Matplotlib 
    Axes). The arrow is drawn from the starting point `P0` to the end point `Pf`, with 
    customizable parameters for color, line style, size, and arrowhead properties.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object where the vector will be plotted.
        P0 (tuple or list): A tuple or list representing the starting point of the vector 
            (x, y) in the 2D plane.
        Pf (tuple or list): A tuple or list representing the endpoint of the vector 
            (x, y) in the 2D plane.
        c (str, optional): The color of the arrow. Default is "k" (black).
        ls (str, optional): The line style of the arrow. Default is solid line ("-").
        s (float, optional): Scaling factor for the vector. Default is 1.
        lw (float, optional): Line width of the arrow. Default is 0.7.
        hw (float, optional): The width of the arrowhead. Default is 0.1.
        hl (float, optional): The length of the arrowhead. Default is 0.2.
        alpha (float, optional): The transparency of the arrow, between 0 (transparent) 
            and 1 (opaque). Default is 1.
        zorder (int, optional): The z-order for layering the arrow on the plot. Default is 1.

    Returns:
        matplotlib.patches.FancyArrowPatch: The arrow patch representing the 2D vector.

    Notes:
        - The `length_includes_head=True` ensures the arrowhead is included in the vector's length.
        - The function uses `ax.arrow()` to draw the arrow and returns the arrow object, 
        allowing further modifications if needed.

    Example:
        vector2d(ax, P0=(0, 0), Pf=(1, 2), c="red", s=2)
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
    Compute a 1D range zoomed around a center.

    This function computes a zoomed range by scaling the distance between the
    center and the range bounds. Adapted from: 
    https://gist.github.com/dukelec/e8d4171ef4d12f9998295cfcbe3027ce.

    Parameters:
        begin (float): The starting bound of the range.
        end (float): The ending bound of the range.
        center (float): The center of the zoom (i.e., the invariant point).
        scale_factor (float): The scale factor to apply to the range.

    Returns:
        tuple: A tuple (min, max) representing the zoomed range.
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
    Apply alpha to the desired colormap.

    This function modifies an existing colormap by adding an alpha channel, which can 
    help avoid issues when using pcolormesh in Matplotlib. Directly applying alpha 
    to pcolormesh can cause various problems. A better approach is to generate and 
    use a pre-diluted colormap on a white background.

    Parameters:
        cmap (matplotlib.colors.Colormap): The original colormap to modify.
        alpha (float): The alpha transparency value to apply, ranging from 0 (fully 
            transparent) to 1 (fully opaque).

    Returns:
        matplotlib.colors.ListedColormap: A new colormap with the specified alpha 
        transparency applied.

    Notes:
        - The alpha value is applied uniformly across all colors in the colormap.
        - The blending is done by interpolating between the colormap colors and a 
        white background, using the alpha value.
        - This approach ensures that the resulting colormap works seamlessly in plots 
        like `pcolormesh`.

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


def config_data_axis(ax, x_step = None, y_step = None, y_right = True, 
                     format_float = False, xlims = None, ylims = None):
    """
    Configure the data axis properties of a Matplotlib Axes object.

    This function customizes various aspects of a Matplotlib axis (`ax`), including 
    tick spacing, axis limits, tick formatting, and whether ticks appear on the 
    right side of the y-axis. It also enables gridlines for better visualization.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object to configure.
        x_step (float, optional): Step size for the major ticks on the x-axis. 
            If provided, minor ticks are set at one-fourth of this value.
        y_step (float, optional): Step size for the major ticks on the y-axis. 
            If provided, minor ticks are set at one-fourth of this value.
        y_right (bool, optional): If True, moves the y-axis ticks to the right side. 
            Defaults to True.
        format_float (bool, optional): If True, formats y-axis tick labels as floats 
            with two decimal places. Defaults to False.
        xlims (tuple, optional): A tuple (xmin, xmax) to set the x-axis limits. 
            If None, the limits remain unchanged.
        ylims (tuple, optional): A tuple (ymin, ymax) to set the y-axis limits. 
            If None, the limits remain unchanged.

    Returns:
        None

    Notes:
        - Minor ticks are automatically calculated as one-fourth of the major tick step size.
        - Gridlines are always enabled after this configuration.
    """
    
    if x_step is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_step / 4))
    if y_step is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_step / 4))

    if y_right:
        ax.yaxis.tick_right()

    if format_float:
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.grid(True)

#######################################################################################
    
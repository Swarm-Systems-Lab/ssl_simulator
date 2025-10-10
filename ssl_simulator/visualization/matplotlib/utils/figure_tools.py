"""
Matplotlib Figure and Axis Initialization Utilities
"""

__all__ = [
    "initialize_plot",
]
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

#######################################################################################

def initialize_plot(ax=None, figsize=(8, 8), projection='3d', **kwargs):
    """
    Initialize a matplotlib figure and axis with optional 3D view.

    Args:
        ax: Existing matplotlib axis (optional).
        figsize: Tuple specifying figure size (default: (8, 8)).
        projection: Projection type for the axis, e.g., '3d' (default: '3d').
        view: Tuple specifying the view as (elev, azim) (default: None).
        **kwargs: Additional keyword arguments for `add_subplot()`.

    Returns:
        fig: The created matplotlib figure (or None if ax is provided).
        ax: The matplotlib axis (either provided or newly created).
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection, **kwargs)
        return fig, ax
    return None, ax
"""Matplotlib Figure and Axis Initialization Utilities."""

__all__ = [
    "initialize_plot",
    "save_paper_figure",
    "set_paper_parameters",
]

import logging
import os

import matplotlib
import matplotlib.pyplot as plt

from ssl_simulator.utils.path_ops import create_dir

logger = logging.getLogger(__name__)


def initialize_plot(ax=None, figsize=(8, 8), projection="3d", **kwargs):
    """
    Initialize a matplotlib figure and axis with optional 3D view.

    Args:
        ax: Existing matplotlib axis (optional).
        figsize: Tuple specifying figure size (default: (8, 8)).
        projection: Projection type for the axis, e.g., '3d' (default: '3d').
        view: Tuple specifying the view as (elev, azim) (default: None).
        **kwargs: Additional keyword arguments for `add_subplot()`.

    Returns
    -------
        fig: The created matplotlib figure (or None if ax is provided).
        ax: The matplotlib axis (either provided or newly created).
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection, **kwargs)
        return fig, ax
    return None, ax


def set_paper_parameters(fontsize=12, fontfamily="serif", uselatex=True):
    """
    Set Matplotlib parameters for consistent, publication-quality plots.

    This function configures font styles, sizes, and LaTeX rendering options for consistent
    figure appearance across plots.

    Parameters
    ----------
        fontsize (int, optional): Global font size. Default is 12.
        fontfamily (str, optional): Font family (e.g., "serif", "Arial"). Default is "serif".
        uselatex (bool, optional): Whether to use LaTeX rendering. Default is True.

    Returns
    -------
        None

    Notes
    -----
        - Requires LaTeX installed if `uselatex=True`.
        - Applies to all future plots in the session.
        - Math rendering uses the AMS math package and Computer Modern fonts.

    Example:
        set_paper_parameters(fontsize=14, fontfamily="Arial", uselatex=False)
    """
    matplotlib.rc("font", **{"size": fontsize, "family": fontfamily})

    matplotlib.rc("text", **{"usetex": uselatex, "latex.preamble": r"\usepackage{amsmath}"})

    matplotlib.rc("mathtext", **{"fontset": "cm"})


def save_paper_figure(
    img_name,
    output_dir,
    fig=None,
    apply_paper_params=True,
    fontsize=12,
    fontfamily="serif",
    uselatex=True,
    figure_dpi=100,
    save_dpi=300,
    low_quality_dpi=100,
    bbox="tight",
    pad_inches=0.1,
    transparent=False,
    save_pdf=True,
    close=False,
):
    """
    Save a Matplotlib figure with publication-friendly defaults for LaTeX papers.

    The function can optionally apply paper parameters, set save-related rcParams,
    create the output directory if needed, and export the figure to:
        - PDF (vector)
        - PNG (high quality)
        - PNG low-quality preview

    Parameters
    ----------
        img_name (str): Base filename (without extension).
        output_dir (str): Directory where files are saved.
        fig (matplotlib.figure.Figure, optional): Figure to save. If None, uses current figure.
        apply_paper_params (bool, optional): Whether to call set_paper_parameters().
        fontsize (int, optional): Font size for set_paper_parameters().
        fontfamily (str, optional): Font family for set_paper_parameters().
        uselatex (bool, optional): Use LaTeX text rendering in set_paper_parameters().
        figure_dpi (int, optional): rcParams['figure.dpi'] value.
        save_dpi (int, optional): DPI for high-quality PNG and rcParams['savefig.dpi'].
        low_quality_dpi (int, optional): DPI for low-quality PNG preview.
        bbox (str, optional): Bounding box mode for savefig.
        pad_inches (float, optional): Padding around saved figure.
        transparent (bool, optional): Save with transparent background if True.
        save_pdf (bool, optional): Save PDF version if True.
        close (bool, optional): Close figure after saving if True.

    Returns
    -------
        dict: Paths of the generated files with keys: 'pdf', 'png', 'png_lq'.
    """
    if apply_paper_params:
        set_paper_parameters(fontsize=fontsize, fontfamily=fontfamily, uselatex=uselatex)

    plt.rcParams["figure.dpi"] = figure_dpi
    plt.rcParams["savefig.dpi"] = save_dpi
    plt.rcParams["savefig.bbox"] = bbox
    plt.rcParams["savefig.pad_inches"] = pad_inches

    os.makedirs(output_dir, exist_ok=True)

    active_fig = fig if fig is not None else plt.gcf()

    pdf_path = os.path.join(output_dir, f"{img_name}.pdf")
    png_path = os.path.join(output_dir, f"{img_name}.png")
    png_lq_path = os.path.join(output_dir, f"{img_name}_lq.png")

    save_kwargs = {
        "bbox_inches": bbox,
        "pad_inches": pad_inches,
        "transparent": transparent,
    }

    result = {}
    formats_saved = []

    if save_pdf:
        active_fig.savefig(pdf_path, **save_kwargs)
        result["pdf"] = pdf_path
        formats_saved.append(".pdf")

    active_fig.savefig(png_path, dpi=save_dpi, **save_kwargs)
    result["png"] = png_path
    formats_saved.append(f".png@{save_dpi}dpi")

    active_fig.savefig(png_lq_path, dpi=low_quality_dpi, **save_kwargs)
    result["png_lq"] = png_lq_path
    formats_saved.append(f"_lq.png@{low_quality_dpi}dpi")

    logger.info(
        "Saved figure '%s' to %s (%s)",
        img_name,
        output_dir,
        ", ".join(formats_saved),
    )

    if close:
        plt.close(active_fig)
        logger.debug("Closed figure after saving: %s", img_name)

    return {
        "pdf": pdf_path,
        "png": png_path,
        "png_lq": png_lq_path,
    }

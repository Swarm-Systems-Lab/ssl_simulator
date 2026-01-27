"""File: _requirements.py."""

#######################################################################################
# Standard Libraries
import shutil

# Graphic Tools (Visualization)
import matplotlib
import matplotlib.pyplot as plt

# Algebra (Numerical computation)
#######################################################################################
import pytest

# Animation Tools (For creating animations)
from matplotlib.animation import FuncAnimation

#######################################################################################
# Third-Party Libraries
#######################################################################################
# Import the Main Module of the Project
import ssl_simulator

# Test some import shortcuts


def test_import():
    assert ssl_simulator is not None


def test_latex_installed():
    if not shutil.which("latex"):
        pytest.skip("LaTeX is not installed or not found in PATH.")
    fontsize = 12
    matplotlib.rc("font", **{"size": fontsize, "family": "serif"})
    matplotlib.rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amsmath}"})
    matplotlib.rc("mathtext", **{"fontset": "cm"})
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, r"Test: $\int_a^b f(x)dx$", fontsize=12, ha="center")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.close(fig)


def test_ffmpeg_installed():
    try:

        def animate(i):
            pass

        fig, _ax = plt.subplots()
        anim = FuncAnimation(fig, animate, frames=1)
        anim.to_html5_video()
        plt.close(fig)
    except Exception as e:
        pytest.skip(f"FFmpeg is missing or not working: {e}")

"""
File: patches.py


"""

# Third-party libraries -------------------

# Algebra
import numpy as np

# Visualization
from matplotlib.path import Path
import matplotlib.patches as patches

# -----------------------------------------

__all__ = [
    "unicycle_patch",
    "fixedwing_patch"
]

#######################################################################################

def unicycle_patch(XY, yaw, size=1, **patch_kwargs):
    """
    Generate a Matplotlib patch representing a unicycle.

    The unicycle is visualized as a triangular patch with a given position (`XY`), 
    heading (`yaw`), and size. Additional keyword arguments are passed to customize 
    patch properties (e.g., color, edge width).

    Parameters:
        XY (tuple or list): The (X, Y) coordinates of the unicycle's center.
        yaw (float): The heading (orientation) in radians.
        size (float, optional): Scaling factor for the unicycle. Default is 1.
        **patch_kwargs: Additional properties for `PathPatch` (e.g., `fc`, `ec`, `lw`).

    Returns:
        matplotlib.patches.PathPatch: A Matplotlib patch representing the unicycle.

    Example:
        ax.add_patch(unicycle_patch([2, 3], np.pi / 4, size=1.5, fc="red", lw=0.5))
    """
    # Rotation matrix for transforming the shape
    Rot = np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]
    ])

    # Define the geometry of the triangle (unicycle shape)
    apex_angle = np.radians(30) # 30-degree apex angle
    side_length = size / np.sin(apex_angle)          # Triangle side length
    base_half = side_length * np.sin(apex_angle / 2) # Half of the base
    height = side_length * np.cos(apex_angle / 2)    # Triangle height

    # Define the local vertices of the unicycle
    vertex_1 = np.array([base_half, -0.3 * height])  # Bottom right
    vertex_2 = np.array([-base_half, -0.3 * height]) # Bottom left
    vertex_3 = np.array([0, 0.6 * height])  # Top (front)

    # Apply rotation
    vertex_1 = Rot.dot(vertex_1)
    vertex_2 = Rot.dot(vertex_2)
    vertex_3 = Rot.dot(vertex_3)

    # Convert to global coordinates
    verts = [
        (XY[0] + vertex_1[1], XY[1] + vertex_1[0]),
        (XY[0] + vertex_2[1], XY[1] + vertex_2[0]),
        (XY[0] + vertex_3[1], XY[1] + vertex_3[0]),
        (0, 0),  # Close the path
    ]

    # Define path codes
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(verts, codes)

    return patches.PathPatch(path, **patch_kwargs)

def fixedwing_patch(XY, yaw, size=1, **patch_kwargs):
    """
    Generate a Matplotlib patch representing a fixed-wing aircraft.

    The fixed-wing aircraft is visualized as an arrow-shaped patch with a given 
    position (`XY`), heading (`yaw`), and size. Additional keyword arguments are 
    passed to customize patch properties (e.g., color, edge width).

    Parameters:
        XY (tuple or list): The (X, Y) coordinates of the aircraft's center.
        yaw (float): The heading (orientation) in radians.
        size (float, optional): Scaling factor for the aircraft. Default is 1.
        **patch_kwargs: Additional properties for `PathPatch` (e.g., `fc`, `ec`, `lw`).

    Returns:
        matplotlib.patches.PathPatch: A Matplotlib patch representing the fixed-wing.

    Example:
        ax.add_patch(fixedwing_patch([2, 3], np.pi / 4, size=1.5, fc="blue", lw=0.5))
    """
    # Rotation matrix for transforming the shape
    yaw -= np.pi/2
    Rot = np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]
    ])

    # Define the aircraft shape in local coordinates (before rotation)
    wing_span = size * 1.2
    fuselage_length = size * 0.6

    points = np.array([
        [0, -fuselage_length],    # Nose
        [-wing_span / 2, -size],  # Left wingtip
        [-wing_span / 3, -size * 0.6],  # Left inner wing
        [0, -size * 0.1],         # Center fuselage
        [wing_span / 3, -size * 0.6],  # Right inner wing
        [wing_span / 2, -size],   # Right wingtip
        [0, -fuselage_length]     # Close shape back to nose
    ]) + np.array([[0, fuselage_length/2]])

    rotated_points = np.dot(points, Rot)

    # Construct the patch vertices and shift to position XY
    verts = [(XY[0] + x, XY[1] + y) for x, y in rotated_points]

    # Define path and create patch
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    path = Path(verts, codes)

    return patches.PathPatch(path, **patch_kwargs)

#######################################################################################
    
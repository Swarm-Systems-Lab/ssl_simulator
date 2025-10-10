"""
Matplotlib Animation Utilities
"""

__all__ = [
    "update_quivers",
    "update_scatters",
    ]

import numpy as np
import matplotlib.pylab as plt

#######################################################################################

def update_scatters(scatter_list, p_frame):
    """
    Update 3D scatter artists in place.

    Args:
        scatter_list : list of Path3DCollection
            Scatter objects, one per agent.
        p_frame : np.ndarray, shape (n_agents, 3)
            Updated positions.
    """
    n_agents = len(scatter_list)
    assert n_agents == p_frame.shape[0], (
        f"Expected {n_agents} scatter artists, got {p_frame.shape[0]} positions."
    )

    for n in range(n_agents):
        sc = scatter_list[n]
        x, y, z = p_frame[n]
        # Matplotlib expects 1-element arrays
        sc._offsets3d = (np.array([x]), np.array([y]), np.array([z]))

def update_quivers(quivers_array, p_frame, R_frame, arr_len=1.0):
    """
    Update quiver artists for multiple agents using set_segments.

    quivers_array: shape (n_agents, 3) → each agent has 3 quivers (x,y,z)
    p_frame: (n_agents, 3) positions
    R_frame: (n_agents, 3, 3) rotation matrices
    arr_len: length scaling for quivers
    """
    n_agents = p_frame.shape[0]
    assert quivers_array.shape == (n_agents, 3), (
        f"Expected quivers_array shape ({n_agents}, 3), got {quivers_array.shape}."
    )

    for n in range(n_agents):
        base = np.asarray(p_frame[n], dtype=float)
        for k in range(3):
            vec = np.asarray(R_frame[n, :, k], dtype=float) * arr_len
            tip = base + vec
            # Convert to list and wrap in a 3D segment list
            new_seg = [[base.tolist(), tip.tolist()]]
            quivers_array[n, k].set_segments(new_seg)
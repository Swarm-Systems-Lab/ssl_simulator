__all__ = ["plot_so3_attitude_vectors", "plot_so3_heading_traj", "plot_so3_omega_traj"]

from typing import Any, cast

import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ssl_simulator.math import check_and_parse_dimensions, so3_log_map, so3_vee
from ssl_simulator.visualization.matplotlib.utils.figure_tools import initialize_plot

#######################################################################################


def plot_3d_sphere_wf(ax: Axes3D, radius: float, projections: bool = False, surface: bool = False):
    # Creating a 3D wireframe sphere with highlighted meridians (0° and 90° longitude)
    # and the equator (0° latitude) using matplotlib.
    artists = {}

    if ax is None:
        raise ValueError("An axis must be provided to plot_3d_sphere_wf")

    # Sphere parameters
    N_lon = 73
    N_lat = 37
    lon = np.linspace(0, 2 * np.pi, N_lon)
    lat = np.linspace(-np.pi / 2, np.pi / 2, N_lat)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Spherical -> Cartesian
    x = radius * np.cos(lat_grid) * np.cos(lon_grid)
    y = radius * np.cos(lat_grid) * np.sin(lon_grid)
    z = radius * np.sin(lat_grid)

    # Compute strides for 15° steps
    rstride = round(15 / (180 / (N_lat - 1)))  # along latitude
    cstride = round(15 / (360 / (N_lon - 1)))  # along longitude

    # Wireframe sphere and surface
    wf = ax.plot_wireframe(
        x, y, z, rstride=rstride, cstride=cstride, linewidth=0.6, alpha=0.2, color="k", zorder=0
    )
    artists["sphere_wireframe"] = wf

    if surface:
        # Surface plot with low alpha so wireframe and lines are visible
        sf = ax.plot_surface(x, y, z, color="lightgrey", alpha=0.1, edgecolor="none", zorder=-1)
        artists["sphere_surface"] = sf

    # Highlight meridians: 0° (lon=0) and 90° (lon=90° = pi/2)
    lats_for_lines = np.linspace(-np.pi / 2, 3 * np.pi / 2, 300)
    lon_0 = 0
    lon_90 = np.pi / 2

    radius_mark = radius * 1.01  # Slightly larger radius for visibility
    x_lon0 = radius_mark * np.cos(lats_for_lines) * np.cos(lon_0)
    y_lon0 = radius_mark * np.cos(lats_for_lines) * np.sin(lon_0)
    z_lon0 = radius_mark * np.sin(lats_for_lines)

    x_lon90 = radius_mark * np.cos(lats_for_lines) * np.cos(lon_90)
    y_lon90 = radius_mark * np.cos(lats_for_lines) * np.sin(lon_90)
    z_lon90 = radius_mark * np.sin(lats_for_lines)

    kw_lines = {"linewidth": 2.5, "color": "grey", "alpha": 1}

    mid_len = x_lon0.shape[0] // 2
    (m1,) = ax.plot(
        x_lon0[0:mid_len], y_lon0[0:mid_len], z_lon0[0:mid_len], **kw_lines, label="Lon 0º"
    )
    (m2,) = ax.plot(
        x_lon90[0:mid_len], y_lon90[0:mid_len], z_lon90[0:mid_len], **kw_lines, label="Lon 90º"
    )
    (m3,) = ax.plot(
        x_lon0[mid_len:], y_lon0[mid_len:], z_lon0[mid_len:], **kw_lines, linestyle="--"
    )
    (m4,) = ax.plot(
        x_lon90[mid_len:], y_lon90[mid_len:], z_lon90[mid_len:], **kw_lines, linestyle="--"
    )

    # Highlight equator: latitude 0°
    lons_for_equator = np.linspace(0, 2 * np.pi, 400)
    lat_eq = 0.0
    x_eq = radius_mark * np.cos(lat_eq) * np.cos(lons_for_equator)
    y_eq = radius_mark * np.cos(lat_eq) * np.sin(lons_for_equator)
    z_eq = radius_mark * np.sin(lat_eq) * np.ones_like(lons_for_equator)
    (eq,) = ax.plot(x_eq, y_eq, z_eq, **kw_lines, label="Lat 0º")

    artists["meridian_lon0"] = m1
    artists["meridian_lon90"] = m2
    artists["meridian_lon0_dashed"] = m3
    artists["meridian_lon90_dashed"] = m4
    artists["equator"] = eq

    # Aspect and legend
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper left", ncol=3)

    if projections:
        # Get actual axis limits for projections
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        # Define contour levels every 15 degrees in z (latitude)
        angles = np.arange(-90, 91, 15)  # -90°, -75°, ..., 75°, 90°
        levels = np.sin(np.radians(angles))  # convert to z = sin(lat)

        # Use axis limits for projection offsets
        ax.contour(
            x,
            y,
            z,
            levels=levels,
            alpha=0.2,
            linewidths=0.7,
            colors=["k"],
            zdir="z",
            offset=zlim[0] * 1.04,
        )
        ax.contour(
            x,
            y,
            z,
            levels=levels,
            alpha=0.2,
            linewidths=0.7,
            colors=["k"],
            zdir="y",
            offset=ylim[1] * 1.04,
        )
        ax.contour(
            x,
            y,
            z,
            levels=levels,
            alpha=0.2,
            linewidths=0.7,
            colors=["k"],
            zdir="x",
            offset=xlim[0] * 1.04,
        )

    return artists


#######################################################################################
# SO(3) Visualization Tools


def plot_so3_attitude_vectors(
    R_data, p_data=None, ax=None, arr_len=1.0, l_list=None, sphere=True, quiver_thickness=2
):
    """
    Plot attitude vectors (body frame axes) at positions p_data using rotations R_data.
    Args:
        ax: matplotlib 3D axis
        p_data: (time_frames, agents, 3) positions
        R_data: (time_frames, agents, 3, 3) rotation matrices
        arr_len: length of quiver arrows
        quiver_thickness: linewidth of the quiver arrows.
    """
    artists = {}
    ax_cols = ["r", "g", "b"]

    # Filtering the input (time_frames, agents, 3, 3)
    R_data = check_and_parse_dimensions(R_data, (None, None, 3, 3), "R_data")
    n_agents = R_data.shape[1]

    if p_data is not None:
        p_data = check_and_parse_dimensions(p_data, (None, n_agents, 3), "p_data")
    else:
        p_data = np.zeros((R_data.shape[0], n_agents, 3))

    # -- Figure init --
    _, main_ax = initialize_plot(ax, figsize=(8, 8), projection="3d")

    # Set labels and limits
    main_ax.set_xlabel(r"$X$ [L]")
    main_ax.set_ylabel(r"$Y$ [L]")
    main_ax.set_zlabel(r"$Z$ [L]")
    main_ax.grid(True)

    # Plot the attitude vectors at specified time frames
    if l_list is None:
        l_list = [0]

    list_len = len(l_list)
    quivers_array = np.empty((list_len, n_agents, 3), dtype=object)
    scatter_array = np.empty((list_len, n_agents), dtype=object)

    for li, l_idx in enumerate(l_list):
        for n in range(n_agents):
            x, y, z = p_data[l_idx, n]  # unpack coordinates
            # Scatter
            scatter_array[li, n] = main_ax.scatter(x, y, z, marker="o", color="k")
            # Quivers for body axes
            quivers_n = []
            for i in range(3):
                q = main_ax.quiver(
                    x,
                    y,
                    z,
                    R_data[l_idx, n, 0, i],
                    R_data[l_idx, n, 1, i],
                    R_data[l_idx, n, 2, i],
                    color=ax_cols[i],
                    length=0.999 * arr_len,
                    normalize=True,
                    alpha=1,
                    linewidth=quiver_thickness,
                )
                quivers_n.append(q)
            quivers_array[li, n, :] = quivers_n

    artists["quivers"] = quivers_array
    artists["scatter"] = scatter_array

    # Optionally plot the sphere
    if sphere:
        sphere_artists = plot_3d_sphere_wf(main_ax, arr_len, projections=False, surface=True)
        artists.update(sphere_artists)

    return artists


def plot_so3_heading_traj(R_data, ax=None, lim=1.6):
    """- Funtion to visualize the 3D heading trajectory -."""
    # Filtering the input (time_frames, agents, 3, 3)
    R_data = check_and_parse_dimensions(R_data, (None, None, 3, 3), "R_data")

    # Generate the 3D heading trajectory
    u = R_data[:, :, 0, :]

    # -- Figure init --
    _, main_ax = initialize_plot(ax, figsize=(8, 8), projection="3d")

    # Format of the axis
    main_ax.set_xlabel(r"$X$ [L]")
    main_ax.set_ylabel(r"$Y$ [L]")
    main_ax.set_zlabel(r"$Z$ [L]")

    main_ax.set_proj_type("ortho")
    main_ax.set_xlim([-lim, lim])
    main_ax.set_ylim([-lim, lim])
    main_ax.set_zlim([-lim, lim])
    main_ax.grid(True)

    # Plot the 3D sphere and its 2D projections
    plot_3d_sphere_wf(main_ax, 1, projections=True, surface=True)

    if R_data.shape[1] > 1:
        for n in range(R_data.shape[1]):
            # Plot the 3D heading trajectories
            main_ax.plot(u[0, n, 0], u[0, n, 1], u[0, n, 2], "or", markersize=2, alpha=0.5)
            main_ax.plot(u[:, n, 0], u[:, n, 1], u[:, n, 2], "b", lw=0.8, alpha=0.8)
            if R_data.shape[0] > 1:
                main_ax.plot(u[-1, n, 0], u[-1, n, 1], u[-1, n, 2], "or", markersize=3, alpha=1)

            # Plot the 2D heading trajectories projections
            main_ax.plot(u[0, n, 0], u[0, n, 1], "og", zdir="z", zs=-lim, markersize=2, alpha=0.5)
            main_ax.plot(u[0, n, 0], u[0, n, 2], "og", zdir="y", zs=lim, markersize=2, alpha=0.5)
            main_ax.plot(u[0, n, 1], u[0, n, 2], "og", zdir="x", zs=-lim, markersize=2, alpha=0.5)

            main_ax.plot(u[:, n, 0], u[:, n, 1], "r", zdir="z", zs=-lim, alpha=0.5)
            main_ax.plot(u[:, n, 0], u[:, n, 2], "r", zdir="y", zs=lim, alpha=0.5)
            main_ax.plot(u[:, n, 1], u[:, n, 2], "r", zdir="x", zs=-lim, alpha=0.5)

            if R_data.shape[0] > 1:
                main_ax.plot(
                    u[-1, n, 0], u[-1, n, 1], "og", zdir="z", zs=-lim, markersize=2, alpha=1
                )
                main_ax.plot(
                    u[-1, n, 0], u[-1, n, 2], "og", zdir="y", zs=lim, markersize=2, alpha=1
                )
                main_ax.plot(
                    u[-1, n, 1], u[-1, n, 2], "og", zdir="x", zs=-lim, markersize=2, alpha=1
                )

    else:
        n = 0
        # Plot the 3D heading point
        main_ax.plot(u[0, n, 0], u[0, n, 1], u[0, n, 2], "or", markersize=4, alpha=1)

        # Plot the 2D heading point projections
        main_ax.plot(u[0, n, 0], u[0, n, 1], "or", zdir="z", zs=-lim, markersize=2, alpha=1)
        main_ax.plot(u[0, n, 0], u[0, n, 2], "or", zdir="y", zs=lim, markersize=2, alpha=1)
        main_ax.plot(u[0, n, 1], u[0, n, 2], "or", zdir="x", zs=-lim, markersize=2, alpha=1)


def plot_so3_omega_traj(R_data, ax=None):
    r"""- Funtion to visualize a trajectory in SO(3) using \omega \in so(3) (associated Lie algebra) -."""
    # Filtering the input (time_frames, agents, 3, 3)
    R_data = check_and_parse_dimensions(R_data, (None, None, 3, 3), "R_data")

    # Generate the SO(3) points
    omega = so3_vee(so3_log_map(R_data)) / np.pi  # Scale to pi

    # -- Figure init --
    _, main_ax = initialize_plot(ax, figsize=(8, 8), projection="3d")

    # Format of the axis
    lims = [-1.5, 1.5]
    main_ax.set_proj_type("ortho")
    main_ax.set_xlim(lims)
    main_ax.set_ylim(lims)
    main_ax.set_zlim(lims)

    main_ax.set_xlabel(r"$w_x/\pi$")
    main_ax.set_ylabel(r"$w_y/\pi$")
    main_ax.set_zlabel(r"$w_z/\pi$")
    main_ax.grid(True)

    # 3D sphere plot and its 2D projections
    plot_3d_sphere_wf(main_ax, 1, projections=True, surface=True)

    for n in range(R_data.shape[1]):
        # 3D SO(3) trajectory plot
        main_ax.plot(omega[0, n, 0], omega[0, n, 1], omega[0, n, 2], "or", markersize=2, alpha=0.5)
        main_ax.plot(
            omega[:, n, 0], omega[:, n, 1], omega[:, n, 2], ".b", markersize=0.5, lw=0.8, alpha=0.4
        )
        if R_data.shape[0] > 1:
            main_ax.plot(
                omega[-1, n, 0], omega[-1, n, 1], omega[-1, n, 2], "og", markersize=3, alpha=0.9
            )

        # Get actual axis limits for projections (use main_ax)
        xlim = main_ax.get_xlim()
        ylim = main_ax.get_ylim()
        zlim = main_ax.get_zlim()

        # 2D SO(3) projection plots
        main_ax.plot(
            omega[:, n, 0],
            omega[:, n, 1],
            ".r",
            zdir="z",
            zs=zlim[0] * 1.04,
            markersize=0.5,
            alpha=0.7,
        )
        main_ax.plot(
            omega[:, n, 0],
            omega[:, n, 2],
            ".r",
            zdir="y",
            zs=ylim[1] * 1.04,
            markersize=0.5,
            alpha=0.7,
        )
        main_ax.plot(
            omega[:, n, 1],
            omega[:, n, 2],
            ".r",
            zdir="x",
            zs=xlim[0] * 1.04,
            markersize=0.5,
            alpha=0.7,
        )

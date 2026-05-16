from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.mapping.voxel_map import downsample_to_max_points


def save_matplotlib_figure(fig, output_path, dpi=300, close=True):
    """
    Save a Matplotlib figure and optionally close it.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)

    if close:
        plt.close(fig)


def set_axes_equal_3d(ax):
    """
    Set equal scale for a 3D Matplotlib axis.
    Useful for point clouds, trajectories, and 3D map plots.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range, 1e-6])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def cloud_to_plot_arrays(
    cloud,
    max_points=120000,
    initial_voxel_size=0.05,
):
    """
    Convert an Open3D cloud to NumPy arrays for plotting.

    Returns:
        points: Nx3 array
        colors: Nx3 array or None
    """
    points = np.asarray(cloud.points)

    if cloud.has_colors():
        colors = np.asarray(cloud.colors)
    else:
        colors = None

    points, colors = downsample_to_max_points(
        points,
        colors_rgb=colors,
        max_points=max_points,
        initial_voxel_size=initial_voxel_size,
    )

    return points, colors


def compute_equal_xy_limits(xy_arrays, margin_ratio=0.05):
    """
    Compute shared equal-aspect XY limits for one or more Nx2 arrays.

    Returns:
        xlim, ylim
    """
    valid_xy = [
        np.asarray(xy, dtype=np.float64)
        for xy in xy_arrays
        if xy is not None and len(xy) > 0
    ]

    if len(valid_xy) == 0:
        return None, None

    all_xy = np.vstack(valid_xy)

    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    span = max(x_max - x_min, y_max - y_min)

    if span < 1e-6:
        span = 1.0

    margin = float(margin_ratio) * span
    half = 0.5 * span + margin

    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    return xlim, ylim
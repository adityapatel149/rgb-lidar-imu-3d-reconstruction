from pathlib import Path
import matplotlib.pyplot as plt

from src.odometry.trajectory_utils import poses_to_xyz
from src.utils.visualization import (
    compute_equal_xy_limits,
    save_matplotlib_figure,
)


def plot_trajectories(
    trajectories,
    output_path,
    title="Trajectory Comparison",
    align_to_origin=False,
    same_axes=True,
):
    """
    Plots each trajectory in its own subplot.

    Example:
    {
        "Ground Truth": gt,
        "IMU": imu,
        "Pure ICP": icp,
        "IMU+LiDAR Fused": fused,
    }
    """
    if len(trajectories) == 0:
        raise ValueError("Expected at least one trajectory to plot")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(trajectories)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5 * n, 5),
    )

    # Handle case when only one subplot exists
    if n == 1:
        axes = [axes]

    colors = {
        "Ground Truth": "black",
        "IMU": "green",
        "Pure ICP": "blue",
        "IMU+LiDAR Fused": "red",
    }

    xyz_by_name = {}

    for name, poses in trajectories.items():
        if poses is None or len(poses) == 0:
            xyz_by_name[name] = None
            continue

        xyz = poses_to_xyz(poses)

        if len(xyz) == 0:
            xyz_by_name[name] = None
            continue

        # Only align here if the input poses are not already relative.
        if align_to_origin:
            xyz = xyz - xyz[0]

        xyz_by_name[name] = xyz

    if same_axes:
        xlim, ylim = compute_equal_xy_limits(
            [
                xyz[:, :2]
                for xyz in xyz_by_name.values()
                if xyz is not None and len(xyz) > 0
            ]
        )
    else:
        xlim = None
        ylim = None


    for ax, (name, poses) in zip(axes, trajectories.items()):
        xyz = xyz_by_name[name]

        if xyz is None or len(xyz) == 0:
            ax.set_title(f"{name}\n(empty)")
            ax.grid(True)
            continue

        color = colors.get(name, None)

        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            color=color,
            linewidth=2,
        )

        # Start point
        ax.scatter(
            xyz[0, 0],
            xyz[0, 1],
            color="lime",
            marker="o",
            s=80,
            label="start",
        )

        # End point
        ax.scatter(
            xyz[-1, 0],
            xyz[-1, 1],
            color="red",
            marker="x",
            s=100,
            label="end",
        )

        ax.set_title(name)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        if same_axes and xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.legend(loc="best")

    fig.suptitle(title, fontsize=16)

    save_matplotlib_figure(fig, output_path, dpi=300)
    print(f"Saved trajectory plot to: {output_path}")
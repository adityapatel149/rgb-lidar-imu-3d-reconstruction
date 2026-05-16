import matplotlib.pyplot as plt

from src.odometry.trajectory_utils import poses_to_xyz
from src.utils.visualization import (
    cloud_to_plot_arrays,
    save_matplotlib_figure,
    set_axes_equal_3d,
)


def save_map_with_trajectory_plot(
    cloud,
    poses,
    output_path,
    title="Colored map with trajectory",
    max_points=120000,
    initial_voxel_size=0.05,
):
    points, colors = cloud_to_plot_arrays(
        cloud,
        max_points=max_points,
        initial_voxel_size=initial_voxel_size,
    )

    xyz = poses_to_xyz(poses)

    fig = plt.figure(figsize=(20, 10))

    # Top-down view
    ax_top = fig.add_subplot(1, 2, 1)

    # cloud
    if len(points) > 0:
        if colors is not None:
            ax_top.scatter(
                points[:, 0],
                points[:, 1],
                c=colors,
                s=0.9,
                linewidths=0,
            )
        else:
            ax_top.scatter(
                points[:, 0],
                points[:, 1],
                s=0.9,
                linewidths=0,
            )
    #trajectory
    if len(xyz) > 0:
        ax_top.plot(
            xyz[:, 0],
            xyz[:, 1],
            linewidth=2,
            label="trajectory",
        )

        ax_top.scatter(
            xyz[0, 0],
            xyz[0, 1],
            s=80,
            label="start",
        )

        ax_top.scatter(
            xyz[-1, 0],
            xyz[-1, 1],
            s=80,
            marker="x",
            label="end",
        )

    ax_top.set_title("Top-down view")
    ax_top.set_xlabel("x [m]")
    ax_top.set_ylabel("y [m]")
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.grid(True)
    ax_top.legend(loc="best")

    # 3D view
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    # Override zorder manually to draw trajectory points on top on cloud points
    ax_3d.computed_zorder = False

    # cloud
    if len(points) > 0:
        if colors is not None:
            ax_3d.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                s=0.9,
                zorder=1,
                linewidths=0,
            )
        else:
            ax_3d.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=0.9,
                zorder=1,
                linewidths=0,
            )
    # trajectory
    if len(xyz) > 0:
        ax_3d.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            linewidth=2,
            zorder=1000,
            label="trajectory",
        )

        ax_3d.scatter(
            xyz[0, 0],
            xyz[0, 1],
            xyz[0, 2],
            s=80,
            zorder=1001,
            label="start",
        )

        ax_3d.scatter(
            xyz[-1, 0],
            xyz[-1, 1],
            xyz[-1, 2],
            s=80,
            zorder=1001,
            marker="x",
            label="end",
        )

    ax_3d.set_title("3D view")
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    ax_3d.legend(loc="best")

    # Optional camera angle for nicer 3D view.
    ax_3d.view_init(elev=35, azim=-45)

    set_axes_equal_3d(ax_3d)
    ax_3d.set_box_aspect((1, 1, 1), zoom=2.0)

    ax_3d.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.suptitle(title, fontsize=16)

    save_matplotlib_figure(fig, output_path, dpi=300)
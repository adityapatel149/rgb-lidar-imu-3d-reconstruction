from pathlib import Path
import numpy as np
import open3d as o3d

from src.mapping.colorize_pointcloud import colorize_lidar_points
from src.mapping.voxel_map import make_point_cloud, voxel_downsample_cloud, downsample_to_max_points
from src.mapping.outlier_filter import remove_statistical_outliers
from src.utils.io import extract_frame_id, load_xyz_points_npy, save_cloud_ply
from src.utils.transforms import compose_transforms, transform_points



def fuse_colored_pointcloud_sequence(
    scene_dir,
    calibration,
    poses,
    camera_names,
    rgb_extension=".jpg",
    frame_stride=1,
    min_range=1.0,
    max_range=80.0,
    max_points_per_frame=None,
    min_depth=0.1,
    border_margin_px=2,
    center_weight=1.0,
    angle_weight=1.0,
    depth_weight=0.05,
    color_mode="best",
    voxel_size=0.15,
    remove_outliers=True,
    outlier_nb_neighbors=20,
    outlier_std_ratio=2.0,
    verbose=True,
):
    """
    Builds a global colored map from LiDAR frames, RGB images, and poses.

    poses must be a list of T_world_vehicle matrices aligned with sorted LiDAR files.
    """
    scene_dir = Path(scene_dir)
    lidar_dir = scene_dir / "lidar"
    lidar_files = sorted(lidar_dir.glob("*.npy"))

    if len(lidar_files) == 0:
        raise RuntimeError(f"No LiDAR files found in {lidar_dir}")

    n = min(len(lidar_files), len(poses))
    lidar_files = lidar_files[:n]
    poses = poses[:n]

    all_points_world = []
    all_colors = []
    frame_reports = []

    for i in range(0, n, int(frame_stride)):
        lidar_path = lidar_files[i]
        frame_id = extract_frame_id(lidar_path)
        T_world_vehicle = poses[i]

        points_lidar = load_xyz_points_npy(lidar_path, min_range=min_range, max_range=max_range)

        colored_lidar, colors_rgb, point_mask, color_debug = colorize_lidar_points(
            points_lidar,
            frame_id,
            scene_dir,
            calibration,
            camera_names,
            rgb_extension,
            min_depth,
            border_margin_px,
            center_weight,
            angle_weight,
            depth_weight,
            color_mode,
        )

        T_world_lidar = compose_transforms(T_world_vehicle, calibration.T_vehicle_lidar)
        points_world = transform_points(colored_lidar, T_world_lidar)

        all_points_world.append(points_world)
        all_colors.append(colors_rgb)

        report = {
            "index": int(i),
            "frame_id": int(frame_id),
            "num_lidar_points": int(len(points_lidar)),
            "num_colored_points": int(len(points_world)),
            "camera_visible_counts": color_debug["camera_visible_counts"],
        }
        frame_reports.append(report)

        if verbose and i % max(1, 50 * int(frame_stride)) == 0:
            print(
                f"[mapping] frame_index={i}/{n} "
                f"frame_id={frame_id} "
                f"colored={len(points_world)}/{len(points_lidar)}"
            )
    if len(all_points_world) == 0:
        cloud = o3d.geometry.PointCloud()
        return cloud, frame_reports

    points_world = np.vstack(all_points_world)
    colors_rgb = np.vstack(all_colors)

    raw_cloud = make_point_cloud(points_world, colors_rgb)
    cloud = voxel_downsample_cloud(raw_cloud, voxel_size)

    if remove_outliers:
        cloud, _ = remove_statistical_outliers(cloud, nb_neighbors=outlier_nb_neighbors, std_ratio=outlier_std_ratio)

    return cloud, frame_reports
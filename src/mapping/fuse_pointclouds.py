from pathlib import Path
import numpy as np
import open3d as o3d

from src.mapping.colorize_pointcloud import colorize_lidar_points
from src.mapping.voxel_map import make_point_cloud, voxel_downsample_cloud, downsample_to_max_points
from src.mapping.outlier_filter import remove_statistical_outliers
from src.utils.io import extract_frame_id, load_xyz_points_npy
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
    dynamic_filter=None,
    dynamic_filter_name="none",
    verbose=True,
):
    """
    Builds a global colored map from LiDAR frames, RGB images, and poses.

    poses must be a list of T_world_vehicle matrices aligned with sorted LiDAR files.
    
    dynamic_filter:
        Optional object with:
            prepare(...)
            filter_points(points_lidar_xyz, frame_id, T_world_vehicle)

    """
    scene_dir = Path(scene_dir)
    lidar_dir = scene_dir / "lidar"
    lidar_files = sorted(lidar_dir.glob("*.npy"))

    if len(lidar_files) == 0:
        raise RuntimeError(f"No LiDAR files found in {lidar_dir}")

    n = min(len(lidar_files), len(poses))
    lidar_files = lidar_files[:n]
    poses = poses[:n]

    frame_stride = int(frame_stride)
    prepare_report = None

    # Dynamic Filtering. First pass to find all stationary vehicles in map 
    if dynamic_filter is not None and hasattr(dynamic_filter, "prepare"):
        prepare_report = dynamic_filter.prepare(
            lidar_files=lidar_files,
            poses=poses,
            min_range=min_range,
            max_range=max_range,
            max_points_per_frame=max_points_per_frame,
            frame_stride=frame_stride,
        )

        if verbose:
            print(f"[mapping] dynamic_filter_prepare={prepare_report}")

    all_points_world = []
    all_colors = []
    frame_reports = []

    for i in range(0, n, frame_stride):
        lidar_path = lidar_files[i]
        frame_id = extract_frame_id(lidar_path)
        T_world_vehicle = poses[i]

        points_lidar = load_xyz_points_npy(lidar_path, min_range=min_range, max_range=max_range)

        if max_points_per_frame is not None:
            points_lidar, _ = downsample_to_max_points(
                points_lidar,
                colors_rgb=None,
                max_points=max_points_per_frame,
                initial_voxel_size=0.05,
            )

        if dynamic_filter is not None:
            filtered_points_lidar, dynamic_report = dynamic_filter.filter_points(
                points_lidar_xyz=points_lidar,
                frame_id=frame_id,
                T_world_vehicle=T_world_vehicle,
            )
        else:
            filtered_points_lidar = points_lidar
            dynamic_report = {
                "method": dynamic_filter_name,
                "num_input_points": int(len(points_lidar)),
                "num_removed_points": 0,
                "num_static_points": int(len(points_lidar)),
                "num_always_dynamic_points": 0,
                "num_vehicle_candidate_points": 0,
                "num_stationary_vehicle_points_kept": 0,
                "num_moving_vehicle_points_removed": 0,
                "removed_ratio": 0.0,
                "camera_reports": {},
            }



        colored_lidar, colors_rgb, point_mask, color_debug = colorize_lidar_points(
            filtered_points_lidar,
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
            "dynamic_filter": dynamic_filter_name,
            "num_lidar_points": int(len(points_lidar)),            
            "num_lidar_points_after_dynamic_filter": int(len(filtered_points_lidar)),
            "num_removed_dynamic_points": int(dynamic_report.get("num_removed_points", 0)),
            "dynamic_removed_ratio": float(dynamic_report.get("removed_ratio", 0.0)),
            "num_always_dynamic_points": int(dynamic_report.get("num_always_dynamic_points", 0)),
            "num_vehicle_candidate_points": int(dynamic_report.get("num_vehicle_candidate_points", 0)),
            "num_stationary_vehicle_points_kept": int(dynamic_report.get("num_stationary_vehicle_points_kept", 0)),
            "num_moving_vehicle_points_removed": int(dynamic_report.get("num_moving_vehicle_points_removed", 0)),
            "num_colored_points": int(len(points_world)),
            "camera_visible_counts": color_debug["camera_visible_counts"],
            "dynamic_report": dynamic_report,
        }
        frame_reports.append(report)

        if verbose and i % max(1, 50 * frame_stride) == 0:
            print(
                f"[mapping] frame_index={i}/{n} "
                f"frame_id={frame_id} "                
                f"filter={dynamic_filter_name} "
                f"input={len(points_lidar)} "
                f"kept={len(filtered_points_lidar)} "
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
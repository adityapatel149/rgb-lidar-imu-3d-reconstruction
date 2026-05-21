from pathlib import Path
import cv2
from cv2.typing import Point
import numpy as np

from src.mapping.camera_selection import projection_quality, score_projection
from src.utils.projection import project_sensor_points_to_camera_image
from src.utils.io import load_rgb_image


def sample_rgb_nearest(image_rgb, uv):
    h, w = image_rgb.shape[:2]
    u = np.rint(uv[:,0]).astype(np.int64)
    v = np.rint(uv[:,1]).astype(np.int64)
    valid = (
        np.isfinite(uv).all(axis=1)
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
    )
    colors = np.zeros((uv.shape[0], 3), dtype=np.float64)
    # Numpy stores row, column so y,x
    colors[valid] = image_rgb[v[valid], u[valid]].astype(np.float64) / 255.0
    return colors, valid



def colorize_lidar_points(
    points_lidar_xyz,
    frame_id,
    scene_dir,
    calibration,
    camera_names,
    rgb_extension=".jpg",
    min_depth=0.1,
    border_margin_px=2,
    center_weight=1.0,
    angle_weight=1.0,
    depth_weight=0.05,
    color_mode="best",
):
    """
    Assigns RGB color to LiDAR points using multiple cameras.
    M is the lidar points that are colored, N is total lidar points
   
    Returns:
        colored_xyz: Mx3 LiDAR-frame points that received color
        colors_rgb: Mx3 float colors in [0, 1]
        point_mask: N bool mask for original LiDAR points
        debug_info: dict
    """
    scene_dir = Path(scene_dir)
    points_lidar_xyz = np.asarray(points_lidar_xyz, dtype=np.float64)

    n = points_lidar_xyz.shape[0]

    if n == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=bool),
            {"camera_visible_counts": {}, "num_colored": 0},
        )

    if color_mode not in {"best", "average"}:
        raise ValueError("color_mode must be 'best' or 'average'")

    all_scores = []
    all_colors = []
    all_valid= []
    camera_visible_counts ={}

    frame_name = f"{int(frame_id):06d}{rgb_extension}"

    for camera_name in camera_names:
        image_path = scene_dir / "rgb" / camera_name / frame_name
        image_rgb = load_rgb_image(image_path)
        image_height, image_width = image_rgb.shape[:2]

        K = calibration.get_camera_K(camera_name)
        D = calibration.get_camera_D(camera_name)
        T_vehicle_camera = calibration.get_T_vehicle_camera(camera_name)
        T_vehicle_lidar = calibration.T_vehicle_lidar

        # Project points to camera and Check which points are visible in this camera
        projection = project_sensor_points_to_camera_image(
            points_lidar_xyz, T_vehicle_lidar, T_vehicle_camera, K, D, image_width, image_height, min_depth, border_margin_px,
        )

        quality = projection_quality(
            points_camera_cv=projection["points_camera_cv"], uv=projection["uv"], K=K,    
        )

        score = score_projection(
            valid_mask=projection["valid_mask"],
            depth=projection["depth"],
            center_distance_norm=quality["center_distance_norm"],
            viewing_angle=quality["viewing_angle"],
            center_weight=center_weight,
            angle_weight=angle_weight,
            depth_weight=depth_weight,
        )

        # sample RGB color for ALL lidar points
        sampled_colors, sample_valid = sample_rgb_nearest(image_rgb, projection["uv"])

        # Final, per-camera visibility mask
        valid = (
            projection["valid_mask"]
            & sample_valid
            & np.isfinite(score)
        )

        # Store colors, scores and valid masks for this camera
        all_scores.append(score)
        all_colors.append(sampled_colors)
        all_valid.append(valid)

        camera_visible_counts[camera_name] = int(np.sum(valid))

    # valid[i, j] = whether LiDAR point i is visible in camera j
    # scores[i, j] = score for LiDAR point i in camera j
    # colors[i, j] = RGB color for LiDAR point i sampled from camera j
    valid = np.stack(all_valid, axis=1)
    scores = np.stack(all_scores, axis=1)
    colors = np.stack(all_colors, axis=1)

    # Look across all cameras(axis=1) for all valid points, and find points that are visible in atleast one camera
    visible = np.any(valid, axis=1)

    if color_mode == "best":
        masked_scores = np.where(valid, scores, np.inf)
        # Best camera idx for all lidar points
        best_camera_idx = np.argmin(masked_scores, axis=1)
        output_colors = np.zeros((n, 3), dtype=np.float64)
        point_indices = np.arange(n)
        # For every point that is visible in atleast one camera, take the point's color from the best camera
        output_colors[visible] = colors[
            point_indices[visible],
            best_camera_idx[visible],
        ]
    else:
        # Average color from all cameras
        # Convert bollenas to numbers (0 or 1)
        valid_f = valid.astype(np.float64)
        # Count how many valid cameras each point has
        denom = np.sum(valid_f, axis=1, keepdims=True)
        denom = np.maximum(denom, 1.0)
        # colors are (N,C,3), valid is (N, C). Using None broadcasts valid_f over RGB channels. So colors * valid_f keeps valid camera colors and zeroes out invalid camera colors
        # Take sum of valid colors and divide by number of cameras
        output_colors = np.sum(colors * valid_f[:,:, None], axis=1) / denom

    # (M, 3) where m is number of points visible in atleast one camera
    colored_xyz = points_lidar_xyz[visible]
    colors_rgb = output_colors[visible]

    debug_info = {
        "camera_visible_counts": camera_visible_counts,
        "num_input_points": int(n),
        "num_colored": int(np.sum(visible)),
        "color_mode": color_mode,
    }

    return colored_xyz, colors_rgb, visible, debug_info
import numpy as np

from src.utils.projection import project_sensor_points_to_camera_image



def sample_mask_nearest(mask, uv):
    """
    Sample a bool image mask at projected pixel coordinates.
    """
    mask = np.asarray(mask, dtype=bool)
    uv = np.asarray(uv, dtype=np.float64)

    h, w = mask.shape[:2]

    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)

    valid = (
        np.isfinite(uv).all(axis=1)
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
    )

    sampled = np.zeros(uv.shape[0], dtype=bool)
    sampled[valid] = mask[v[valid], u[valid]]

    return sampled, valid




def lidar_semantic_masks_camera(
    points_lidar_xyz,
    semantic_image_masks,
    calibration,
    camera_name,
    min_depth=0.1,
    border_margin_px=2,
):
    """
    Project LiDAR points into one camera and sample semantic masks.

    semantic_image_masks:
        {
            "always_dynamic": HxW bool,
            "vehicle": HxW bool,
        }

    Returns:
        {
            "always_dynamic": N bool,
            "vehicle": N bool,
            "visible": N bool,
        }
    """
    points_lidar_xyz = np.asarray(points_lidar_xyz, dtype=np.float64)

    any_mask = next(iter(semantic_image_masks.values()))
    image_height, image_width = any_mask.shape[:2]

    K = calibration.get_camera_K(camera_name)
    D = calibration.get_camera_D(camera_name)
    T_vehicle_camera = calibration.get_T_vehicle_camera(camera_name)
    T_vehicle_lidar = calibration.T_vehicle_lidar

    projection = project_sensor_points_to_camera_image(
        points_sensor_xyz=points_lidar_xyz,
        T_vehicle_sensor=T_vehicle_lidar,
        T_vehicle_camera=T_vehicle_camera,
        K=K,
        D=D,
        image_width=image_width,
        image_height=image_height,
        min_depth=min_depth,
        border_margin_px=border_margin_px,
    )

    visible = projection["valid_mask"].copy()

    output = {
        "visible": visible,    
    }

    for category_name, image_mask in semantic_image_masks.items():
        sampled, sample_valid = sample_mask_nearest(mask=image_mask, uv=projection["uv"])
        output[category_name] = visible & sample_valid & sampled

    return output



def lidar_semantic_masks_multicamera(
    points_lidar_xyz,
    camera_names,
    semantic_masks_by_camera,
    calibration,
    min_depth=0.1,
    border_margin_px=2,
):
    """
    Combine per-camera masks into LiDAR-frame point masks.

    A point is in a category if any camera sees it in that category.
    """
    points_lidar_xyz = np.asarray(points_lidar_xyz, dtype=np.float64)
    n = len(points_lidar_xyz)
    combined_visible = np.zeros(n, dtype=bool)
    combined_always_dynamic = np.zeros(n, dtype=bool)
    combined_vehicle = np.zeros(n, dtype=bool)
    camera_reports = {}

    for camera_name in camera_names:
        if camera_name not in semantic_masks_by_camera:
            continue
        result = lidar_semantic_masks_camera(
            points_lidar_xyz=points_lidar_xyz,
            semantic_image_masks=semantic_masks_by_camera[camera_name],
            calibration=calibration,
            camera_name=camera_name,
            min_depth=min_depth,
            border_margin_px=border_margin_px,
        )
        visible = result["visible"]
        always_dynamic = result.get("always_dynamic", np.zeros(n, dtype=bool))
        vehicle = result.get("vehicle", np.zeros(n, dtype=bool))

        # Bitwise OR
        combined_visible |= visible
        combined_always_dynamic |= always_dynamic
        combined_vehicle |= vehicle

        camera_reports[camera_name] = {
            "num_visible": int(np.sum(visible)),
            "num_always_dynamic": int(np.sum(always_dynamic)),
            "num_vehicle": int(np.sum(vehicle)),
        }

    return {
        "visible_mask": combined_visible,
        "always_dynamic_mask": combined_always_dynamic,
        "vehicle_mask": combined_vehicle,
        "camera_reports": camera_reports,
    }
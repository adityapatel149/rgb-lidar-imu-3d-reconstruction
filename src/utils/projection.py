import numpy as np
import cv2
from src.utils.camera_geometry import carla_camera_to_opencv
from src.utils.transforms import relative_transform, transform_points


def project_camera_points(points_camera_cv, K, D=None, min_depth=0.1):
    """
    Project OpenCV camera-frame 3D points into image pixels.

    Inputs:
        points_camera_cv: Nx3 points in OpenCV camera convention.
        K: 3x3 intrinsic matrix.
        D: Distortion coefficients. Uses zeros if None.

    Returns:
        uv: Nx2 pixel coordinates. Invalid rows are NaN.
        depth: N depth values.
        valid_depth: N bool mask for positive-depth points.
    """
    points_camera_cv = np.asarray(points_camera_cv, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    if points_camera_cv.ndim != 2 or points_camera_cv.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points_camera_cv.shape}")

    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 K, got {K.shape}")

    if D is None:
        D = np.zeros(5, dtype=np.float64)
    else:
        D = np.asarray(D, dtype=np.float64).reshape(-1)

    # Number of points
    n = points_camera_cv.shape[0]
    # [u,v] values for each point
    uv = np.full((n, 2), np.nan, dtype=np.float64)
    depth = points_camera_cv[:, 2].copy()

    valid_depth = depth > float(min_depth)

    # If invalid depth, return uv full of NaN
    if not np.any(valid_depth):
        return uv, depth, valid_depth

    # Select only valid points. cv.projectPoints expects points shaped like Nx1x3, instead of just Nx3
    object_points = points_camera_cv[valid_depth].reshape(-1, 1, 3)

    projected, _ = cv2.projectPoints(
        object_points,
        rvec=np.zeros(3, dtype=np.float64), # Do not rotate points, since points aree already in camera coordinate system
        tvec=np.zeros(3, dtype=np.float64), # Do not translate points
        cameraMatrix=K,
        distCoeffs=D,
    )

    # Convert OpenCV result into (N,2) shape
    uv[valid_depth] = projected.reshape(-1, 2)

    return uv, depth, valid_depth


def image_bounds_mask(
    uv,
    image_width,
    image_height,
    border_margin_px=0,
):
    uv = np.asarray(uv, dtype=np.float64)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"Expected Nx2 uv coordinates, got {uv.shape}")

    u = uv[:, 0]
    v = uv[:, 1]

    margin = int(border_margin_px)

    return (
        np.isfinite(uv).all(axis=1)
        & (u >= margin)
        & (u < int(image_width) - margin)
        & (v >= margin)
        & (v < int(image_height) - margin)
    )


def project_sensor_points_to_camera_image(
    points_sensor_xyz,
    T_vehicle_sensor,
    T_vehicle_camera,
    K,
    D,
    image_width,
    image_height,
    min_depth=0.1,
    border_margin_px=0,
):
    """
    General sensor-to-camera projection.
    This works for LiDAR, radar, or any 3D sensor whose extrinsic is known.

    Inputs:
        points_sensor_xyz: Nx3 points in source sensor frame.
        T_vehicle_sensor: source sensor frame -> vehicle frame.
        T_vehicle_camera: camera frame -> vehicle frame.

    Returns:
        dict containing:
            valid_mask
            uv
            depth
            points_camera_carla
            points_camera_cv
    """

    # sensor frame -> camera frame
    T_camera_sensor = relative_transform(
        T_world_source=T_vehicle_sensor,
        T_world_target=T_vehicle_camera,
    )

    # transform points to camera frame
    points_camera_carla = transform_points(
        points_xyz=points_sensor_xyz,
        T_target_source=T_camera_sensor,
    )

    points_camera_cv = carla_camera_to_opencv(points_camera_carla)

    uv, depth, valid_depth = project_camera_points(
        points_camera_cv=points_camera_cv,
        K=K,
        D=D,
        min_depth=min_depth,
    )

    valid_bounds = image_bounds_mask(
        uv=uv,
        image_width=image_width,
        image_height=image_height,
        border_margin_px=border_margin_px,
    )

    valid_mask = valid_depth & valid_bounds

    return {
        "valid_mask": valid_mask,
        "uv": uv,
        "depth": depth,
        "points_camera_carla": points_camera_carla,
        "points_camera_cv": points_camera_cv,
    }
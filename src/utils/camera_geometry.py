import math
import numpy as np



def camera_intrinsics(width, height, fov_degrees):
    """
    Build pinhole camera intrinsics from image size and horizontal FOV.
    """
    width = int(width)
    height = int(height)
    fov = math.radians(float(fov_degrees))

    fx = width / (2.0 * math.tan(fov / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    D = np.zeros(5, dtype=np.float64)

    return K, D




def carla_camera_to_opencv(points_camera_carla):
    """
    Convert points from CARLA camera convention to OpenCV camera convention.

    CARLA camera:
        x forward
        y right
        z up

    OpenCV camera:
        x right
        y down
        z forward
    """
    points_camera_carla = np.asarray(points_camera_carla, dtype=np.float64)
    if points_camera_carla.ndim != 2 or points_camera_carla.shape[1] != 3:
        raise ValueError(
            f"Expected Nx3 camera points, got {points_camera_carla.shape}"
        )

    x = points_camera_carla[:, 0]
    y = points_camera_carla[:, 1]
    z = points_camera_carla[:, 2]

    return np.stack([y, -z, x], axis=1)




def opencv_camera_to_carla(points_camera_cv):
    """
    Convert points from OpenCV camera convention to CARLA camera convention.

    OpenCV camera:
        x right
        y down
        z forward

    CARLA camera:
        x forward
        y right
        z up
    """
    points_camera_cv = np.asarray(points_camera_cv, dtype=np.float64)
    if points_camera_cv.ndim != 2 or points_camera_cv.shape[1] != 3:
        raise ValueError(
            f"Expected Nx3 camera points, got {points_camera_cv.shape}"
        )

    x_cv = points_camera_cv[:, 0]
    y_cv = points_camera_cv[:, 1]
    z_cv = points_camera_cv[:, 2]

    return np.stack([z_cv, x_cv, -y_cv], axis=1)
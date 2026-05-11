import math
from xml.dom import VALIDATION_ERR
import numpy as np
import cv2


# Output T = 
# [ R | t ]
# [ 0 | 1 ]
# R = rotation matrix, t = translation vector
def carla_transform_to_matrix(transform):
    location = transform.location
    rotation = transform.rotation

    cy = math.cos(math.radians(rotation.yaw))
    sy = math.sin(math.radians(rotation.yaw))
    cp = math.cos(math.radians(rotation.pitch))
    sp = math.sin(math.radians(rotation.pitch))
    cr = math.cos(math.radians(rotation.roll))
    sr = math.sin(math.radians(rotation.roll))

    # Create a matrix with ones on diagonals, zeros everywhere
    matrix = np.eye(4, dtype=np.float64)
    
    matrix[0, 0] = cp * cy
    matrix[0, 1] = cy * sp * sr - sy * cr
    matrix[0, 2] = -cy * sp * cr - sy * sr

    matrix[1, 0] = sy * cp
    matrix[1, 1] = sy * sp * sr + cy * cr
    matrix[1, 2] = -sy * sp * cr + cy * sr

    matrix[2, 0] = sp
    matrix[2, 1] = -cp * sr
    matrix[2, 2] = cp * cr

    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z

    return matrix



def camera_intrinsics(width, height, fov_degrees):
    fov = math.radians(fov_degrees)
    fx = width / (2.0 * math.tan(fov/2.0))
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



def serialize_matrix(matrix):
    return np.asarray(matrix).tolist()



def build_calibration(camera_cfgs, camera_actors, lidar_actor, imu_actor):
    calib = {
        "frame_convention": {
            "vehicle_frame": "CARLA vehicle local frame",
            "sensor_extrinsics": "T_vehicle_sensor, 4x4 transform from sensor frame to vehicle frame",
            "camera_projection_note": "LiDAR projection uses T_camera_lidar = inverse(T_vehicle_camera) @ T_vehicle_lidar",
        },
        "cameras": {},
        "lidar": {},
        "imu": {}, 
    }

    for cfg in camera_cfgs:
        name = cfg["name"]
        width = int(cfg["width"])
        height = int(cfg["height"])
        fov = float(cfg["fov"])

        K, D = camera_intrinsics(width, height, fov)
        T_vehicle_camera = carla_transform_to_matrix(camera_actors[name].get_transform())

        calib["cameras"][name] = {
            "image_width": width,
            "image_height": height,
            "fov_degrees": fov,
            "K": serialize_matrix(K),
            "D": D.tolist(),
            "T_vehicle_camera": serialize_matrix(T_vehicle_camera),
        }

    T_vehicle_lidar = carla_transform_to_matrix(lidar_actor.get_transform())
    T_vehicle_imu = carla_transform_to_matrix(imu_actor.get_transform())

    calib["lidar"] = {
        "T_vehicle_lidar": serialize_matrix(T_vehicle_lidar),
    }

    calib["imu"] = {
        "T_vehicle_imu": serialize_matrix(T_vehicle_imu),
    }

    return calib



def carla_lidar_to_camera_points(points_lidar):
    """
    CARLA lidar points are x-forward, y-right, z-up.
    Camera projection expects x-right, y-down, z-forward.
    """
    x = points_lidar[:, 0]
    y = points_lidar[:, 1]
    z = points_lidar[:, 2]

    points_cam = np.stack([y, -z, x], axis=1)
    return points_cam



def project_lidar_to_camera(points_lidar, K, T_vehicle_lidar, T_vehicle_camera):
    # Lidar points may contain intensity too, we only need xyz
    points_xyz = points_lidar[:, :3]
    # Convert to homogeneous coordinates (x,y,z) -> (x,y,z,1). Allows translation + rotation using matrix multiplication 
    points_h = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    # We know lidar to vehicle transform and camera to vehicle vehicle, now we need lidar to camera
    T_camera_lidar = np.linalg.inv(T_vehicle_camera) @ T_vehicle_lidar
    # convert lidar points to camera coordinatees
    points_camera_raw = (T_camera_lidar @ points_h.T).T[:, :3]
    # Convert coordinate convention
    points_camera = carla_lidar_to_camera_points(points_camera_raw)

    # Keep points in front of camera
    z = points_camera[:, 2]
    valid_depth = z > 0.1

    points_camera = points_camera[valid_depth]
    z = z[valid_depth]

    # Project points on image, equivalent to u=fx*x/z + cx, v=fy*y/z + cy
    projected = (K @ points_camera.T).T
    # Perspective projection. Convert homogenous to coordinates to actual pixel coordinates. divide by z to get perspective. farther objects look smaller and closer to center of camera.
    uv = projected[:, :2] / projected[:, 2:3]

    return uv, z



def save_lidar_projection_debug(rgb_image, lidar_points, camera_calib, lidar_calib, output_path):
    image = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
    image = image.reshape(rgb_image.height, rgb_image.width, 4)[:, :, :3].copy()

    K = np.array(camera_calib["K"], dtype=np.float64)
    T_vehicle_camera = np.array(camera_calib["T_vehicle_camera"], dtype=np.float64)
    T_vehicle_lidar = np.array(lidar_calib["T_vehicle_lidar"], dtype=np.float64)

    uv, depth = project_lidar_to_camera(
        lidar_points,
        K,
        T_vehicle_lidar,
        T_vehicle_camera,
    )

    # Keep points in the view of the camera
    height, width = image.shape[:2]
    valid = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height) 
    )

    uv = uv[valid].astype(np.int32)
    depth = depth[valid]

    if len(depth) > 0:
        depth_norm = np.clip(depth / 80.0, 0.0, 1.0)
        colors = (255 * (1.0 - depth_norm)).astype(np.uint8)

        for (u, v), c in zip(uv[::3], colors[::3]):
            cv2.circle(image, (int(u), int(v)), 1, (0, int(c), 255 - int(c)), -1)

    cv2.imwrite(str(output_path), image)

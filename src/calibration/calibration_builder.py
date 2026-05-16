import numpy as np
from src.utils.camera_geometry import camera_intrinsics
from src.utils.transforms import relative_actor_transform



def serialize_matrix(matrix):
    return np.asarray(matrix, dtype=np.float64).tolist()



def build_calibration(vehicle, camera_cfgs, camera_actors, lidar_actor, imu_actor):
    """
    Build vehicle-relative sensor calibration.

    Stored convention:
        T_vehicle_sensor maps sensor-frame points into vehicle frame.
    """
    calibration = {
        "frame_convention": {
            "vehicle_frame": "CARLA vehicle local frame",
            "sensor_extrinsics": "T_vehicle_sensor maps sensor frame to vehicle frame",
            "camera_frame": "CARLA camera frame: x-forward, y-right, z-up",
            "opencv_camera_frame": "OpenCV camera frame: x-right, y-down, z-forward",
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

        T_vehicle_camera = relative_actor_transform(
            source_actor=camera_actors[name],
            target_actor=vehicle,
        )

        calibration["cameras"][name] = {
            "image_width": width,
            "image_height": height,
            "fov_degrees": fov,
            "K": serialize_matrix(K),
            "D": D.tolist(),
            "T_vehicle_camera": serialize_matrix(T_vehicle_camera),
        }

    T_vehicle_lidar = relative_actor_transform(
        source_actor=lidar_actor,
        target_actor=vehicle,
    )

    T_vehicle_imu = relative_actor_transform(
        source_actor=imu_actor,
        target_actor=vehicle,
    )

    calibration["lidar"] = {
        "T_vehicle_lidar": serialize_matrix(T_vehicle_lidar),
    }

    calibration["imu"] = {
        "T_vehicle_imu": serialize_matrix(T_vehicle_imu),
    }

    return calibration
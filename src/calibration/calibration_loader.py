from pathlib import Path
import json
import numpy as np
from src.utils.transforms import transform_points


class Calibration:
    def __init__(self, calib_path):

        with open(Path(calib_path), "r") as f:
            self.data = json.load(f)

        self.cameras = self.data["cameras"]
        self.T_vehicle_lidar = np.array(self.data["lidar"]["T_vehicle_lidar"], dtype=np.float64)
        self.T_vehicle_imu = np.array(self.data["imu"]["T_vehicle_imu"], dtype=np.float64)



    def get_camera_K(self, camera_name):
        return np.array(self.cameras[camera_name]["K"], dtype=np.float64)



    def get_camera_D(self, camera_name):
        return np.array(self.cameras[camera_name]["D"], dtype=np.float64)




    def get_T_vehicle_camera(self, camera_name):
        return np.array(self.cameras[camera_name]["T_vehicle_camera"], dtype=np.float64)



    def transform_points(self, points_xyz, T_target_source):
        return transform_points(points_xyz, T_target_source)



    def T_lidar_vehicle(self):
        return np.linalg.inv(self.T_vehicle_lidar)



    def T_imu_vehicle(self):
        return np.linalg.inv(self.T_vehicle_imu)



    def T_camera_vehicle(self, camera_name):
        return np.linalg.inv(self.get_T_vehicle_camera(camera_name))



    def T_camera_lidar(self, camera_name):
        return self.T_camera_vehicle(camera_name) @ self.T_vehicle_lidar
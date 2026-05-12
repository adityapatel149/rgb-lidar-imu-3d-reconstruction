import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from src.odometry.trajectory_utils import make_transform



def integrate_imu(
    imu_csv,
    initial_pose=None,
    gravity_world=np.array([0.0, 0.0, -9.81]),
):
    imu_df = pd.read_csv(imu_csv).sort_values("timestamp").reset_index(drop=True)

    if initial_pose is None:
        T = np.eye(4, dtype=np.float64)
    else:
        T = initial_pose.copy()

    rotation = R.from_matrix(T[:3, :3])
    position = T[:3, 3].copy()
    velocity = np.zeros(3, dtype=np.float64)

    poses = [T.copy()]
    timestamps = [imu_df.loc[0, "timestamp"]]

    for i in range(1, len(imu_df)):
        prev = imu_df.loc[i-1]
        curr = imu_df.loc[i]

        dt = float(curr["timestamp"] - prev["timestamp"])
        if dt <= 0.0 or dt > 1.0:
            dt = 0.05     # fixed_delta_seconds in sensors.yaml 
        
        dt = np.clip(dt, 1e-3, 0.1)

        gyro = np.array(
            [curr["gyro_x"], curr["gyro_y"], curr["gyro_z"]],
            dtype=np.float64,
        )
        
        acc_body = np.array(
            [curr["acc_x"], curr["acc_y"], curr["acc_z"]],
            dtype=np.float64,
        )

        # Initial IMU samples may have enormous values for acceleration.
        if not np.isfinite(acc_body).all() or np.linalg.norm(acc_body) > 50.0:
            poses.append(T.copy())
            timestamps.append(float(curr["timestamp"]))
            continue
        
        # Update rotation
        delta_rot = R.from_rotvec(gyro * dt)
        rotation = rotation * delta_rot
        
        acc_world = rotation.apply(acc_body)

        # CARLA IMU acceleration usually includes gravity-like effects depending on setup.
        # This subtraction is the standard strapdown form.
        acc_world = acc_world + gravity_world

        position = position + velocity * dt + 0.5 * acc_world * dt * dt
        velocity = velocity + acc_world * dt

        T = make_transform(rotation.as_matrix(), position)
        poses.append(T)
        timestamps.append(float(curr["timestamp"]))

    return timestamps, poses
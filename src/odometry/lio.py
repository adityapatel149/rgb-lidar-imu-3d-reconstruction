from pathlib import Path
import numpy as np
import pandas as pd

from src.calibration.calibration_loader import Calibration
from src.odometry.eskf import ErrorStateKalmanFilter
from src.odometry.icp_odometry import load_lidar_cloud, run_icp_pair, lidar_files
from src.odometry.trajectory_utils import pose_row_to_matrix
from src.utils.io import extract_frame_id



def load_ground_truth_by_frame(scene_dir):
    df = pd.read_csv(Path(scene_dir) / "poses.csv")
    result = {}
    for _, row in df.iterrows():
        frame = int(row["frame"])
        result[frame] = {
            "timestamp": float(row["timestamp"]),
            "pose": pose_row_to_matrix(row),
        }
    return result



def load_imu(scene_dir):
    df = pd.read_csv(Path(scene_dir) / "imu.csv")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df



def find_imu_range(imu_df, t0, t1, start_index):
    rows = []
    i = start_index
    while i < len(imu_df) and float(imu_df.loc[i, "timestamp"]) <= t0:
        i += 1
    while i < len(imu_df) and float(imu_df.loc[i, "timestamp"]) <= t1:
        rows.append(imu_df.loc[i])
        i += 1
    return rows, i



def imu_only_trajectory(scene_dir, initial_pose=None, max_acc_norm=50.0):
    imu_df = load_imu(scene_dir)
    eskf = ErrorStateKalmanFilter(initial_pose=initial_pose)
    poses = [eskf.pose_matrix()]
    timestamps = [float(imu_df.loc[0, "timestamp"])]

    for i in range(1, len(imu_df)):
        prev = imu_df.loc[i - 1]
        curr = imu_df.loc[i]
        dt = float(curr["timestamp"] - prev["timestamp"])
        if dt <= 0.0 or dt > 1.0:
            dt = 0.05
        dt = float(np.clip(dt, 1e-3, 0.1))

        acc = np.array(
            [curr["acc_x"], curr["acc_y"], curr["acc_z"]], dtype=np.float64,
        )
        gyro = np.array(
            [curr["gyro_x"], curr["gyro_y"], curr["gyro_z"]], dtype=np.float64,
        )

        # Match imu_odometry.py behavior:
        # skip bad CARLA IMU spikes instead of propagating them.
        if not np.isfinite(acc).all() or np.linalg.norm(acc) > max_acc_norm:
            poses.append(eskf.pose_matrix())
            timestamps.append(float(curr["timestamp"]))
            continue

        if not np.isfinite(gyro).all():
            poses.append(eskf.pose_matrix())
            timestamps.append(float(curr["timestamp"]))
            continue

        eskf.propagate(acc, gyro, dt)

        poses.append(eskf.pose_matrix())
        timestamps.append(float(curr["timestamp"]))

    return timestamps, poses



def run_lio(scene_dir, calib_path=None, voxel_size=0.25, max_correspondence_distance=0.75, max_acc_norm=50.0,):
    scene_dir = Path(scene_dir)

    if calib_path is None:
        calib_path = scene_dir / "calib" / "calibration.json"

    calib = Calibration(calib_path)

    gt_by_frame = load_ground_truth_by_frame(scene_dir)
    imu_df = load_imu(scene_dir)

    files = lidar_files(scene_dir)
    frame_ids = [extract_frame_id(f) for f in files]

    valid_pairs = [
        (f, frame)
        for f, frame in zip(files, frame_ids)
        if frame in gt_by_frame
    ]

    files = [x[0] for x in valid_pairs]
    frame_ids = [x[1] for x in valid_pairs]

    if len(files) < 2:
        raise RuntimeError("Need at least two LiDAR frames with matching poses.")

    first_frame = frame_ids[0]
    initial_pose = np.eye(4, dtype=np.float64)

    eskf = ErrorStateKalmanFilter(initial_pose=initial_pose)
    
    fused_poses = [eskf.pose_matrix()]
    icp_poses = [np.eye(4, dtype=np.float64)]
    gt_poses = [np.eye(4, dtype=np.float64)]
    timestamps = [gt_by_frame[first_frame]["timestamp"]]

    prev_cloud = load_lidar_cloud(files[0], voxel_size=voxel_size)
    prev_timestamp = timestamps[0]
    imu_index = 0

    T_vehicle_lidar = calib.T_vehicle_lidar
    T_lidar_vehicle = calib.T_lidar_vehicle()

    icp_stats = []

    for i in range(1, len(files)):
        frame= frame_ids[i]
        timestamp= gt_by_frame[frame]["timestamp"]
        imu_rows, imu_index = find_imu_range(
            imu_df,
            prev_timestamp,
            timestamp,
            imu_index,
        )
        last_t = prev_timestamp
        for row in imu_rows:
            curr_t = float(row["timestamp"])
            dt = curr_t - last_t
            if dt <= 0.0 or dt > 1.0:
                dt = 0.05
            dt = float(np.clip(dt, 1e-3, 0.1))
            acc = np.array(
                [row["acc_x"], row["acc_y"], row["acc_z"]], dtype=np.float64,
            )
            gyro = np.array(
                [row["gyro_x"], row["gyro_y"], row["gyro_z"]], dtype=np.float64,
            )
            # handle spikes and outliers
            if not np.isfinite(acc).all() or np.linalg.norm(acc) > max_acc_norm:
                last_t = curr_t
                continue
            if not np.isfinite(gyro).all():
                last_t = curr_t
                continue

            eskf.propagate(acc, gyro, dt)
            last_t = curr_t

        curr_cloud = load_lidar_cloud(files[i], voxel_size=voxel_size)

        # Previous fused vehicle pose in world
        T_world_vehicle_prev = fused_poses[-1]

        # Current IMU-predicted vehicle pose in world
        T_world_vehicle_curr = eskf.pose_matrix()

        # ICP expects current -> previous, This relative transform maps:
        # vehicle_curr -> vehicle_prev
        T_vehicle_prev_vehicle_curr = np.linalg.inv(T_world_vehicle_prev) @ T_world_vehicle_curr

        # Convert vehicle-frame motion into lidar-frame motion. Result maps:
        # lidar_curr -> lidar_prev
        T_lidar_prev_lidar_curr_prior = (
            T_lidar_vehicle
            @ T_vehicle_prev_vehicle_curr
            @ T_vehicle_lidar
        )

        # Open3D returns transform from source/current to target/previous.
        T_lidar_prev_lidar_curr_icp, fitness, rmse = run_icp_pair(
            source_cloud=curr_cloud,
            target_cloud=prev_cloud,
            init_transform=T_lidar_prev_lidar_curr_prior,
            max_correspondence_distance=max_correspondence_distance,
        )
        # ICP result maps lidar_curr -> lidar_prev. Keep this direction.        
        # T_vehicle_prev_vehicle_curr_icp maps vehicle_curr -> vehicle_prev.
        # Left-multiplying by T_world_vehicle_prev gives vehicle_curr -> world.
        T_vehicle_prev_vehicle_curr_icp = (
            T_vehicle_lidar
            @ T_lidar_prev_lidar_curr_icp
            @ T_lidar_vehicle
        )

        # Previous fused pose in world.
        T_world_vehicle_prev = fused_poses[-1]

        # LiDAR measurement of the current vehicle pose in world.
        # T_world_vehicle_prev maps vehicle_prev -> world, and
        # T_vehicle_prev_vehicle_curr_icp maps vehicle_curr -> vehicle_prev.
        T_world_vehicle_curr_meas = (
            T_world_vehicle_prev
            @ T_vehicle_prev_vehicle_curr_icp
        )

        residual = eskf.update_pose(
            T_world_vehicle_curr_meas,
            pos_noise=0.35,
            rot_noise=0.65,
            update_velocity=True,
            update_bias=False,
        )
        fused_poses.append(eskf.pose_matrix())
        icp_poses.append(icp_poses[-1] @ T_vehicle_prev_vehicle_curr_icp)

        # Ground truth relative to first frame.
        T_gt0 = gt_by_frame[first_frame]["pose"]
        T_gt = np.linalg.inv(T_gt0) @ gt_by_frame[frame]["pose"]
        gt_poses.append(T_gt)

        timestamps.append(timestamp)

        icp_stats.append(
            {
                "frame": int(frame),
                "timestamp": float(timestamp),
                "fitness": float(fitness),
                "rmse": float(rmse),
                "residual_norm": float(np.linalg.norm(residual)),
            }
        )

        prev_cloud = curr_cloud
        prev_timestamp = timestamp

        if i % 50 == 0:
            print(
                f"[LIO] {i}/{len(files)} "
                f"fitness={fitness:.3f} rmse={rmse:.3f} "
                f"residual={np.linalg.norm(residual):.3f}"
            )

    imu_timestamps, imu_poses = imu_only_trajectory(
        scene_dir,
        initial_pose=np.eye(4, dtype=np.float64),
    )

    return {
        "timestamps": timestamps,
        "gt_poses": gt_poses,
        "imu_timestamps": imu_timestamps,
        "imu_poses": imu_poses,
        "icp_poses": icp_poses,
        "fused_poses": fused_poses,
        "icp_stats": icp_stats,
    }
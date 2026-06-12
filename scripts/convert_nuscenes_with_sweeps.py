#!/usr/bin/env python3
"""
Project-specific nuScenes -> project scene converter.

Default dataset layout:
    data/nuscenes/
      v1.0-mini/
      samples/
      sweeps/
      maps/
      can_bus/

Normal keyframe conversion:
    python scripts/convert_nuscenes_with_sweeps.py --scene scene-0061

Dense conversion using LiDAR sample_data sweeps and nearest RGB sample_data:
    python scripts/convert_nuscenes_with_sweeps.py --scene scene-0061 --include-sweeps
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
LIDAR_CHANNEL = "LIDAR_TOP"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def transform_from_quat_translation(rotation_wxyz: Iterable[float], translation_xyz: Iterable[float]) -> np.ndarray:
    """Return T_parent_child, mapping child-frame points into parent frame."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Quaternion(rotation_wxyz).rotation_matrix
    T[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return T



def carla_from_opencv_camera_matrix() -> np.ndarray:
    """
    Convert OpenCV camera coordinates to CARLA camera coordinates.

    OpenCV camera:
        x right, y down, z forward

    CARLA camera:
        x forward, y right, z up

    p_carla = T_carla_cv @ p_cv
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([
        [0.0, 0.0, 1.0],    # x_carla = z_cv
        [1.0, 0.0, 0.0],    # y_carla = x_cv
        [0.0, -1.0, 0.0],   # z_carla = -y_cv
    ], dtype=np.float64)
    return T


def rotation_matrix_to_rpy_degrees(R: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def so3_log(R: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    vee = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64)
    return (theta / (2.0 * math.sin(theta))) * vee


def list_scenes(nusc: NuScenes) -> None:
    print("Available scenes:")
    for s in nusc.scene:
        print(f"  {s['name']:>10}  samples={s['nbr_samples']:>3}  {s.get('description', '')}")


def choose_scene(nusc: NuScenes, scene_name: str) -> dict:
    for scene in nusc.scene:
        if scene["name"] == scene_name:
            return scene
    available = ", ".join(scene["name"] for scene in nusc.scene)
    raise ValueError(f"Scene {scene_name!r} not found. Available scenes: {available}")


def iter_scene_samples(nusc: NuScenes, scene: dict) -> List[dict]:
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = sample["next"]
    return samples


def collect_channel_sample_data(
    nusc: NuScenes,
    first_token: str,
    start_time_us: int,
    end_time_us: int,
) -> List[dict]:
    """
    Follow a channel's sample_data linked list and collect keyframes + sweeps
    within the scene time window.
    """
    records: List[dict] = []
    token = first_token
    while token:
        sd = nusc.get("sample_data", token)
        ts = int(sd["timestamp"])
        if ts > end_time_us:
            break
        if ts >= start_time_us:
            records.append(sd)
        token = sd.get("next", "")
    return records


def nearest_sample_data(records: List[dict], timestamp_us: int) -> dict:
    if not records:
        raise ValueError("No sample_data records available for this camera.")
    times = np.asarray([int(r["timestamp"]) for r in records], dtype=np.int64)
    idx = int(np.argmin(np.abs(times - int(timestamp_us))))
    return records[idx]


def copy_or_symlink(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            print(f"[warn] symlink failed for {dst}; copying instead")
    shutil.copy2(src, dst)


def build_calibration_json(nusc: NuScenes, first_sample: dict) -> dict:
    calibration = {
        "frame_convention": {
            "vehicle_frame": "nuScenes ego vehicle frame",
            "sensor_extrinsics": "T_vehicle_sensor maps sensor-frame points into ego/vehicle frame",
            "camera_frame": "Stored camera frame is CARLA-style: x-forward, y-right, z-up, matching existing projection.py.",
            "opencv_camera_frame": "OpenCV camera frame: x-right, y-down, z-forward",
        },
        "cameras": {},
        "lidar": {},
        "imu": {
            "T_vehicle_imu": np.eye(4, dtype=np.float64).tolist(),
            "note": "imu.csv uses CAN bus MS_IMU when a matching file with overlapping timestamps exists; otherwise pose-derived fallback is used.",
        },
    }

    for cam in CAMERA_CHANNELS:
        sd = nusc.get("sample_data", first_sample["data"][cam])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        # nuScenes calibrated_sensor gives:
        #     T_vehicle_camera_cv: OpenCV-camera frame -> ego/vehicle frame
        #
        # Existing CARLA-compatible project code expects:
        #     T_vehicle_camera: CARLA-camera frame -> vehicle frame
        #
        # Therefore:
        #     p_cv      = T_cv_carla @ p_carla
        #     p_vehicle = T_vehicle_camera_cv @ p_cv
        #
        # so:
        #     T_vehicle_camera_carla = T_vehicle_camera_cv @ T_cv_carla
        T_vehicle_camera_cv = transform_from_quat_translation(
            calib["rotation"],
            calib["translation"],
        )

        T_carla_cv = carla_from_opencv_camera_matrix()
        T_cv_carla = np.linalg.inv(T_carla_cv)

        T_vehicle_camera = T_vehicle_camera_cv @ T_cv_carla

        calibration["cameras"][cam] = {
            "image_width": int(sd["width"]),
            "image_height": int(sd["height"]),
            "K": np.asarray(calib["camera_intrinsic"], dtype=np.float64).tolist(),
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "T_vehicle_camera": T_vehicle_camera.tolist(),

            # optional debug info
            "T_nuscenes_vehicle_camera_cv_raw": T_vehicle_camera_cv.tolist(),
            "extrinsic_note": (
                "nuScenes raw camera extrinsic is OpenCV-camera->ego. "
                "Stored T_vehicle_camera is converted to CARLA-camera->vehicle "
                "so existing projection.py and CARLA datasets remain unchanged."
            ),
        }

    lidar_sd = nusc.get("sample_data", first_sample["data"][LIDAR_CHANNEL])
    lidar_calib = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    T_vehicle_lidar = transform_from_quat_translation(lidar_calib["rotation"], lidar_calib["translation"])
    calibration["lidar"] = {"T_vehicle_lidar": T_vehicle_lidar.tolist()}
    return calibration


def write_poses_csv(path: Path, pose_rows: List[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp", "x", "y", "z", "roll", "pitch", "yaw"])
        for row in pose_rows:
            T = row["T_world_vehicle"]
            roll, pitch, yaw = rotation_matrix_to_rpy_degrees(T[:3, :3])
            writer.writerow([
                int(row["frame"]), float(row["timestamp_sec"]),
                float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
                float(roll), float(pitch), float(yaw),
            ])


def nearest_frame_for_time(timestamp_sec: float, pose_rows: List[dict]) -> int:
    times = np.array([r["timestamp_sec"] for r in pose_rows], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - float(timestamp_sec))))
    return int(pose_rows[idx]["frame"])


def find_canbus_ms_imu_file(root: Path, scene_name: str) -> Path | None:
    candidates = [
        root / "can_bus" / f"{scene_name}_ms_imu.json",
        root / f"{scene_name}_ms_imu.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(root.rglob(f"{scene_name}_ms_imu.json"))
    return matches[0] if matches else None



def find_canbus_pose_file(root: Path, scene_name: str) -> Path | None:
    candidates = [
        root / "can_bus" / f"{scene_name}_pose.json",
        root / f"{scene_name}_pose.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = sorted(root.rglob(f"{scene_name}_pose.json"))
    return matches[0] if matches else None


def initial_velocity_from_canbus_pose(
    root: Path,
    scene_name: str,
    start_time_sec: float,
) -> dict:
    pose_path = find_canbus_pose_file(root, scene_name)

    if pose_path is None:
        return {
            "source": "missing",
            "value": [0.0, 0.0, 0.0],
            "frame": "vehicle",
        }

    with pose_path.open("r") as f:
        pose_rows = json.load(f)

    best = None
    best_dt = float("inf")

    for row in pose_rows:
        if "utime" not in row or "vel" not in row:
            continue

        t = float(row["utime"]) / 1e6
        dt = abs(t - float(start_time_sec))

        if dt < best_dt:
            best = row
            best_dt = dt

    if best is None:
        return {
            "source": str(pose_path),
            "value": [0.0, 0.0, 0.0],
            "frame": "vehicle",
        }

    v = np.asarray(best["vel"], dtype=np.float64)

    if v.shape != (3,) or not np.isfinite(v).all():
        v = np.zeros(3, dtype=np.float64)

    return {
        "source": str(pose_path),
        "value": v.tolist(),
        "frame": "vehicle",
        "timestamp": float(best["utime"]) / 1e6,
        "scene_start_timestamp": float(start_time_sec),
        "dt_from_scene_start": float(best_dt),
    }



def canbus_imu_rows(ms_imu_path: Path, pose_rows: List[dict]) -> tuple[list[list[float]], dict]:
    with ms_imu_path.open("r") as f:
        ms_imu = json.load(f)

    t0 = float(pose_rows[0]["timestamp_sec"])
    t1 = float(pose_rows[-1]["timestamp_sec"])
    rows: list[list[float]] = []
    dropped = 0

    for r in ms_imu:
        if "utime" not in r:
            dropped += 1
            continue

        t = float(r["utime"]) / 1e6
        if t < t0 or t > t1:
            dropped += 1
            continue

        acc = np.asarray(r.get("linear_accel", []), dtype=np.float64)
        gyro = np.asarray(r.get("rotation_rate", []), dtype=np.float64)
        if acc.shape != (3,) or gyro.shape != (3,):
            dropped += 1
            continue
        if not np.isfinite(acc).all() or not np.isfinite(gyro).all():
            dropped += 1
            continue

        compass = 0.0
        q = r.get("q")
        if q is not None and len(q) == 4:
            try:
                _, _, yaw_deg = rotation_matrix_to_rpy_degrees(Quaternion(q).rotation_matrix)
                compass = math.radians(yaw_deg)
            except Exception:
                compass = 0.0

        rows.append([
            nearest_frame_for_time(t, pose_rows), t,
            float(acc[0]), float(acc[1]), float(acc[2]),
            float(gyro[0]), float(gyro[1]), float(gyro[2]),
            float(compass),
        ])

    return rows, {
        "source": "canbus_ms_imu",
        "path": str(ms_imu_path),
        "input_rows": len(ms_imu),
        "written_rows": len(rows),
        "dropped_rows_outside_scene_time_or_invalid": dropped,
        "scene_time_start": t0,
        "scene_time_end": t1,
    }


def pose_derived_imu_rows(pose_rows: List[dict]) -> tuple[list[list[float]], dict]:
    n = len(pose_rows)
    t = np.array([r["timestamp_sec"] for r in pose_rows], dtype=np.float64)
    p = np.vstack([r["T_world_vehicle"][:3, 3] for r in pose_rows])
    R = np.stack([r["T_world_vehicle"][:3, :3] for r in pose_rows])

    v = np.zeros_like(p)
    for i in range(n):
        j0, j1 = (0, 1) if i == 0 else ((n - 2, n - 1) if i == n - 1 else (i - 1, i + 1))
        dt = max(t[j1] - t[j0], 1e-6)
        v[i] = (p[j1] - p[j0]) / dt

    rows: list[list[float]] = []
    for i in range(n):
        j0, j1 = (0, 1) if i == 0 else ((n - 2, n - 1) if i == n - 1 else (i - 1, i + 1))
        dt = max(t[j1] - t[j0], 1e-6)
        acc_world = (v[j1] - v[j0]) / dt
        acc_body = R[i].T @ acc_world
        if i < n - 1:
            dt_rot = max(t[i + 1] - t[i], 1e-6)
            gyro_body = so3_log(R[i].T @ R[i + 1]) / dt_rot
        else:
            gyro_body = np.asarray(rows[-1][5:8], dtype=np.float64) if rows else np.zeros(3)
        _, _, yaw = rotation_matrix_to_rpy_degrees(R[i])
        rows.append([
            int(pose_rows[i]["frame"]), float(t[i]),
            float(acc_body[0]), float(acc_body[1]), float(acc_body[2]),
            float(gyro_body[0]), float(gyro_body[1]), float(gyro_body[2]),
            float(math.radians(yaw)),
        ])

    return rows, {"source": "pose_derived_fallback", "written_rows": len(rows)}


def write_imu_csv(path: Path, root: Path, scene_name: str, pose_rows: List[dict], use_canbus: bool) -> dict:
    if use_canbus:
        ms_imu_path = find_canbus_ms_imu_file(root, scene_name)
        if ms_imu_path is not None:
            rows, info = canbus_imu_rows(ms_imu_path, pose_rows)
            if rows:
                print(f"[imu] using CAN bus MS_IMU: {ms_imu_path} ({len(rows)} rows)")
            else:
                print(f"[imu] found {ms_imu_path}, but no timestamps overlap. Falling back to pose-derived IMU.")
                rows, info = pose_derived_imu_rows(pose_rows)
        else:
            print(f"[imu] no {scene_name}_ms_imu.json found under {root}. Falling back to pose-derived IMU.")
            rows, info = pose_derived_imu_rows(pose_rows)
    else:
        print("[imu] CAN bus disabled. Using pose-derived IMU fallback.")
        rows, info = pose_derived_imu_rows(pose_rows)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "compass"])
        writer.writerows(rows)
    return info


def save_pose_npz(path: Path, pose_rows: List[dict]) -> None:
    poses_abs = [r["T_world_vehicle"] for r in pose_rows]
    T0_inv = np.linalg.inv(poses_abs[0])
    poses_rel = np.stack([T0_inv @ T for T in poses_abs]).astype(np.float64)
    np.savez(path, poses_rel, poses=poses_rel)


def convert_scene(
    root: Path,
    version: str,
    scene_name: str,
    output_root: Path,
    copy_mode: str,
    max_frames: int | None,
    use_canbus: bool,
    include_sweeps: bool,
    list_only: bool = False,
) -> None:
    nusc = NuScenes(version=version, dataroot=str(root), verbose=not list_only)

    if list_only:
        list_scenes(nusc)
        return

    scene = choose_scene(nusc, scene_name)
    scene_name = scene["name"]
    key_samples = iter_scene_samples(nusc, scene)

    first_sample = key_samples[0]
    last_sample = key_samples[-1]

    first_lidar_sd = nusc.get("sample_data", first_sample["data"][LIDAR_CHANNEL])
    last_lidar_sd = nusc.get("sample_data", last_sample["data"][LIDAR_CHANNEL])
    start_time_us = int(first_lidar_sd["timestamp"])
    end_time_us = int(last_lidar_sd["timestamp"])

    if include_sweeps:
        lidar_records = collect_channel_sample_data(
            nusc=nusc,
            first_token=first_sample["data"][LIDAR_CHANNEL],
            start_time_us=start_time_us,
            end_time_us=end_time_us,
        )
    else:
        lidar_records = [
            nusc.get("sample_data", sample["data"][LIDAR_CHANNEL])
            for sample in key_samples
        ]

    if max_frames is not None:
        lidar_records = lidar_records[: int(max_frames)]

    out_name = scene_name.replace("-", "_")
    out = output_root / out_name
    ensure_dir(out / "lidar")
    ensure_dir(out / "calib")
    for cam in CAMERA_CHANNELS:
        ensure_dir(out / "rgb" / cam)

    calibration = build_calibration_json(nusc, first_sample)
    with (out / "calib" / "calibration.json").open("w") as f:
        json.dump(calibration, f, indent=2)

    camera_records_by_channel = {}
    for cam in CAMERA_CHANNELS:
        if include_sweeps:
            camera_records_by_channel[cam] = collect_channel_sample_data(
                nusc=nusc,
                first_token=first_sample["data"][cam],
                start_time_us=start_time_us,
                end_time_us=end_time_us,
            )
        else:
            camera_records_by_channel[cam] = [
                nusc.get("sample_data", sample["data"][cam])
                for sample in key_samples
            ]

    pose_rows = []
    for frame_idx, lidar_sd in enumerate(lidar_records):
        lidar_token = lidar_sd["token"]
        lidar_path = Path(nusc.get_sample_data_path(lidar_token))

        pc = LidarPointCloud.from_file(str(lidar_path))
        points = pc.points.T.astype(np.float32)
        np.save(out / "lidar" / f"{frame_idx:06d}.npy", points)

        ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        T_world_vehicle = transform_from_quat_translation(ego["rotation"], ego["translation"])
        pose_rows.append({
            "frame": frame_idx,
            "timestamp_sec": float(lidar_sd["timestamp"]) / 1e6,
            "T_world_vehicle": T_world_vehicle,
            "sample_data_token": lidar_token,
            "is_key_frame": bool(lidar_sd.get("is_key_frame", False)),
        })

        for cam in CAMERA_CHANNELS:
            if include_sweeps:
                cam_sd = nearest_sample_data(
                    camera_records_by_channel[cam],
                    timestamp_us=int(lidar_sd["timestamp"]),
                )
            else:
                cam_sd = camera_records_by_channel[cam][frame_idx]

            cam_path = Path(nusc.get_sample_data_path(cam_sd["token"]))
            suffix = cam_path.suffix.lower() or ".jpg"
            copy_or_symlink(cam_path, out / "rgb" / cam / f"{frame_idx:06d}{suffix}", mode=copy_mode)

        mode = "sweeps" if include_sweeps else "keyframes"
        print(f"[convert:{mode}] {scene_name}: frame {frame_idx + 1}/{len(lidar_records)}")

    write_poses_csv(out / "poses.csv", pose_rows)
    imu_info = write_imu_csv(out / "imu.csv", root, scene_name, pose_rows, use_canbus=use_canbus)
 
    initial_velocity_info = initial_velocity_from_canbus_pose(
        root=root,
        scene_name=scene_name,
        start_time_sec=float(pose_rows[0]["timestamp_sec"]),
    )
    traj_dir = out / "trajectory_reference"
    ensure_dir(traj_dir)
    save_pose_npz(traj_dir / "ground_truth_poses.npz", pose_rows)
    
    metadata = {
        "source": "nuScenes",
        "version": version,
        "scene_name": scene_name,
        "scene_token": scene["token"],
        "description": scene.get("description", ""),
        "include_sweeps": bool(include_sweeps),
        "num_frames": len(lidar_records),
        "num_keyframe_samples": len(key_samples),
        "camera_channels": CAMERA_CHANNELS,
        "lidar_channel": LIDAR_CHANNEL,
        "copy_mode": copy_mode,
        "imu_info": imu_info,
        "initial_velocity": initial_velocity_info,
        "camera_records_per_channel": {
            cam: len(records) for cam, records in camera_records_by_channel.items()
        },
    }
    with (out / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone")
    print(f"Converted scene: {out}")
    print(f"Frames written: {len(lidar_records)}")
    print(f"Reference poses: {traj_dir}")
    print("\nRun LIO:")
    print(f"python scripts/run_lio.py --scene {out} --output outputs/trajectories/nuscenes/{out_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert nuScenes mini into this project's scene format.")
    parser.add_argument("--scene", default="scene-0061", help="nuScenes scene name, e.g. scene-0061.")
    parser.add_argument("--list-scenes", action="store_true", help="List scenes and exit.")
    parser.add_argument("--root", default="data/nuscenes", help="Dataset root containing v1.0-mini, samples, sweeps, maps, and can_bus.")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output-root", default=None, help="Default: <root>/converted.")
    parser.add_argument("--copy-mode", default="symlink", choices=["symlink", "copy"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-canbus", action="store_true")
    parser.add_argument("--include-sweeps", action="store_true", help="Convert dense LiDAR sample_data sweeps and nearest RGB sample_data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_root = Path(args.output_root) if args.output_root else root / "converted"
    convert_scene(
        root=root,
        version=args.version,
        scene_name=args.scene,
        output_root=output_root,
        copy_mode=args.copy_mode,
        max_frames=args.max_frames,
        use_canbus=not args.no_canbus,
        include_sweeps=args.include_sweeps,
        list_only=args.list_scenes,
    )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import yaml
import open3d as o3d
import numpy as np

from src.calibration.calibration_loader import Calibration
from src.evaluation.reconstruction_eval import (
    compute_reconstruction_metrics_table,
    save_reconstruction_metrics_csv,
)
from src.mapping.fuse_pointclouds import fuse_colored_pointcloud_sequence
from src.utils.io import save_json, save_cloud_ply, load_pose_matrices_npz
from src.visualization.vis_map import save_map_with_trajectory_plot



def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)




def load_saved_lio_trajectories(trajectory_dir):
    trajectory_dir = Path(trajectory_dir)

    pose_files = {
        "ground_truth": trajectory_dir / "ground_truth_poses.npz",
        "imu_only": trajectory_dir / "imu_only_poses.npz",
        "icp_only": trajectory_dir / "icp_only_poses.npz",
        "fused": trajectory_dir / "tightly_fused_lio_poses.npz",
    }

    missing = [
        path for path in pose_files.values()
        if not path.exists()
    ]

    if len(missing) > 0:
        raise FileNotFoundError(
            "Missing saved LIO trajectory files. Run run_lio.py first. "
            "Missing files: "
            + ", ".join(str(path) for path in missing)
        )

    return {
        "ground_truth": load_pose_matrices_npz(pose_files["ground_truth"]),
        "imu_only": load_pose_matrices_npz(pose_files["imu_only"]),
        "icp_only": load_pose_matrices_npz(pose_files["icp_only"]),
        "fused": load_pose_matrices_npz(pose_files["fused"]),
    }




def verify_scene_inputs(scene_dir, camera_names, rgb_extension, calibration_path=None):
    scene_dir = Path(scene_dir)

    required = [
        scene_dir / "lidar",
        scene_dir / "rgb",
        scene_dir / "poses.csv",
        scene_dir / "imu.csv",
    ]

    if calibration_path is None:
        required.append(scene_dir / "calib" / "calibration.json")
    else:
        required.append(Path(calibration_path))

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required scene input: {path}")

    lidar_files = sorted((scene_dir / "lidar").glob("*.npy"))

    if len(lidar_files) == 0:
        raise RuntimeError(f"No LiDAR frames found: {scene_dir / 'lidar'}")

    missing_report = {}

    for camera_name in camera_names:
        camera_dir = scene_dir / "rgb" / camera_name

        if not camera_dir.exists():
            raise FileNotFoundError(f"Missing RGB camera directory: {camera_dir}")

        missing = []

        for lidar_path in lidar_files:
            frame = lidar_path.stem
            rgb_path = camera_dir / f"{frame}{rgb_extension}"

            if not rgb_path.exists():
                missing.append(frame)

        missing_report[camera_name] = {
            "num_missing": len(missing),
            "first_missing": missing[:10],
        }

    return {
        "num_lidar_frames": len(lidar_files),
        "missing_rgb_by_camera": missing_report,
    }




def verify_calibration(calibration, camera_names):
    report = {
        "valid": True,
        "cameras": {},
    }

    for camera_name in camera_names:
        K = calibration.get_camera_K(camera_name)
        D = calibration.get_camera_D(camera_name)
        T_vehicle_camera = calibration.get_T_vehicle_camera(camera_name)

        cam_report = {
            "K_shape": list(K.shape),
            "D_shape": list(D.shape),
            "T_vehicle_camera_shape": list(T_vehicle_camera.shape),
            "K_finite": bool(np.isfinite(K).all()),
            "D_finite": bool(np.isfinite(D).all()),
            "T_vehicle_camera_finite": bool(np.isfinite(T_vehicle_camera).all()),
        }

        ok = (
            K.shape == (3, 3)
            and T_vehicle_camera.shape == (4, 4)
            and cam_report["K_finite"]
            and cam_report["D_finite"]
            and cam_report["T_vehicle_camera_finite"]
        )

        cam_report["valid"] = bool(ok)

        if not ok:
            report["valid"] = False

        report["cameras"][camera_name] = cam_report

    T_vehicle_lidar = calibration.T_vehicle_lidar

    report["lidar"] = {
        "T_vehicle_lidar_shape": list(T_vehicle_lidar.shape),
        "T_vehicle_lidar_finite": bool(np.isfinite(T_vehicle_lidar).all()),
        "valid": bool(
            T_vehicle_lidar.shape == (4, 4)
            and np.isfinite(T_vehicle_lidar).all()
        ),
    }

    if not report["lidar"]["valid"]:
        report["valid"] = False

    return report




def build_map_for_pose_source(
    name,
    poses,
    cfg,
    calibration,
    output_dir,
):
    scene_dir = cfg["scene_dir"]
    camera_names = cfg["camera_names"]

    lidar_cfg = cfg["lidar"]
    proj_cfg = cfg["projection"]
    map_cfg = cfg["mapping"]

    cloud, frame_reports = fuse_colored_pointcloud_sequence(
        scene_dir=scene_dir,
        calibration=calibration,
        poses=poses,
        camera_names=camera_names,
        rgb_extension=cfg.get("rgb_extension", ".jpg"),
        frame_stride=map_cfg.get("frame_stride", 1),
        min_range=lidar_cfg.get("min_range", 1.0),
        max_range=lidar_cfg.get("max_range", 80.0),
        max_points_per_frame=lidar_cfg.get("max_points_per_frame", None),
        min_depth=proj_cfg.get("min_depth", 0.1),
        border_margin_px=proj_cfg.get("border_margin_px", 2),
        center_weight=proj_cfg.get("center_weight", 1.0),
        angle_weight=proj_cfg.get("angle_weight", 1.0),
        depth_weight=proj_cfg.get("depth_weight", 0.05),
        color_mode=proj_cfg.get("color_mode", "best"),
        voxel_size=map_cfg.get("voxel_size", 0.15),
        remove_outliers=map_cfg.get("remove_outliers", True),
        outlier_nb_neighbors=map_cfg.get("outlier_nb_neighbors", 20),
        outlier_std_ratio=map_cfg.get("outlier_std_ratio", 2.0),
        verbose=True,
    )

    if name == "fused":
        ply_name = "baseline_colored_map.ply"
    else:
        ply_name = f"baseline_colored_map_{name}.ply"

    ply_path = output_dir / ply_name
    save_cloud_ply(cloud, ply_path)

    plot_path = output_dir / f"map_with_trajectory_{name}.png"

    save_map_with_trajectory_plot(
        cloud=cloud,
        poses=poses,
        output_path=plot_path,
        title=f"Colored map with {name} trajectory",
        max_points=cfg["visualization"].get("max_points_plot", 120000),
        initial_voxel_size=cfg["visualization"].get("initial_voxel_size_plot", 0.05),
    )

    return cloud, {
        "ply_path": str(ply_path),
        "plot_path": str(plot_path),
        "frame_reports": frame_reports,
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mapping.yaml")
    parser.add_argument("--scene_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--trajectory_dir", default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    if args.scene_dir is not None:
        cfg["scene_dir"] = args.scene_dir

    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    if args.trajectory_dir is not None:
        cfg["trajectory_dir"] = args.trajectory_dir

    scene_dir = Path(cfg["scene_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_names = cfg["camera_names"]
    rgb_extension = cfg.get("rgb_extension", ".jpg")

    calib_path = cfg.get("calibration_path")

    if calib_path is None:
        calib_path = scene_dir / "calib" / "calibration.json"
    else:
        calib_path = Path(calib_path)

    input_report = verify_scene_inputs(
        scene_dir=scene_dir,
        camera_names=camera_names,
        rgb_extension=rgb_extension,
        calibration_path=calib_path,
    )

    calibration = Calibration(calib_path)
    calibration_report = verify_calibration(calibration, camera_names)

    save_json(
        output_dir / "sync_calibration_report.json",
        {
            "input_report": input_report,
            "calibration_report": calibration_report,
        },
    )

    trajectory_dir = Path(cfg["trajectory_dir"])

    print(f"Loading saved LIO trajectories from: {trajectory_dir}")

    pose_sources = load_saved_lio_trajectories(trajectory_dir)

    map_results = {}
    clouds = {}

    for name, poses in pose_sources.items():
        print(f"Building colored map using pose source: {name}")

        cloud, result = build_map_for_pose_source(
            name=name,
            poses=poses,
            cfg=cfg,
            calibration=calibration,
            output_dir=output_dir,
        )

        clouds[name] = cloud
        map_results[name] = result

    eval_cfg = cfg.get("evaluation", {})

    thresholds = eval_cfg.get(
        "distance_thresholds_m",
        [0.10, 0.25, 0.50, 1.00],
    )

    voxel_sizes = eval_cfg.get(
        "voxel_sizes_m",
        [0.10, 0.25, 0.50],
    )

    print("Computing reconstruction metrics against ground-truth cloud.")

    reconstruction_rows = compute_reconstruction_metrics_table(
        clouds=clouds,
        gt_key="ground_truth",
        pose_sources=("ground_truth", "imu_only", "icp_only", "fused"),
        thresholds=thresholds,
        voxel_sizes=voxel_sizes,
        max_points=eval_cfg.get("max_eval_points", 100000),
        eval_initial_voxel_size=eval_cfg.get("eval_initial_voxel_size", 0.05),
        normal_radius=eval_cfg.get("normal_radius", 0.5),
        normal_max_nn=eval_cfg.get("normal_max_nn", 30),
    )

    metrics_csv_path = output_dir / "reconstruction_metrics.csv"

    save_reconstruction_metrics_csv(
        rows=reconstruction_rows,
        output_path=metrics_csv_path,
    )

    save_json(
        output_dir / "map_outputs.json",
        {
            "map_results": map_results,
            "reconstruction_metrics_csv": str(metrics_csv_path),
        },
    )

    if args.visualize:
        main_cloud = clouds[cfg.get("pose_source_for_main_output", "fused")]
        o3d.visualization.draw_geometries([main_cloud])

    print(f"Saved outputs to: {output_dir}")
    print(f"Saved reconstruction metrics CSV to: {metrics_csv_path}")



if __name__ == "__main__":
    main()
import argparse
from pathlib import Path

import yaml
import open3d as o3d

from src.calibration.calibration_loader import Calibration
from src.dynamic_removal.dynamic_point_filter import DynamicPointFilter
from src.evaluation.reconstruction_eval import (
    compute_cloud_to_cloud_distance,
    evaluate_frame_to_frame_alignment,
)
from src.mapping.fuse_pointclouds import fuse_colored_pointcloud_sequence
from src.mapping.voxel_map import estimate_map_density
from src.utils.io import (
    save_json,
    save_cloud_ply,
    load_pose_matrices_npz,
)
from src.visualization.vis_map import save_map_with_trajectory_plot


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_fused_poses(trajectory_dir):
    trajectory_dir = Path(trajectory_dir)
    path = trajectory_dir / "tightly_fused_lio_poses.npz"
    #path = trajectory_dir / "ground_truth_poses.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing fused LIO poses: {path}. Run scripts/run_lio.py first."
        )

    return load_pose_matrices_npz(path)


def make_dynamic_cfg(base_dynamic_cfg, method, keep_stationary_vehicles=None):
    cfg = dict(base_dynamic_cfg)
    cfg["method"] = method

    vehicle_cfg = dict(cfg.get("vehicle_filtering", {}))

    if keep_stationary_vehicles is not None:
        vehicle_cfg["keep_stationary_vehicles"] = bool(keep_stationary_vehicles)

    cfg["vehicle_filtering"] = vehicle_cfg

    return cfg


def summarize_frame_reports(frame_reports):
    if len(frame_reports) == 0:
        return {
            "num_frames": 0,
            "total_input_points": 0,
            "total_kept_points": 0,
            "total_removed_points": 0,
            "total_always_dynamic_points": 0,
            "total_vehicle_candidate_points": 0,
            "total_stationary_vehicle_points_kept": 0,
            "total_moving_vehicle_points_removed": 0,
            "mean_removed_ratio": 0.0,
        }

    total_input = sum(
        int(r.get("num_lidar_points", 0))
        for r in frame_reports
    )

    total_kept = sum(
        int(r.get("num_lidar_points_after_dynamic_filter", 0))
        for r in frame_reports
    )

    total_removed = sum(
        int(r.get("num_removed_dynamic_points", 0))
        for r in frame_reports
    )

    total_always_dynamic = sum(
        int(r.get("num_always_dynamic_points", 0))
        for r in frame_reports
    )

    total_vehicle_candidate = sum(
        int(r.get("num_vehicle_candidate_points", 0))
        for r in frame_reports
    )

    total_stationary_kept = sum(
        int(r.get("num_stationary_vehicle_points_kept", 0))
        for r in frame_reports
    )

    total_moving_removed = sum(
        int(r.get("num_moving_vehicle_points_removed", 0))
        for r in frame_reports
    )

    mean_removed_ratio = (
        float(total_removed / total_input)
        if total_input > 0
        else 0.0
    )

    return {
        "num_frames": int(len(frame_reports)),
        "total_input_points": int(total_input),
        "total_kept_points": int(total_kept),
        "total_removed_points": int(total_removed),
        "total_always_dynamic_points": int(total_always_dynamic),
        "total_vehicle_candidate_points": int(total_vehicle_candidate),
        "total_stationary_vehicle_points_kept": int(total_stationary_kept),
        "total_moving_vehicle_points_removed": int(total_moving_removed),
        "mean_removed_ratio": mean_removed_ratio,
    }


def build_one_variant(
    method,
    variant_name,
    mapping_cfg,
    dynamic_cfg,
    calibration,
    poses,
    output_dir,
):
    scene_dir = Path(mapping_cfg["scene_dir"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_names = mapping_cfg["camera_names"]
    lidar_cfg = mapping_cfg["lidar"]
    proj_cfg = mapping_cfg["projection"]
    map_cfg = mapping_cfg["mapping"]

    if method == "none":
        dynamic_filter = None
    else:
        dynamic_filter = DynamicPointFilter(
            scene_dir=scene_dir,
            calibration=calibration,
            camera_names=camera_names,
            cfg=dynamic_cfg,
            rgb_extension=mapping_cfg.get("rgb_extension", ".jpg"),
        )

    cloud, frame_reports = fuse_colored_pointcloud_sequence(
        scene_dir=scene_dir,
        calibration=calibration,
        poses=poses,
        camera_names=camera_names,
        rgb_extension=mapping_cfg.get("rgb_extension", ".jpg"),
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
        dynamic_filter=dynamic_filter,
        dynamic_filter_name=variant_name,
    )

    ply_path = output_dir / f"colored_map_{variant_name}.ply"
    plot_path = output_dir / f"map_with_trajectory_{variant_name}.png"
    frame_reports_path = output_dir / f"frame_reports_{variant_name}.json"

    save_cloud_ply(cloud, ply_path)

    save_map_with_trajectory_plot(
        cloud=cloud,
        poses=poses,
        output_path=plot_path,
        title=f"Fused map with dynamic filter: {variant_name}",
        max_points=mapping_cfg["visualization"].get(
            "max_points_plot",
            120000,
        ),
        initial_voxel_size=mapping_cfg["visualization"].get(
            "initial_voxel_size_plot",
            0.05,
        ),
    )

    save_json(frame_reports_path, frame_reports)

    summary = summarize_frame_reports(frame_reports)
    density = estimate_map_density(cloud)

    result = {
        "method": method,
        "variant_name": variant_name,
        "ply_path": str(ply_path),
        "plot_path": str(plot_path),
        "frame_reports_path": str(frame_reports_path),
        "frame_summary": summary,
        "map_density": density,
        "dynamic_config": dynamic_cfg,
    }

    return cloud, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_config", default="configs/mapping.yaml")
    parser.add_argument("--dynamic_config", default="configs/dynamic_removal.yaml")
    parser.add_argument("--scene_dir", default=None)
    parser.add_argument("--trajectory_dir", default=None)
    parser.add_argument("--output_dir", default=None)

    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "none",
            "semantic_keep_stationary",
            "semantic_remove_all_vehicles",
            "yolo_keep_stationary",
            "yolo_remove_all_vehicles",
        ],
        choices=[
            "none",
            "semantic_keep_stationary",
            "semantic_remove_all_vehicles",
            "yolo_keep_stationary",
            "yolo_remove_all_vehicles",
        ],
    )

    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    mapping_cfg = load_yaml(args.mapping_config)
    dynamic_base_cfg = load_yaml(args.dynamic_config)

    if args.scene_dir is not None:
        mapping_cfg["scene_dir"] = args.scene_dir

    if args.trajectory_dir is not None:
        mapping_cfg["trajectory_dir"] = args.trajectory_dir

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(mapping_cfg["output_dir"]) / "dynamic_ablation"

    output_dir.mkdir(parents=True, exist_ok=True)

    scene_dir = Path(mapping_cfg["scene_dir"])

    calib_path = mapping_cfg.get("calibration_path")
    if calib_path is None:
        calib_path = scene_dir / "calib" / "calibration.json"
    else:
        calib_path = Path(calib_path)

    calibration = Calibration(calib_path)
    poses = load_fused_poses(mapping_cfg["trajectory_dir"])

    variant_specs = {
        "none": {
            "method": "none",
            "keep_stationary_vehicles": None,
        },
        "semantic_keep_stationary": {
            "method": "semantic",
            "keep_stationary_vehicles": True,
        },
        "semantic_remove_all_vehicles": {
            "method": "semantic",
            "keep_stationary_vehicles": False,
        },
        "yolo_keep_stationary": {
            "method": "yolo",
            "keep_stationary_vehicles": True,
        },
        "yolo_remove_all_vehicles": {
            "method": "yolo",
            "keep_stationary_vehicles": False,
        },
    }

    clouds = {}
    results = {}

    for variant_name in args.variants:
        spec = variant_specs[variant_name]

        method = spec["method"]

        print(f"Building dynamic-removal variant: {variant_name}")

        method_output_dir = output_dir / variant_name

        dynamic_cfg = make_dynamic_cfg(
            base_dynamic_cfg=dynamic_base_cfg,
            method=method,
            keep_stationary_vehicles=spec["keep_stationary_vehicles"],
        )

        cloud, result = build_one_variant(
            method=method,
            variant_name=variant_name,
            mapping_cfg=mapping_cfg,
            dynamic_cfg=dynamic_cfg,
            calibration=calibration,
            poses=poses,
            output_dir=method_output_dir,
        )

        clouds[variant_name] = cloud
        results[variant_name] = result

    comparison = {}

    if "none" in clouds:
        baseline_cloud = clouds["none"]

        for variant_name, cloud in clouds.items():
            comparison[variant_name] = {
                "distance_to_no_filter": compute_cloud_to_cloud_distance(
                    source_cloud=cloud,
                    target_cloud=baseline_cloud,
                    sample_size=50000,
                ),
                "density": estimate_map_density(cloud),
                "frame_summary": results[variant_name]["frame_summary"],
            }

    alignment = evaluate_frame_to_frame_alignment(
        scene_dir=scene_dir,
        poses=poses,
        calibration=calibration,
        voxel_size=mapping_cfg["mapping"].get("voxel_size", 0.3),
        stride=10,
        max_corr=1.0,
        min_range=mapping_cfg["lidar"].get("min_range", 1.0),
        max_range=mapping_cfg["lidar"].get("max_range", 80.0),
    )

    save_json(
        output_dir / "dynamic_ablation_results.json",
        {
            "variant_outputs": results,
            "comparison": comparison,
            "alignment_quality": alignment,
        },
    )

    if args.visualize and len(clouds) > 0:
        o3d.visualization.draw_geometries(list(clouds.values()))

    print(f"Saved dynamic ablation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
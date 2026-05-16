from pathlib import Path
import csv

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from src.mapping.voxel_map import (
    make_point_cloud,
    voxel_downsample_cloud,
    estimate_map_density,
    downsample_to_max_points,
)
from src.utils.io import load_xyz_points_npy
from src.utils.transforms import compose_transforms, transform_points


def summarize_cloud(cloud):
    summary = estimate_map_density(cloud)

    if len(cloud.points) == 0:
        summary.update(
            {
                "has_colors": False,
                "mean_color_rgb": None,
            }
        )
        return summary

    summary["has_colors"] = bool(cloud.has_colors())

    if cloud.has_colors():
        colors = np.asarray(cloud.colors)
        summary["mean_color_rgb"] = colors.mean(axis=0).tolist()
    else:
        summary["mean_color_rgb"] = None

    return summary



def compute_reconstruction_metrics(
    pred_cloud,
    gt_cloud,
    pose_source,
    thresholds=(0.10, 0.25, 0.50, 1.00),
    voxel_sizes=(0.10, 0.25, 0.50),
    max_points=100000,
    eval_initial_voxel_size=0.05,
    normal_radius=0.5,
    normal_max_nn=30,
):
    """
    Compute reconstruction metrics by comparing a predicted cloud to a GT cloud.

    P = predicted reconstructed cloud
    G = ground-truth/reference cloud

    Accuracy:
        P -> G nearest-neighbor distances.

    Completeness:
        G -> P nearest-neighbor distances.

    Chamfer:
        Bidirectional nearest-neighbor distance.
    """
    pred_points = np.asarray(pred_cloud.points, dtype=np.float64)
    gt_points = np.asarray(gt_cloud.points, dtype=np.float64)

    pred_colors = (
        np.asarray(pred_cloud.colors, dtype=np.float64)
        if pred_cloud.has_colors()
        else None
    )

    gt_colors = (
        np.asarray(gt_cloud.colors, dtype=np.float64)
        if gt_cloud.has_colors()
        else None
    )

    pred_points, pred_colors = downsample_to_max_points(
        points_xyz=pred_points,
        colors_rgb=pred_colors,
        max_points=max_points,
        initial_voxel_size=eval_initial_voxel_size,
    )

    gt_points, gt_colors = downsample_to_max_points(
        points_xyz=gt_points,
        colors_rgb=gt_colors,
        max_points=max_points,
        initial_voxel_size=eval_initial_voxel_size,
    )

    row = {
        "pose_source": pose_source,
        "num_pred_points_eval": int(len(pred_points)),
        "num_gt_points_eval": int(len(gt_points)),
    }

    if len(pred_points) == 0 or len(gt_points) == 0:
        row.update(
            {
                "chamfer_l1": np.nan,
                "chamfer_l2": np.nan,
                "pred_to_gt_mean": np.nan,
                "gt_to_pred_mean": np.nan,
                "pred_to_gt_rmse": np.nan,
                "gt_to_pred_rmse": np.nan,
                "accuracy_mean": np.nan,
                "accuracy_median": np.nan,
                "accuracy_rmse": np.nan,
                "accuracy_p95": np.nan,
                "completeness_mean": np.nan,
                "completeness_median": np.nan,
                "completeness_rmse": np.nan,
                "completeness_p95": np.nan,
                "normal_consistency_mean": np.nan,
                "normal_consistency_median": np.nan,
                "color_mae_rgb": np.nan,
                "color_rmse_rgb": np.nan,
                "color_mae_lab": np.nan,
                "color_rmse_lab": np.nan,
            }
        )

        for threshold in thresholds:
            key = f"{float(threshold):.2f}m"
            row[f"precision@{key}"] = np.nan
            row[f"recall@{key}"] = np.nan
            row[f"fscore@{key}"] = np.nan

        for voxel_size in voxel_sizes:
            key = f"{float(voxel_size):.2f}m"
            row[f"voxel_iou@{key}"] = np.nan
            row[f"voxel_precision@{key}"] = np.nan
            row[f"voxel_recall@{key}"] = np.nan

        return row

    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)

    pred_to_gt_distances, pred_to_gt_indices = gt_tree.query(
        pred_points,
        k=1,
        workers=-1,
    )

    gt_to_pred_distances, gt_to_pred_indices = pred_tree.query(
        gt_points,
        k=1,
        workers=-1,
    )

    pred_to_gt_distances = pred_to_gt_distances.astype(np.float64)
    gt_to_pred_distances = gt_to_pred_distances.astype(np.float64)

    pred_to_gt_mean = float(np.mean(pred_to_gt_distances))
    gt_to_pred_mean = float(np.mean(gt_to_pred_distances))

    pred_to_gt_rmse = float(np.sqrt(np.mean(pred_to_gt_distances ** 2)))
    gt_to_pred_rmse = float(np.sqrt(np.mean(gt_to_pred_distances ** 2)))

    row.update(
        {
            "chamfer_l1": float(pred_to_gt_mean + gt_to_pred_mean),
            "chamfer_l2": float(
                np.mean(pred_to_gt_distances ** 2)
                + np.mean(gt_to_pred_distances ** 2)
            ),

            "pred_to_gt_mean": pred_to_gt_mean,
            "gt_to_pred_mean": gt_to_pred_mean,
            "pred_to_gt_rmse": pred_to_gt_rmse,
            "gt_to_pred_rmse": gt_to_pred_rmse,

            "accuracy_mean": pred_to_gt_mean,
            "accuracy_median": float(np.median(pred_to_gt_distances)),
            "accuracy_rmse": pred_to_gt_rmse,
            "accuracy_p95": float(np.percentile(pred_to_gt_distances, 95)),

            "completeness_mean": gt_to_pred_mean,
            "completeness_median": float(np.median(gt_to_pred_distances)),
            "completeness_rmse": gt_to_pred_rmse,
            "completeness_p95": float(np.percentile(gt_to_pred_distances, 95)),
        }
    )

    for threshold in thresholds:
        threshold = float(threshold)
        key = f"{threshold:.2f}m"

        # precision@threshold = fraction of predicted points close to ground truth
        precision = float(np.mean(pred_to_gt_distances <= threshold))

        # recall@threshold = fraction of ground-truth points recovered by prediction
        recall = float(np.mean(gt_to_pred_distances <= threshold))

        if precision + recall > 1e-12:
            fscore = float(2.0 * precision * recall / (precision + recall))
        else:
            fscore = np.nan

        row[f"precision@{key}"] = precision
        row[f"recall@{key}"] = recall
        row[f"fscore@{key}"] = fscore

    pred_normal_cloud = make_point_cloud(pred_points)
    gt_normal_cloud = make_point_cloud(gt_points)

    pred_normal_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius),
            max_nn=int(normal_max_nn),
        )
    )

    gt_normal_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius),
            max_nn=int(normal_max_nn),
        )
    )

    pred_normal_cloud.normalize_normals()
    gt_normal_cloud.normalize_normals()

    pred_normals = np.asarray(pred_normal_cloud.normals, dtype=np.float64)
    gt_normals = np.asarray(gt_normal_cloud.normals, dtype=np.float64)

    matched_gt_normals = gt_normals[pred_to_gt_indices]

    # abs dot product because Open3D normals are usually unoriented.
    normal_consistency = np.abs(
        np.clip(
            np.sum(pred_normals * matched_gt_normals, axis=1),
            -1.0,
            1.0,
        )
    )

    row["normal_consistency_mean"] = float(np.mean(normal_consistency))
    row["normal_consistency_median"] = float(np.median(normal_consistency))

    if pred_colors is not None and gt_colors is not None:
        pred_colors = np.clip(pred_colors, 0.0, 1.0)
        gt_colors = np.clip(gt_colors, 0.0, 1.0)

        matched_gt_colors = gt_colors[pred_to_gt_indices]

        rgb_diff = pred_colors - matched_gt_colors
        rgb_l2 = np.linalg.norm(rgb_diff, axis=1)

        pred_rgb_u8 = (pred_colors * 255.0).round().astype(np.uint8)
        gt_rgb_u8 = (matched_gt_colors * 255.0).round().astype(np.uint8)

        pred_lab = cv2.cvtColor(
            pred_rgb_u8.reshape(1, -1, 3),
            cv2.COLOR_RGB2LAB,
        ).reshape(-1, 3).astype(np.float64)

        gt_lab = cv2.cvtColor(
            gt_rgb_u8.reshape(1, -1, 3),
            cv2.COLOR_RGB2LAB,
        ).reshape(-1, 3).astype(np.float64)

        lab_diff = pred_lab - gt_lab
        lab_l2 = np.linalg.norm(lab_diff, axis=1)

        row["color_mae_rgb"] = float(np.mean(np.abs(rgb_diff)))
        row["color_rmse_rgb"] = float(np.sqrt(np.mean(rgb_l2 ** 2)))
        row["color_mae_lab"] = float(np.mean(np.abs(lab_diff)))
        row["color_rmse_lab"] = float(np.sqrt(np.mean(lab_l2 ** 2)))
    else:
        row["color_mae_rgb"] = np.nan
        row["color_rmse_rgb"] = np.nan
        row["color_mae_lab"] = np.nan
        row["color_rmse_lab"] = np.nan

    for voxel_size in voxel_sizes:
        voxel_size = float(voxel_size)
        key = f"{voxel_size:.2f}m"

        pred_voxel_indices = np.floor(pred_points / voxel_size).astype(np.int64)
        gt_voxel_indices = np.floor(gt_points / voxel_size).astype(np.int64)

        pred_voxels = set(map(tuple, pred_voxel_indices))
        gt_voxels = set(map(tuple, gt_voxel_indices))

        intersection = len(pred_voxels & gt_voxels)
        union = len(pred_voxels | gt_voxels)

        if len(pred_voxels) > 0:
            voxel_precision = float(intersection / len(pred_voxels))
        else:
            voxel_precision = np.nan

        if len(gt_voxels) > 0:
            voxel_recall = float(intersection / len(gt_voxels))
        else:
            voxel_recall = np.nan

        if union > 0:
            voxel_iou = float(intersection / union)
        else:
            voxel_iou = np.nan

        row[f"voxel_iou@{key}"] = voxel_iou
        row[f"voxel_precision@{key}"] = voxel_precision
        row[f"voxel_recall@{key}"] = voxel_recall

    return row



def compute_reconstruction_metrics_table(
    clouds,
    gt_key="ground_truth",
    pose_sources=("ground_truth", "imu_only", "icp_only", "fused"),
    thresholds=(0.10, 0.25, 0.50, 1.00),
    voxel_sizes=(0.10, 0.25, 0.50),
    max_points=100000,
    eval_initial_voxel_size=0.05,
    normal_radius=0.5,
    normal_max_nn=30,
):
    """
    Build one CSV-ready metrics row per pose source.

    The ground_truth row compares GT cloud to itself. It should be near-perfect
    and acts as a sanity check.
    """
    if gt_key not in clouds:
        raise KeyError(f"GT cloud key '{gt_key}' not found in clouds.")

    gt_cloud = clouds[gt_key]

    rows = []

    for pose_source in pose_sources:
        if pose_source not in clouds:
            continue

        row = compute_reconstruction_metrics(
            pred_cloud=clouds[pose_source],
            gt_cloud=gt_cloud,
            pose_source=pose_source,
            thresholds=thresholds,
            voxel_sizes=voxel_sizes,
            max_points=max_points,
            eval_initial_voxel_size=eval_initial_voxel_size,
            normal_radius=normal_radius,
            normal_max_nn=normal_max_nn,
        )

        rows.append(row)

    return rows



def save_reconstruction_metrics_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        raise RuntimeError("No reconstruction metric rows to save.")

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            clean_row = {}

            for key, value in row.items():
                if isinstance(value, np.generic):
                    value = value.item()

                if isinstance(value, float) and not np.isfinite(value):
                    clean_row[key] = ""
                else:
                    clean_row[key] = value

            writer.writerow(clean_row)



def compute_cloud_to_cloud_distance(
    source_cloud,
    target_cloud,
    sample_size=50000,
):
    """
    Computes nearest-neighbor distance from source to target.
    """
    if len(source_cloud.points) == 0 or len(target_cloud.points) == 0:
        return {
            "mean": None,
            "median": None,
            "rmse": None,
            "max": None,
            "num_points": 0,
        }

    source = source_cloud

    if len(source.points) > sample_size:
        source = source.random_down_sample(float(sample_size) / float(len(source.points)))

    distances = np.asarray(source.compute_point_cloud_distance(target_cloud))
    rmse = float(np.sqrt(np.mean(distances ** 2)))

    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "rmse": rmse,
        "max": float(np.max(distances)),
        "num_points": int(len(distances)),
    }



def evaluate_frame_to_frame_alignment(
    scene_dir,
    poses,
    calibration,
    voxel_size=0.3,
    stride=10,
    max_corr=1.0,
    min_range=1.0,
    max_range=80.0,
):
    """
    Lightweight frame-to-frame alignment quality check.

    It transforms consecutive LiDAR frames into world using the chosen poses,
    then runs ICP with identity init. Good poses should need only small ICP correction.
    """
    
    scene_dir = Path(scene_dir)
    lidar_files = sorted((scene_dir / "lidar").glob("*.npy"))

    n = min(len(lidar_files), len(poses))
    lidar_files = lidar_files[:n]
    poses = poses[:n]

    stats = []

    for i in range(0, n - stride, stride):
        p0 = load_xyz_points_npy(lidar_files[i], min_range=min_range, max_range=max_range)
        p1 = load_xyz_points_npy(lidar_files[i + stride], min_range=min_range, max_range=max_range)

        T_world_lidar_0 = compose_transforms(poses[i], calibration.T_vehicle_lidar)
        T_world_lidar_1 = compose_transforms(poses[i + stride], calibration.T_vehicle_lidar)

        w0 = transform_points(points_xyz=p0, T_target_source=T_world_lidar_0)
        w1 = transform_points(points_xyz=p1, T_target_source=T_world_lidar_1)

        c0 = make_point_cloud(w0)
        c0 = voxel_downsample_cloud(c0, voxel_size)

        c1 = make_point_cloud(w1)
        c1 = voxel_downsample_cloud(c1, voxel_size)

        if len(c0.points) < 50 or len(c1.points) < 50:
            continue

        result = o3d.pipelines.registration.registration_icp(
            c1,
            c0,
            max_corr,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        correction_translation = np.linalg.norm(result.transformation[:3, 3])

        stats.append(
            {
                "source_index": int(i + stride),
                "target_index": int(i),
                "fitness": float(result.fitness),
                "inlier_rmse": float(result.inlier_rmse),
                "correction_translation_norm": float(correction_translation),
            }
        )

    if len(stats) == 0:
        return {
            "pairs": [],
            "mean_fitness": None,
            "mean_inlier_rmse": None,
            "mean_correction_translation_norm": None,
        }

    return {
        "pairs": stats,
        "mean_fitness": float(np.mean([s["fitness"] for s in stats])),
        "mean_inlier_rmse": float(np.mean([s["inlier_rmse"] for s in stats])),
        "mean_correction_translation_norm": float(
            np.mean([s["correction_translation_norm"] for s in stats])
        ),
    }
#!/usr/bin/env python3
"""
Aggregate scene-wise evaluation outputs into README-ready CSV and Markdown tables.

Expected existing outputs per scene:
  outputs/trajectories/scene_XXX/metrics.json
  outputs/maps/scene_XXX/reconstruction_metrics.csv
  outputs/maps/scene_XXX/dynamic_ablation/dynamic_ablation_results.json

Example:
  python scripts/aggregate_readme_metrics.py \
    --trajectory_root outputs/trajectories \
    --map_root outputs/maps \
    --output_dir outputs/readme_eval_summary

If --scene_ids is not provided, the script auto-discovers all scene_* folders
under outputs/trajectories and outputs/maps.

This script does not recompute point-cloud metrics. It aggregates the metrics your existing
pipeline has already saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


TRAJECTORY_METHODS = [
    ("imu_only", "IMU only"),
    ("icp_only", "ICP only"),
    ("tightly_fused_lio", "Tightly fused LIO"),
]

RECON_POSE_LABELS = {
    "ground_truth": "Ground truth",
    "imu_only": "IMU only",
    "icp_only": "ICP only",
    "fused": "Tightly fused LIO",
    "tightly_fused_lio": "Tightly fused LIO",
}

DYNAMIC_VARIANT_LABELS = {
    "none": "none",
    "semantic_keep_stationary": "semantic_keep_stationary",
    "semantic_remove_all_vehicles": "semantic_remove_all_vehicles",
    "yolo_keep_stationary": "yolo_keep_stationary",
    "yolo_remove_all_vehicles": "yolo_remove_all_vehicles",
}

DYNAMIC_NOTES = {
    "none": "Baseline with dynamic trails",
    "semantic_keep_stationary": "Oracle-style dynamic filtering with parked vehicle preservation",
    "semantic_remove_all_vehicles": "Removes all vehicle points",
    "yolo_keep_stationary": "Practical segmentation-based filtering",
    "yolo_remove_all_vehicles": "Aggressive learned filtering",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_ids",
        nargs="+",
        default=None,
        help=(
            "Optional scene ids without the scene_ prefix. "
            "If omitted, all scene_* folders under trajectory_root and map_root are auto-discovered."
        ),
    )
    parser.add_argument("--trajectory_root", default="outputs/trajectories")
    parser.add_argument("--map_root", default="outputs/maps")
    parser.add_argument("--output_dir", default="outputs/readme_eval_summary")
    parser.add_argument(
        "--frame_bins",
        nargs="+",
        default=["0:600", "600:1200", "1200:1800", "1800:2400"],
        help="Pose-step ranges for ICP diagnostics, formatted start:end.",
    )
    parser.add_argument("--float_digits", type=int, default=4)
    return parser.parse_args()



def normalize_scene_id(scene_id: str) -> str:
    scene_id = str(scene_id).strip()
    if scene_id.startswith("scene_"):
        scene_id = scene_id[len("scene_"):]
    return scene_id


def discover_scene_ids(trajectory_root: Path, map_root: Path) -> List[str]:
    """
    Discover all scene ids from outputs/trajectories and outputs/maps.

    Uses the union of scene_* directories from both roots so partially completed
    scenes are still included. Individual metric loaders will skip missing files.
    """
    scene_ids = set()

    for root in [trajectory_root, map_root]:
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("scene_"):
                scene_ids.add(normalize_scene_id(path.name))

    def sort_key(x: str):
        return (0, int(x)) if x.isdigit() else (1, x)

    return sorted(scene_ids, key=sort_key)

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def mean_std(values: Iterable[Any]) -> Tuple[Optional[float], Optional[float], int]:
    xs = [safe_float(v) for v in values]
    xs = [v for v in xs if v is not None]
    if len(xs) == 0:
        return None, None, 0
    series = pd.Series(xs, dtype="float64")
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(xs) > 1 else 0.0
    return mean, std, len(xs)


def fmt_value(value: Any, digits: int = 4) -> str:
    v = safe_float(value)
    if v is None:
        return "TODO"
    return f"{v:.{digits}f}"


def fmt_mean_std(mean: Any, std: Any, digits: int = 4) -> str:
    m = safe_float(mean)
    s = safe_float(std)
    if m is None:
        return "TODO"
    if s is None:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def load_trajectory_scene_rows(scene_ids: Sequence[str], trajectory_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    metric_rows: List[Dict[str, Any]] = []
    icp_rows: List[Dict[str, Any]] = []

    for scene_id in scene_ids:
        scene_name = f"scene_{scene_id}"
        metrics_path = trajectory_root / scene_name / "metrics.json"
        data = read_json(metrics_path)
        if data is None:
            continue

        for method_key, method_label in TRAJECTORY_METHODS:
            ate = data.get(f"{method_key}_ate", {}) or {}
            row = {
                "scene": scene_name,
                "method_key": method_key,
                "method": method_label,
                "ate_rmse_m": ate.get("rmse"),
                "ate_mean_m": ate.get("mean"),
                "ate_median_m": ate.get("median"),
                "ate_max_m": ate.get("max"),
                "final_drift_m": data.get(f"{method_key}_final_drift"),
            }
            metric_rows.append(row)

        for idx, stat in enumerate(data.get("icp_stats", []) or []):
            icp_rows.append(
                {
                    "scene": scene_name,
                    "pose_step": idx + 1,
                    "frame": stat.get("frame"),
                    "timestamp": stat.get("timestamp"),
                    "fitness": stat.get("fitness"),
                    "rmse": stat.get("rmse"),
                    "residual_norm": stat.get("residual_norm"),
                }
            )

    return metric_rows, icp_rows


def summarize_trajectory(metric_rows: List[Dict[str, Any]], digits: int) -> Tuple[List[Dict[str, Any]], str]:
    out_rows: List[Dict[str, Any]] = []
    md_rows: List[List[str]] = []

    for method_key, method_label in TRAJECTORY_METHODS:
        rows = [r for r in metric_rows if r["method_key"] == method_key]
        summary: Dict[str, Any] = {"method": method_label}
        md_row = [method_label]
        for col in ["ate_rmse_m", "ate_mean_m", "ate_median_m", "ate_max_m", "final_drift_m"]:
            mean, std, n = mean_std(r.get(col) for r in rows)
            summary[f"{col}_mean"] = mean
            summary[f"{col}_std"] = std
            summary[f"{col}_n"] = n
            md_row.append(fmt_mean_std(mean, std, digits))
        out_rows.append(summary)
        md_rows.append(md_row)

    md = markdown_table(
        ["Method", "ATE RMSE [m]", "ATE Mean [m]", "ATE Median [m]", "ATE Max [m]", "Final Drift [m]"],
        md_rows,
    )
    return out_rows, md


def parse_bins(bin_specs: Sequence[str]) -> List[Tuple[int, int]]:
    bins: List[Tuple[int, int]] = []
    for spec in bin_specs:
        start_s, end_s = spec.split(":", 1)
        bins.append((int(start_s), int(end_s)))
    return bins


def summarize_icp_diagnostics(icp_rows: List[Dict[str, Any]], bins: List[Tuple[int, int]], digits: int) -> Tuple[List[Dict[str, Any]], str]:
    out_rows: List[Dict[str, Any]] = []
    md_rows: List[List[str]] = []

    for start, end in bins:
        rows = [r for r in icp_rows if start <= int(r.get("pose_step", -1)) < end]
        fit_mean, fit_std, fit_n = mean_std(r.get("fitness") for r in rows)
        rmse_mean, rmse_std, rmse_n = mean_std(r.get("rmse") for r in rows)
        res_mean, res_std, res_n = mean_std(r.get("residual_norm") for r in rows)
        label = f"{start}-{end}"
        out_rows.append(
            {
                "frame_range": label,
                "mean_icp_fitness": fit_mean,
                "std_icp_fitness": fit_std,
                "n_icp_fitness": fit_n,
                "mean_icp_rmse": rmse_mean,
                "std_icp_rmse": rmse_std,
                "n_icp_rmse": rmse_n,
                "mean_residual_norm": res_mean,
                "std_residual_norm": res_std,
                "n_residual_norm": res_n,
                "notes": "",
            }
        )
        md_rows.append(
            [
                label,
                fmt_mean_std(fit_mean, fit_std, digits),
                fmt_mean_std(rmse_mean, rmse_std, digits),
                fmt_mean_std(res_mean, res_std, digits),
                "TODO",
            ]
        )

    md = markdown_table(
        ["Frame Range", "Mean ICP Fitness", "Mean ICP RMSE", "Mean Residual Norm", "Notes"],
        md_rows,
    )
    return out_rows, md


def load_reconstruction_rows(scene_ids: Sequence[str], map_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene_id in scene_ids:
        scene_name = f"scene_{scene_id}"
        path = map_root / scene_name / "reconstruction_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            d = row.to_dict()
            d["scene"] = scene_name
            rows.append(d)
    return rows


def summarize_reconstruction(rows: List[Dict[str, Any]], digits: int) -> Tuple[List[Dict[str, Any]], str, str, str]:
    pose_order = ["ground_truth", "imu_only", "icp_only", "fused"]
    summary_rows: List[Dict[str, Any]] = []
    core_md_rows: List[List[str]] = []
    pr_md_rows: List[List[str]] = []
    voxel_md_rows: List[List[str]] = []

    for pose_source in pose_order:
        pose_rows = [r for r in rows if str(r.get("pose_source")) == pose_source]
        label = RECON_POSE_LABELS.get(pose_source, pose_source)
        summary: Dict[str, Any] = {"pose_source": pose_source, "label": label}

        all_cols = [
            "chamfer_l1", "chamfer_l2", "accuracy_mean", "completeness_mean",
            "normal_consistency_mean", "color_mae_rgb", "color_rmse_rgb",
            "precision@0.10m", "recall@0.10m", "fscore@0.10m",
            "precision@0.25m", "recall@0.25m", "fscore@0.25m",
            "precision@0.50m", "recall@0.50m", "fscore@0.50m",
            "precision@1.00m", "recall@1.00m", "fscore@1.00m",
            "voxel_iou@0.10m", "voxel_precision@0.10m", "voxel_recall@0.10m",
            "voxel_iou@0.25m", "voxel_precision@0.25m", "voxel_recall@0.25m",
            "voxel_iou@0.50m", "voxel_precision@0.50m", "voxel_recall@0.50m",
        ]
        for col in all_cols:
            mean, std, n = mean_std(r.get(col) for r in pose_rows)
            summary[f"{col}_mean"] = mean
            summary[f"{col}_std"] = std
            summary[f"{col}_n"] = n
        summary_rows.append(summary)

        core_md_rows.append(
            [
                label,
                fmt_mean_std(summary["chamfer_l1_mean"], summary["chamfer_l1_std"], digits),
                fmt_mean_std(summary["chamfer_l2_mean"], summary["chamfer_l2_std"], digits),
                fmt_mean_std(summary["accuracy_mean_mean"], summary["accuracy_mean_std"], digits),
                fmt_mean_std(summary["completeness_mean_mean"], summary["completeness_mean_std"], digits),
                fmt_mean_std(summary["normal_consistency_mean_mean"], summary["normal_consistency_mean_std"], digits),
                fmt_mean_std(summary["color_mae_rgb_mean"], summary["color_mae_rgb_std"], digits),
                fmt_mean_std(summary["color_rmse_rgb_mean"], summary["color_rmse_rgb_std"], digits),
            ]
        )

        pr_md_rows.append(
            [
                label,
                fmt_mean_std(summary["precision@0.10m_mean"], summary["precision@0.10m_std"], digits),
                fmt_mean_std(summary["recall@0.10m_mean"], summary["recall@0.10m_std"], digits),
                fmt_mean_std(summary["fscore@0.10m_mean"], summary["fscore@0.10m_std"], digits),

                fmt_mean_std(summary["precision@0.25m_mean"], summary["precision@0.25m_std"], digits),
                fmt_mean_std(summary["recall@0.25m_mean"], summary["recall@0.25m_std"], digits),
                fmt_mean_std(summary["fscore@0.25m_mean"], summary["fscore@0.25m_std"], digits),

                fmt_mean_std(summary["precision@0.50m_mean"], summary["precision@0.50m_std"], digits),
                fmt_mean_std(summary["recall@0.50m_mean"], summary["recall@0.50m_std"], digits),
                fmt_mean_std(summary["fscore@0.50m_mean"], summary["fscore@0.50m_std"], digits),

                fmt_mean_std(summary["precision@1.00m_mean"], summary["precision@1.00m_std"], digits),
                fmt_mean_std(summary["recall@1.00m_mean"], summary["recall@1.00m_std"], digits),
                fmt_mean_std(summary["fscore@1.00m_mean"], summary["fscore@1.00m_std"], digits),
            ]
        )

        voxel_md_rows.append(
            [
                label,
                fmt_mean_std(summary["voxel_iou@0.10m_mean"], summary["voxel_iou@0.10m_std"], digits),
                fmt_mean_std(summary["voxel_precision@0.10m_mean"], summary["voxel_precision@0.10m_std"], digits),
                fmt_mean_std(summary["voxel_recall@0.10m_mean"], summary["voxel_recall@0.10m_std"], digits),
                fmt_mean_std(summary["voxel_iou@0.25m_mean"], summary["voxel_iou@0.25m_std"], digits),
                fmt_mean_std(summary["voxel_precision@0.25m_mean"], summary["voxel_precision@0.25m_std"], digits),
                fmt_mean_std(summary["voxel_recall@0.25m_mean"], summary["voxel_recall@0.25m_std"], digits),
            ]
        )

    core_md = markdown_table(
        ["Pose Source", "Chamfer L1", "Chamfer L2", "Accuracy Mean [m]", "Completeness Mean [m]", "Normal Consistency", "Color MAE RGB", "Color RMSE RGB"],
        core_md_rows,
    )
    pr_md = markdown_table(
        [
            "Pose Source",
            "Precision @ 0.10 m",
            "Recall @ 0.10 m",
            "F-Score @ 0.10 m",
            "Precision @ 0.25 m",
            "Recall @ 0.25 m",
            "F-Score @ 0.25 m",
            "Precision @ 0.50 m",
            "Recall @ 0.50 m",
            "F-Score @ 0.50 m",
            "Precision @ 1.00 m",
            "Recall @ 1.00 m",
            "F-Score @ 1.00 m",
        ],
        pr_md_rows,
    )
    voxel_md = markdown_table(
        ["Pose Source", "Voxel IoU @ 0.10 m", "Voxel Precision @ 0.10 m", "Voxel Recall @ 0.10 m", "Voxel IoU @ 0.25 m", "Voxel Precision @ 0.25 m", "Voxel Recall @ 0.25 m"],
        voxel_md_rows,
    )
    return summary_rows, core_md, pr_md, voxel_md


def load_dynamic_scene_rows(scene_ids: Sequence[str], map_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    removal_rows: List[Dict[str, Any]] = []
    map_quality_rows: List[Dict[str, Any]] = []
    alignment_rows: List[Dict[str, Any]] = []

    for scene_id in scene_ids:
        scene_name = f"scene_{scene_id}"
        path = map_root / scene_name / "dynamic_ablation" / "dynamic_ablation_results.json"
        data = read_json(path)
        if data is None:
            continue

        variant_outputs = data.get("variant_outputs", {}) or {}
        comparison = data.get("comparison", {}) or {}

        for variant_name, variant_data in variant_outputs.items():
            frame_summary = variant_data.get("frame_summary", {}) or {}
            density = variant_data.get("map_density", {}) or {}
            removal_rows.append(
                {
                    "scene": scene_name,
                    "variant": variant_name,
                    "total_input_points": frame_summary.get("total_input_points"),
                    "total_kept_points": frame_summary.get("total_kept_points"),
                    "total_removed_points": frame_summary.get("total_removed_points"),
                    "removed_ratio": frame_summary.get("mean_removed_ratio"),
                    "always_dynamic_points": frame_summary.get("total_always_dynamic_points"),
                    "vehicle_candidate_points": frame_summary.get("total_vehicle_candidate_points"),
                    "stationary_vehicle_points_kept": frame_summary.get("total_stationary_vehicle_points_kept"),
                    "moving_vehicle_points_removed": frame_summary.get("total_moving_vehicle_points_removed"),
                }
            )
            dist = (comparison.get(variant_name, {}) or {}).get("distance_to_no_filter", {}) or {}
            map_quality_rows.append(
                {
                    "scene": scene_name,
                    "variant": variant_name,
                    "map_points": density.get("num_points"),
                    "bbox_volume": density.get("bbox_volume"),
                    "density_points_per_m3": density.get("density_points_per_m3"),
                    "distance_to_no_filter_mean": dist.get("mean"),
                    "distance_to_no_filter_rmse": dist.get("rmse"),
                    "notes": DYNAMIC_NOTES.get(variant_name, ""),
                }
            )

        alignment = data.get("alignment_quality", {}) or {}
        alignment_rows.append(
            {
                "scene": scene_name,
                "mean_fitness": alignment.get("mean_fitness"),
                "mean_inlier_rmse": alignment.get("mean_inlier_rmse"),
                "mean_correction_translation_norm": alignment.get("mean_correction_translation_norm"),
            }
        )

    return removal_rows, map_quality_rows, alignment_rows


def summarize_dynamic(removal_rows: List[Dict[str, Any]], map_quality_rows: List[Dict[str, Any]], alignment_rows: List[Dict[str, Any]], digits: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str, str, str]:
    variant_order = [
        "none",
        "semantic_keep_stationary",
        "semantic_remove_all_vehicles",
        "yolo_keep_stationary",
        "yolo_remove_all_vehicles",
    ]

    removal_summary: List[Dict[str, Any]] = []
    removal_md_rows: List[List[str]] = []
    for variant in variant_order:
        rows = [r for r in removal_rows if r.get("variant") == variant]
        summary: Dict[str, Any] = {"variant": variant}
        md_row = [variant]
        for col in [
            "total_input_points",
            "total_kept_points",
            "total_removed_points",
            "removed_ratio",
            "always_dynamic_points",
            "vehicle_candidate_points",
            "stationary_vehicle_points_kept",
            "moving_vehicle_points_removed",
        ]:
            mean, std, n = mean_std(r.get(col) for r in rows)
            summary[f"{col}_mean"] = mean
            summary[f"{col}_std"] = std
            summary[f"{col}_n"] = n
            md_row.append(fmt_mean_std(mean, std, digits))
        removal_summary.append(summary)
        removal_md_rows.append(md_row)

    removal_md = markdown_table(
        ["Variant", "Total Input Points", "Total Kept Points", "Total Removed Points", "Removed Ratio", "Always Dynamic Points", "Vehicle Candidate Points", "Stationary Vehicle Points Kept", "Moving Vehicle Points Removed"],
        removal_md_rows,
    )

    map_summary: List[Dict[str, Any]] = []
    map_md_rows: List[List[str]] = []
    for variant in variant_order:
        rows = [r for r in map_quality_rows if r.get("variant") == variant]
        summary = {"variant": variant, "notes": DYNAMIC_NOTES.get(variant, "")}
        md_row = [variant]
        for col in [
            "map_points",
            "bbox_volume",
            "density_points_per_m3",
            "distance_to_no_filter_mean",
            "distance_to_no_filter_rmse",
        ]:
            mean, std, n = mean_std(r.get(col) for r in rows)
            summary[f"{col}_mean"] = mean
            summary[f"{col}_std"] = std
            summary[f"{col}_n"] = n
            md_row.append(fmt_mean_std(mean, std, digits))
        md_row.append(DYNAMIC_NOTES.get(variant, ""))
        map_summary.append(summary)
        map_md_rows.append(md_row)

    map_md = markdown_table(
        ["Variant", "Map Points", "Bounding Box Volume", "Density [points/m3]", "Distance to No Filter Mean", "Distance to No Filter RMSE", "Notes"],
        map_md_rows,
    )

    alignment_summary: List[Dict[str, Any]] = []
    alignment_md_rows: List[List[str]] = []
    for col, desc in [
        ("mean_fitness", "Average ICP fitness between transformed frame pairs"),
        ("mean_inlier_rmse", "Average alignment error between frame pairs"),
        ("mean_correction_translation_norm", "Average correction needed after pose transform"),
    ]:
        mean, std, n = mean_std(r.get(col) for r in alignment_rows)
        alignment_summary.append(
            {
                "metric": col,
                "description": desc,
                "mean": mean,
                "std": std,
                "n": n,
            }
        )
        pretty = {
            "mean_fitness": "Mean fitness",
            "mean_inlier_rmse": "Mean inlier RMSE",
            "mean_correction_translation_norm": "Mean correction translation norm",
        }[col]
        alignment_md_rows.append([pretty, desc, fmt_mean_std(mean, std, digits)])

    alignment_md = markdown_table(["Metric", "Description", "Value"], alignment_md_rows)
    return removal_summary, map_summary, alignment_summary, removal_md, map_md, alignment_md


def main() -> None:
    args = parse_args()
    trajectory_root = Path(args.trajectory_root)
    map_root = Path(args.map_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scene_ids is None:
        scene_ids = discover_scene_ids(trajectory_root, map_root)
    else:
        scene_ids = [normalize_scene_id(x) for x in args.scene_ids]

    if not scene_ids:
        raise RuntimeError(
            "No scenes found. Provide --scene_ids, or make sure trajectory_root/map_root "
            "contain scene_* directories."
        )

    bins = parse_bins(args.frame_bins)
    digits = args.float_digits

    trajectory_scene_rows, icp_scene_rows = load_trajectory_scene_rows(scene_ids, trajectory_root)
    trajectory_summary_rows, trajectory_md = summarize_trajectory(trajectory_scene_rows, digits)
    icp_summary_rows, icp_md = summarize_icp_diagnostics(icp_scene_rows, bins, digits)

    reconstruction_rows = load_reconstruction_rows(scene_ids, map_root)
    reconstruction_summary_rows, reconstruction_md, precision_recall_md, voxel_md = summarize_reconstruction(reconstruction_rows, digits)

    dynamic_removal_scene_rows, dynamic_map_scene_rows, alignment_scene_rows = load_dynamic_scene_rows(scene_ids, map_root)
    dynamic_removal_summary_rows, dynamic_map_summary_rows, alignment_summary_rows, dynamic_removal_md, dynamic_map_md, alignment_md = summarize_dynamic(
        dynamic_removal_scene_rows,
        dynamic_map_scene_rows,
        alignment_scene_rows,
        digits,
    )

    write_csv(output_dir / "trajectory_scene_level.csv", trajectory_scene_rows)
    write_csv(output_dir / "trajectory_summary.csv", trajectory_summary_rows)
    write_csv(output_dir / "icp_diagnostics_scene_level.csv", icp_scene_rows)
    write_csv(output_dir / "icp_diagnostics_summary.csv", icp_summary_rows)
    write_csv(output_dir / "reconstruction_scene_level.csv", reconstruction_rows)
    write_csv(output_dir / "reconstruction_summary.csv", reconstruction_summary_rows)
    write_csv(output_dir / "dynamic_removal_scene_level.csv", dynamic_removal_scene_rows)
    write_csv(output_dir / "dynamic_removal_summary.csv", dynamic_removal_summary_rows)
    write_csv(output_dir / "dynamic_map_quality_scene_level.csv", dynamic_map_scene_rows)
    write_csv(output_dir / "dynamic_map_quality_summary.csv", dynamic_map_summary_rows)
    write_csv(output_dir / "frame_to_frame_alignment_scene_level.csv", alignment_scene_rows)
    write_csv(output_dir / "frame_to_frame_alignment_summary.csv", alignment_summary_rows)

    md_sections = [
        "# README Evaluation Tables",
        "",
        "Values are reported as mean +/- standard deviation across available scenes.",
        "",
        "## Trajectory Evaluation",
        trajectory_md,
        "",
        "## ICP and Fusion Diagnostics",
        icp_md,
        "",
        "## Reconstruction Evaluation",
        reconstruction_md,
        "",
        "## Precision, Recall, and F-Score",
        precision_recall_md,
        "",
        "## Voxel Occupancy Evaluation",
        voxel_md,
        "",
        "## Dynamic Object Removal Evaluation",
        dynamic_removal_md,
        "",
        "## Dynamic Ablation Map Quality",
        dynamic_map_md,
        "",
        "## Frame-to-Frame Alignment Quality",
        alignment_md,
        "",
    ]
    (output_dir / "readme_tables.md").write_text("\n".join(md_sections))

    manifest = {
        "scene_ids_used": scene_ids,
        "scene_discovery_mode": "auto" if args.scene_ids is None else "cli",
        "num_trajectory_metric_rows": len(trajectory_scene_rows),
        "num_icp_rows": len(icp_scene_rows),
        "num_reconstruction_rows": len(reconstruction_rows),
        "num_dynamic_removal_rows": len(dynamic_removal_scene_rows),
        "num_dynamic_map_quality_rows": len(dynamic_map_scene_rows),
        "num_alignment_rows": len(alignment_scene_rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print(f"Saved README-ready tables to: {output_dir / 'readme_tables.md'}")


if __name__ == "__main__":
    main()

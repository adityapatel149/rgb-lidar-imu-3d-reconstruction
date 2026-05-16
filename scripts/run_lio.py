import argparse
from pathlib import Path
import json

from src.odometry.lio import run_lio
from src.evaluation.trajectory_eval import compute_ate, compute_final_drift
from src.visualization.vis_trajectory import plot_trajectories
from src.utils.io import save_trajectory_csv, save_pose_matrices_npz

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="data/carla/raw/scene_001")
    parser.add_argument("--voxel_size", type=float, default=0.25)
    parser.add_argument("--max_corr", type=float, default=0.75)
    parser.add_argument("--output", default="outputs/trajectories/scene_001")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_lio(
        scene_dir=args.scene,
        voxel_size=args.voxel_size,
        max_correspondence_distance=args.max_corr,
    )

    gt = results["gt_poses"]
    icp = results["icp_poses"]
    fused = results["fused_poses"]
    imu = results["imu_poses"][: len(gt)]

    metrics = {
        "imu_only_ate": compute_ate(gt, imu),
        "icp_only_ate": compute_ate(gt, icp),
        "tightly_fused_lio_ate": compute_ate(gt, fused),
        "imu_only_final_drift": compute_final_drift(gt, imu),
        "icp_only_final_drift": compute_final_drift(gt, icp),
        "tightly_fused_lio_final_drift": compute_final_drift(gt, fused),
        "icp_stats": results["icp_stats"],
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_trajectories(
        {
            "ground_truth": gt,
            "imu_only": imu,
            "icp_only": icp,
            "tightly_fused_lio": fused,
        },
        output_dir / "trajectory_comparison.png",
    )

    save_trajectory_csv(gt, output_dir / "ground_truth.csv")
    save_trajectory_csv(imu, output_dir / "imu_only.csv")
    save_trajectory_csv(icp, output_dir / "icp_only.csv")
    save_trajectory_csv(fused, output_dir / "tightly_fused_lio.csv")

    save_pose_matrices_npz(output_dir / "ground_truth_poses.npz", gt)
    save_pose_matrices_npz(output_dir / "imu_only_poses.npz", imu)
    save_pose_matrices_npz(output_dir / "icp_only_poses.npz", icp)
    save_pose_matrices_npz(output_dir / "tightly_fused_lio_poses.npz", fused)
    # print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
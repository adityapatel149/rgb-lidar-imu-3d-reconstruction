import numpy as np

from src.odometry.trajectory_utils import poses_to_xyz



def align_by_start(poses):
    T0_inv = np.linalg.inv(poses[0])
    return [T0_inv @ T for T in poses]



def compute_ate(gt_poses, pred_poses):
    n = min(len(gt_poses), len(pred_poses))

    gt = align_by_start(gt_poses[:n])
    pred = align_by_start(pred_poses[:n])

    gt_xyz = poses_to_xyz(gt)
    pred_xyz = poses_to_xyz(pred)

    errors = np.linalg.norm(gt_xyz - pred_xyz, axis=1)

    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
    }



def compute_final_drift(gt_poses, pred_poses):
    n = min(len(gt_poses), len(pred_poses))

    gt = align_by_start(gt_poses[:n])
    pred = align_by_start(pred_poses[:n])

    gt_final = gt[-1][:3, 3]
    pred_final = pred[-1][:3, 3]

    return float(np.linalg.norm(gt_final - pred_final))
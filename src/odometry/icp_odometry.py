from pathlib import Path
import numpy as np
import open3d as o3d



def load_lidar_cloud(path, voxel_size=0.25, max_range=80.0):
    points = np.load(path)
    xyz = points[:, :3]
    distances = np.linalg.norm(xyz, axis=1)
    valid = (
        np.isfinite(xyz).all(axis=1)
        & (distances > 1.0)
        & (distances < max_range)
    )

    xyz=xyz[valid]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)

    if voxel_size is not None and voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)

    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius = voxel_size * 4.0,
            max_nn=30,
        )    
    )

    return cloud



def run_icp_pair(
    source_cloud,
    target_cloud,
    init_transform=np.eye(4),
    max_correspondence_distance = 1.0,
):
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),        
        # o3d.pipelines.registration.TransformationEstimationPointToPoint(),        
    )
    return result.transformation, result.fitness, result.inlier_rmse



def lidar_files(scene_dir):
    lidar_dir = Path(scene_dir) / "lidar"
    return sorted(lidar_dir.glob("*.npy"))
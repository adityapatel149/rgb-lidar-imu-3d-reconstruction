import numpy as np
import open3d as o3d



def make_point_cloud(points_xyz, colors_rgb=None):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_xyz)
    if colors_rgb is not None:
        colors_rgb = np.asarray(colors_rgb, dtype=np.float64)
        colors_rgb = np.clip(colors_rgb, 0.0, 1.0)
        cloud.colors = o3d.utility.Vector3dVector(colors_rgb)

    return cloud



def voxel_downsample_cloud(cloud, voxel_size):
    if voxel_size is None or voxel_size <= 0:
        return cloud
    if len(cloud.points) == 0:
        return cloud
    return cloud.voxel_down_sample(float(voxel_size))



def voxel_downsample_arrays(points_xyz, colors_rgb=None, voxel_size=0.15):
    cloud = make_point_cloud(points_xyz, colors_rgb)
    cloud = voxel_downsample_cloud(cloud, voxel_size)
    
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None

    return points, colors



def downsample_to_max_points(
    points_xyz,
    colors_rgb=None,
    max_points=100000,
    initial_voxel_size=0.05,
    growth_rate=1.5,
    max_iters=20,
):
    if max_points is None or len(points_xyz) <= max_points:
        return points_xyz, colors_rgb

    voxel_size = float(initial_voxel_size)

    for _ in range(max_iters):
        pts, cols = voxel_downsample_arrays(points_xyz, colors_rgb, voxel_size=voxel_size)
        if len(pts) <= max_points:
            return pts, cols
        voxel_size *= growth_rate

    return pts, cols



def estimate_map_density(cloud):
    points = np.asarray(cloud.points)
    if len(points) == 0:
        return {
            "num_points": 0,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_extent": None,
            "bbox_volume": 0.0,
            "density_points_per_m3": 0.0,
        }

    bbox = cloud.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    volume = float(np.prod(np.maximum(extent, 1e-9)))

    return {
        "num_points": int(len(points)),
        "bbox_min": bbox.get_min_bound().tolist(),
        "bbox_max": bbox.get_max_bound().tolist(),
        "bbox_extent": extent.tolist(),
        "bbox_volume": volume,
        "density_points_per_m3": float(len(points) / volume),
    }
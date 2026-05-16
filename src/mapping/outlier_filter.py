import open3d as o3d



def remove_statistical_outliers(cloud, nb_neighbors=20, std_ratio=2.0):
    if len(cloud.points) == 0:
        return cloud, []
    filtered, inlier_indices = cloud.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
    return filtered, inlier_indices



def remove_radius_outliers(cloud, nb_points=8, radius=0.5):
    if len(cloud.points) == 0:
        return cloud, []
    filtered, inlier_indices = cloud.remove_radius_outlier(nb_points=int(nb_points), radius=float(radius))
    return filtered, inlier_indices
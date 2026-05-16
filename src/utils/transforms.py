import math
import numpy as np



def carla_transform_to_matrix(transform):
    """
    Convert a CARLA Transform to a 4x4 homogeneous transform matrix.

    Output:
        T_world_actor if the input transform is actor.get_transform().
    """
    location = transform.location
    rotation = transform.rotation

    cy = math.cos(math.radians(rotation.yaw))
    sy = math.sin(math.radians(rotation.yaw))
    cp = math.cos(math.radians(rotation.pitch))
    sp = math.sin(math.radians(rotation.pitch))
    cr = math.cos(math.radians(rotation.roll))
    sr = math.sin(math.radians(rotation.roll))

    T = np.eye(4, dtype=np.float64)

    T[0, 0] = cp * cy
    T[0, 1] = cy * sp * sr - sy * cr
    T[0, 2] = -cy * sp * cr - sy * sr

    T[1, 0] = sy * cp
    T[1, 1] = sy * sp * sr + cy * cr
    T[1, 2] = -sy * sp * cr + cy * sr

    T[2, 0] = sp
    T[2, 1] = -cp * sr
    T[2, 2] = cp * cr

    T[0, 3] = location.x
    T[1, 3] = location.y
    T[2, 3] = location.z

    return T



def relative_transform(T_world_source, T_world_target):
    """
    Compute transform from source frame to target frame.

    Inputs:
        T_world_source: source frame -> world frame
        T_world_target: target frame -> world frame

    Output:
        T_target_source: source frame -> target frame
    """
    return np.linalg.inv(T_world_target) @ T_world_source



def compose_transforms(*transforms):
    """
    Compose transforms left-to-right.

    Example:
        compose_transforms(
            T_a_b,
            T_b_c,
        )
        returns T_a_c
    """
    T_out = np.eye(4, dtype=np.float64)
    for T in transforms:
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Expected 4x4 transform, got {T.shape}")
        T_out = T_out @ T

    return T_out



def to_homogeneous(points_xyz):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points_xyz.shape}")

    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    return np.hstack([points_xyz, ones])



def transform_points(points_xyz, T_target_source):
    """
    Transform Nx3 points from source frame to target frame.
    """
    points_h = to_homogeneous(points_xyz)
    T_target_source = np.asarray(T_target_source, dtype=np.float64)

    if T_target_source.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {T_target_source.shape}")

    transformed = (T_target_source @ points_h.T).T
    return transformed[:, :3]



def relative_actor_transform(source_actor, target_actor):
    """
    Compute transform from source actor frame
    to target actor frame.

    Returns:
        T_target_source
    """
    T_world_source = carla_transform_to_matrix(source_actor.get_transform())
    T_world_target = carla_transform_to_matrix(target_actor.get_transform())

    return relative_transform(
        T_world_source=T_world_source,
        T_world_target=T_world_target,
    )
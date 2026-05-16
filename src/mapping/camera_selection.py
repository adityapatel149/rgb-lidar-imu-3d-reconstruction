import numpy as np



def projection_quality(
    points_camera_cv,
    uv,
    K,
):
    """
    Compute camera-view quality terms for projected 3D points.
    Lower values are better.

    Returns:
        center_distance_norm: Normalized distance from projected pixel to image center. 
        viewing_angle: Angle between camera optical axis and 3D point ray.
    """
    points_camera_cv = np.asarray(points_camera_cv, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    n = points_camera_cv.shape[0]

    center_distance_norm = np.full(n, np.nan, dtype=np.float64)
    viewing_angle = np.full(n, np.nan, dtype=np.float64)

    finite_uv = np.isfinite(uv).all(axis=1)

    cx = float(K[0, 2])
    cy = float(K[1, 2])
    half_diag = np.sqrt(cx * cx + cy * cy) + 1e-12

    center_distance_norm[finite_uv] = (
        np.sqrt(
            (uv[finite_uv, 0] - cx) ** 2
            + (uv[finite_uv, 1] - cy) ** 2
        )
        / half_diag
    )

    ray_norm = np.linalg.norm(points_camera_cv, axis=1) + 1e-12
    # cos angle between camera axis(z-forward) and ray
    cos_angle = np.clip(points_camera_cv[:, 2] / ray_norm, -1.0, 1.0)
    # angle in radians
    viewing_angle[:] = np.arccos(cos_angle)

    return {
        "center_distance_norm": center_distance_norm,
        "viewing_angle": viewing_angle,
    }



def score_projection(
    valid_mask,
    depth,
    center_distance_norm,
    viewing_angle,
    center_weight=1.0,
    angle_weight=1.0,
    depth_weight=0.05,
):
    """
    Score projected points for camera selection.
    Lower score is better.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    depth = np.asarray(depth, dtype=np.float64)
    center_distance_norm = np.asarray(center_distance_norm, dtype=np.float64)
    viewing_angle = np.asarray(viewing_angle, dtype=np.float64)

    score = np.full(valid_mask.shape[0], np.inf, dtype=np.float64)

    finite = (
        valid_mask
        & np.isfinite(depth)
        & np.isfinite(center_distance_norm)
        & np.isfinite(viewing_angle)
    )

    if not np.any(finite):
        return score

    # normalize depth values, divide by median depth
    depth_norm = depth / (np.nanmedian(depth[finite]) + 1e-12)
    # Compute score
    score[finite] = (
        center_weight * center_distance_norm[finite] # center distance penalty
        + angle_weight * viewing_angle[finite] # viewing ngle penalty
        + depth_weight * depth_norm[finite] # depth penalty
    )

    return score
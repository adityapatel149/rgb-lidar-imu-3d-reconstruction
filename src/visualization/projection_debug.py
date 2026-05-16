import numpy as np
import cv2
from src.utils.projection import project_sensor_points_to_camera_image


def save_projection_debug_image(
    image_bgra,
    points_sensor_xyz,
    T_vehicle_sensor,
    T_vehicle_camera,
    K,
    D,
    output_path,
    image_width,
    image_height,
    max_depth_for_color=80.0,
):
    """
    Save a debug image showing projected 3D sensor points on an RGB image.

    image_bgra:
        CARLA raw BGRA image array or an already-shaped HxWx4 array.
    """
    image_bgra = np.asarray(image_bgra)

    if image_bgra.ndim == 1:
        image_bgra = image_bgra.reshape(image_height, image_width, 4)

    image = image_bgra[:, :, :3].copy()

    projection = project_sensor_points_to_camera_image(
        points_sensor_xyz=points_sensor_xyz,
        T_vehicle_sensor=T_vehicle_sensor,
        T_vehicle_camera=T_vehicle_camera,
        K=K,
        D=D,
        image_width=image_width,
        image_height=image_height,
        min_depth=0.1,
        border_margin_px=2,
    )

    valid = projection["valid_mask"]
    uv = projection["uv"][valid].astype(np.int32)
    depth = projection["depth"][valid]

    if len(depth) > 0:
        depth_norm = np.clip(depth / float(max_depth_for_color), 0.0, 1.0)
        colors = (255 * (1.0 - depth_norm)).astype(np.uint8)

        for (u, v), c in zip(uv[::3], colors[::3]):
            cv2.circle(
                image,
                (int(u), int(v)),
                1,
                (0, int(c), 255 - int(c)),
                -1,
            )

    cv2.imwrite(str(output_path), image)
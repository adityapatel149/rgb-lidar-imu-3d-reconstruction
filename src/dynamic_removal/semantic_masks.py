from pathlib import Path
import cv2
import numpy as np



def load_semantic_label(scene_dir, camera_name, frame_id, semantic_dir = "semantic"):
    """
    Load a saved CARLA semantic segmentation label image.

    Expected path:
        scene_dir/semantic/{camera_name}/{frame_id:06d}.png
    """
    scene_dir = Path(scene_dir)
    path = scene_dir / semantic_dir / camera_name / f"{int(frame_id):06d}.png"
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(f"Could not read semantic label image: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.uint8)



def make_class_mask(label_image, class_ids):
    label_image = np.asarray(label_image)
    class_ids = np.asarray(class_ids, dtype=np.uint8)
    if label_image.ndim != 2:
        raise ValueError(f"Expected single-channel label image, got shape {label_image.shape}")

    if len(class_ids) == 0:
        return np.zeros(label_image.shape, dtype=bool)

    return np.isin(label_image, class_ids)



def dilate_mask(mask, dilation_px):
    mask = np.asarray(mask, dtype=bool)
    dilation_px = int(dilation_px)
    if dilation_px <= 0:
        return mask
    kernel_size = 2 * dilation_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)



def load_semantic_masks(
    scene_dir,
    camera_name,
    frame_id,
    always_dynamic_class_ids,
    vehicle_class_ids,
    semantic_dir="semantic",
    mask_dilation_px=0,
):
    """
    Load CARLA semantic label image and return two masks:
    - always_dynamic_mask: pedestrians, generic dynamic objects, etc.
    - vehicle: vehicles, handled separately for parked/moving filtering.
    """
    label = load_semantic_label(scene_dir, camera_name, frame_id, semantic_dir)
    always_dynamic_mask = make_class_mask(label_image=label, class_ids=always_dynamic_class_ids)
    vehicle_mask = make_class_mask(label_image=label, class_ids=vehicle_class_ids)

    always_dynamic_mask = dilate_mask(always_dynamic_mask, mask_dilation_px)
    vehicle_mask = dilate_mask(vehicle_mask, mask_dilation_px)

    return {
        "always_dynamic": always_dynamic_mask,
        "vehicle": vehicle_mask,
    }
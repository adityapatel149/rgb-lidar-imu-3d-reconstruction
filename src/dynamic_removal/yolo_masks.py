from pathlib import Path
import cv2
import numpy as np

from src.utils.io import load_bgr_image
from src.dynamic_removal.semantic_masks import dilate_mask



class YoloDynamicMasker:
    """
    YOLO-based mask provider.

    model_path may be:
        .pt
        .onnx
        .engine

    """
    def __init__(
        self,
        model_path="yolo26n-seg.pt",
        always_dynamic_class_names=None,
        vehicle_class_names=None,
        confidence=0.35,
        iou=0.50,
        mask_dilation_px=0,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for YOLO dynamic removal."
                "Install it with: pip install ultralytics"
            ) from exc

        self.model = YOLO(model_path)

        if always_dynamic_class_names is None:
            always_dynamic_class_names = [
                "person",
            ]

        if vehicle_class_names is None:
            vehicle_class_names = [
                "car",
                "bus",
                "truck",
                "bicycle",
                "motorcycle",
            ]

        self.always_dynamic_class_names = set(always_dynamic_class_names)
        self.vehicle_class_names = set(vehicle_class_names)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.mask_dilation_px = int(mask_dilation_px)



    def __class_ids_for_names(self, target_names):
        ids = set()
        for class_id, class_name in self.model.names.items():
            if class_name in target_names:
                ids.add(int(class_id))
        return ids



    def predict_semantic_masks(self, image_path):
        """
        Predict two masks:
        - always_dynamic
        - vehicle

        Returns:
            dict with HxW bool arrays.
        """
        image_path = Path(image_path)
        image_bgr = load_bgr_image(image_path)
        h, w = image_bgr.shape[:2]
        always_dynamic_mask = np.zeros((h, w), dtype=bool)
        vehicle_mask = np.zeros((h, w), dtype=bool)
        always_dynamic_ids = self.__class_ids_for_names(self.always_dynamic_class_names)
        vehicle_ids = self.__class_ids_for_names(self.vehicle_class_names)

        results = self.model.predict(image_bgr, conf=self.confidence, iou=self.iou, verbose=False)

        if len(results) == 0:
            return {
                "always_dynamic": always_dynamic_mask,
                "vehicle": vehicle_mask,
            }

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return {
                "always_dynamic": always_dynamic_mask,
                "vehicle": vehicle_mask,
            }

        if result.masks is None or result.masks.data is None:
            raise RuntimeError(
                "YOLO model did not return segmentation masks. "
                "Use an instance-segmentation model such as yolov26n-seg.pt, "
                "or an exported segmentation model in .onnx/.engine format."
            )

        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        masks = result.masks.data.detach().cpu().numpy()

        for det_idx, class_id in enumerate(classes):
            is_always_dynamic = class_id in always_dynamic_ids
            is_vehicle = class_id in vehicle_ids

            if not is_always_dynamic and not is_vehicle:
                continue

            instance_mask = (masks[det_idx] > 0.5).astype(np.uint8)
            if instance_mask.shape != (h, w):
                instance_mask = cv2.resize(instance_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            instance_mask = instance_mask.astype(bool)

            # Bitwise OR
            if is_always_dynamic:
                always_dynamic_mask |= instance_mask
            if is_vehicle:
                vehicle_mask |= instance_mask

        always_dynamic_mask = dilate_mask(always_dynamic_mask, self.mask_dilation_px)
        vehicle_mask = dilate_mask(vehicle_mask, self.mask_dilation_px)

        return {
            "always_dynamic": always_dynamic_mask,
            "vehicle": vehicle_mask,
        }
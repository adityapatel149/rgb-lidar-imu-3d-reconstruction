from pathlib import Path
import numpy as np

from src.dynamic_removal.semantic_masks import load_semantic_masks
from src.dynamic_removal.yolo_masks import YoloDynamicMasker
from src.dynamic_removal.lidar_masking import lidar_semantic_masks_multicamera
from src.dynamic_removal.vehicle_motion_filter import VehicleMotionFilter
from src.utils.io import extract_frame_id, load_xyz_points_npy
from src.mapping.voxel_map import downsample_to_max_points


class DynamicPointFilter:
    """
    Dynamic LiDAR point filter.

    Supports:
    - none
    - semantic
    - yolo

    Vehicle handling:
    - remove all vehicles
    - or keep stationary vehicles using world-frame voxel consistency
    """
    def __init__(self, scene_dir, calibration, camera_names, cfg, rgb_extension=".jpg"):
        self.scene_dir = Path(scene_dir)
        self.calibration = calibration
        self.camera_names = list(camera_names)
        self.cfg = cfg
        self.rgb_extension = rgb_extension

        self.method = cfg.get("method", "none")
        self._yolo_masker = None
        self._vehicle_motion_filter = None

        if self.method == "yolo":
            yolo_cfg = cfg.get("yolo", {})
            self._yolo_masker = YoloDynamicMasker(
                model_path=yolo_cfg.get("model_path", "yolo26n-seg.pt"),
                always_dynamic_class_names=yolo_cfg.get("always_dynamic_class_names", ["person"]),
                vehicle_class_names=yolo_cfg.get("vehicle_class_names", ["car", "bus", "truck", "bicycle", "motorcycle"]),
                confidence=yolo_cfg.get("confidence", 0.35),
                iou=yolo_cfg.get("iou", 0.5),
                mask_dilation_px=yolo_cfg.get("mask_dilation_px", 0),
                imgsz=yolo_cfg.get("imgsz", 1280),
                half=yolo_cfg.get("half", False),
                device=yolo_cfg.get("device", None),
                retina_masks=yolo_cfg.get("retina_masks", True),
            )



    def uses_vehicle_filtering(self):
        motion_cfg = self.cfg.get("vehicle_filtering", {})
        return (
            self.method in {"semantic", "yolo"}
            and bool(motion_cfg.get("keep_stationary_vehicles", False))
        )



    def prepare(
        self,
        lidar_files,
        poses,
        min_range=1.0,
        max_range=80.0,
        max_points_per_frame=None,
        frame_stride=1,
    ):
        """
        Optional first pass over the sequence.

        For vehicle stationarity:
        - find vehicle candidate points per frame
        - transform them into world frame
        - count repeated world-voxel observations
        """

        if not self.uses_vehicle_filtering():
            return {
                "used": False,
                "reason": "vehicle motion filter disabled. Either method=none or keep_stationary_vehicles=False",
            }
        
        motion_cfg = self.cfg.get("vehicle_filtering", {})
        self._vehicle_motion_filter = VehicleMotionFilter(
            calibration=self.calibration,
            voxel_size_m=motion_cfg.get("voxel_size_m", 0.5),
            min_observations_for_stationary=motion_cfg.get("min_observations_for_stationary", 3),
            min_frame_gap=motion_cfg.get("min_frame_gap", 2),
        )

        n = min(len(lidar_files), len(poses))
        lidar_files = lidar_files[:n]
        poses = poses[:n]
        frame_stride = int(frame_stride)

        for i in range(0, n, frame_stride):
            lidar_path = lidar_files[i]
            frame_id = extract_frame_id(lidar_path)
            points_lidar = load_xyz_points_npy(lidar_path, min_range=min_range, max_range=max_range)

            if (max_points_per_frame is not None):
                points_lidar, _ = downsample_to_max_points(points_lidar, colors_rgb=None, max_points=max_points_per_frame)

            category_masks = self._lidar_masks_for_frame(points_lidar, frame_id)
            vehicle_mask = category_masks["vehicle_mask"]
            vehicle_points = points_lidar[vehicle_mask]
            self._vehicle_motion_filter.add_vehicle_points(vehicle_points, T_world_vehicle=poses[i], frame_index=i)

        self._vehicle_motion_filter.finalize()

        return {
            "used": True,
            "summary": self._vehicle_motion_filter.summary(),
        }



    def _semantic_masks_for_frame(self, frame_id):
        semantic_cfg = self.cfg.get("semantic", {})

        always_dynamic_class_ids = semantic_cfg.get("always_dynamic_class_ids", [4, 20])
        vehicle_class_ids = semantic_cfg.get("vehicle_class_ids", [10])
        semantic_dir = semantic_cfg.get("semantic_dir", "semantic")
        mask_dilation_px = semantic_cfg.get("mask_dilation_px", 0)

        masks = {}
        for camera_name in self.camera_names:
            masks[camera_name] = load_semantic_masks(
                scene_dir=self.scene_dir,
                camera_name=camera_name,
                frame_id=frame_id,
                always_dynamic_class_ids=always_dynamic_class_ids,
                vehicle_class_ids=vehicle_class_ids,
                semantic_dir=semantic_dir,
                mask_dilation_px=mask_dilation_px,
            )

        return masks



    def _yolo_masks_for_frame(self, frame_id):
        if self._yolo_masker is None:
            raise RuntimeError("YOLO masker was not initialized.")

        masks = {}
        for camera_name in self.camera_names:
            image_path = (
                self.scene_dir
                / "rgb"
                / camera_name
                / f"{int(frame_id):06d}{self.rgb_extension}"
            )

            masks[camera_name] = self._yolo_masker.predict_semantic_masks(image_path=image_path)

        return masks



    def _image_masks_for_frame(self, frame_id):
        if self.method == "none":
            return None

        if self.method == "semantic":
            return self._semantic_masks_for_frame(frame_id)

        if self.method == "yolo":
            return self._yolo_masks_for_frame(frame_id)

        raise ValueError(
            f"Unknown dynamic removal method '{self.method}'. "
            "Expected one of: none, semantic, yolo."
        )



    def _lidar_masks_for_frame(self, points_lidar_xyz, frame_id):
        projection_cfg = self.cfg.get("projection", {})

        image_masks = self._image_masks_for_frame(frame_id)

        return lidar_semantic_masks_multicamera(
            points_lidar_xyz=points_lidar_xyz,
            camera_names=self.camera_names,
            semantic_masks_by_camera=image_masks,
            calibration=self.calibration,
            min_depth=projection_cfg.get("min_depth", 0.1),
            border_margin_px=projection_cfg.get("border_margin_px", 2),
        )



    def filter_points(self, points_lidar_xyz, frame_id, T_world_vehicle=None):
        """
        Remove dynamic LiDAR points for one frame.

        Args:
            points_lidar_xyz:
                Nx3 LiDAR-frame points.
            frame_id:
                CARLA frame ID.
            T_world_vehicle:
                Required if stationary-vehicle filtering is enabled.

        Returns:
            filtered_points:
                Mx3 static or kept points.
            report:
                Dict with debug counts.
        """
        points_lidar_xyz = np.asarray(points_lidar_xyz, dtype=np.float64)

        if self.method == "none":
            return points_lidar_xyz, {
                "method": "none",
                "num_input_points": int(len(points_lidar_xyz)),
                "num_removed_points": 0,
                "num_static_points": int(len(points_lidar_xyz)),
                "num_always_dynamic_points": 0,
                "num_vehicle_candidate_points": 0,
                "num_stationary_vehicle_points_kept": 0,
                "num_moving_vehicle_points_removed": 0,
                "removed_ratio": 0.0,
                "camera_reports": {},
            }

        category_masks = self._lidar_masks_for_frame(points_lidar_xyz, frame_id)

        always_dynamic_mask = category_masks["always_dynamic_mask"]
        vehicle_mask = category_masks["vehicle_mask"]
        visible_mask = category_masks["visible_mask"]

        filtering_cfg = self.cfg.get("filtering", {})
        motion_cfg = self.cfg.get("vehicle_filtering", {})

        keep_unprojected_points = filtering_cfg.get("keep_unprojected_points", True)
        keep_stationary_vehicles = bool(motion_cfg.get("keep_stationary_vehicles", False))

        if keep_stationary_vehicles:
            if T_world_vehicle is None:
                raise ValueError("T_world_vehicle is required when keep_stationary_vehicles=True.")

            if self._vehicle_motion_filter is None:
                raise RuntimeError("Vehicle motion filter was not prepared. Call prepare(...) before filtering frames.")

            vehicle_points = points_lidar_xyz[vehicle_mask]

            stationary_vehicle_local = self._vehicle_motion_filter.classify_stationary(points_lidar_xyz=vehicle_points, T_world_vehicle=T_world_vehicle)

            stationary_vehicle_mask = np.zeros(len(points_lidar_xyz), dtype=bool)

            vehicle_indices = np.flatnonzero(vehicle_mask)
            stationary_vehicle_mask[vehicle_indices] = stationary_vehicle_local

            moving_vehicle_mask = vehicle_mask & (~stationary_vehicle_mask)

            remove_mask = always_dynamic_mask | moving_vehicle_mask

        else:
            stationary_vehicle_mask = np.zeros(len(points_lidar_xyz), dtype=bool)
            moving_vehicle_mask = vehicle_mask.copy()
            remove_mask = always_dynamic_mask | vehicle_mask


        if not keep_unprojected_points:
            remove_mask = remove_mask | (~visible_mask)

        keep_mask = ~remove_mask

        filtered_points = points_lidar_xyz[keep_mask]

        num_input = int(len(points_lidar_xyz))
        num_removed = int(np.sum(remove_mask))
        num_static = int(len(filtered_points))
        removed_ratio = (
            float(num_removed / num_input)
            if num_input > 0
            else 0.0
        )

        report = {
            "method": self.method,
            "num_input_points": num_input,
            "num_removed_points": num_removed,
            "num_static_points": num_static,
            "num_visible_points": int(np.sum(visible_mask)),
            "num_always_dynamic_points": int(np.sum(always_dynamic_mask)),
            "num_vehicle_candidate_points": int(np.sum(vehicle_mask)),
            "num_stationary_vehicle_points_kept": int(np.sum(stationary_vehicle_mask & keep_mask)),
            "num_moving_vehicle_points_removed": int(np.sum(moving_vehicle_mask & remove_mask)),
            "removed_ratio": removed_ratio,
            "keep_unprojected_points": bool(keep_unprojected_points),
            "keep_stationary_vehicles": bool(keep_stationary_vehicles),
            "camera_reports": category_masks["camera_reports"],
        }

        return filtered_points, report

        


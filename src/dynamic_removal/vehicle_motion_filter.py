from collections import defaultdict
import numpy as np

from src.utils.transforms import compose_transforms, transform_points



# Convert 3D point coordinatess into voxel grid indices
def voxel_keys(points_world, voxel_size):
    points_world = np.asarray(points_world, dtype=np.float64)
    if len(points_world) == 0:
        return np.zeros((0,3), dtype=np.int64)

    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")
    return np.floor(points_world/voxel_size).astype(np.int64)



def keys_to_tuples(keys):
    if len(keys) == 0:
        return []

    return [tuple(k) for k in keys.tolist()]



class VehicleMotionFilter:
    """
    World-frame temporal consistency filter for vehicle points.

    It treats vehicle points as stationary if their world-space voxels are
    observed repeatedly across frames. Transient vehicle voxels are assumed to be
    moving vehicles or ghost trails.
    """

    def __init__(
        self,
        calibration,
        voxel_size_m=0.50,
        min_observations_for_stationary=3,
        min_frame_gap=2,
    ):
        self.calibration = calibration
        self.voxel_size_m = float(voxel_size_m)
        self.min_observations_for_stationary = int(min_observations_for_stationary)
        self.min_frame_gap = int(min_frame_gap)

        self._last_seen_frame_by_voxel = {}
        self._observation_count_by_voxel = defaultdict(int)
        self.stationary_voxels = set()
        self.prepared = False



    def add_vehicle_points(self, points_lidar_xyz, T_world_vehicle, frame_index):
        points_lidar_xyz = np.asarray(points_lidar_xyz, dtype=np.float64)
        if len(points_lidar_xyz) == 0:
            return

        T_world_lidar = compose_transforms(
            T_world_vehicle,
            self.calibration.T_vehicle_lidar,
        )

        points_world = transform_points(
            points_lidar_xyz,
            T_world_lidar,
        )

        keys = keys_to_tuples(
            voxel_keys(points_world, self.voxel_size_m)    
        )

        for key in keys:
            last_seen = self._last_seen_frame_by_voxel.get(key)
            if last_seen is not None:
                if int(frame_index) - int(last_seen) < self.min_frame_gap:
                    continue

            self._observation_count_by_voxel[key] += 1
            self._last_seen_frame_by_voxel[key] = int(frame_index)



    def finalize(self):
        self.stationary_voxels = {
            key
            for key, count in self._observation_count_by_voxel.items()
            if count >= self.min_observations_for_stationary
        }

        self.prepared = True



    def classify_stationary(self, points_lidar_xyz, T_world_vehicle):
        """
        Returns:
            N bool mask. True means vehicle point appears stationary.
        """
        if not self.prepared:
            raise RuntimeError(
                "VehicleMotionFilter must be finalized before use."
            )

        points_lidar_xyz = np.asarray(points_lidar_xyz, dtype=np.float64)

        if len(points_lidar_xyz) == 0:
            return np.zeros(0, dtype=bool)

        T_world_lidar = compose_transforms(
            T_world_vehicle,
            self.calibration.T_vehicle_lidar,
        )

        points_world = transform_points(
            points_lidar_xyz,
            T_world_lidar,
        )

        keys = keys_to_tuples(
            voxel_keys(points_world, self.voxel_size_m)
        )

        stationary = np.array(
            [key in self.stationary_voxels for key in keys],
            dtype=bool,
        )

        return stationary



    def summary(self):
        return {
            "voxel_size_m": self.voxel_size_m,
            "min_observations_for_stationary": self.min_observations_for_stationary,
            "min_frame_gap": self.min_frame_gap,
            "num_observed_vehicle_voxels": int(len(self._observation_count_by_voxel)),
            "num_stationary_vehicle_voxels": int(len(self.stationary_voxels)),
            "prepared": bool(self.prepared),
        }

from pathlib import Path
import csv
import json
import numpy as np
import cv2
import open3d as o3d




def extract_frame_id(path):
    return int(Path(path).stem)



def load_xyz_points_npy(path, min_range=None, max_range=None):
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Invalid point file shape {points.shape}: {path}")

    xyz = points[:, :3].astype(np.float64)
    valid = np.isfinite(xyz).all(axis=1)

    if min_range is not None or max_range is not None:
        distance = np.linalg.norm(xyz, axis=1)
        if min_range is not None:
            valid &= distance >= float(min_range)
        if max_range is not None:
            valid &= distance <= float(max_range)

    return xyz[valid]



def make_scene_dirs(scene_dir: str, camera_names=None):
    scene_dir = Path(scene_dir)

    dirs = {
        "rgb": scene_dir / "rgb",
        "semantic": scene_dir / "semantic",
        "lidar": scene_dir / "lidar",
        "calib": scene_dir / "calib",
        "calib_validation": scene_dir / "calib" / "validation",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    rgb_camera_dirs = {}
    semantic_camera_dirs = {}
    if camera_names is not None:
        for name in camera_names:
            rgb_camera_dirs[name] = dirs["rgb"] / name
            rgb_camera_dirs[name].mkdir(parents=True, exist_ok=True)

            semantic_camera_dirs[name] = dirs["semantic"] / name
            semantic_camera_dirs[name].mkdir(parents=True, exist_ok=True)

    dirs["camera_dirs"] = rgb_camera_dirs
    dirs["semantic_camera_dirs"] = semantic_camera_dirs

    return dirs



def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)



def pack_image(image):
    """
    Copy CARLA image data into a pickleable object for multiprocessing.
    Do this in the main process because CARLA image objects should not be
    passed directly to worker processes.
    """
    return {
        "width": int(image.width),
        "height": int(image.height),
        "raw": bytes(image.raw_data),
    }



def pack_lidar(lidar_data):
    """
    Copy CARLA lidar raw data into a pickleable object for multiprocessing.
    """
    return {
        "raw": bytes(lidar_data.raw_data),
    }



def save_rgb_packet(image_packet, path, jpeg_quality=90):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array = np.frombuffer(image_packet["raw"], dtype=np.uint8)
    array = array.reshape((image_packet["height"], image_packet["width"], 4))

    # CARLA image raw data is BGRA. OpenCV expects BGR.
    bgr = array[:, :, :3].copy()

    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        cv2.imwrite(
            str(path),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        )
    else:
        cv2.imwrite(str(path), bgr)



def save_semantic_label_packet(image_packet, path):
    """
    Save semantic segmentation as a single-channel label PNG.

    CARLA semantic raw image is BGRA-like. The semantic tag is commonly
    stored in the red channel, which is channel index 2 after BGRA loading.

    This is faster and more useful for ML than CityScapesPalette RGB images.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array = np.frombuffer(image_packet["raw"], dtype=np.uint8)
    array = array.reshape((image_packet["height"], image_packet["width"], 4))

    label = array[:, :, 2].copy()
    cv2.imwrite(str(path), label)



def save_lidar_packet(lidar_packet, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.frombuffer(lidar_packet["raw"], dtype=np.float32)
    points = points.reshape((-1, 4))

    np.save(str(path), points)



def save_cloud_ply(cloud, output_path):
    output_path = str(output_path)
    ok = o3d.io.write_point_cloud(output_path, cloud, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to save point cloud: {output_path}")



def save_pose_matrices_npz(path, poses):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    poses_array = np.stack([np.asarray(T, dtype=np.float64) for T in poses], axis=0)
    np.savez_compressed(path, poses=poses_array)



def load_pose_matrices_npz(path):
    path = Path(path)
    data = np.load(path)
    poses = data["poses"]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Invalid pose array shape {poses.shape}: {path}")
    return [poses[i] for i in range(poses.shape[0])]



def save_trajectory_csv(poses, output_path, align_to_origin=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.array([T[:3, 3] for T in poses], dtype=np.float64)
    if align_to_origin and len(xyz) > 0:
        xyz = xyz - xyz[0]
    with open(output_path, "w") as f:
        f.write("index,x,y,z\n")
        for i, p in enumerate(xyz):
            f.write(
                f"{i},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n"
            )
    print(f"Saved trajectory CSV to: {output_path}")



def write_csv_rows(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)



def frame_writer_worker(save_queue, jpeg_quality=90, rgb_ext=".jpg"):
    """
    Multiprocessing worker.

    Receives one full synchronized frame at a time and writes:
    - all RGB cameras
    - all semantic label cameras
    - LiDAR NPY

    CSV is intentionally not written here. It is buffered in the main process
    and written once at the end for speed and deterministic ordering.
    """
    while True:
        item = save_queue.get()

        try:
            if item is None:
                return

            frame = item["frame"]
            scene_dirs = item["scene_dirs"]
            camera_names = item["camera_names"]

            for name in camera_names:
                rgb_path = (scene_dirs["camera_dirs"][name] / f"{frame:06d}{rgb_ext}")
                save_rgb_packet(item["rgb"][name], rgb_path, jpeg_quality=jpeg_quality)
                
                semantic_path = (scene_dirs["semantic_camera_dirs"][name] / f"{frame:06d}.png")
                save_semantic_label_packet(item["semantic"][name], semantic_path)
                                
            lidar_path = scene_dirs["lidar"] / f"{frame:06d}.npy"
            save_lidar_packet(item["lidar"], lidar_path)

        except Exception as e:
            frame = (item.get("frame", "unknown") if isinstance(item, dict) else "unknown")
            print(f"[writer] Failed while saving frame {frame}: {e}", flush=True)
            raise

        finally:
            save_queue.task_done()
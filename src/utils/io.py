from pathlib import Path
import csv
import json
import numpy as np
import cv2


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

    camera_dirs = {}
    if camera_names is not None:
        for name in camera_names:
            camera_dirs[name] = dirs["rgb"] / name
            camera_dirs[name].mkdir(parents=True, exist_ok=True)

    dirs["camera_dirs"] = camera_dirs
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
    - semantic label PNG
    - lidar NPY

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
                rgb_path = scene_dirs["camera_dirs"][name] / f"{frame:06d}{rgb_ext}"
                save_rgb_packet(
                    item["rgb"][name],
                    rgb_path,
                    jpeg_quality=jpeg_quality,
                )

            semantic_path = scene_dirs["semantic"] / f"{frame:06d}.png"
            save_semantic_label_packet(
                item["semantic"],
                semantic_path,
            )

            lidar_path = scene_dirs["lidar"] / f"{frame:06d}.npy"
            save_lidar_packet(
                item["lidar"],
                lidar_path,
            )

        except Exception as e:
            frame = item.get("frame", "unknown") if isinstance(item, dict) else "unknown"
            print(f"[writer] Failed while saving frame {frame}: {e}", flush=True)
            raise

        finally:
            save_queue.task_done()
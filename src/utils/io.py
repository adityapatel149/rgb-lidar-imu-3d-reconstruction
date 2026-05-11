from pathlib import Path
import csv
import json
import numpy as np
import cv2
import carla



def make_scene_dirs(scene_dir: str, camera_names=None):
    scene_dir = Path(scene_dir)
    dirs = {
        "rgb": scene_dir/ "rgb",
        "semantic": scene_dir/ "semantic",   
        "lidar": scene_dir/ "lidar",
        "calib": scene_dir/ "calib",
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



def save_rgb(image, path):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    bgr = array[:,:,:3]
    cv2.imwrite(str(path), bgr)



def save_semantic(image, path):
    image.save_to_disk(str(path), carla.ColorConverter.CityScapesPalette)



def save_lidar(lidar_data, path):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    points = points.reshape((-1, 4))
    np.save(str(path), points)



def append_csv(path, row, header=None):
    path = Path(path)
    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if header is not None and not file_exists:
            writer.writerow(header)
        writer.writerow(row)



def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
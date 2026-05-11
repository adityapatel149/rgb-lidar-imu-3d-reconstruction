from pathlib import Path
import random
import numpy as np
import yaml
import carla

from src.data_collection.sync_mode import CarlaSyncMode

from src.data_collection.sensor_setup import (
    spawn_rgb_cameras, spawn_semantic, spawn_lidar, spawn_imu
)

from src.utils.io import (
    make_scene_dirs, save_rgb, save_lidar, save_semantic, append_csv, save_json,
)

from src.utils.calibration import (
    build_calibration, save_lidar_projection_debug
)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)



def spawn_vehicle(world, vehicle_blueprint_name, traffic_manager=None):
    blueprint = world.get_blueprint_library().find(vehicle_blueprint_name)
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(blueprint, spawn_point)
    if traffic_manager is not None:
        vehicle.set_autopilot(True, traffic_manager.get_port())
    else:
        vehicle.set_autopilot(True)
    return vehicle



def log_pose(scene_dir, frame, timestamp, vehicle):
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation

    append_csv(
        Path(scene_dir) / "poses.csv",
        [
            frame,
            timestamp,
            location.x,
            location.y,
            location.z,
            rotation.roll,
            rotation.pitch,
            rotation.yaw,
        ],
        header=[
            "frame",
            "timestamp",
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
        ],
    )



def log_imu(scene_dir, frame, timestamp, imu_data):
    acc = imu_data.accelerometer
    gyro = imu_data.gyroscope
    compass = imu_data.compass

    append_csv(
        Path(scene_dir) / "imu.csv",
        [
            frame,
            timestamp,
            acc.x,
            acc.y,
            acc.z,
            gyro.x,
            gyro.y,
            gyro.z,
            compass,
        ],
        header=[
            "frame",
            "timestamp",
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "compass",
        ],
    )



def collect_carla_scene(config_path, output_dir, num_frames=600, host="localhost", port=2000):
    cfg = load_config(config_path)
    
    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.load_world(cfg["world"]["town"])

    # 1. Initialize Traffic Manager
    tm = client.get_trafficmanager()
    # 2. Match TM synchrony with your world settings
    tm.set_synchronous_mode(True) 

    spectator = world.get_spectator()

    vehicle = None
    all_actors = []

    try:
        vehicle = spawn_vehicle(world, cfg["vehicle"]["blueprint"], traffic_manager=tm)

        camera_cfgs = cfg["sensors"]["cameras"]
        camera_names = [cam["name"] for cam in camera_cfgs]
        rgb_cameras = spawn_rgb_cameras(world, vehicle, camera_cfgs)
        semantic = spawn_semantic(world, vehicle, cfg["sensors"]["semantic"])
        lidar = spawn_lidar(world, vehicle, cfg["sensors"]["lidar"])
        imu = spawn_imu(world, vehicle, cfg["sensors"]["imu"])

        sensors = {}
        for name, actor in rgb_cameras.items():
            sensors[f"rgb_{name}"] = actor
        sensors["semantic"] = semantic
        sensors["lidar"] = lidar
        sensors["imu"] = imu

        all_actors = list(rgb_cameras.values()) + [semantic, lidar, imu]
        
        scene_dirs = make_scene_dirs(output_dir, camera_names=camera_names)
        
        # Save calibration data
        calibration = build_calibration(camera_cfgs=camera_cfgs, camera_actors=rgb_cameras, lidar_actor=lidar, imu_actor=imu)
        save_json(Path(output_dir) / "calib" / "calibration.json", calibration)

        with CarlaSyncMode(world, sensors, fixed_delta_seconds=cfg["world"]["fixed_delta_seconds"]) as sync:
            
            validation_saved = False

            for i in range(num_frames):
                data = sync.tick()

                # Attach spectator camera (10m back, 5m up)
                v_transform = vehicle.get_transform()     
                camera_pos = v_transform.location + v_transform.get_forward_vector() * -10 + carla.Location(z=5)
                camera_rot = v_transform.rotation
                camera_rot.pitch = -15  # Tilt the camera down
                spectator.set_transform(carla.Transform(camera_pos, camera_rot))   
                
                frame = data["lidar"].frame  # frame number
                timestamp = data["lidar"].timestamp

                for name in camera_names:
                    save_rgb(
                        data[f"rgb_{name}"],
                        scene_dirs["camera_dirs"][name] / f"{frame:06d}.png",
                    )

                save_semantic(
                    data["semantic"],
                    scene_dirs["semantic"] / f"{frame:06d}.png",
                )

                save_lidar(
                    data["lidar"],
                    scene_dirs["lidar"] / f"{frame:06d}.npy",
                )

                log_pose(output_dir, frame, timestamp, vehicle)
                log_imu(output_dir, frame, timestamp, data["imu"])
                
                if not validation_saved:
                    lidar_points_path = scene_dirs["lidar"] / f"{frame:06d}.npy"
                    lidar_points = np.load(lidar_points_path)

                    for cam_name in camera_names:
                        save_lidar_projection_debug(
                            rgb_image=data[f"rgb_{cam_name}"],
                            lidar_points=lidar_points,
                            camera_calib=calibration["cameras"][cam_name],
                            lidar_calib=calibration["lidar"],
                            output_path=scene_dirs["calib_validation"] / f"lidar_projection_{cam_name}.png",
                        )

                    validation_saved = True
                if i % 50 == 0:
                    print(f"Saved synchronized frame {i}/{num_frames}")

    finally:
        for actor in all_actors:
            actor.destroy()

        if vehicle is not None:
            vehicle.destroy()
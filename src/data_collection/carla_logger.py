import multiprocessing as mp
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
    make_scene_dirs,
    save_json,
    pack_image,
    pack_lidar,
    write_csv_rows,
    frame_writer_worker,
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



def collect_carla_scene(
    config_path,
    output_dir,
    num_frames=600,
    host="localhost",
    port=2000,
    num_writer_processes=4,
    save_rgb_ext=".jpg",
    jpeg_quality=90,
):
    cfg = load_config(config_path)

    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.load_world(cfg["world"]["town"])

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    spectator = world.get_spectator()

    vehicle = None
    all_actors = []

    save_queue = None
    writer_processes = []

    pose_rows = []
    imu_rows = []

    pose_header = [
        "frame",
        "timestamp",
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
    ]

    imu_header = [
        "frame",
        "timestamp",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "compass",
    ]

    try:
        vehicle = spawn_vehicle(
            world,
            cfg["vehicle"]["blueprint"],
            traffic_manager=tm,
        )

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

        # IMPORTANT:
        # build_calibration now needs vehicle because calibration must be
        # vehicle-relative, not world-relative.
        calibration = build_calibration(
            vehicle=vehicle,
            camera_cfgs=camera_cfgs,
            camera_actors=rgb_cameras,
            lidar_actor=lidar,
            imu_actor=imu,
        )

        save_json(Path(output_dir) / "calib" / "calibration.json", calibration)

        # Multiprocessing writer queue.
        # If this queue fills up, collection will block, which means the writers
        # or disk are still not keeping up.
        save_queue = mp.JoinableQueue(maxsize=32)

        for _ in range(num_writer_processes):
            p = mp.Process(
                target=frame_writer_worker,
                args=(save_queue,),
                kwargs={
                    "jpeg_quality": jpeg_quality,
                    "rgb_ext": save_rgb_ext,
                },
                daemon=True,
            )
            p.start()
            writer_processes.append(p)

        with CarlaSyncMode(
            world,
            sensors,
            fixed_delta_seconds=cfg["world"]["fixed_delta_seconds"],
        ) as sync:

            validation_saved = False

            for i in range(num_frames):
                data = sync.tick()

                # Optional spectator update. This is not a huge cost, but it is
                # unnecessary for maximum-speed logging. Leave commented unless
                # you need live visualization.
                #
                # v_transform = vehicle.get_transform()
                # camera_pos = (
                #     v_transform.location
                #     + v_transform.get_forward_vector() * -10
                #     + carla.Location(z=5)
                # )
                # camera_rot = v_transform.rotation
                # camera_rot.pitch = -15
                # spectator.set_transform(carla.Transform(camera_pos, camera_rot))

                frame = data["lidar"].frame
                timestamp = data["lidar"].timestamp

                # Copy CARLA sensor buffers immediately in the main process.
                # Do not send CARLA objects themselves to subprocesses.
                rgb_packets = {
                    name: pack_image(data[f"rgb_{name}"])
                    for name in camera_names
                }

                semantic_packet = pack_image(data["semantic"])
                lidar_packet = pack_lidar(data["lidar"])

                # Pose row.
                transform = vehicle.get_transform()
                location = transform.location
                rotation = transform.rotation

                pose_rows.append(
                    [
                        frame,
                        timestamp,
                        location.x,
                        location.y,
                        location.z,
                        rotation.roll,
                        rotation.pitch,
                        rotation.yaw,
                    ]
                )

                # IMU row.
                imu_data = data["imu"]
                acc = imu_data.accelerometer
                gyro = imu_data.gyroscope
                compass = imu_data.compass

                imu_rows.append(
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
                    ]
                )

                # Optional one-time calibration validation.
                # This still runs in the main process because it uses CARLA image
                # object format and only happens once.
                if not validation_saved:
                    lidar_points = np.frombuffer(
                        lidar_packet["raw"],
                        dtype=np.float32,
                    ).reshape((-1, 4))

                    for cam_name in camera_names:
                        save_lidar_projection_debug(
                            rgb_image=data[f"rgb_{cam_name}"],
                            lidar_points=lidar_points,
                            camera_calib=calibration["cameras"][cam_name],
                            lidar_calib=calibration["lidar"],
                            output_path=(
                                scene_dirs["calib_validation"]
                                / f"lidar_projection_{cam_name}.png"
                            ),
                        )

                    validation_saved = True

                # Send frame to async disk writers.
                save_queue.put(
                    {
                        "frame": frame,
                        "timestamp": timestamp,
                        "scene_dirs": scene_dirs,
                        "camera_names": camera_names,
                        "rgb": rgb_packets,
                        "semantic": semantic_packet,
                        "lidar": lidar_packet,
                    }
                )

                if i % 50 == 0:
                    try:
                        qsize = save_queue.qsize()
                    except NotImplementedError:
                        qsize = -1

                    print(
                        f"Captured synchronized frame {i}/{num_frames} "
                        f"| queue_size={qsize}"
                    )

        # Wait until all pending frames are written.
        save_queue.join()

        # Write CSV files once at the end.
        write_csv_rows(
            Path(output_dir) / "poses.csv",
            pose_header,
            pose_rows,
        )

        write_csv_rows(
            Path(output_dir) / "imu.csv",
            imu_header,
            imu_rows,
        )

        print(f"Saved outputs to: {output_dir}")

    finally:
        # Stop writer processes.
        if save_queue is not None:
            for _ in writer_processes:
                save_queue.put(None)

            save_queue.join()

            for p in writer_processes:
                p.join(timeout=10.0)

                if p.is_alive():
                    p.terminate()
                    p.join()

        for actor in all_actors:
            actor.destroy()

        if vehicle is not None:
            vehicle.destroy()
import carla



def make_transform(cfg):
    location = carla.Location(
        x=cfg.get("x", 0.0), 
        y=cfg.get("y", 0.0), 
        z=cfg.get("z", 0.0), 
    )
    rotation = carla.Rotation(
        pitch=cfg.get("pitch", 0.0), # y
        yaw=cfg.get("yaw", 0.0), # z
        roll=cfg.get("roll", 0.0), # x
    )
    return carla.Transform(location, rotation)



def spawn_rgb(world, vehicle, cfg):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(cfg["width"]))
    blueprint.set_attribute("image_size_y", str(cfg["height"]))
    blueprint.set_attribute("fov", str(cfg["fov"]))
    
    return world.spawn_actor(
        blueprint,
        make_transform(cfg),
        attach_to=vehicle,
    )



def spawn_rgb_cameras(world, vehicle, camera_cfgs):
    cameras = {}
    for cfg in camera_cfgs:
        name = cfg["name"]
        cameras[name] = spawn_rgb(world, vehicle, cfg)
    return cameras



def spawn_semantic(world, vehicle, cfg):
    blueprint = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
    blueprint.set_attribute("image_size_x", str(cfg["width"]))
    blueprint.set_attribute("image_size_y", str(cfg["height"]))
    blueprint.set_attribute("fov", str(cfg["fov"]))
    
    return world.spawn_actor(
        blueprint,
        make_transform(cfg),
        attach_to=vehicle,
    )



def spawn_lidar(world, vehicle, cfg):
    blueprint = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    blueprint.set_attribute("channels", str(cfg["channels"]))
    blueprint.set_attribute("range", str(cfg["range"]))
    blueprint.set_attribute("points_per_second", str(cfg["points_per_second"]))
    blueprint.set_attribute("rotation_frequency", str(cfg["rotation_frequency"]))
    blueprint.set_attribute("upper_fov", str(cfg["upper_fov"]))
    blueprint.set_attribute("lower_fov", str(cfg["lower_fov"]))
    
    return world.spawn_actor(
        blueprint,
        make_transform(cfg),
        attach_to=vehicle,
    )



def spawn_imu(world, vehicle, cfg):
    blueprint = world.get_blueprint_library().find("sensor.other.imu")
    
    return world.spawn_actor(
        blueprint,
        make_transform(cfg),
        attach_to=vehicle,
    )
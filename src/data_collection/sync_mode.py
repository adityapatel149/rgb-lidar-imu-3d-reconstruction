import queue



class CarlaSyncMode:

    def __init__(self, world, sensors, fixed_delta_seconds=0.05):
        self.world = world
        self.sensors = sensors
        self.fixed_delta_seconds = fixed_delta_seconds
        self.frame = None
        self.queues = {}



    # Runs automatically when entering: "with CarlaSyncMode(...) as sync:"
    def __enter__(self):
        self.settings = self.world.get_settings()         # Store settings to reset them before exiting

        new_settings = self.world.get_settings()
        new_settings.synchronous_mode = True        # World only advances when world.tick() is called manually
        new_settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(new_settings)

        # Each sensor gets its own queue
        self.queues = {}
        for name, sensor in self.sensors.items():
            q = queue.Queue()
            sensor.listen(q.put)    # CARLA sensors use callbacks, so data pushed into individual sensor queues automatically
            self.queues[name] = q

        return self



    def tick(self, timeout=5.0):
        self.frame = self.world.tick()  # returns current frame number

        data = {}
        for name, q in self.queues.items():   # for each sensor queue
            item = q.get(timeout=timeout)   # wait for data. Blocks until data arrives. Timeout prevents infinite hanging.
            while item.frame != self.frame: # discard old frames or anything not matching current world frame
                item = q.get(timeout=timeout)
            data[name] = item   # Save synchronized item from all sensors

        return data



    # Called automatically when leaving the  "with" block
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.world.apply_settings(self.settings)
        for sensor in self.sensors.values():
            sensor.stop()
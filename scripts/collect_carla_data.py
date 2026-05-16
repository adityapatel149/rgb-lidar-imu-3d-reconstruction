import argparse

from src.data_collection.carla_logger import collect_carla_scene



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scene.yaml")
    parser.add_argument("--output", default="data/carla/raw/scene_001")
    parser.add_argument("--num_frames", type=int, default=600)
    args = parser.parse_args()

    collect_carla_scene(
        config_path=args.config,
        output_dir=args.output,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
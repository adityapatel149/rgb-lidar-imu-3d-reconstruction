import argparse
import os
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
import shutil

# Import core pipeline elements to ensure identical results to the mapping backend
from src.calibration.calibration_loader import Calibration
from src.mapping.colorize_pointcloud import colorize_lidar_points
from src.mapping.voxel_map import voxel_downsample_cloud




fontsize = 0.65
fontwidth = 1   # has to be Int






def render_global_bev(
    pcd,
    current_pose=None,
    width=640,
    height=480,
    padding_m=10.0
):
    """
    Global accumulated BEV renderer.

    Mimics Open3D accumulation-map behavior:
      - Entire accumulated map remains visible
      - Scale expands automatically as map grows
      - Highest Z wins per pixel
      - Optional vehicle marker
    """

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    if len(pts) == 0:
        return canvas

    xmin = pts[:, 0].min() - padding_m
    xmax = pts[:, 0].max() + padding_m

    ymin = pts[:, 1].min() - padding_m
    ymax = pts[:, 1].max() + padding_m

    dx = xmax - xmin
    dy = ymax - ymin

    extent = max(dx, dy)

    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5

    xmin = cx - extent * 0.5
    xmax = cx + extent * 0.5

    ymin = cy - extent * 0.5
    ymax = cy + extent * 0.5

    px = (
        (ymax - pts[:, 1])
        / (ymax - ymin)
        * (width - 1)
    ).astype(np.int32)

    py = (
        (xmax - pts[:, 0])
        / (xmax - xmin)
        * (height - 1)
    ).astype(np.int32)

    z_buffer = np.full(
        (height, width),
        -np.inf,
        dtype=np.float32
    )

    rgb = (cols * 255).astype(np.uint8)

    for x, y, z, c in zip(px, py, pts[:, 2], rgb):

        if (
            x < 0 or x >= width or
            y < 0 or y >= height
        ):
            continue

        if z > z_buffer[y, x]:
            z_buffer[y, x] = z
            canvas[y, x] = c[::-1]

    canvas = cv2.dilate(
        canvas,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    if current_pose is not None:

        vx = current_pose[0, 3]
        vy = current_pose[1, 3]

        vehicle_px = int(
            (ymax - vy)
            / (ymax - ymin)
            * (width - 1)
        )

        vehicle_py = int(
            (xmax - vx)
            / (xmax - xmin)
            * (height - 1)
        )

        cv2.circle(
            canvas,
            (vehicle_px, vehicle_py),
            5,
            (0, 255, 255),
            -1
        )

        cv2.circle(
            canvas,
            (vehicle_px, vehicle_py),
            8,
            (0, 0, 0),
            2
        )

    return canvas



def render_following_bev(
    pcd,
    current_pose,
    width=640,
    height=480,
    window_size_m=80.0
):
    """
    Vehicle-following BEV renderer.

    Highest Z wins per pixel.
    Camera follows current_pose exactly like Open3D lookat.
    """

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    if len(pts) == 0:
        return canvas

    center_x = current_pose[0, 3]
    center_y = current_pose[1, 3]

    half_window = window_size_m * 0.5

    xmin = center_x - half_window
    xmax = center_x + half_window

    ymin = center_y - half_window
    ymax = center_y + half_window

    mask = (
        (pts[:,0] >= xmin) &
        (pts[:,0] <= xmax) &
        (pts[:,1] >= ymin) &
        (pts[:,1] <= ymax)
    )

    pts = pts[mask]
    cols = cols[mask]

    if len(pts) == 0:
        return canvas

    px = (
        (ymax - pts[:,1])
        / (ymax - ymin)
        * (width - 1)
    ).astype(np.int32)

    py = (
        (xmax - pts[:,0])
        / (xmax - xmin)
        * (height - 1)
    ).astype(np.int32)

    z_buffer = np.full(
        (height, width),
        -np.inf,
        dtype=np.float32
    )

    rgb = (cols * 255).astype(np.uint8)

    for x, y, z, c in zip(px, py, pts[:,2], rgb):

        if (
            x < 0 or x >= width or
            y < 0 or y >= height
        ):
            continue

        if z > z_buffer[y, x]:
            z_buffer[y, x] = z
            canvas[y, x] = c[::-1]   # RGB->BGR

    canvas = cv2.dilate(
        canvas,
        np.ones((2,2), np.uint8),
        iterations=1
    )

    vehicle_px = width // 2
    vehicle_py = height // 2
    meters_per_pixel = window_size_m / width
    vehicle_radius_m = 1.0   # 1 m radius
    vehicle_radius_px = max(5, int(vehicle_radius_m / meters_per_pixel))

    cv2.circle(
        canvas,
        (vehicle_px, vehicle_py),
        vehicle_radius_px,
        (0,255,255),
        -1
    )

    cv2.circle(
        canvas,
        (vehicle_px, vehicle_py),
        vehicle_radius_px + 3,
        (0,0,0),
        2
    )

    return canvas




def load_fused_poses(pose_path, flip_matrix=None):
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    data = np.load(pose_path)
    key = 'poses' if 'poses' in data.files else data.files[0]
    poses = data[key]
    
    if flip_matrix is not None:
        flipped_poses = []
        inv_flip = np.linalg.inv(flip_matrix)
        for pose in poses:
            flipped_pose = flip_matrix @ pose @ inv_flip
            flipped_poses.append(flipped_pose)
        return np.array(flipped_poses)
        
    return poses

def draw_trajectory_widget(width, height, fused_poses, imu_poses, current_idx):
    canvas = np.zeros((height, width, 3), dtype=np.uint8) + 20
    fused_xy = fused_poses[:, :2, 3]
    imu_xy = imu_poses[:, :2, 3]
    
    all_x = np.concatenate([fused_xy[:, 0], imu_xy[:, 0]])
    all_y = np.concatenate([fused_xy[:, 1], imu_xy[:, 1]])
    min_x, max_x = all_x.min() - 5, all_x.max() + 5
    min_y, max_y = all_y.min() - 5, all_y.max() + 5
    
    def to_pixels(pt):
        px = int((max_y - pt[1]) / (max_y - min_y) * (width - 40)) + 20
        py = int((pt[0] - min_x) / (max_x - min_x) * (height - 60)) + 40
        py = height - py
        return (px, py)

    for idx in range(1, current_idx + 1):
        p1_f = to_pixels(fused_xy[idx-1])
        p2_f = to_pixels(fused_xy[idx])
        cv2.line(canvas, p1_f, p2_f, (0, 215, 0), 2, cv2.LINE_AA)
        
        p1_i = to_pixels(imu_xy[idx-1])
        p2_i = to_pixels(imu_xy[idx])
        cv2.line(canvas, p1_i, p2_i, (0, 0, 225), 1, cv2.LINE_AA)

    cx, cy = to_pixels(fused_xy[current_idx])
    cv2.circle(canvas, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 7, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(canvas, "LiDAR-Inertial Odometry and Trajectory Estimation", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255, 255, 255), fontwidth, cv2.LINE_AA)
    cv2.line(canvas, (15, 45), (35, 45), (0, 215, 0), 2)
    cv2.putText(canvas, "Tightly Fused LIO (Ours)", (45, 50), cv2.FONT_HERSHEY_SIMPLEX, fontsize*0.75, (255, 255, 255), fontwidth, cv2.LINE_AA)
    cv2.line(canvas, (15, 65), (35, 65), (0, 0, 225), 2)
    cv2.putText(canvas, "Raw IMU Data (Drifting)", (45, 70), cv2.FONT_HERSHEY_SIMPLEX, fontsize*0.75, (255, 255, 255), fontwidth, cv2.LINE_AA)
    
    return canvas

def build_camera_grid_by_id(scene_dir, frame_id, camera_names, extension=".jpg"):
    frames = []

    for cam_name in camera_names:
        img_path = Path(scene_dir) / "rgb" / cam_name / f"{int(frame_id):06d}{extension}"

        img = cv2.imread(str(img_path))

        if img is None:
            alt_ext = ".png" if extension == ".jpg" else ".jpg"
            img_path = Path(scene_dir) / "rgb" / cam_name / f"{int(frame_id):06d}{alt_ext}"
            img = cv2.imread(str(img_path))

        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        # For 6 cameras: 3 columns x 2 rows -> final 640x480
        if len(camera_names) == 6:
            img_res = cv2.resize(img, (213, 240))
        else:
            img_res = cv2.resize(img, (320, 240))

        cv2.putText(
            img_res,
            cam_name,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontsize * 0.75,
            (255, 255, 255),
            fontwidth,
            cv2.LINE_AA,
        )

        frames.append(img_res)

    if len(frames) == 6:
        top_row = np.hstack(frames[:3])
        bottom_row = np.hstack(frames[3:6])
        grid = np.vstack((top_row, bottom_row))
        grid = cv2.resize(grid, (640, 480))
        return grid

    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    return np.vstack((top_row, bottom_row))

def patch_colorize_runtime_warnings():
    import src.mapping.colorize_pointcloud as cp
    
    def safe_sample_rgb_nearest(image_rgb, uv):
        h, w = image_rgb.shape[:2]
        finite_mask = np.isfinite(uv).all(axis=1)
        
        u = np.zeros(uv.shape[0], dtype=np.int64)
        v = np.zeros(uv.shape[0], dtype=np.int64)
        
        u[finite_mask] = np.rint(uv[finite_mask, 0]).astype(np.int64)
        v[finite_mask] = np.rint(uv[finite_mask, 1]).astype(np.int64)
        
        valid = (
            finite_mask
            & (u >= 0)
            & (u < w)
            & (v >= 0)
            & (v < h)
        )
        colors = np.zeros((uv.shape[0], 3), dtype=np.float64)
        colors[valid] = image_rgb[v[valid], u[valid]].astype(np.float64) / 255.0
        return colors, valid

    cp.sample_rgb_nearest = safe_sample_rgb_nearest

def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Progressive Dashboard and Reconstruction Mapping Pipeline")
    parser.add_argument("--scene", default="scene_001", help="Target identifier (e.g. scene_001)") 
    
    parser.add_argument("--carla_raw_dir", default=None, help="Override root dataset directory path")
    parser.add_argument("--lidar_dir", default=None, help="Override path to raw .npy LiDAR scan directory") 
    parser.add_argument("--fused_poses", default=None, help="Override path to tightly_fused_lio_poses.npz")
    parser.add_argument("--imu_poses", default=None, help="Override path to imu_only_poses.npz")
    parser.add_argument("--output_video", default=None, help="Override output path for dashboard video")
    parser.add_argument("--output_map_video", default=None, help="Override output path for dual reconstruction map video")
    
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    patch_colorize_runtime_warnings()

    scene_id = args.scene
    carla_raw_dir = Path(args.carla_raw_dir) if args.carla_raw_dir else Path(f"data/carla/raw/{scene_id}")
    lidar_dir = Path(args.lidar_dir) if args.lidar_dir else carla_raw_dir / "lidar"
    fused_poses_path = Path(args.fused_poses) if args.fused_poses else Path(f"outputs/trajectories/{scene_id}/tightly_fused_lio_poses.npz")
    imu_poses_path = Path(args.imu_poses) if args.imu_poses else Path(f"outputs/trajectories/{scene_id}/imu_only_poses.npz")
    output_video_path = Path(args.output_video) if args.output_video else Path(f"outputs/progressive_dashboard_{scene_id}.mp4")
    output_map_video_path = Path(args.output_map_video) if args.output_map_video else Path(f"outputs/colored_map_progression_{scene_id}.mp4")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_map_video_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("outputs/temp_panels")
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[-] Initializing workspace assets for scene target: {scene_id}")
    
    is_nuscenes = (carla_raw_dir / "rgb" / "CAM_FRONT").exists()
    if is_nuscenes:
        flip_matrix = np.eye(4, dtype=np.float64)
    else:
        flip_matrix = np.array([
            [1,  0, 0, 0],
            [0, -1, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ], dtype=np.float64)
    
    calib_path = carla_raw_dir / "calib" / "calibration.json"
    if not calib_path.exists():
        print(f"[!] Calibration file missing at destination path: {calib_path}")
        sys.exit(1)
    calibration = Calibration(calib_path)

    try:
        fused_poses = load_fused_poses(str(fused_poses_path), flip_matrix=flip_matrix)
        imu_poses = load_fused_poses(str(imu_poses_path), flip_matrix=flip_matrix)
    except Exception as e:
        print(f"[!] Error loading trajectory asset files: {e}")
        sys.exit(1)
    
    #camera_names = ["front", "left", "rear", "right"]
    #camera_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    if (carla_raw_dir / "rgb" / "CAM_FRONT").exists():
        camera_names = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]
    else:
        camera_names = ["front", "left", "rear", "right"]
    if not lidar_dir.exists():
        print(f"[!] Error: LiDAR directory path does not exist: {lidar_dir}")
        sys.exit(1)
        
    lidar_files = sorted([f for f in lidar_dir.iterdir() if f.suffix.lower() == '.npy'])

    num_frames = min(len(fused_poses), len(imu_poses), len(lidar_files))
    if num_frames == 0:
        print("[!] Sync failed. Zero files matched across target directories.")
        sys.exit(1)

    print(f"[+] System synchronized: Processing {num_frames} frames.")

    # Distinct tracking containers for map projections
    progressive_accumulated_rgb_map = o3d.geometry.PointCloud()
    progressive_accumulated_gradient_map = o3d.geometry.PointCloud()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Render Window Engine", width=640, height=480, visible=False)
    render_opt = vis.get_render_option()
    render_opt.point_size = 0.5
    render_opt.background_color = np.array([0.05, 0.05, 0.05])
    
    video_writer_dashboard = None
    video_writer_map = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print("[-] Compiling progressive map videos...")
    for i in range(0, num_frames, args.stride):
        current_pose = fused_poses[i]
        frame_path = lidar_files[i]
        
        frame_id = int(frame_path.stem.split("_")[-1] if "_" in frame_path.stem else frame_path.stem)        
        
        try:
            raw_data = np.load(frame_path)
            points_lidar_xyz = raw_data[:, :3]
            
            # Save pristine raw data to bypass camera field-of-view restrictions for tracking
            pristine_raw_xyz = points_lidar_xyz.copy()
            
            # Unpack color results safely from the tuple return format
            colored_xyz, colors_rgb, _, _ = colorize_lidar_points(
                points_lidar_xyz=points_lidar_xyz,
                frame_id=frame_id,
                scene_dir=str(carla_raw_dir),
                calibration=calibration,
                camera_names=camera_names,
                color_mode="best"
            )
            
            current_frame_rgb_pcd = o3d.geometry.PointCloud()
            current_frame_rgb_pcd.points = o3d.utility.Vector3dVector(colored_xyz)
            current_frame_rgb_pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
            current_frame_rgb_pcd.transform(calibration.T_vehicle_lidar)
            
        except Exception as e:
            print(f"\n[!] Failed to match or color frame {frame_path.name}: {e}. Skipping.")
            continue

        # Transform RGB cloud into world coordinate frames
        current_frame_rgb_pcd.transform(flip_matrix)
        current_frame_rgb_pcd.transform(current_pose)
        
        # -------------------------------------------------------------
        # DISTANCE GRADIENT COLOR MATRIX CALCULATION (0 - 255 Normalized)
        # -------------------------------------------------------------
        raw_distances = np.linalg.norm(pristine_raw_xyz, axis=1)
        min_d, max_d = raw_distances.min(), raw_distances.max()
        
        if max_d > min_d:
            norm_distances = ((raw_distances - min_d) / (max_d - min_d) * 255.0).astype(np.uint8)
        else:
            norm_distances = np.zeros(raw_distances.shape[0], dtype=np.uint8)
            
        gradient_colors = cv2.applyColorMap(norm_distances, cv2.COLORMAP_JET).squeeze()
        gradient_colors_rgb = gradient_colors[:, [2, 1, 0]].astype(np.float64) / 255.0
        
        raw_gradient_pcd = o3d.geometry.PointCloud()
        raw_gradient_pcd.points = o3d.utility.Vector3dVector(pristine_raw_xyz)
        raw_gradient_pcd.colors = o3d.utility.Vector3dVector(gradient_colors_rgb)
        
        raw_gradient_pcd.transform(calibration.T_vehicle_lidar)
        raw_gradient_pcd.transform(flip_matrix)
        raw_gradient_pcd.transform(current_pose)

        # -------------------------------------------------------------
        # SNAPSHOT-ISOLATION DOWN-SAMPLING
        # -------------------------------------------------------------
        #current_frame_rgb_pcd = voxel_downsample_cloud(current_frame_rgb_pcd, voxel_size=0.15)
        #raw_gradient_pcd = voxel_downsample_cloud(raw_gradient_pcd, voxel_size=0.15)
        
        local_view_rgb_cloud = o3d.geometry.PointCloud(current_frame_rgb_pcd)
        
        # -------------------------------------------------------------
        # FORCE DEPTH RETENTION VIA STRICT NUMPY VSTACK ARRAY STRATEGY
        # -------------------------------------------------------------
        # Stacking as raw arrays completely disables Open3D's voxel bin collision replacements,
        # forcing different Z coordinates at the same X/Y cell to coexist safely.
        if len(progressive_accumulated_rgb_map.points) == 0:
            progressive_accumulated_rgb_map.points = current_frame_rgb_pcd.points
            progressive_accumulated_rgb_map.colors = current_frame_rgb_pcd.colors
        else:
            combined_rgb_pts = np.vstack([np.asarray(progressive_accumulated_rgb_map.points), np.asarray(current_frame_rgb_pcd.points)])
            combined_rgb_cls = np.vstack([np.asarray(progressive_accumulated_rgb_map.colors), np.asarray(current_frame_rgb_pcd.colors)])
            progressive_accumulated_rgb_map.points = o3d.utility.Vector3dVector(combined_rgb_pts)
            progressive_accumulated_rgb_map.colors = o3d.utility.Vector3dVector(combined_rgb_cls)

        if len(progressive_accumulated_gradient_map.points) == 0:
            progressive_accumulated_gradient_map.points = raw_gradient_pcd.points
            progressive_accumulated_gradient_map.colors = raw_gradient_pcd.colors
        else:
            combined_grad_pts = np.vstack([np.asarray(progressive_accumulated_gradient_map.points), np.asarray(raw_gradient_pcd.points)])
            combined_grad_cls = np.vstack([np.asarray(progressive_accumulated_gradient_map.colors), np.asarray(raw_gradient_pcd.colors)])
            progressive_accumulated_gradient_map.points = o3d.utility.Vector3dVector(combined_grad_pts)
            progressive_accumulated_gradient_map.colors = o3d.utility.Vector3dVector(combined_grad_cls)

        # -------------------------------------------------------------
        # PANEL 3 RENDER: RAW UNCOLORED LIDAR SCAN WITH DISTANCE GRADIENT
        # -------------------------------------------------------------
        vis.clear_geometries()
        vis.add_geometry(raw_gradient_pcd)
        view_ctl = vis.get_view_control()
        view_ctl.set_lookat(current_pose[0:3, 3])
        view_ctl.set_front(np.array([0.0, 0.0, -1.0]))  
        view_ctl.set_up(np.array([1.0, 0.0, 0.0]))
        view_ctl.set_zoom(0.25)
        
        vis.update_geometry(raw_gradient_pcd)
        vis.poll_events()
        vis.update_renderer()
        
        panel3_frame_path = temp_dir / f"raw_gradient_{i:04d}.png"
        vis.capture_screen_image(str(panel3_frame_path), do_render=True)
        img_panel3 = cv2.imread(str(panel3_frame_path))
        img_panel3 = cv2.flip(img_panel3, 1)

        # -------------------------------------------------------------
        # PANEL 4 RENDER: ACCUMULATED DISTANCE GRADIENT MAP (TOP-DOWN)
        # -------------------------------------------------------------
        # vis.clear_geometries()
        # vis.add_geometry(progressive_accumulated_gradient_map)
        # view_ctl = vis.get_view_control()
        # view_ctl.set_lookat(current_pose[0:3, 3]) 
        # view_ctl.set_front(np.array([0.0, 0.0, -1.0])) 
        # view_ctl.set_up(np.array([1.0, 0.0, 0.0])) 
        # view_ctl.set_zoom(0.55)
        
        # vis.update_geometry(progressive_accumulated_gradient_map)
        # vis.poll_events()
        # vis.update_renderer()
        
        # panel4_frame_path = temp_dir / f"accumulated_gradient_{i:04d}.png"
        # vis.capture_screen_image(str(panel4_frame_path), do_render=True)
        # img_panel4 = cv2.imread(str(panel4_frame_path))
        # img_panel4 = cv2.flip(img_panel4, 1)
        vis.clear_geometries()
        vis.add_geometry(local_view_rgb_cloud)
        view_ctl = vis.get_view_control()
        view_ctl.set_lookat(current_pose[0:3, 3])
        view_ctl.set_front(np.array([0.0, 0.0, -1.0])) 
        view_ctl.set_up(np.array([1.0, 0.0, 0.0]))
        view_ctl.set_zoom(0.25)
        
        vis.update_geometry(local_view_rgb_cloud)
        vis.poll_events()
        vis.update_renderer()
        
        panel4_frame_path = temp_dir / f"panel4_frame_path_{i:04d}.png"
        vis.capture_screen_image(str(panel4_frame_path), do_render=True)
        img_panel4 = cv2.imread(str(panel4_frame_path))
        img_panel4 = cv2.flip(img_panel4, 1)
        # -------------------------------------------------------------
        # VIDEO 2 RENDER: LEFT HALF (CURRENT RGB) & RIGHT HALF (GLOBAL RGB)
        # -------------------------------------------------------------
        # vis.clear_geometries()
        # vis.add_geometry(local_view_rgb_cloud)
        # view_ctl = vis.get_view_control()
        # view_ctl.set_lookat(current_pose[0:3, 3])
        # view_ctl.set_front(np.array([0.0, 0.0, -1.0])) 
        # view_ctl.set_up(np.array([1.0, 0.0, 0.0]))
        # view_ctl.set_zoom(0.25)
        
        # vis.update_geometry(local_view_rgb_cloud)
        # vis.poll_events()
        # vis.update_renderer()
        
        vid2_left_path = temp_dir / f"v2_left_{i:04d}.png"
        img_v2_left = render_following_bev(
            progressive_accumulated_rgb_map,
            current_pose,
            width=640,
            height=480,
            window_size_m=50.0
        )
        # vis.capture_screen_image(str(vid2_left_path), do_render=True)
        # img_v2_left = cv2.imread(str(vid2_left_path))
        # img_v2_left = cv2.flip(img_v2_left, 1)
        
        vid2_right_path = temp_dir / f"v2_right_{i:04d}.png"
        # img_v2_right = render_following_bev(
        #     progressive_accumulated_rgb_map,
        #     current_pose,
        #     width=640,
        #     height=480,
        #     window_size_m=200.0
        # )
        img_v2_right = render_global_bev(
            progressive_accumulated_rgb_map,
            current_pose,
            width=640,
            height=480,
        )
        # -------------------------------------------------------------
        # STITCH AND WRITE VIDEO 1 (Progressive Dashboard)
        # -------------------------------------------------------------
        img_camera_grid = build_camera_grid_by_id(carla_raw_dir, frame_id, camera_names)
        img_telemetry = draw_trajectory_widget(640, 480, fused_poses, imu_poses, i)

        # cv2.putText(img_camera_grid, "Multi-Camera Surround View", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        #cv2.putText(img_telemetry, "LIO and Trajectory Estimation", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(img_panel3, "Raw LiDAR Scan", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1, cv2.LINE_AA)
        # cv2.putText(img_panel4, f"Camera-Colored LiDAR", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img_panel3, "Raw LiDAR Scan", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255, 255, 255), fontwidth, cv2.LINE_AA)
        cv2.putText(img_panel4, f"Camera-Colored LiDAR", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255, 255, 255), fontwidth, cv2.LINE_AA)

        top_row = np.hstack((img_camera_grid, img_telemetry))
        bottom_row = np.hstack((img_panel3, img_panel4))
        dashboard_canvas = np.vstack((top_row, bottom_row))

        cv2.line(dashboard_canvas, (640, 0), (640, 960), (50, 50, 50), 3)
        cv2.line(dashboard_canvas, (0, 480), (1280, 480), (50, 50, 50), 3)

        if video_writer_dashboard is None:
            h, w, _ = dashboard_canvas.shape
            video_writer_dashboard = cv2.VideoWriter(str(output_video_path), fourcc, args.fps, (w, h))
        video_writer_dashboard.write(dashboard_canvas)
        
        # -------------------------------------------------------------
        # STITCH AND WRITE VIDEO 2 (Side-By-Side RGB Pair Progression)
        # -------------------------------------------------------------
        # cv2.putText(img_v2_left, "Local Reconstruction View", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img_v2_right, f"Global Reconstruction Map ({len(progressive_accumulated_rgb_map.points):,} pts)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_v2_left, "Local Reconstruction View", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255, 255, 255), fontwidth, cv2.LINE_AA)
        cv2.putText(img_v2_right, f"Global Reconstruction Map ({len(progressive_accumulated_rgb_map.points):,} pts)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255, 255, 255), fontwidth, cv2.LINE_AA)
        
        dual_map_canvas = np.hstack((img_v2_left, img_v2_right))
        cv2.line(dual_map_canvas, (640, 0), (640, 480), (100, 100, 100), 2) 

        if video_writer_map is None:
            hm, wm, _ = dual_map_canvas.shape
            video_writer_map = cv2.VideoWriter(str(output_map_video_path), fourcc, args.fps, (wm, hm))
        video_writer_map.write(dual_map_canvas)
        
        # Clear temporary screen captures from frame sequence step
        for path in [panel3_frame_path, panel4_frame_path, vid2_left_path, vid2_right_path]:
            if os.path.exists(path): 
                os.remove(path)

        print(f"\rPipeline processing reconstruction video frames: {i}/{num_frames}...", end="")

    print("\n[-] Cleaning up temporary session assets...")
    vis.destroy_window()
    if video_writer_dashboard is not None: video_writer_dashboard.release()
    if video_writer_map is not None: video_writer_map.release()

    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[!] Warning: Could not auto-delete temp directory: {e}")

    print(f"[========] Completed! Dashboard video saved to: {output_video_path}")
    print(f"[========] Completed! Dual Map view accumulation reconstruction video saved to: {output_map_video_path}")

if __name__ == "__main__":
    main()
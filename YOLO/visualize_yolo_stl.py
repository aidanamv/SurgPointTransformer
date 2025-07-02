import argparse
import os
import re
import cv2
import numpy as np
from PIL import Image
import open3d as o3d
from ultralytics import YOLO


def extract_camera_intrinsics_from_file(conf_file_path, camera_type='LEFT_CAM_FHD'):
    with open(conf_file_path, 'r') as file:
        data = file.read()

    pattern = re.compile(rf'\[{camera_type}\](.*?)\n(?:\[|\Z)', re.DOTALL)
    match = pattern.search(data)

    if not match:
        raise ValueError(f"Camera type '{camera_type}' not found in file.")

    section = match.group(1)
    intrinsics = {}
    for line in section.strip().split('\n'):
        key, value = line.split('=')
        intrinsics[key.strip()] = float(value.strip())
    return intrinsics


def run_inference(args):
    model = YOLO(args.model)
    files = os.listdir(args.rgb_dir)
    intrinsics_left = extract_camera_intrinsics_from_file(args.calib_dir)

    for file in files:
        print(f"📷 Processing {file}")
        _, specimen_str, _, recording_str, _, cam_str, _, frame_str = file[:-4].split('_')
        specimen = int(specimen_str)
        recording = int(recording_str)
        cam = int(cam_str)
        frame = int(frame_str)

        if specimen != args.specimen:
            continue  # Skip files not matching the specified specimen

        rgb_path = os.path.join(args.rgb_dir, file)
        depth_path = os.path.join(args.depth_root, f"Specimen_{specimen}", f"recording_{recording}", f"cam_{cam}", f"frame_{frame}", "depth.png")

        results = model([rgb_path])
        color_image = np.asarray(Image.open(rgb_path))
        depth_image = np.asarray(Image.open(depth_path))

        color_pallete = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
        stl_base = os.path.join(args.stl_root, f"Specimen_{specimen}", f"recording_{recording}", f"cam_{cam}", f"frame_{frame}")
        vis = []

        for el, box in enumerate(results[0].boxes):
            cls = int(box.cls.item())
            xyxy = box.xyxy.cpu().numpy()[0]

            stl_path = os.path.join(stl_base, f"transformed_vertebra{cls + 1}.stl")
            stl = o3d.io.read_triangle_mesh(stl_path)
            stl.compute_vertex_normals()

            # Mask image outside bounding box
            mask = np.ones(color_image.shape[:2], dtype=np.uint8) * 255
            cv2.rectangle(mask, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), 0, -1)
            mask_inv = cv2.bitwise_not(mask)
            white_img = np.ones_like(color_image) * 255
            masked_rgb = cv2.add(cv2.bitwise_and(color_image, color_image, mask=mask_inv),
                                 cv2.bitwise_and(white_img, white_img, mask=mask))

            h, w = depth_image.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            x = (u - intrinsics_left["cx"]) * depth_image / intrinsics_left["fx"]
            y = (v - intrinsics_left["cy"]) * depth_image / intrinsics_left["fy"]
            z = depth_image

            pc = np.stack((x, y, z, masked_rgb[:, :, 0] / 255, masked_rgb[:, :, 1] / 255, masked_rgb[:, :, 2] / 255), axis=-1).reshape(-1, 6)
            pc = np.nan_to_num(pc)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6])

            colors = np.asarray(pcd.colors)
            points = np.asarray(pcd.points)
            non_white = np.all(colors < 0.95, axis=1)

            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(points[non_white])
            filtered_pcd.colors = o3d.utility.Vector3dVector(colors[non_white])

            bbox = stl.get_axis_aligned_bounding_box()
            indices = bbox.get_point_indices_within_bounding_box(filtered_pcd.points)
            segment = filtered_pcd.select_by_index(indices)
            segment.paint_uniform_color(color_pallete[el])
            vis.append(segment)
            vis.append(stl)

        o3d.visualization.draw_geometries(vis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO + STL prediction for one specimen")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 model checkpoint")
    parser.add_argument("--specimen", type=int, required=True, help="Specimen number to process (e.g., 2)")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Path to directory with RGB images")
    parser.add_argument("--depth_root", type=str, required=True, help="Root path to depth maps")
    parser.add_argument("--stl_root", type=str, required=True, help="Root path to STL meshes")
    parser.add_argument("--calib_dir", type=str, required=True, help="Full path to .conf calibration file")

    args = parser.parse_args()
    run_inference(args)

import os
import matplotlib
matplotlib.use("TkAgg")
import open3d as o3d
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import shutil
import pyzed.sl as sl
import re

def extract_camera_intrinsics_from_file(conf_file_path, camera_type='LEFT_CAM_FHD'):
    # Read the contents of the .conf file
    with open(conf_file_path, 'r') as file:
        data = file.read()

    # Define a regular expression to find the camera parameters
    pattern = re.compile(rf'\[{camera_type}\](.*?)\n(?:\[|\Z)', re.DOTALL)

    # Search for the camera parameters section
    match = pattern.search(data)

    if not match:
        raise ValueError(f"Camera type '{camera_type}' not found in the file.")

    # Extract the relevant section
    section = match.group(1)

    # Define a dictionary to store the intrinsics
    intrinsics = {}

    # Extract parameters using regular expressions
    for line in section.strip().split('\n'):
        key, value = line.split('=')
        intrinsics[key.strip()] = float(value.strip())

    return intrinsics

fold = 2
ckpt_path = "/home/aidana/Documents/YOLO/detect/fold_{}/weights/best.pt".format(fold)
# Load the model
model = YOLO(ckpt_path)
data_dir ="/home/aidana/Documents/YOLO/fold_{}/test/images".format(fold)
files = os.listdir(data_dir)
dir = "/media/aidana/aidana/SpineDepth"

for file in files:
    print(file)
    _, specimen,_,recording,_,cam,_, frame = file[:-4].split('_')
    pcd_dir = os.path.join("/home/aidana/Documents/PointNet_data/1",file[:-4]+".npz")


    stls_dir = "/home/aidana/Documents/stls_transformed"
    depth_dir = os.path.join("/media/aidana/SpineDepth/PoinTr_dataset/segmented_spinedepth_new", "Specimen_"+specimen, "recording_"+recording,"cam_"+ cam, "frame_"+frame, "depth.png")
    results = model([os.path.join(data_dir,file)])  # return a list of Results objects
    color_image = np.asarray(Image.open(os.path.join(data_dir,file)))
    depth_image = np.asarray(Image.open(depth_dir))

    if specimen == 2 and recording >= 32:
        calib_src_dir = os.path.join(dir, "Specimen_"+specimen, "Calib_b")
    if specimen == 5 and recording >= 8:
        calib_src_dir = os.path.join(dir, "Specimen_"+specimen, "Calib_b")
    if specimen == 7 and recording >= 12:
        calib_src_dir = os.path.join(dir, "Specimen_"+specimen, "Calib_b")
    if specimen == 9 and recording >= 12:
        calib_src_dir = os.path.join(dir, "Specimen_"+specimen, "Calib_b")
    else:
        calib_src_dir = os.path.join(dir, "Specimen_"+specimen, "Calib")
    if cam ==0:
        calib_dir = os.path.join(calib_src_dir,"SN10027879.conf")
    else:
        calib_dir = os.path.join(calib_src_dir,"SN10028650.conf")



    left_camera_intrinsic =extract_camera_intrinsics_from_file(calib_dir)






    # Process results list
    for result in results:
        color_pallete = [(0,0,1), (0,1,0), (1,0,0), (1,1,0), (0,1,1), (1,0,1)]
        vis =[]
        boxes = result.boxes
        for el,box in enumerate(boxes):
            cls = box.cls
            print(cls)
            conf = box.conf
            xyxy = box.xyxy.cpu().numpy()

            if cls == 0:
                stl = o3d.io.read_triangle_mesh(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                                                "recording_" + str(recording),
                                                                "cam_" + str(cam), "frame_" + str(frame),
                                                                "transformed_vertebra1.stl"))
            if cls == 1:
                stl = o3d.io.read_triangle_mesh(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                                                "recording_" + str(recording),
                                                                "cam_" + str(cam), "frame_" + str(frame),
                                                                "transformed_vertebra2.stl"))
            if cls == 2:
                stl = o3d.io.read_triangle_mesh(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                                                "recording_" + str(recording),
                                                                "cam_" + str(cam), "frame_" + str(frame),
                                                                "transformed_vertebra3.stl"))
            if cls == 3:
                stl = o3d.io.read_triangle_mesh(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                                                "recording_" + str(recording),
                                                                "cam_" + str(cam), "frame_" + str(frame),
                                                                "transformed_vertebra4.stl"))
            if cls == 4:
                stl = o3d.io.read_triangle_mesh(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                                                "recording_" + str(recording),
                                                                "cam_" + str(cam), "frame_" + str(frame),
                                                                "transformed_vertebra5.stl"))

            start_point = (int(xyxy[0,0]), int(xyxy[0,1]))
            end_point = (int(xyxy[0,2]), int(xyxy[0,3]))

            # Create a white image with the same dimensions as the original
            mask = np.ones(color_image.shape[:2], dtype=np.uint8) * 255

            # Draw a black rectangle on the mask where the original rectangle will be
            cv2.rectangle(mask, start_point, end_point, 0, -1)

            # Invert the mask
            mask_inv = cv2.bitwise_not(mask)

            # Create a white image of the same size as the original
            white_image = np.ones_like(color_image) * 255

            # Use the mask to make the region outside the rectangle white
            image_outside_white = cv2.bitwise_and(color_image, color_image, mask=mask_inv)
            white_area = cv2.bitwise_and(white_image, white_image, mask=mask)

            # Combine the white area with the original image
            result = cv2.add(image_outside_white, white_area)


            height, width = result.shape[:2]
            # Convert depth image to point cloud
            u, v = np.meshgrid(np.arange(width), np.arange(height))

            x = (u - left_camera_intrinsic["cx"]) * depth_image / left_camera_intrinsic["fx"]
            y = (v - left_camera_intrinsic["cy"]) * depth_image / left_camera_intrinsic["fy"]
            z = depth_image

            # Create point cloud
            point_cloud_orig = np.stack((x, y, z, result[:, :, 0] / 255, result[:, :, 1] / 255,
                                         result[:, :, 2] / 255), axis=-1).reshape(
                -1, 6)
            point_cloud_orig = np.nan_to_num(point_cloud_orig)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_orig[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(point_cloud_orig[:, 3:6])

            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            # Define a threshold to consider a point as "white"
            threshold = 0.95

            # Create a mask for points that are not white
            # White points will have R, G, B values close to 1.0
            non_white_mask = np.all(colors < threshold, axis=1)

            # Filter out the white points
            filtered_points = points[non_white_mask]
            filtered_colors = colors[non_white_mask]

            # Create a new point cloud without white points
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)


            stl.compute_vertex_normals()

            # Compute the bounding box of the STL mesh
            bounding_box = stl.get_axis_aligned_bounding_box()

            # Filter points inside the bounding box
            indices = bounding_box.get_point_indices_within_bounding_box(filtered_pcd.points)
            filtered_point_cloud = filtered_pcd.select_by_index(indices)
            filtered_point_cloud.paint_uniform_color(color_pallete[el])
            vis.append(filtered_point_cloud)
            vis.append(stl)



            # Visualize the result
        o3d.visualization.draw_geometries(vis)



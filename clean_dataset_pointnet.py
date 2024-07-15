import os
import numpy as np
import open3d as o3d

def update_labels(cropped_pcd, filtered_data, label):
    cropped_points = np.asarray(cropped_pcd.points)

    # Get indices for x, y, z coordinates in filtered_data
    ind_x = np.where(np.isin(filtered_data[:, 0], cropped_points[:, 0]))[0]
    ind_y = np.where(np.isin(filtered_data[:, 1], cropped_points[:, 1]))[0]
    ind_z = np.where(np.isin(filtered_data[:, 2], cropped_points[:, 2]))[0]

    # Find common indices using set intersection
    common_xy = np.intersect1d(ind_x, ind_y)
    common_xyz = np.intersect1d(common_xy, ind_z)

    # Set values in filtered_data at common indices
    filtered_data[common_xyz, 6] = label
    return filtered_data

dir = "/Volumes/SpineDepth/PointNet_data/1"
stls_dir = "/Volumes/SpineDepth/PoinTr_dataset/stls_transformed"
files = os.listdir(dir)

save_dir = "/Volumes/SpineDepth/PointNet_data/1_updated"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for file in files:
    if "._" in file:
        continue
    print(file)
    _, specimen,_, recording,_, cam,_, frame = file[:-4].split("_")
    data = np.load(os.path.join(dir, file))["arr_0"]
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)

    # Calculate the Z-scores for each point
    z_scores = (data - mean) / std_dev

    # Set a Z-score threshold (e.g., 3 standard deviations)
    threshold = 3

    # Identify points with Z-scores within the threshold
    non_outliers = (np.abs(z_scores) < threshold).all(axis=1)

    # Filter out the outliers
    filtered_data = data[non_outliers]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_data[:,0:3])
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_data[:,3:6])

    L1_ind = np.where(filtered_data[:,6] == 1.00)
    L2_ind = np.where(filtered_data[:,6] == 2.00)
    L3_ind = np.where(filtered_data[:,6] == 3.00)
    L4_ind = np.where(filtered_data[:,6] == 4.00)
    L5_ind = np.where(filtered_data[:,6] == 5.00)


    L1_pcd = o3d.geometry.PointCloud()
    L2_pcd = o3d.geometry.PointCloud()
    L3_pcd = o3d.geometry.PointCloud()
    L4_pcd = o3d.geometry.PointCloud()
    L5_pcd = o3d.geometry.PointCloud()

    L1_pcd.points = o3d.utility.Vector3dVector(filtered_data[L1_ind[0],0:3])
    L2_pcd.points = o3d.utility.Vector3dVector(filtered_data[L2_ind[0],0:3])
    L3_pcd.points = o3d.utility.Vector3dVector(filtered_data[L3_ind[0],0:3])
    L4_pcd.points = o3d.utility.Vector3dVector(filtered_data[L4_ind[0],0:3])
    L5_pcd.points = o3d.utility.Vector3dVector(filtered_data[L5_ind[0],0:3])

    L1_pcd.paint_uniform_color((1,0,0))
    L2_pcd.paint_uniform_color((0,1,0))
    L3_pcd.paint_uniform_color((0,0,1))
    L4_pcd.paint_uniform_color((1,1,0))
    L5_pcd.paint_uniform_color((1,0,1))



    L1_stl = o3d.io.read_triangle_mesh(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                                    "recording_" + str(recording),
                                                    "cam_" + str(cam), "frame_" + str(frame),
                                                    "transformed_vertebra1.stl"))

    L2_stl = o3d.io.read_triangle_mesh(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                                    "recording_" + str(recording),
                                                    "cam_" + str(cam), "frame_" + str(frame),
                                                    "transformed_vertebra2.stl"))

    L3_stl = o3d.io.read_triangle_mesh(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                                    "recording_" + str(recording),
                                                    "cam_" + str(cam), "frame_" + str(frame),
                                                    "transformed_vertebra3.stl"))

    L4_stl = o3d.io.read_triangle_mesh(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                                    "recording_" + str(recording),
                                                    "cam_" + str(cam), "frame_" + str(frame),
                                                    "transformed_vertebra4.stl"))
    L5_stl = o3d.io.read_triangle_mesh(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                                    "recording_" + str(recording),
                                                    "cam_" + str(cam), "frame_" + str(frame),
                                                    "transformed_vertebra5.stl"))

    L1_stl.compute_vertex_normals()
    L2_stl.compute_vertex_normals()
    L3_stl.compute_vertex_normals()
    L4_stl.compute_vertex_normals()
    L5_stl.compute_vertex_normals()

    bbox1 = L1_stl.get_oriented_bounding_box()
    bbox2 = L2_stl.get_oriented_bounding_box()
    bbox3 = L3_stl.get_oriented_bounding_box()
    bbox4 = L4_stl.get_oriented_bounding_box()
    bbox5 = L5_stl.get_oriented_bounding_box()

    bbox1.color = (1,0,0)
    bbox2.color = (1,0,0)
    bbox3.color = (1,0,0)
    bbox4.color = (1,0,0)
    bbox5.color = (1,0,0)


    # Step 3: Crop the point cloud using the bounding box
    cropped_L1_pcd = L1_pcd.crop(bbox1)
    cropped_L2_pcd = L2_pcd.crop(bbox2)
    cropped_L3_pcd = L3_pcd.crop(bbox3)
    cropped_L4_pcd = L4_pcd.crop(bbox4)
    cropped_L5_pcd = L5_pcd.crop(bbox5)



    threshold = 5000
    if np.asarray(cropped_L1_pcd.points).shape[0] > threshold and np.asarray(cropped_L2_pcd.points).shape[0] > threshold and np.asarray(cropped_L3_pcd.points).shape[0] > threshold and np.asarray(cropped_L4_pcd.points).shape[0] > threshold and np.asarray(cropped_L5_pcd.points).shape[0] > threshold:

        filtered_data[:, 6] = 0.0

        filtered_data = update_labels(cropped_L1_pcd, filtered_data,1)
        filtered_data = update_labels(cropped_L2_pcd, filtered_data,2)
        filtered_data = update_labels(cropped_L3_pcd, filtered_data,3)
        filtered_data = update_labels(cropped_L4_pcd, filtered_data,4)
        filtered_data = update_labels(cropped_L5_pcd, filtered_data,5)

        np.savez(os.path.join(save_dir, file), filtered_data)




            # L1_ind = np.where(filtered_data[:, 6] == 1.00)
            # L2_ind = np.where(filtered_data[:, 6] == 2.00)
            # L3_ind = np.where(filtered_data[:, 6] == 3.00)
            # L4_ind = np.where(filtered_data[:, 6] == 4.00)
            # L5_ind = np.where(filtered_data[:, 6] == 5.00)
            # background_ind = np.where(filtered_data[:,6] == 0.0)
            #
            # L1_pcd = o3d.geometry.PointCloud()
            # L2_pcd = o3d.geometry.PointCloud()
            # L3_pcd = o3d.geometry.PointCloud()
            # L4_pcd = o3d.geometry.PointCloud()
            # L5_pcd = o3d.geometry.PointCloud()
            # background_pcd = o3d.geometry.PointCloud()
            #
            #
            # L1_pcd.points = o3d.utility.Vector3dVector(filtered_data[L1_ind[0], 0:3])
            # L2_pcd.points = o3d.utility.Vector3dVector(filtered_data[L2_ind[0], 0:3])
            # L3_pcd.points = o3d.utility.Vector3dVector(filtered_data[L3_ind[0], 0:3])
            # L4_pcd.points = o3d.utility.Vector3dVector(filtered_data[L4_ind[0], 0:3])
            # L5_pcd.points = o3d.utility.Vector3dVector(filtered_data[L5_ind[0], 0:3])
            # background_pcd.points = o3d.utility.Vector3dVector(filtered_data[background_ind[0], 0:3])
            #
            # L1_pcd.paint_uniform_color((1, 0, 0))
            # L2_pcd.paint_uniform_color((0, 1, 0))
            # L3_pcd.paint_uniform_color((0, 0, 1))
            # L4_pcd.paint_uniform_color((1, 1, 0))
            # L5_pcd.paint_uniform_color((1, 0, 1))
            # background_pcd.paint_uniform_color((0, 0, 0))
            #
            # o3d.visualization.draw_geometries([
            #     L1_stl, L2_stl, L3_stl, L4_stl, L5_stl,
            #     L1_pcd, L2_pcd, L3_pcd, L4_pcd, L5_pcd, background_pcd
            # ])
            #
            #
            #
            #
            #

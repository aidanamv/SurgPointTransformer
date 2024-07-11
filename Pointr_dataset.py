import os
import numpy as np
import open3d as o3d


dir = r"G:\PointNet_data\1"
stls_dir = r"G:\PoinTr_dataset\stls_transformed"
files = os.listdir(dir)

#Specimen_6_recording_14_cam_0_frame_0.npz


for file in files:
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


    # o3d.visualization.draw_geometries([L1_pcd,L2_pcd, L3_pcd, L4_pcd, L5_pcd,
    #                                      L1_stl, L2_stl, L3_stl, L4_stl, L5_stl])

    L1_stl_dwp = L1_stl.sample_points_uniformly(4096)
    L2_stl_dwp = L2_stl.sample_points_uniformly(4096)
    L3_stl_dwp = L3_stl.sample_points_uniformly(4096)
    L4_stl_dwp = L4_stl.sample_points_uniformly(4096)
    L5_stl_dwp = L5_stl.sample_points_uniformly(4096)



    if not os.path.exists(os.path.join( "complete")):
        os.makedirs(os.path.join( "complete"))

    o3d.io.write_point_cloud(os.path.join( "complete", file[:-4] + "_L1.pcd"), L1_stl_dwp)
    o3d.io.write_point_cloud(os.path.join( "complete", file[:-4] + "_L2.pcd"), L2_stl_dwp)
    o3d.io.write_point_cloud(os.path.join( "complete", file[:-4] + "_L3.pcd"), L3_stl_dwp)
    o3d.io.write_point_cloud(os.path.join( "complete", file[:-4] + "_L4.pcd"), L4_stl_dwp)
    o3d.io.write_point_cloud(os.path.join( "complete", file[:-4] + "_L5.pcd"), L5_stl_dwp)

    if not os.path.exists(os.path.join( "partial", file[:-4] + "_L1")):
        os.makedirs(os.path.join( "partial", file[:-4] + "_L1"))
    if not os.path.exists(os.path.join( "partial", file[:-4] + "_L2")):
        os.makedirs(os.path.join( "partial", file[:-4] + "_L2"))
    if not os.path.exists(os.path.join( "partial", file[:-4] + "_L3")):
        os.makedirs(os.path.join( "partial", file[:-4] + "_L3"))
    if not os.path.exists(os.path.join( "partial", file[:-4] + "_L4")):
        os.makedirs(os.path.join( "partial", file[:-4] + "_L4"))
    if not os.path.exists(os.path.join( "partial", file[:-4] + "_L5")):
        os.makedirs(os.path.join( "partial", file[:-4] + "_L5"))

    o3d.io.write_point_cloud(os.path.join( "partial", file[:-4] + "_L1", "00.pcd"), L1_pcd)
    o3d.io.write_point_cloud(os.path.join( "partial", file[:-4] + "_L2", "00.pcd"), L2_pcd)
    o3d.io.write_point_cloud(os.path.join( "partial", file[:-4] + "_L3", "00.pcd"), L3_pcd)
    o3d.io.write_point_cloud(os.path.join( "partial", file[:-4] + "_L4", "00.pcd"), L4_pcd)
    o3d.io.write_point_cloud(os.path.join( "partial", file[:-4] + "_L5", "00.pcd"), L5_pcd)





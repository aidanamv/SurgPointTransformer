import open3d as o3d
import os
import numpy as np


dir = "/Users/aidanamassalimova/Documents/FinalDataset_4096/fold_1/val"

files = os.listdir(os.path.join(dir, "predictions"))

for file in files:

    partial = o3d.io.read_point_cloud(os.path.join(dir, "partial", "10102023",file[:-4], "00.pcd"))
    complete = o3d.io.read_point_cloud(os.path.join(dir, "complete","10102023", file[:-4] +".pcd"))
    prediction_np = np.load(os.path.join(dir, "predictions", file))["arr_0"]

    prediction = o3d.geometry.PointCloud()
    prediction.points = o3d.utility.Vector3dVector(prediction_np.squeeze(0))
    # prediction.estimate_normals()
    # prediction.orient_normals_consistent_tangent_plane(100)
    #
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #     prediction, depth=14)
    # mesh.paint_uniform_color([200/255,200/255,200/255])
    # mesh.compute_vertex_normals()

    prediction.paint_uniform_color([0,0,1])
    complete.paint_uniform_color([0,1,0])
    partial.paint_uniform_color([1,1,0])
    o3d.visualization.draw_geometries([complete,prediction, partial])

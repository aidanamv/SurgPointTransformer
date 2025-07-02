from PoinTr.tools.utils.model_utils import calc_cd
from PointNet.dataloader import SpineDepthDataset
from PointNet.model import PointNetDenseCls
import logging
import importlib
import argparse
import munch
import yaml
from utils.vis_utils import plot_single_pcd
from utils.train_utils import *
from spinedepth_dataset import ShapeNetDataset
from torch.autograd import Variable
import numpy as np
import pandas as pd
import sys
import os
import pyvista as pv
from scipy.spatial import KDTree
import re
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

import open3d as o3d
import numpy as np
import pyvista as pv

def poisson_reconstruction_visualization(stl, pred):
    # Convert STL mesh to point cloud
    pcd_complete = o3d.geometry.PointCloud()
    pcd_complete.points = o3d.utility.Vector3dVector(np.array(stl.vertices))
    normals = stl.vertex_normals

    # Use KDTree to assign normals from the ground truth STL to the predicted point cloud
    gt_kdtree = o3d.geometry.KDTreeFlann(pcd_complete)
    assigned_normals = []
    for point in pred.points:
        [_, idx, _] = gt_kdtree.search_knn_vector_3d(point, 1)
        closest_normal = normals[idx[0]]
        assigned_normals.append(closest_normal)

    # Set the assigned normals to the predicted point cloud
    pred.normals = o3d.utility.Vector3dVector(np.array(assigned_normals))

    # Run Poisson surface reconstruction
    print("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pred, depth=9)
    mesh.paint_uniform_color((0, 0, 1))  # Blue color for the reconstructed mesh
    mesh.compute_vertex_normals()

    # Save the predicted mesh and the ground truth STL
    o3d.io.write_triangle_mesh("mesh.stl", mesh)
    o3d.io.write_triangle_mesh("gt.stl", stl)

    # Load the meshes using PyVista for advanced visualization
    mesh_pv = pv.read("mesh.stl")
    gt_pv = pv.read("gt.stl")

    # Set up the PyVista plotter
    p = pv.Plotter()

    # Add predicted mesh with specular lighting and shading
    p.add_mesh(mesh_pv, color='orange', opacity=0.5, specular=1.0, specular_power=10, smooth_shading=True)

    # Add ground truth mesh with contrasting color and similar lighting
    p.add_mesh(gt_pv, color='teal', opacity=0.5,specular=1.0, specular_power=10, smooth_shading=True)

    # Adjust the camera to a 3D oblique view
    p.camera_position = 'iso'  # This sets an oblique view

    # Show the plot with proper lighting and shading
    p.show()


# Function to estimate noise variance from the point cloud
def estimate_noise_variance(noisy_cloud, clean_cloud):
    # Use Nearest Neighbors to find corresponding points in clean cloud
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(clean_cloud)
    distances, indices = nbrs.kneighbors(noisy_cloud)

    # Calculate the residuals (differences between noisy and clean points)
    residuals = noisy_cloud - clean_cloud[indices.flatten()]

    # Estimate noise variance as the mean squared residuals
    noise_variance = np.mean(np.sum(residuals ** 2, axis=1))

    return noise_variance


# Function to estimate signal variance from the clean cloud
def estimate_signal_variance(clean_cloud):
    # Signal variance is the variance of the clean point coordinates
    signal_variance = np.var(clean_cloud, axis=0).mean()

    return signal_variance


def calculate_snr(gt_cloud, pred_cloud):
    # Convert Open3D point clouds to numpy arrays
    clean_points = np.asarray(gt_cloud.points)
    noisy_points = np.asarray(pred_cloud.points)

    # Compute signal power (mean squared distance of clean points from the origin)
    signal_power = np.mean(np.linalg.norm(clean_points, axis=1) ** 2)

    # Compute noise power (mean squared error between noisy and clean points)
    noise_power = np.mean(np.linalg.norm(clean_points - noisy_points, axis=1) ** 2)

    # Compute SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)
    return snr




def halve_point_clouds(input, pred, gt, split_normal):


    split_origin = gt.get_center()

    gt_pv = pv.PolyData(np.array(gt.points))
    input_pv = pv.PolyData(np.array(input.points))
    pred_pv = pv.PolyData(np.array(pred.points))
    split_plane = pv.Plane(center=split_origin, direction=split_normal, i_size=200, j_size=200)
    gt_part1 = gt_pv.clip(normal=split_normal, origin=split_origin, invert= False )
    gt_part2 = gt_pv.clip(normal=split_normal, origin=split_origin, invert= True )

    pred_part1 = pred_pv.clip(normal=split_normal, origin=split_origin, invert=False)
    pred_part2 = pred_pv.clip(normal=split_normal, origin=split_origin, invert=True)


    cd1 = calc_cd(torch.from_numpy(np.array(gt_part1.points)).unsqueeze(0).float().cuda(),torch.from_numpy(np.array(input_pv.points)).unsqueeze(0).float().cuda())
    cd2 =  calc_cd(torch.from_numpy(np.array(gt_part2.points)).unsqueeze(0).float().cuda(),torch.from_numpy(np.array(input_pv.points)).unsqueeze(0).float().cuda())

    gt_top = np.empty((3,3))
    gt_bottom = np.empty((3,3))
    pred_top = np.empty((3,3))
    pred_bottom = np.empty((3,3))
    if cd1 > cd2:
        gt_top =gt_part2
        gt_bottom = gt_part1
        pred_top = pred_part2
        pred_bottom = pred_part1
    else:
        gt_top = gt_part1
        gt_bottom = gt_part2
        pred_top = pred_part1
        pred_bottom = pred_part2

    cd_top = calc_cd(torch.from_numpy(np.array(gt_top.points)).unsqueeze(0).float().cuda(),torch.from_numpy(np.array(pred_top.points)).unsqueeze(0).float().cuda())
    cd_bottom =  calc_cd(torch.from_numpy(np.array(gt_bottom.points)).unsqueeze(0).float().cuda(),torch.from_numpy(np.array(pred_bottom.points)).unsqueeze(0).float().cuda())


    # p = pv.Plotter()
    # p.add_points(gt_part1, color="green")
    # p.add_points(pred_part1, color ="blue")
    # p.show()


    return cd_top[0].detach().cpu().numpy(), cd_bottom[0].detach().cpu().numpy()






def read_matrices(file_path):
    matrices = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into individual numbers and convert to float
            if re.search(r',', line):
                numbers = list(map(float, line.split(',')))
            else:
                numbers = list(map(float, line.split()))
            # Reshape the list of numbers into a 4x4 matrix
            matrix = np.array(numbers).reshape((4, 4))
            matrices.append(matrix)
    return matrices
def calculate_tre(pred, target):

    kdtree1 = KDTree(np.array(pred.points))

    closest_point_index1= kdtree1.query(target[0])[1]
    closest_point_index2= kdtree1.query(target[1])[1]
    closest_point_index3= kdtree1.query(target[2])[1]


    closest_distance1 = np.linalg.norm(np.array(pred.points)[closest_point_index1] - target[0])
    closest_distance2 = np.linalg.norm(np.array(pred.points)[closest_point_index2] - target[1])
    closest_distance3 = np.linalg.norm(np.array(pred.points)[closest_point_index3] - target[2])



    return closest_distance1, closest_distance2, closest_distance3
def pick_points(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()
def transform_pcds_to_ct_frame(gt, pred, trans,specimen, recording, cam, frame, level):
    poses_file = "/media/aidana/aidana/SpineDepth/Specimen_{}/recordings/recording{}/Poses_{}.txt".format(specimen,
                                                                                                          recording,
                                                                                                          cam)
    trans_mat_file = "/media/aidana/aidana/SpineDepth/Specimen_{}/transformation_matrices.txt".format(specimen)

    trans_mats = read_matrices(trans_mat_file)
    poses = read_matrices(poses_file)

    inverse_matrix = np.linalg.inv(poses[5 * frame+level-1])


    gt.transform(inverse_matrix)
    pred.transform(inverse_matrix)

    if level == 1:
        trans = gt.get_center()
        return trans, gt, pred
    else:
        gt.transform(trans_mats[level-2])
        pred.transform(trans_mats[level-2])
        gt.translate(trans, relative=False)
        pred.translate(trans, relative=False)
        return gt, pred

def pick_target_points(gt):
    target_points_indices = pick_points(gt)
    selected_points1=gt.points[target_points_indices[0]]
    selected_points2=gt.points[target_points_indices[1]]
    selected_points3=gt.points[target_points_indices[2]]
    return selected_points1, selected_points2, selected_points3





def take_screenshot(vis_list):
    import datetime

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    for obj in vis_list:
        vis.add_geometry(obj)
    opt = vis.get_render_option()
    opt.point_size = 7

    frames_dir = "frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    frame_filenames = []

    def capture_frame(vis):
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{frames_dir}/frame_{timestamp}.png"
        vis.capture_screen_image(filename)
        frame_filenames.append(filename)
        return False

    vis.register_key_callback(ord("S"), capture_frame)

    vis.run()
    vis.destroy_window()

def compute_metrics(predictions, ground_truth, num_classes):

    iou_per_class = []
    dice_per_class = []
    accuracy_per_class = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for class_id in range(num_classes):
        tp = np.sum((predictions == class_id) & (ground_truth == class_id))
        fp = np.sum((predictions == class_id) & (ground_truth != class_id))
        fn = np.sum((predictions != class_id) & (ground_truth == class_id))

        total_tp += tp
        total_fp += fp
        total_fn += fn

        union = tp + fp + fn
        if union == 0:
            iou = 0
        else:
            iou = tp / union
        iou_per_class.append(iou)

        denom = 2 * tp + fp + fn
        if denom == 0:
            dice = 0
        else:
            dice = 2 * tp / denom
        dice_per_class.append(dice)

        class_points = np.sum(ground_truth == class_id)
        if class_points == 0:
            accuracy = 0
        else:
            accuracy = tp / class_points
        accuracy_per_class.append(accuracy)

    overall_accuracy = np.sum(predictions == ground_truth) / len(ground_truth)

    overall_union = total_tp + total_fp + total_fn
    if overall_union == 0:
        overall_iou = 0
    else:
        overall_iou = total_tp / overall_union

    overall_denom = 2 * total_tp + total_fp + total_fn
    if overall_denom == 0:
        overall_dice = 0
    else:
        overall_dice = 2 * total_tp / overall_denom

    metrics = {
        'IoU': iou_per_class,
        'Dice': dice_per_class,
        'Accuracy': accuracy_per_class,
        'Overall Accuracy': overall_accuracy,
        'Overall IoU': overall_iou,
        'Overall Dice': overall_dice
    }

    return metrics

def denoise_point_cloud(point_cloud):
    min_distance = 2

    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)

    # Create a list to store the indices of points to keep
    indices_to_keep = []

    # Loop through each point and only keep those that are sufficiently far from their neighbors
    for i in range(len(point_cloud.points)):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point_cloud.points[i], min_distance)

        # Only keep the point if no other point within the radius has been processed yet
        if len(idx) <= 1:  # means it's not clustered with already considered points
            indices_to_keep.append(i)

    # Select points to keep for a more uniform distribution
    cleaned_point_cloud = point_cloud.select_by_index(indices_to_keep)
    return cleaned_point_cloud

def calculate_iou(gt: o3d.geometry.PointCloud, pr: o3d.geometry.PointCloud):
    point_cloud1 = np.asarray(gt.points)
    point_cloud2 = np.asarray(pr.points)
    min1, max1 = np.min(point_cloud1, axis=0), np.max(point_cloud1, axis=0)
    min2, max2 = np.min(point_cloud2, axis=0), np.max(point_cloud2, axis=0)

    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)
    intersection_volume = np.maximum(0, intersection_max - intersection_min).prod()

    union_volume = (max1 - min1).prod() + (max2 - min2).prod() - intersection_volume

    iou = intersection_volume / union_volume if union_volume > 0 else 0

    return iou



def point_test():
    fold = arg.fold

    model = "/home/aidana/PycharmProjects/SurgPointTransformer/PointNet/checkpoints/fold_{}/ckpt-best.pth".format(fold)
    dataset = "/home/aidana/Documents/PointNet_data"
    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(arg.model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()



    logging.info('Testing...')

    cd_list = []
    f1score_list = []
    emd_list = []
    levels = []
    iou_input_list = []

    iou_seg_list = []
    accuracy_seg_list = []
    cd_top_list = []
    cd_bottom_list = []

    snr_list = []

    val_dataset = SpineDepthDataset(
        root=dataset,
        fold=fold,
        num_channels=6,
        class_choice=["spine"],
        split="val",
        data_augmentation=False)

    stls_dir = "/home/aidana/Documents/stls_transformed"

    for idx in range(len(val_dataset)):
        print("Processing ", idx, "th point cloud from", len(val_dataset), "point clouds.")
        torch.cuda.empty_cache()
        data, gt, filepath, dist, trans = val_dataset[idx]
        point = data[:, :3]
        filename = filepath.split("/")[-1][:-4]
        _, specimen, _, recording, _, cam, _, frame = filename.split("_")
        if recording == '39' and cam == '0' and frame == '1':
            import open3d as o3d

            print(filename)
            point_np = point.numpy()
            import  open3d as o3d
            import  numpy as np
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
            state_dict = torch.load(model, map_location=torch.device('cpu'))
            classifier = PointNetDenseCls(k=6, number_channels=6)
            classifier.load_state_dict(state_dict)
            classifier.eval()

            data = data.transpose(1, 0)
            data = Variable(data.view(1, data.size()[0], data.size()[1]))
            pred, _, _ = classifier(data)
            pred_choice = pred.data.max(2)[1].cpu().numpy()

            metrics = compute_metrics(pred_choice[0], gt.numpy(), 6)

            L1_ind = np.where(gt == 1)[0]
            L2_ind = np.where(gt == 2)[0]
            L3_ind = np.where(gt == 3)[0]
            L4_ind = np.where(gt == 4)[0]
            L5_ind = np.where(gt == 5)[0]
            background = np.where(gt == 0)[0]

            point_np = point_np * dist + trans

            L1_points = point_np[L1_ind]
            L2_points = point_np[L2_ind]
            L3_points = point_np[L3_ind]
            L4_points = point_np[L4_ind]
            L5_points = point_np[L5_ind]
            background_points = point_np[background]

            L1_pcd = o3d.geometry.PointCloud()
            L1_pcd.points = o3d.utility.Vector3dVector(L1_points)

            L2_pcd = o3d.geometry.PointCloud()
            L2_pcd.points = o3d.utility.Vector3dVector(L2_points)

            L3_pcd = o3d.geometry.PointCloud()
            L3_pcd.points = o3d.utility.Vector3dVector(L3_points)

            L4_pcd = o3d.geometry.PointCloud()
            L4_pcd.points = o3d.utility.Vector3dVector(L4_points)

            L5_pcd = o3d.geometry.PointCloud()
            L5_pcd.points = o3d.utility.Vector3dVector(L5_points)

            background_pcd = o3d.geometry.PointCloud()
            background_pcd.points = o3d.utility.Vector3dVector(background_points)

            L1_pcd.paint_uniform_color([1, 0, 0])
            L2_pcd.paint_uniform_color([0, 1, 0])
            L3_pcd.paint_uniform_color([0, 0, 1])
            L4_pcd.paint_uniform_color([1, 1, 0])
            L5_pcd.paint_uniform_color([1, 0, 1])
            background_pcd.paint_uniform_color([0, 0, 0])

            take_screenshot([L1_pcd, L2_pcd, L3_pcd, L4_pcd, L5_pcd, background_pcd])


            segs = [L1_points, L2_points, L3_points, L4_points, L5_points]


            L1_stl_dwp = L1_stl.sample_points_uniformly(4096)
            L2_stl_dwp = L2_stl.sample_points_uniformly(4096)
            L3_stl_dwp = L3_stl.sample_points_uniformly(4096)
            L4_stl_dwp = L4_stl.sample_points_uniformly(4096)
            L5_stl_dwp = L5_stl.sample_points_uniformly(4096)



            gts = [L1_stl_dwp, L2_stl_dwp, L3_stl_dwp, L4_stl_dwp, L5_stl_dwp]
            input_pcds = [L1_pcd, L2_pcd, L3_pcd, L4_pcd, L5_pcd]
            stls = [L1_stl, L2_stl, L3_stl, L4_stl, L5_stl]

            with torch.no_grad():
                for el, seg in enumerate(segs):
                    if len(seg) > 2048:
                        choice = np.random.choice(len(seg), 2048, replace=True)
                        seg = seg[choice]
                        gt_points = torch.from_numpy(np.array(gts[el].points)).cuda().float()/1000
                        gt = gts[el]

                        iou_input_list.append(calculate_iou(gt, input_pcds[el]))

                        inputs = torch.from_numpy(seg).float().unsqueeze(0).cuda()/1000
                        inputs = inputs.transpose(2, 1).contiguous()
                        result_dict = net(inputs,gt_points.unsqueeze(0),is_training=False)
                        pred = o3d.geometry.PointCloud()
                        pred.points = o3d.utility.Vector3dVector(result_dict['out2'][0].cpu().numpy() * 1000)
                        pred.paint_uniform_color([0,0,1])

                        snr = calculate_snr(gt, pred)
                        snr_list.append(snr)

                        pred = denoise_point_cloud(pred)


                        gt.paint_uniform_color([0,1,0])
                        if el == 0:
                            bbox_gt = gt.get_oriented_bounding_box()

                            obb_matrix = bbox_gt.R
                            orientation_matrix = obb_matrix[:3, :3]

                            # The principal axes are the columns of the orientation matrix
                            axis_x = orientation_matrix[:, 0]
                            axis_y = orientation_matrix[:, 1]
                            axis_z = orientation_matrix[:, 2]
                            split_normal = axis_x

                      #  poisson_reconstruction_visualization(stls[el], pred)
                        cd_top, cd_bottom= halve_point_clouds(input_pcds[el],pred, gt, split_normal)
                        cd = calc_cd(torch.from_numpy(np.array(gt.points)).unsqueeze(0).float().cuda(),
                                torch.from_numpy(np.array(pred.points)).unsqueeze(0).float().cuda())
                        cd = cd[0].detach().cpu().numpy()
                        cd_top_list.append(cd_top[0])
                        cd_bottom_list.append(cd_bottom[0])
                        cd_list.append(cd[0])
                        f1score_list.append(result_dict["f1"].detach().cpu().numpy()[0])
                        emd_list.append(result_dict["emd"].detach().cpu().numpy()[0])
                        levels.append(el + 1)
                        iou_seg_list.append(metrics['IoU'][el + 1])
                        accuracy_seg_list.append(metrics['Accuracy'][el + 1])


        data = {
            'IoU_input': iou_input_list,
            'IoU_seg': iou_seg_list,
            'Accuracy': accuracy_seg_list,
            'CD': cd_list,
            'F1': f1score_list,
            'EMD': emd_list,
            'Level': levels,
            'CD_top': cd_top_list,
            'CD_bottom': cd_bottom_list,
            'SNR': snr_list,

        }

        # Create a DataFrame from the dictionary
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        df.to_csv('filtered_vrcnet_fold{}_test.csv'.format(fold), index=False)

        print("DataFrame saved ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-f', '--fold',required=True)
    parser.add_argument('-m', '--model',required=True)


    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))



    exp_name = os.path.basename(arg.model)

    log_dir = os.path.dirname(arg.model)

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                       logging.StreamHandler(sys.stdout)])

    point_test()

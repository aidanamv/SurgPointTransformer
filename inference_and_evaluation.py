import argparse
import os
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from dataloader import ShapeNetDataset
from model import PointNetDenseCls
import numpy as np
import open3d as o3d

def compute_metrics(predictions, ground_truth, num_classes):
    """
    Compute IoU, Dice score, accuracy, overall IoU, and overall Dice score for point cloud segmentation.

    Parameters:
    predictions (numpy array): Array of predicted class labels.
    ground_truth (numpy array): Array of ground truth class labels.
    num_classes (int): Number of classes.

    Returns:
    dict: Dictionary containing IoU, Dice score, and accuracy for each class, overall IoU, overall Dice score, and overall accuracy.
    """
    iou_per_class = []
    dice_per_class = []
    accuracy_per_class = []

    # Aggregate counts for overall IoU and Dice score
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for class_id in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((predictions == class_id) & (ground_truth == class_id))
        fp = np.sum((predictions == class_id) & (ground_truth != class_id))
        fn = np.sum((predictions != class_id) & (ground_truth == class_id))

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # IoU calculation
        union = tp + fp + fn
        if union == 0:
            iou = 0  # If there are no ground truth points for this class
        else:
            iou = tp / union
        iou_per_class.append(iou)

        # Dice score calculation
        denom = 2 * tp + fp + fn
        if denom == 0:
            dice = 0  # If there are no ground truth points for this class
        else:
            dice = 2 * tp / denom
        dice_per_class.append(dice)

        # Accuracy calculation for each class
        class_points = np.sum(ground_truth == class_id)
        if class_points == 0:
            accuracy = 0  # If there are no points for this class in the ground truth
        else:
            accuracy = tp / class_points
        accuracy_per_class.append(accuracy)

    # Overall accuracy
    overall_accuracy = np.sum(predictions == ground_truth) / len(ground_truth)

    # Overall IoU calculation
    overall_union = total_tp + total_fp + total_fn
    if overall_union == 0:
        overall_iou = 0
    else:
        overall_iou = total_tp / overall_union

    # Overall Dice score calculation
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

def voxel_grid_downsample(points, num_voxels):
    bounds = np.ptp(points, axis=0)  # Compute the range of the points
    voxel_size = np.cbrt(bounds.prod() / num_voxels)  # Compute voxel size
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)  # Compute voxel indices for each point
    voxel_dict = {}  # Dictionary to store one point per voxel

    for point, voxel_index in zip(points, voxel_indices):
        voxel_key = tuple(voxel_index)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = point

    downsampled_points = np.array(list(voxel_dict.values()))
    return downsampled_points


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="./checkpoints/fold_1/ckpt-best.pth", help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='Users/aidanamassalimova/Documents/PointNet_data', help='dataset path')
parser.add_argument('--class_choice', type=str, default='spine', help='class choice')
parser.add_argument('--fold', type=int, default=1)

opt = parser.parse_args()
print(opt)


val_dataset = ShapeNetDataset(
    root=opt.dataset,
    fold = opt.fold,
    class_choice=[opt.class_choice],
    split='val',
    data_augmentation=False)


IoU_total = []
Accuracy_total = []
Dice_total = []

iou_L1_total = []
iou_L2_total = []
iou_L3_total = []
iou_L4_total = []
iou_L5_total = []

acc_L1_total = []
acc_L2_total = []
acc_L3_total = []
acc_L4_total = []
acc_L5_total = []

dice_L1_total = []
dice_L2_total = []
dice_L3_total = []
dice_L4_total = []
dice_L5_total = []

stls_dir = "G:/PoinTr_dataset/stls_transformed"
save_val_dir =r"G:\\pointr_data\\fold_{}\\val".format(opt.fold)
sets = ["complete", "partial"]
for set in sets:
    if not os.path.exists(os.path.join(save_val_dir,set)):
        os.makedirs(os.path.join(save_val_dir,set))

for idx in range(len(val_dataset)):
    print("model %d/%d" % (idx, len(val_dataset)))
    point, gt,filepath,dist,trans = val_dataset[idx]

    filename = filepath.split("\\")[-1][:-4]
    print(filename)
    _,specimen, _, recording, _, cam,_, frame = filename.split("_")
    point_np = point.numpy()
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

    state_dict = torch.load(opt.model,map_location=torch.device('cuda'))
    classifier = PointNetDenseCls(k= 6).cuda()
    classifier.load_state_dict(state_dict)
    classifier.eval()

    point = point.transpose(1, 0).contiguous().cuda()

    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _ = classifier(point)
    pred_choice = pred.data.max(2)[1].cpu().numpy()


    # Calculate the mean and standard deviation for each dimension
    mean = np.mean(point_np, axis=0)
    std_dev = np.std(point_np, axis=0)

    # Calculate the Z-scores for each point
    z_scores = (point_np - mean) / std_dev

    # Set a Z-score threshold (e.g., 3 standard deviations)
    threshold = 3

    # Identify points with Z-scores within the threshold
    non_outliers = (np.abs(z_scores) < threshold).all(axis=1)







    # Filter out the outliers
    filtered_point_cloud = point_np[non_outliers]
    filtered_pred = pred_choice[0][non_outliers]
    filtered_gt = gt[non_outliers]

    L1 = np.where(filtered_pred == 1)[0]
    L2 = np.where(filtered_pred == 2)[0]
    L3 = np.where(filtered_pred == 3)[0]
    L4 = np.where(filtered_pred == 4)[0]
    L5 = np.where(filtered_pred == 5)[0]
    background = np.where(filtered_pred == 0)[0]

    if len(L1) and len(L2) and len(L3) and len(L4) and len(L5):



        #ground truth labels

        L1_gt = np.where(filtered_gt == 1)[0]
        L2_gt = np.where(filtered_gt == 2)[0]
        L3_gt = np.where(filtered_gt == 3)[0]
        L4_gt = np.where(filtered_gt == 4)[0]
        L5_gt = np.where(filtered_gt == 5)[0]
        background_gt = np.where(filtered_gt == 0)[0]

        colors = np.zeros((filtered_point_cloud.shape[0], 3))
        colors[L1, :] =(1,0,0)
        colors[L2, :] = (0,1,0)
        colors[L3, :] = (0,0,1)
        colors[L4, :] = (1,1,0)
        colors[L5, :] = (1,0,1)
        colors[background, :] = (0,0,0)

        filtered_point_cloud = filtered_point_cloud * dist + trans
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_point_cloud)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors)


        L1_pcd = o3d.geometry.PointCloud()
        L2_pcd = o3d.geometry.PointCloud()
        L3_pcd = o3d.geometry.PointCloud()
        L4_pcd = o3d.geometry.PointCloud()
        L5_pcd = o3d.geometry.PointCloud()
        background_pcd = o3d.geometry.PointCloud()

        L1_pcd_gt = o3d.geometry.PointCloud()
        L2_pcd_gt = o3d.geometry.PointCloud()
        L3_pcd_gt = o3d.geometry.PointCloud()
        L4_pcd_gt = o3d.geometry.PointCloud()
        L5_pcd_gt = o3d.geometry.PointCloud()
        background_pcd_gt = o3d.geometry.PointCloud()


        L1_pcd_gt.points = o3d.utility.Vector3dVector(filtered_point_cloud[L1_gt,:])
        L2_pcd_gt.points = o3d.utility.Vector3dVector(filtered_point_cloud[L2_gt,:])
        L3_pcd_gt.points = o3d.utility.Vector3dVector(filtered_point_cloud[L3_gt,:])
        L4_pcd_gt.points = o3d.utility.Vector3dVector(filtered_point_cloud[L4_gt,:])
        L5_pcd_gt.points = o3d.utility.Vector3dVector(filtered_point_cloud[L5_gt,:])
        background_pcd_gt.points = o3d.utility.Vector3dVector(filtered_point_cloud[background_gt,:])


        L1_pcd.points = o3d.utility.Vector3dVector(filtered_point_cloud[L1,:])
        L2_pcd.points = o3d.utility.Vector3dVector(filtered_point_cloud[L2,:])
        L3_pcd.points = o3d.utility.Vector3dVector(filtered_point_cloud[L3,:])
        L4_pcd.points = o3d.utility.Vector3dVector(filtered_point_cloud[L4,:])
        L5_pcd.points = o3d.utility.Vector3dVector(filtered_point_cloud[L5,:])
        background_pcd.points = o3d.utility.Vector3dVector(filtered_point_cloud[background,:])


        L1_pcd.paint_uniform_color((1,0,0))
        L2_pcd.paint_uniform_color((1,0,0))
        L3_pcd.paint_uniform_color((1,0,0))
        L4_pcd.paint_uniform_color((1,0,0))
        L5_pcd.paint_uniform_color((1,0,0))



        L1_pcd_gt.paint_uniform_color((0,0,1))
        L2_pcd_gt.paint_uniform_color((0,0,1))
        L3_pcd_gt.paint_uniform_color((0,0,1))
        L4_pcd_gt.paint_uniform_color((0,0,1))
        L5_pcd_gt.paint_uniform_color((0,0,1))

        background_pcd_gt.paint_uniform_color((0,0,0))
        background_pcd.paint_uniform_color((0,1,0))

        o3d.visualization.draw_geometries([L1_pcd,L2_pcd, L3_pcd, L4_pcd, L5_pcd])

#         metrics = compute_metrics(pred_choice[0], gt.numpy(), 6)
#
#         IoU_total.append(metrics['Overall IoU'])
#         Accuracy_total.append(metrics['Overall Accuracy'])
#         Dice_total.append(metrics['Overall Dice'])
#
#         acc_L1_total.append(metrics['Accuracy'][1])
#         acc_L2_total.append(metrics['Accuracy'][2])
#         acc_L3_total.append(metrics['Accuracy'][3])
#         acc_L4_total.append(metrics['Accuracy'][4])
#         acc_L5_total.append(metrics['Accuracy'][5])
#
#         iou_L1_total.append(metrics['IoU'][1])
#         iou_L2_total.append(metrics['IoU'][2])
#         iou_L3_total.append(metrics['IoU'][3])
#         iou_L4_total.append(metrics['IoU'][4])
#         iou_L5_total.append(metrics['IoU'][5])
#
#         dice_L1_total.append(metrics['Dice'][1])
#         dice_L2_total.append(metrics['Dice'][2])
#         dice_L3_total.append(metrics['Dice'][3])
#         dice_L4_total.append(metrics['Dice'][4])
#         dice_L5_total.append(metrics['Dice'][5])
#
#         L1_stl_dwp = L1_stl.sample_points_uniformly(4096)
#         L2_stl_dwp = L2_stl.sample_points_uniformly(4096)
#         L3_stl_dwp = L3_stl.sample_points_uniformly(4096)
#         L4_stl_dwp = L4_stl.sample_points_uniformly(4096)
#         L5_stl_dwp = L5_stl.sample_points_uniformly(4096)
#
#         o3d.io.write_point_cloud(os.path.join(save_val_dir,"complete",filename+"_L1.pcd"),L1_stl_dwp)
#         o3d.io.write_point_cloud(os.path.join(save_val_dir,"complete",filename+"_L2.pcd"),L2_stl_dwp)
#         o3d.io.write_point_cloud(os.path.join(save_val_dir,"complete",filename+"_L3.pcd"),L3_stl_dwp)
#         o3d.io.write_point_cloud(os.path.join(save_val_dir,"complete",filename+"_L4.pcd"),L4_stl_dwp)
#         o3d.io.write_point_cloud(os.path.join(save_val_dir,"complete",filename+"_L5.pcd"),L5_stl_dwp)
#
#         if not os.path.exists(os.path.join(save_val_dir, "partial", filename + "_L1")):
#             os.makedirs(os.path.join(save_val_dir, "partial", filename + "_L1"))
#         if not os.path.exists(os.path.join(save_val_dir, "partial", filename + "_L2")):
#             os.makedirs(os.path.join(save_val_dir, "partial", filename + "_L2"))
#         if not os.path.exists(os.path.join(save_val_dir, "partial", filename + "_L3")):
#             os.makedirs(os.path.join(save_val_dir, "partial", filename + "_L3"))
#         if not os.path.exists(os.path.join(save_val_dir, "partial", filename + "_L4")):
#             os.makedirs(os.path.join(save_val_dir, "partial", filename + "_L4"))
#         if not os.path.exists(os.path.join(save_val_dir, "partial", filename + "_L5")):
#             os.makedirs(os.path.join(save_val_dir, "partial", filename + "_L5"))
#
#         o3d.io.write_point_cloud(os.path.join(save_val_dir, "partial", filename + "_L1","00.pcd"), L1_pcd)
#         o3d.io.write_point_cloud(os.path.join(save_val_dir, "partial", filename + "_L2","00.pcd"), L2_pcd)
#         o3d.io.write_point_cloud(os.path.join(save_val_dir, "partial", filename + "_L3","00.pcd"), L3_pcd)
#         o3d.io.write_point_cloud(os.path.join(save_val_dir, "partial", filename + "_L4","00.pcd"), L4_pcd)
#         o3d.io.write_point_cloud(os.path.join(save_val_dir, "partial", filename + "_L5","00.pcd"), L5_pcd)
#
# print(len(iou_L1_total))
#
# print("mean L1 IoU:", np.mean(iou_L1_total))
# print("mean L1 Accuracy:", np.mean(acc_L1_total))
# print("mean L1 dice:", np.mean(dice_L1_total))
#
#
# print("mean L2 IoU:", np.mean(iou_L2_total))
# print("mean L2 Accuracy:", np.mean(acc_L2_total))
# print("mean L2 dice:", np.mean(dice_L2_total))
#
#
# print("mean L3 IoU:", np.mean(iou_L3_total))
# print("mean L3 Accuracy:", np.mean(acc_L3_total))
# print("mean L3 dice:", np.mean(dice_L3_total))
#
#
# print("mean L4 IoU:", np.mean(iou_L4_total))
# print("mean L4 Accuracy:", np.mean(acc_L4_total))
# print("mean L4 dice:", np.mean(dice_L4_total))
#
#
# print("mean L5 IoU:", np.mean(iou_L5_total))
# print("mean L5 Accuracy:", np.mean(acc_L5_total))
# print("mean L5 dice:", np.mean(dice_L5_total))
#
# print("mean IoU:", np.mean(IoU_total))
# print("mean Accuracy:", np.mean(Accuracy_total))
# print("mean Dice:",np.mean(Dice_total))
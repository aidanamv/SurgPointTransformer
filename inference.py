import argparse
import os
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from dataloader import ShapeNetDataset
from model import PointNetDenseCls
import pyvista as pv
import numpy as np


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

parser.add_argument('--model', type=str, default='/Users/aidanamassalimova/PycharmProjects/pointnet.pytorch/seg_model_spine_249.pth', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='/Users/aidanamassalimova/Documents/PointNet_data', help='dataset path')
parser.add_argument('--class_choice', type=str, default='spine', help='class choice')
parser.add_argument('--fold', type=int, default=1)

opt = parser.parse_args()
print(opt)
train_dataset = ShapeNetDataset(
    root=opt.dataset,
    fold = opt.fold,
    class_choice=[opt.class_choice],
    split='train',
    data_augmentation=False)

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

stls_dir = "../stls_transformed"
save_val_dir ="./pointr_data/fold_{}/val".format(opt.fold)
save_train_dir ="./pointr_data/fold_{}/train".format(opt.fold)
sets = ["point_cloud", "stls"]
for set in sets:
    if not os.path.exists(os.path.join(save_val_dir,set)):
        os.makedirs(os.path.join(save_val_dir,set))
    if not os.path.exists(os.path.join(save_train_dir,set)):
        os.makedirs(os.path.join(save_train_dir,set))
try:
    for idx in range(len(train_dataset)):
        print("model %d/%d" % (idx, len(train_dataset)))
        point, seg,filepath,dist,trans = train_dataset[idx]

        filename = filepath.split("/")[-1][:-4]
        print(filename)
        if not os.path.exists(os.path.join(save_train_dir,"point_cloud", filename+".vtp")):
            _,specimen, _, recording, _, cam,_, frame = filename.split("_")
            point_np = point.numpy()
            L1_stl = pv.read(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                          "recording_" + str(recording),
                                          "cam_" + str(cam), "frame_" + str(frame),
                                          "transformed_vertebra1.stl"))

            L2_stl = pv.read(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                          "recording_" + str(recording),
                                          "cam_" + str(cam), "frame_" + str(frame),
                                          "transformed_vertebra2.stl"))

            L3_stl = pv.read(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                          "recording_" + str(recording),
                                          "cam_" + str(cam), "frame_" + str(frame),
                                          "transformed_vertebra3.stl"))

            L4_stl = pv.read(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                          "recording_" + str(recording),
                                          "cam_" + str(cam), "frame_" + str(frame),
                                          "transformed_vertebra4.stl"))
            L5_stl = pv.read(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                          "recording_" + str(recording),
                                          "cam_" + str(cam), "frame_" + str(frame),
                                          "transformed_vertebra5.stl"))

            state_dict = torch.load(opt.model,map_location=torch.device('cpu'))
            classifier = PointNetDenseCls(k= 6)
            classifier.load_state_dict(state_dict)
            classifier.eval()

            point = point.transpose(1, 0).contiguous()

            point = Variable(point.view(1, point.size()[0], point.size()[1]))
            pred, _, _ = classifier(point)
            pred_choice = pred.data.max(2)[1]


            # Calculate the mean and standard deviation for each dimension
            mean = np.mean(point_np, axis=0)
            std_dev = np.std(point_np, axis=0)

            # Calculate the Z-scores for each point
            z_scores = (point_np - mean) / std_dev

            # Set a Z-score threshold (e.g., 3 standard deviations)
            threshold = 3

            # Identify points with Z-scores within the threshold
            non_outliers = (np.abs(z_scores) < threshold).all(axis=1)





            metrics = compute_metrics(pred_choice[0].numpy() , seg.numpy(), 6)

            IoU_total.append(metrics['Overall IoU'])
            Accuracy_total.append(metrics['Overall Accuracy'])
            Dice_total.append(metrics['Overall Dice'])

            acc_L1_total.append(metrics['Accuracy'][1])
            acc_L2_total.append(metrics['Accuracy'][2])
            acc_L3_total.append(metrics['Accuracy'][3])
            acc_L4_total.append(metrics['Accuracy'][4])
            acc_L5_total.append(metrics['Accuracy'][5])

            iou_L1_total.append(metrics['IoU'][1])
            iou_L2_total.append(metrics['IoU'][2])
            iou_L3_total.append(metrics['IoU'][3])
            iou_L4_total.append(metrics['IoU'][4])
            iou_L5_total.append(metrics['IoU'][5])

            dice_L1_total.append(metrics['Dice'][1])
            dice_L2_total.append(metrics['Dice'][2])
            dice_L3_total.append(metrics['Dice'][3])
            dice_L4_total.append(metrics['Dice'][4])
            dice_L5_total.append(metrics['Dice'][5])

            # Filter out the outliers
            filtered_point_cloud = point_np[non_outliers]
            filtered_pred = pred_choice[0][non_outliers]
            L1 = np.where(filtered_pred == 1)[0]
            L2 = np.where(filtered_pred == 2)[0]
            L3 = np.where(filtered_pred == 3)[0]
            L4 = np.where(filtered_pred == 4)[0]
            L5 = np.where(filtered_pred == 5)[0]
            background = np.where(filtered_pred == 0)[0]

            colors = np.zeros((filtered_point_cloud.shape[0], 3))
            colors[L1, :] = [115, 21, 19]
            colors[L2, :] = [155, 224, 138]
            colors[L3, :] = [181, 10, 236]
            colors[L4, :] = [75, 75, 251]
            colors[L5, :] = [246, 103, 178]
            colors[background, :] = [0, 0, 0]
            filtered_point_cloud = filtered_point_cloud * dist + trans
            pcd = pv.PolyData(filtered_point_cloud)
            pcd["colors"] = colors

            pcd.save(os.path.join(save_train_dir,"point_cloud", filename+".vtp"))
            L1_stl.save(os.path.join(save_train_dir,"stls", filename+"_L1.stl"))
            L2_stl.save(os.path.join(save_train_dir,"stls", filename+"_L2.stl"))
            L3_stl.save(os.path.join(save_train_dir,"stls", filename+"_L3.stl"))
            L4_stl.save(os.path.join(save_train_dir,"stls", filename+"_L4.stl"))
            L5_stl.save(os.path.join(save_train_dir,"stls", filename+"_L5.stl"))

    print("mean L1 IoU:", np.mean(iou_L1_total))
    print("mean L1 Accuracy:", np.mean(acc_L1_total))
    print("mean L1 dice:", np.mean(dice_L1_total))


    print("mean L2 IoU:", np.mean(iou_L2_total))
    print("mean L2 Accuracy:", np.mean(acc_L2_total))
    print("mean L2 dice:", np.mean(dice_L2_total))


    print("mean L3 IoU:", np.mean(iou_L3_total))
    print("mean L3 Accuracy:", np.mean(acc_L3_total))
    print("mean L3 dice:", np.mean(dice_L3_total))


    print("mean L4 IoU:", np.mean(iou_L4_total))
    print("mean L4 Accuracy:", np.mean(acc_L4_total))
    print("mean L4 dice:", np.mean(dice_L4_total))


    print("mean L5 IoU:", np.mean(iou_L5_total))
    print("mean L5 Accuracy:", np.mean(acc_L5_total))
    print("mean L5 dice:", np.mean(dice_L5_total))

    print("mean IoU:", np.mean(IoU_total))
    print("mean Accuracy:", np.mean(Accuracy_total))
    print("mean Dice:",np.mean(Dice_total))
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

    for idx in range(len(val_dataset)):
        print("model %d/%d" % (idx, len(val_dataset)))
        point, seg,filepath,dist,trans = val_dataset[idx]
        filename = filepath.split("/")[-1][:-4]
        print(filename)
        _,specimen, _, recording, _, cam,_, frame = filename.split("_")
        point_np = point.numpy()
        L1_stl = pv.read(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                      "recording_" + str(recording),
                                      "cam_" + str(cam), "frame_" + str(frame),
                                      "transformed_vertebra1.stl"))

        L2_stl = pv.read(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                      "recording_" + str(recording),
                                      "cam_" + str(cam), "frame_" + str(frame),
                                      "transformed_vertebra2.stl"))

        L3_stl = pv.read(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                      "recording_" + str(recording),
                                      "cam_" + str(cam), "frame_" + str(frame),
                                      "transformed_vertebra3.stl"))

        L4_stl = pv.read(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                      "recording_" + str(recording),
                                      "cam_" + str(cam), "frame_" + str(frame),
                                      "transformed_vertebra4.stl"))
        L5_stl = pv.read(os.path.join(stls_dir, "Specimen_" + str(specimen),
                                      "recording_" + str(recording),
                                      "cam_" + str(cam), "frame_" + str(frame),
                                      "transformed_vertebra5.stl"))

        state_dict = torch.load(opt.model,map_location=torch.device('cpu'))
        classifier = PointNetDenseCls(k= 6)
        classifier.load_state_dict(state_dict)
        classifier.eval()

        point = point.transpose(1, 0).contiguous()

        point = Variable(point.view(1, point.size()[0], point.size()[1]))
        pred, _, _ = classifier(point)
        pred_choice = pred.data.max(2)[1]




        # Calculate the mean and standard deviation for each dimension
        mean = np.mean(point_np, axis=0)
        std_dev = np.std(point_np, axis=0)

        # Calculate the Z-scores for each point
        z_scores = (point_np - mean) / std_dev

        # Set a Z-score threshold (e.g., 3 standard deviations)
        threshold = 3

        # Identify points with Z-scores within the threshold
        non_outliers = (np.abs(z_scores) < threshold).all(axis=1)





        metrics = compute_metrics(pred_choice[0].numpy() , seg.numpy(), 6)

        IoU_total.append(metrics['Overall IoU'])
        Accuracy_total.append(metrics['Overall Accuracy'])
        Dice_total.append(metrics['Overall Dice'])

        acc_L1_total.append(metrics['Accuracy'][1])
        acc_L2_total.append(metrics['Accuracy'][2])
        acc_L3_total.append(metrics['Accuracy'][3])
        acc_L4_total.append(metrics['Accuracy'][4])
        acc_L5_total.append(metrics['Accuracy'][5])

        iou_L1_total.append(metrics['IoU'][1])
        iou_L2_total.append(metrics['IoU'][2])
        iou_L3_total.append(metrics['IoU'][3])
        iou_L4_total.append(metrics['IoU'][4])
        iou_L5_total.append(metrics['IoU'][5])

        dice_L1_total.append(metrics['Dice'][1])
        dice_L2_total.append(metrics['Dice'][2])
        dice_L3_total.append(metrics['Dice'][3])
        dice_L4_total.append(metrics['Dice'][4])
        dice_L5_total.append(metrics['Dice'][5])

        # Filter out the outliers
        filtered_point_cloud = point_np[non_outliers]
        filtered_pred = pred_choice[0][non_outliers]
        L1 = np.where(filtered_pred == 1)[0]
        L2 = np.where(filtered_pred == 2)[0]
        L3 = np.where(filtered_pred == 3)[0]
        L4 = np.where(filtered_pred == 4)[0]
        L5 = np.where(filtered_pred == 5)[0]
        background = np.where(filtered_pred == 0)[0]

        colors = np.zeros((filtered_point_cloud.shape[0], 3))
        colors[L1, :] = [115, 21, 19]
        colors[L2, :] = [155, 224, 138]
        colors[L3, :] = [181, 10, 236]
        colors[L4, :] = [75, 75, 251]
        colors[L5, :] = [246, 103, 178]
        colors[background, :] = [0, 0, 0]
        filtered_point_cloud = filtered_point_cloud * dist + trans
        pcd = pv.PolyData(filtered_point_cloud)
        pcd["colors"] = colors

        pcd.save(os.path.join(save_val_dir,"point_cloud", filename+".vtp"))
        L1_stl.save(os.path.join(save_val_dir,"stls", filename+"_L1.stl"))
        L2_stl.save(os.path.join(save_val_dir,"stls", filename+"_L2.stl"))
        L3_stl.save(os.path.join(save_val_dir,"stls", filename+"_L3.stl"))
        L4_stl.save(os.path.join(save_val_dir,"stls", filename+"_L4.stl"))
        L5_stl.save(os.path.join(save_val_dir,"stls", filename+"_L5.stl"))

except:
    print("file error")

print("mean L1 IoU:", np.mean(iou_L1_total))
print("mean L1 Accuracy:", np.mean(acc_L1_total))
print("mean L1 dice:", np.mean(dice_L1_total))


print("mean L2 IoU:", np.mean(iou_L2_total))
print("mean L2 Accuracy:", np.mean(acc_L2_total))
print("mean L2 dice:", np.mean(dice_L2_total))


print("mean L3 IoU:", np.mean(iou_L3_total))
print("mean L3 Accuracy:", np.mean(acc_L3_total))
print("mean L3 dice:", np.mean(dice_L3_total))


print("mean L4 IoU:", np.mean(iou_L4_total))
print("mean L4 Accuracy:", np.mean(acc_L4_total))
print("mean L4 dice:", np.mean(dice_L4_total))


print("mean L5 IoU:", np.mean(iou_L5_total))
print("mean L5 Accuracy:", np.mean(acc_L5_total))
print("mean L5 dice:", np.mean(dice_L5_total))

print("mean IoU:", np.mean(IoU_total))
print("mean Accuracy:", np.mean(Accuracy_total))
print("mean Dice:",np.mean(Dice_total))
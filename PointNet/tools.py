import numpy as np
import os
import open3d as o3d

def visualize_results(stls_dir, specimen, recording, cam, frame, point_cloud, pred):

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

    L1 = np.where(pred == 1)[0]
    L2 = np.where(pred == 2)[0]
    L3 = np.where(pred == 3)[0]
    L4 = np.where(pred == 4)[0]
    L5 = np.where(pred == 5)[0]
    background = np.where(pred == 0)[0]



    colors = np.zeros((point_cloud.shape[0], 3))
    colors[L1, :] = (1, 0, 0)
    colors[L2, :] = (0, 1, 0)
    colors[L3, :] = (0, 0, 1)
    colors[L4, :] = (1, 1, 0)
    colors[L5, :] = (1, 0, 1)
    colors[background, :] = (0, 0, 0)

    predicted_pcd = o3d.geometry.PointCloud()
    predicted_pcd.points = o3d.utility.Vector3dVector(point_cloud)
    predicted_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([predicted_pcd, L1_stl, L2_stl, L3_stl, L4_stl, L5_stl])




def filter_point_clouds(point_np, pred, gt):
    mean = np.mean(point_np, axis=0)
    std_dev = np.std(point_np, axis=0)
    z_scores = (point_np - mean) / std_dev
    threshold = 3
    non_outliers = (np.abs(z_scores) < threshold).all(axis=1)
    filtered_point_cloud = point_np[non_outliers]
    filtered_pred = pred[non_outliers]
    filtered_gt = gt[non_outliers]
    return filtered_point_cloud,filtered_pred,filtered_gt


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

def voxel_grid_downsample(points, num_voxels):
    bounds = np.ptp(points, axis=0)
    voxel_size = np.cbrt(bounds.prod() / num_voxels)
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    voxel_dict = {}

    for point, voxel_index in zip(points, voxel_indices):
        voxel_key = tuple(voxel_index)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = point

    downsampled_points = np.array(list(voxel_dict.values()))
    return downsampled_points
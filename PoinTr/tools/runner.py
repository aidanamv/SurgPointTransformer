import open3d as o3d
import os
import json
import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import torch.utils.data
from torch.autograd import Variable
from dataloader import SpineDepthDataset
from model import PointNetDenseCls
import open3d as o3d
import numpy as np
import pandas as pd
import pyvista as pv
import re
from tools.utils.model_utils import calc_cd, calc_emd
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors



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


    cd_top = calc_cd(torch.from_numpy(np.array(pred_top.points)).unsqueeze(0).float().cuda(),torch.from_numpy(np.array(gt_top.points)).unsqueeze(0).float().cuda())
    print(cd_top)

    cd_bottom =  calc_cd(torch.from_numpy(np.array(gt_bottom.points)).unsqueeze(0).float().cuda(),torch.from_numpy(np.array(pred_bottom.points)).unsqueeze(0).float().cuda())



    return cd_top[0].detach().cpu().numpy(), cd_bottom[0].detach().cpu().numpy()


def compute_metrics(predictions, ground_truth, num_classes):

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



def calculate_iou(gt: o3d.geometry.PointCloud, pr: o3d.geometry.PointCloud):
    # Calculate the bounding boxes for each point cloud
    point_cloud1 = np.asarray(gt.points)
    point_cloud2 = np.asarray(pr.points)
    min1, max1 = np.min(point_cloud1, axis=0), np.max(point_cloud1, axis=0)
    min2, max2 = np.min(point_cloud2, axis=0), np.max(point_cloud2, axis=0)

    # Calculate the intersection
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)
    intersection_volume = np.maximum(0, intersection_max - intersection_min).prod()

    # Calculate the union
    union_volume = (max1 - min1).prod() + (max2 - min2).prod() - intersection_volume

    # Calculate IoU
    iou = intersection_volume / union_volume if union_volume > 0 else 0

    return iou

def poisson_reconstruction_visualization(stl, pred):
    pcd_complete = o3d.geometry.PointCloud()
    pcd_complete.points = o3d.utility.Vector3dVector(np.array(stl.vertices))
    normals = stl.vertex_normals
    gt_kdtree = o3d.geometry.KDTreeFlann(pcd_complete)
    assigned_normals = []
    for point in pred.points:
        [_, idx, _] = gt_kdtree.search_knn_vector_3d(point, 1)
        closest_normal = normals[idx[0]]  # Get the normal of the closest point
        assigned_normals.append(closest_normal)

    pred.normals = o3d.utility.Vector3dVector(np.array(assigned_normals))
    print("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pred, depth=9)
    mesh.paint_uniform_color((0, 0, 1))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("mesh.stl", mesh)
    o3d.io.write_triangle_mesh("gt.stl", stl)
    mesh_pv = pv.read("mesh.stl")
    gt_pv = pv.read("gt.stl")
    p = pv.Plotter()
    p.add_mesh(mesh_pv, color='fuchsia', opacity=0.5)
    p.add_mesh(gt_pv, color='green', opacity=0.5)
    p.show()

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
    selected_points1 = gt.points[target_points_indices[0]]
    selected_points2 = gt.points[target_points_indices[1]]
    selected_points3 = gt.points[target_points_indices[2]]


    return selected_points1, selected_points2, selected_points3




def calculate_tre(pred, target):

    kdtree1 = KDTree(np.array(pred.points))

    closest_point_index1= kdtree1.query(target[0])[1]
    closest_point_index2= kdtree1.query(target[1])[1]
    closest_point_index3= kdtree1.query(target[2])[1]


    closest_distance1 = np.linalg.norm(np.array(pred.points)[closest_point_index1] - target[0])
    closest_distance2 = np.linalg.norm(np.array(pred.points)[closest_point_index2] - target[1])
    closest_distance3 = np.linalg.norm(np.array(pred.points)[closest_point_index3] - target[2])



    return closest_distance1, closest_distance2, closest_distance3







def run_net(args, config):
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                              builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts)


    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
    else:
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch - 1)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Completion3D' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()


            num_iter += 1

            ret = base_model(partial)

            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt, epoch)

            _loss = sparse_loss + dense_loss
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10),
                                               norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item() , dense_loss.item() ])
            else:
                losses.update([sparse_loss.item() , dense_loss.item() ])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']))

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()


        print('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]))

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, args, config)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args)
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                    args)


def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, args, config, logger=None):
    print(f"[VALIDATION] Start validating epoch {epoch}")
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)  # bs is 1

    interval = n_samples // 10

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Completion3D' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()


            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[-1]

            sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
            dense_loss_l1 = ChamferDisL1(dense_points, gt)
            dense_loss_l2 = ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() , sparse_loss_l2.item() , dense_loss_l1.item() ,
                                dense_loss_l2.item() ])

            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(dense_points, gt)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)

            if (idx + 1) % interval == 0:
                print('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                          (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()],
                           ['%.4f' % m for m in _metrics]))
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]))

        if args.distributed:
            torch.cuda.synchronize()

    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print('============================ TEST RESULTS ============================')
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print(msg)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print(msg)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print(msg, logger)



    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1 / 4,
    'median': 1 / 2,
    'hard': 3 / 4
}


def test_net(args, config):

    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)
    ckpt_dir=args.ckpts
    fold = ckpt_dir.split('/')[1].split("_")[-1][-1]
    print(fold)


    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()


    base_model.eval()  # set model to eval mode



    iou_list = []
    cd_list = []
    f1score_list = []
    emd_list = []
    levels = []

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]
            _,specimen,_, recording,_, cam,_, frame, level = model_id.split("_")

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()

                ret = base_model(partial)
                coarse_points = ret[0].detach().cpu().numpy()
                partial_np = partial.detach().cpu().numpy()
                gt = data[1]
                _metrics = Metrics.get(ret[-1],gt.cuda(),require_emd=False)
                dense_points = ret[-1].detach().cpu().numpy()


                pred_pcd = o3d.geometry.PointCloud()
                pred_pcd.points = o3d.utility.Vector3dVector(dense_points[0,:,:])
                pred_pcd.paint_uniform_color([174/255, 198/255, 207/255])


                input_pcd = o3d.geometry.PointCloud()
                input_pcd.points = o3d.utility.Vector3dVector(partial_np[0, :, :])
                input_pcd.paint_uniform_color([0, 1, 0])


                gt_pcd = o3d.geometry.PointCloud()
                gt_pcd.points = o3d.utility.Vector3dVector(gt[0, :, :])
                gt_pcd.paint_uniform_color([119/255, 221/255, 119/255])

                o3d.visualization.draw_geometries([pred_pcd, gt_pcd])


                input = ret[-1]/1000
                output =gt/1000
                cd,_, fscore = calc_cd(input.cuda(), output.cuda(),calc_f1 = True)
                emd = calc_emd(input, output)
                iou_list.append(calculate_iou(pred_pcd, gt_pcd))
                cd_list.append(cd.detach().cpu().numpy()[0])
                f1score_list.append(fscore.detach().cpu().numpy()[0])
                emd_list.append(emd.detach().cpu().numpy()[0])
                levels.append(level)


    # Sample data
    data = {
        'IoU': iou_list,
        'CD': cd_list,
        'F1': f1score_list,
        'EMD': emd_list,
        'Level': levels

    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv('fold{}_val.csv'.format(fold), index=False)

    print("DataFrame saved ")


def test_pointnet(args, config):
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts)
    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        raise NotImplementedError()

    base_model.eval()  # set model to eval mode

    fold = int(args.ckpts.split("/")[2].split("_")[-1][-1])
    if fold ==0:
        fold = 10
    print(fold)

    model = "/home/aidana/PycharmProjects/RGBDSeg/PointNet/checkpoints/fold_{}/ckpt-best.pth".format(fold)
    dataset = "/home/aidana/Documents/PointNet_data"
    stls_dir = "/home/aidana/Documents/stls_transformed"



    iou_list = []
    cd_list = []
    f1score_list = []
    emd_list = []
    levels = []

    iou_seg_list =[]
    dice_seg_list = []
    accuracy_seg_list = []
    cd_top_list = []
    cd_bottom_list = []

    val_dataset = SpineDepthDataset(
        root=dataset,
        fold=fold,
        num_channels=6,
        class_choice=["spine"],
        split="val",
        data_augmentation=False)



    for idx in range(len(val_dataset)):
        print("Processing ", idx, "th point cloud from", len(val_dataset), "point clouds.")
        torch.cuda.empty_cache()
        data, gt, filepath, dist, trans = val_dataset[idx]
        print(filepath)
        point = data[:, :3]

        filename = filepath.split("/")[-1][:-4]
        _, specimen, _, recording, _, cam, _, frame = filename.split("_")
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




        state_dict = torch.load(model, map_location=torch.device('cpu'))
        classifier = PointNetDenseCls(k=6, number_channels=6)
        classifier.load_state_dict(state_dict)
        classifier.eval()

        data = data.transpose(1, 0)
        data = Variable(data.view(1, data.size()[0], data.size()[1]))
        pred, _, _ = classifier(data)
        pred_choice = pred.data.max(2)[1].cpu().numpy()


        metrics = compute_metrics(pred_choice[0], gt.numpy(), 6)




        L1_ind = np.where(pred_choice[0]== 1)[0]
        L2_ind = np.where( pred_choice[0]== 2)[0]
        L3_ind = np.where(pred_choice[0] == 3)[0]
        L4_ind = np.where(pred_choice[0] == 4)[0]
        L5_ind = np.where(pred_choice[0] == 5)[0]
        background = np.where(pred_choice[0] == 0)[0]

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




        segs = [L1_points,L2_points, L3_points, L4_points, L5_points]


        L1_stl_dwp = L1_stl.sample_points_uniformly(4096)
        L2_stl_dwp = L2_stl.sample_points_uniformly(4096)
        L3_stl_dwp = L3_stl.sample_points_uniformly(4096)
        L4_stl_dwp = L4_stl.sample_points_uniformly(4096)
        L5_stl_dwp = L5_stl.sample_points_uniformly(4096)





        gts = [L1_stl_dwp, L2_stl_dwp, L3_stl_dwp, L4_stl_dwp, L5_stl_dwp]
        input_pcds = [L1_pcd, L2_pcd, L3_pcd, L4_pcd, L5_pcd]
        stls = [L1_stl,L2_stl, L3_stl, L4_stl, L5_stl]


        with torch.no_grad():
            for el,seg in enumerate(segs):
                if len(seg) > 2048:
                    print(el)
                    choice = np.random.choice(len(seg), 2048, replace=True)
                    seg = seg[choice]

                    partial = torch.from_numpy(seg).float().unsqueeze(0).cuda()
                    ret = base_model(partial)
                    dense_points = ret[-1].detach().cpu().numpy()
                    sparse_points = ret[0].detach().cpu()
                    partial = partial.detach().cpu().numpy()
                    gt = gts[el]
                    gt.paint_uniform_color([0,1,0])

                    pred = o3d.geometry.PointCloud()
                    pred.points = o3d.utility.Vector3dVector(dense_points[0,:,:])
                    pred.paint_uniform_color([1,0,1])

                    stl = stls[el]

                    poisson_reconstruction_visualization(stl, pred)




                    input = ret[-1]/ 1000
                    output = torch.Tensor(gt.points).unsqueeze(0) / 1000
                    cd, _, fscore = calc_cd(input.cuda(), output.cuda(), calc_f1=True)
                    emd = calc_emd(input, output)
                    if el == 0:
                        bbox_gt = gt.get_oriented_bounding_box()

                        obb_matrix = bbox_gt.R
                        orientation_matrix = obb_matrix[:3, :3]

                        axis_x = orientation_matrix[:, 0]
                        axis_y = orientation_matrix[:, 1]
                        axis_z = orientation_matrix[:, 2]
                        split_normal = axis_x

                    cd_top, cd_bottom = halve_point_clouds(input_pcds[el], pred, gt, split_normal)
                    cd_top_list.append(cd_top[0])
                    cd_bottom_list.append(cd_bottom[0])

                    iou_list.append(calculate_iou(pred, gt))
                    cd_list.append(cd.detach().cpu().numpy()[0])
                    f1score_list.append(fscore.detach().cpu().numpy()[0])
                    emd_list.append(emd.detach().cpu().numpy()[0])
                    levels.append(el+1)
                    iou_seg_list.append(metrics['IoU'][el+1])
                    dice_seg_list.append(metrics['Dice'][el+1])
                    accuracy_seg_list.append(metrics['Accuracy'][el+1])






    data = {

        'IoU_seg': iou_seg_list,
        'Dice': dice_seg_list,
        'Accuracy': accuracy_seg_list,
        'IoU': iou_list,
        'CD': cd_list,
        'F1': f1score_list,
        'EMD': emd_list,
        'Level': levels,
        'CD_top': cd_top_list,
        'CD_bottom': cd_bottom_list,


    }

    df = pd.DataFrame(data)

    df.to_csv('fold{}_test.csv'.format(fold), index=False)

    print("DataFrame saved ")


def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())

    category_metrics = dict()
    n_samples = len(test_dataloader)  # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[-1]

                sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                dense_loss_l1 = ChamferDisL1(dense_points, gt)
                dense_loss_l2 = ChamferDisL2(dense_points, gt)

                test_losses.update(
                    [sparse_loss_l1.item() , sparse_loss_l2.item() , dense_loss_l1.item() ,
                     dense_loss_l2.item() ])

                _metrics = Metrics.get(dense_points, gt, require_emd=False)
                #test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]),
                          torch.Tensor([-1, 1, 1]),
                          torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]),
                          torch.Tensor([-1, -1, -1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points=item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[-1]

                    sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 = ChamferDisL1(dense_points, gt)
                    dense_loss_l2 = ChamferDisL2(dense_points, gt)

                    test_losses.update(
                        [sparse_loss_l1.item() , sparse_loss_l2.item() , dense_loss_l1.item() ,
                         dense_loss_l2.item() ])

                    _metrics = Metrics.get(dense_points, gt)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[-1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx + 1) % 200 == 0:
                print('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                          (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()],
                           ['%.4f' % m for m in _metrics]))
        if dataset_name == 'KITTI':
            return
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]))

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print(msg)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print(msg)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print(msg)
    return

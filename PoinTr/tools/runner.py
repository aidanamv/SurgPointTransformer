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
from PointNet.dataloader import SpineDepthDataset
from PointNet.model import PointNetDenseCls
import open3d as o3d
import numpy as np
import pandas as pd
import pyvista as pv
import re
from tools.utils.model_utils import calc_cd, calc_emd
from scipy.spatial import KDTree

import pandas as pd
import numpy as np



def calculate_snr(gt_cloud, pred_cloud):
    # Convert Open3D point clouds to numpy arrays
    clean_points = np.asarray(gt_cloud)
    noisy_points = np.asarray(pred_cloud)

    # Compute signal power (mean squared distance of clean points from the origin)
    signal_power = np.mean(np.linalg.norm(clean_points, axis=1) ** 2)

    # Compute noise power (mean squared error between noisy and clean points)
    noise_power = np.mean(np.linalg.norm(clean_points - noisy_points, axis=1) ** 2)

    # Compute SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def run_net(args, config):
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                              builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)


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
            print(ret[0].size())
            print(ret[1].size())

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









def test(args, config):

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=None)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    base_model.eval()  # set model to eval mode
    snr_list =[]
    cd_list =[]
    f1score_list =[]
    emd_list = []
    levels = []
    with (torch.no_grad()):
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            level = model_ids[0][-1]
            levels.append(int(level))
            partial = data[0].cuda()
            gt = data[1].cuda()
            gt_np =gt.squeeze(0).detach().cpu().numpy()
            input_np = partial.squeeze(0).detach().cpu().numpy()
            ret = base_model(partial)
            pred = ret[-1]
            pred_np = ret[-1].squeeze(0).detach().cpu().numpy()
            snr = calculate_snr(gt_np, pred_np)
            snr_list.append(snr)
            cd, _, fscore = calc_cd(gt/1000, pred/1000, calc_f1=True)
            emd = calc_emd(gt/1000, pred/1000)
            cd_list.append(cd.detach().cpu().numpy()[0])
            f1score_list.append(fscore.detach().cpu().numpy()[0])
            emd_list.append(emd.detach().cpu().numpy()[0])

        data = {
            'CD': cd_list,
            'F1': f1score_list,
            'EMD': emd_list,
            'SNR': snr_list,
            "Level": levels

        }

        df = pd.DataFrame(data)
        df.to_csv('results.csv', index=False)






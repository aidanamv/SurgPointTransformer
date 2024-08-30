import argparse
import torch.utils.data
from torch.autograd import Variable
from dataloader import SpineDepthDataset
from model import PointNetDenseCls
import pandas as pd
from tools import *


parser = argparse.ArgumentParser()
opt = parser.parse_args()
dataset = "/home/aidana/Documents/PointNet_data"
stls_dir = "/home/aidana/Documents/stls_transformed"

for fold in range(2,11):
    print(fold)
    model ="./checkpoints/three_channel/fold_{}/ckpt-best.pth".format(fold)

    val_dataset = SpineDepthDataset(
        root=dataset,
        fold = fold,
        num_channels=3,
        class_choice=["spine"],
        split="val",
        data_augmentation=False)




    iou_L1_total = []
    iou_L2_total = []
    iou_L3_total = []
    iou_L4_total = []
    iou_L5_total = []
    iou_background_total = []


    acc_L1_total = []
    acc_L2_total = []
    acc_L3_total = []
    acc_L4_total = []
    acc_L5_total = []
    acc_background_total = []

    dice_L1_total = []
    dice_L2_total = []
    dice_L3_total = []
    dice_L4_total = []
    dice_L5_total = []
    dice_background_total = []




    for idx in range(len(val_dataset)):
        data, gt,filepath,dist,trans = val_dataset[idx]
        point = data[:, :3]


        filename = filepath.split("/")[-1][:-4]
        print(filename)
        _,specimen, _, recording, _, cam,_, frame = filename.split("_")
        point_np = point.numpy()

        state_dict = torch.load(model,map_location=torch.device('cpu'))
        classifier = PointNetDenseCls(k= 6, number_channels=3)
        classifier.load_state_dict(state_dict)
        classifier.eval()

        data = data.transpose(1, 0)
        data = Variable(data.view(1, data.size()[0], data.size()[1]))
        pred, _, _ = classifier(data)
        pred_choice = pred.data.max(2)[1].cpu().numpy()

        filtered_point_cloud,filtered_pred,filtered_gt = filter_point_clouds(point_np, pred_choice[0],gt)

        filtered_point_cloud = filtered_point_cloud * dist + trans

        metrics = compute_metrics(pred_choice[0], gt.numpy(), 6)

        acc_background_total.append(metrics['Accuracy'][0])
        acc_L1_total.append(metrics['Accuracy'][1])
        acc_L2_total.append(metrics['Accuracy'][2])
        acc_L3_total.append(metrics['Accuracy'][3])
        acc_L4_total.append(metrics['Accuracy'][4])
        acc_L5_total.append(metrics['Accuracy'][5])

        iou_background_total.append(metrics['IoU'][0])
        iou_L1_total.append(metrics['IoU'][1])
        iou_L2_total.append(metrics['IoU'][2])
        iou_L3_total.append(metrics['IoU'][3])
        iou_L4_total.append(metrics['IoU'][4])
        iou_L5_total.append(metrics['IoU'][5])

        dice_background_total.append(metrics['Dice'][0])
        dice_L1_total.append(metrics['Dice'][1])
        dice_L2_total.append(metrics['Dice'][2])
        dice_L3_total.append(metrics['Dice'][3])
        dice_L4_total.append(metrics['Dice'][4])
        dice_L5_total.append(metrics['Dice'][5])



    data = {

        'acc_background': acc_background_total,
        'acc_L1': acc_L1_total,
        'acc_L2': acc_L2_total,
        'acc_L3': acc_L3_total,
        'acc_L4': acc_L4_total,
        'acc_L5': acc_L5_total,
        'iou_background': iou_background_total,
        'iou_L1': iou_L1_total,
        'iou_L2': iou_L2_total,
        'iou_L3': iou_L3_total,
        'iou_L4': iou_L4_total,
        'iou_L5': iou_L5_total,
        'dice_background': dice_background_total,
        'dice_L1': dice_L1_total,
        'dice_L2': dice_L2_total,
        'dice_L3': dice_L3_total,
        'dice_L4': dice_L4_total,
        'dice_L5': dice_L5_total,


    }

    df = pd.DataFrame(data)

    df.to_csv('../results/three_channel_segmentation_results_fold_{}.csv'.format(fold), index=False)

    print("DataFrame saved for fold {}".format(fold))
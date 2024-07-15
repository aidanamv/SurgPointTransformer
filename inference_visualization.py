import argparse
import os
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from dataloader import ShapeNetDataset
from model import PointNetDenseCls
import numpy as np
import pyvista as pv



parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="./checkpoints/fold_1/ckpt-best.pth", help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='/Users/aidanamassalimova/Documents/PointNet_data', help='dataset path')
parser.add_argument('--class_choice', type=str, default='spine', help='class choice')
parser.add_argument('--fold', type=int, default=1)

opt = parser.parse_args()
print(opt)

stls_dir = "/Volumes/SpineDepth/PoinTr_dataset/stls_transformed"

val_dataset = ShapeNetDataset(
    root=opt.dataset,
    fold = opt.fold,
    class_choice=[opt.class_choice],
    split='val',
    data_augmentation=False)




for idx in range(len(val_dataset)):
    print("model %d/%d" % (idx, len(val_dataset)))
    point, gt,filepath,dist,trans = val_dataset[idx]
    filename = filepath.split("/")[-1][:-4]
    print(filename)
    _, specimen, _, recording, _, cam, _, frame = filename.split("_")
    point_np = point.numpy()

    state_dict = torch.load(opt.model,map_location=torch.device('cpu'))
    classifier = PointNetDenseCls(k= 6)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    point = point.transpose(1, 0)

    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _ = classifier(point)
    pred_choice = pred.data.max(2)[1]

    if len(np.unique(pred_choice)) > 0:
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

        point_np = point_np * dist + trans

        L1_ind = np.where(pred_choice[0] == 1)
        L2_ind = np.where(pred_choice[0] == 2)
        L3_ind = np.where(pred_choice[0] == 3)
        L4_ind = np.where(pred_choice[0] == 4)
        L5_ind = np.where(pred_choice[0] == 5)
        rest_ind = np.where(pred_choice[0] == 0)




        pv.global_theme.allow_empty_mesh = True

        p = pv.Plotter()

        p.add_points(point_np[L1_ind[0],:],color = "red")
        p.add_points(point_np[L2_ind[0],:],color = "blue")
        p.add_points(point_np[L3_ind[0],:],color = "yellow")
        p.add_points(point_np[L4_ind[0],:],color = "green")
        p.add_points(point_np[L5_ind[0],:],color = "brown")
        p.add_points(point_np[rest_ind[0],:],color = "black")
        p.add_mesh(L1_stl)
        p.add_mesh(L2_stl)
        p.add_mesh(L3_stl)
        p.add_mesh(L4_stl)
        p.add_mesh(L5_stl)


        p.show()



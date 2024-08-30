import torch
import numpy as np
import torch.utils.data as data
import open3d as o3d
import os
import json
import random

class ShapeNetDataset(data.Dataset):
    def __init__(self, fold=2,train=True, npoints=2048):
        self.input_path = "/home/aidana/PycharmProjects/RGBDSeg/PoinTr/data/PointTr_data/partial"
        self.gt_path = "/home/aidana/PycharmProjects/RGBDSeg/PoinTr/data/PointTr_data/complete"
        self.fold = fold
        # Load the JSON data from a file
        with open('../data/PointTr_data/fold_{}.json'.format(fold), 'r') as file:
            data = json.load(file)
            # Extract the relevant data
        for item in data:
            train_records = item['train']
            val_records = item['val']
        if train:
            self.filenames = train_records
        else:
            self.filenames = val_records

        self.npoints = npoints
        self.train = train

        self.len = len(self.filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input_data = o3d.io.read_point_cloud(os.path.join(self.input_path, self.filenames[index],"00.pcd"))
        gt_data = o3d.io.read_point_cloud(os.path.join(self.gt_path, self.filenames[index]+".pcd"))
        input_np = np.array(input_data.points)/1000
        gt_np = np.array(gt_data.points)/1000
        choice = np.random.choice(len(input_np), self.npoints, replace=True)
        partial = torch.from_numpy(input_np[choice])
        choice = np.random.choice(len(gt_np), self.npoints, replace=True)
        complete = torch.from_numpy(gt_np[choice])
        assert partial.size(0) == self.npoints
        assert complete.size(0) == self.npoints

        return  partial, complete

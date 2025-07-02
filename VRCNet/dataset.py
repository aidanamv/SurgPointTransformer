import torch
import numpy as np
import torch.utils.data as data
import open3d as o3d
import os
import json
import random

import os
import json
import torch
import numpy as np
import open3d as o3d
from torch.utils import data


class ShapeNetDataset(data.Dataset):
    def __init__(self, dataset, fold=2, train=True, npoints=2048):
        self.dataset = dataset
        self.input_path = os.path.join(dataset, "partial")
        self.gt_path = os.path.join(dataset, "complete")
        self.fold = fold
        self.train = train
        self.npoints = npoints

        json_path = os.path.join(dataset_root, f"fold_{fold}.json")
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Extract the relevant train/val lists
        for item in data:
            train_records = item['train']
            val_records = item['val']

        self.filenames = train_records if train else val_records
        self.len = len(self.filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        filename = self.filenames[index]
        input_pcd_path = os.path.join(self.input_path, filename, "00.pcd")
        gt_pcd_path = os.path.join(self.gt_path, filename + ".pcd")

        input_data = o3d.io.read_point_cloud(input_pcd_path)
        gt_data = o3d.io.read_point_cloud(gt_pcd_path)

        input_np = np.array(input_data.points) / 1000
        gt_np = np.array(gt_data.points) / 1000

        # Sample points
        choice = np.random.choice(len(input_np), self.npoints, replace=True)
        partial = torch.from_numpy(input_np[choice]).float()

        choice = np.random.choice(len(gt_np), self.npoints, replace=True)
        complete = torch.from_numpy(gt_np[choice]).float()

        return partial, complete

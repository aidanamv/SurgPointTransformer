import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from dataloader import SpineDepthDataset
from model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='seg', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', default ="/home/aidana/Documents/PointNet_data", type=str,  help="dataset path")
    parser.add_argument('--class_choice', type=str, default='spine', help="class_choice")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--fold', default=2)
    parser.add_argument('--channels', default=3)

    opt = parser.parse_args()

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    num_channels =  opt.channels

    dataset = SpineDepthDataset(
        root=opt.dataset,
        fold = opt.fold,
        num_channels=num_channels,
        classification=False,
        class_choice=[opt.class_choice])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    test_dataset = SpineDepthDataset(
        root=opt.dataset,
        fold = opt.fold,
        num_channels=num_channels,
        classification=False,
        class_choice=[opt.class_choice],
        split='val',
        data_augmentation=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))
    num_classes = dataset.num_seg_classes
    print('classes', num_classes)
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    blue = lambda x: '\033[94m' + x + '\033[0m'

    classifier = PointNetDenseCls(k=num_classes, number_channels =num_channels,feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_batch = len(dataset) / opt.batchSize

    for epoch in range(opt.nepoch):
        scheduler.step()
        for i, data in enumerate(dataloader, 0):
            points, target,_,_,_ = data
            points = points.transpose(2, 1)
            print(points.size())
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)

            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target,_,_,_ = data
                points = points.transpose(2, 1)
             #   points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0]
                loss = F.nll_loss(pred, target)
                loss = F.cross_entropy(pred,target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))

        torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

    shape_ious = []
    for i,data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(2)[1]

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))



import argparse
import os
import torch
import torch.utils.data
from torch.autograd import Variable
import pandas as pd
from dataloader import SpineDepthDataset
from model import PointNetDenseCls
from utils import compute_metrics, filter_point_clouds


def evaluate_fold(fold, dataset_root, stl_root, model_path, output_dir, channels):
    print(f"\n📂 Evaluating fold {fold}...")

    val_dataset = SpineDepthDataset(
        root=dataset_root,
        fold=fold,
        num_channels=channels,
        class_choice=["spine"],
        split="val",
        data_augmentation=False
    )

    classifier = PointNetDenseCls(k=6, feature_transform=False)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()

    metrics_acc, metrics_iou, metrics_dice = [[] for _ in range(6)], [[] for _ in range(6)], [[] for _ in range(6)]

    for idx in range(len(val_dataset)):
        data, gt, filepath, dist, trans = val_dataset[idx]
        filename = os.path.basename(filepath)[:-4]
        print(f"🧾 {filename}")

        point = data[:, :channels]
        point_np = point.numpy()
        data = Variable(data.transpose(1, 0).unsqueeze(0))

        pred, _, _ = classifier(data)
        pred_choice = pred.data.max(2)[1].cpu().numpy()[0]

        filtered_point_cloud, filtered_pred, filtered_gt = filter_point_clouds(point_np, pred_choice, gt)
        filtered_point_cloud = filtered_point_cloud * dist + trans

        metrics = compute_metrics(pred_choice, gt.numpy(), num_classes=6)

        for i in range(6):
            metrics_acc[i].append(metrics['Accuracy'][i])
            metrics_iou[i].append(metrics['IoU'][i])
            metrics_dice[i].append(metrics['Dice'][i])

    df = pd.DataFrame({
        'acc_background': metrics_acc[0], 'acc_L1': metrics_acc[1], 'acc_L2': metrics_acc[2],
        'acc_L3': metrics_acc[3], 'acc_L4': metrics_acc[4], 'acc_L5': metrics_acc[5],
        'iou_background': metrics_iou[0], 'iou_L1': metrics_iou[1], 'iou_L2': metrics_iou[2],
        'iou_L3': metrics_iou[3], 'iou_L4': metrics_iou[4], 'iou_L5': metrics_iou[5],
        'dice_background': metrics_dice[0], 'dice_L1': metrics_dice[1], 'dice_L2': metrics_dice[2],
        'dice_L3': metrics_dice[3], 'dice_L4': metrics_dice[4], 'dice_L5': metrics_dice[5],
    })

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"three_channel_segmentation_results_fold_{fold}.csv")
    df.to_csv(result_path, index=False)
    print(f"✅ Saved results to {result_path}")


def parse_args():
    epilog_text = """
Example directory structure:

├── dataset_root/
│   ├── specimen_01/
│   │   └── frame_0001.pcd
│   └── ...
├── stl_root/
│   └── specimen_01/
│       └── vertebra_L1.stl
├── checkpoints/
│   └── three_channel/
│       ├── fold_2/ckpt-best.pth
│       ├── fold_3/ckpt-best.pth
│       └── ...
"""

    parser = argparse.ArgumentParser(
        description="Evaluate PointNet segmentation on SpineDepth dataset",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to the dataset root')
    parser.add_argument('--stl_root', type=str, required=True, help='Path to STL files')
    parser.add_argument('--checkpoints_dir', type=str, required=True, help='Directory containing checkpoint folders')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for CSV files')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(2, 11)), help='Folds to evaluate (e.g. 2 3 4)')
    parser.add_argument('--channels', default=3)

    return parser.parse_args()


def main():
    args = parse_args()

    for fold in args.folds:
        ckpt_path = os.path.join(args.checkpoints_dir, f"fold_{fold}", "ckpt-best.pth")
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint missing for fold {fold}: {ckpt_path}")
            continue
        evaluate_fold(fold, args.dataset_root, args.stl_root, ckpt_path, args.output_dir, args.channels)


if __name__ == '__main__':
    main()

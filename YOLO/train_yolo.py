import argparse
from ultralytics import YOLO

def main(opt):
    # Load model from YAML config (e.g., yolov10n.yaml)
    model = YOLO(opt.model)

    # Start training
    model.train(
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=opt.device,
        name=opt.name,
        workers=opt.workers
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv10 model using Ultralytics")
    parser.add_argument("--model", type=str, default="yolov10n.yaml", help="Model config file or pretrained checkpoint")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per device")
    parser.add_argument("--device", type=str, default="0", help="CUDA device to use, e.g., 0 or 'cpu'")
    parser.add_argument("--name", type=str, default="exp", help="Name of the training run (for saving results)")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers")

    opt = parser.parse_args()
    main(opt)

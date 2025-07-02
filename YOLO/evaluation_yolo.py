from ultralytics import YOLO

ckpt_path = "/home/aidana/Documents/YOLO/detect/fold_10/weights/best.pt"
# Load the model
model = YOLO(ckpt_path)

# Run the evaluation
results = model.val(data="dataset.yaml")
for result in results:
    # Print specific metrics
    print("F1 score:", results.box.f1)
    print("Mean precision:", results.box.mp)
    print("Mean average precision:", results.box.map)
    print("Recall:", results.box.r)
    print("Mean recall:", results.box.mr)
    print("Precision:", results.box.p)
    print("Precision values:", results.box.prec_values)
import argparse
from ultralytics import YOLO

def main(opt):
    # Load the YOLO model
    model = YOLO(opt.model)

    # Run validation
    results = model.val(data=opt.data)

    # Print specific evaluation metrics
    print("📊 Evaluation Results:")
    print("F1 Score:", results.box.f1)
    print("Mean Precision:", results.box.mp)
    print("Mean Average Precision (mAP@0.5):", results.box.map)
    print("Recall:", results.box.r)
    print("Mean Recall:", results.box.mr)
    print("Precision:", results.box.p)
    print("Precision values per class:", results.box.prec_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model using Ultralytics")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml file")

    opt = parser.parse_args()
    main(opt)

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

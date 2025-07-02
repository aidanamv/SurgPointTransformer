# SurgPointTransformer

A transformer-based pipeline for vertebra shape completion using RGB-D imaging. This method enables 3D spine reconstruction in surgical settings without ionizing radiation by leveraging sparse RGB-D data and transformer architectures.

## 🔬 Overview

SurgPointTransformer is a novel approach to reconstruct complete vertebral anatomy from partial RGB-D data captured intraoperatively. The pipeline:

* Segments visible spinal regions from RGB-D images.
* Performs vertebra-wise shape completion using a transformer-based network.
* Outputs high-quality 3D mesh reconstructions without relying on radiation-based imaging like CT or fluoroscopy.

> 📌 This project was developed at the University Hospital Balgrist, Zurich.

## 📷 Motivation

Intraoperative imaging methods like CT and fluoroscopy expose patients and clinicians to radiation. SurgPointTransformer addresses this by enabling radiation-free 3D shape reconstruction using RGB-D cameras, inspired by how surgeons mentally reconstruct anatomy from partial exposure.

## 🧠 Key Features

* 🧠 Transformer-based shape completion (built on AdaPointr)
* 📦 Integrated vertebra segmentation using YOLOv8 + SAM + PointNet++
* 🔄 No registration to preoperative data required
* 💻 Real-time capable pipeline (GPU-accelerated)
* 📊 Evaluated on SpineDepth dataset (ex vivo RGB-D surgeries)

## 🏗️ Pipeline

```
RGB-D Images → Spine Localization → Vertebra Segmentation → Shape Completion → 3D Mesh
```

### Components:

* **Localization**: YOLOv8 trained on RGB images
* **Segmentation**: SAM + PointNet++ for vertebra-level point clouds
* **Shape Completion**: SurgPointTransformer using geometric transformer blocks
* **3D Reconstruction**: Poisson surface reconstruction for mesh generation

## 📊 Results

| Metric                 | SurgPointTransformer | VRCNet (Baseline) |
| ---------------------- | -------------------- | ----------------- |
| Chamfer Distance (mm)  | **5.39**             | 6.17              |
| F-score                | 0.85                 | **0.86**          |
| Earth Mover’s Distance | **11.00**            | 20.00             |
| SNR (dB)               | **22.90**            | 22.89             |

> 🎯 Significantly better shape fidelity and robustness to noise and occlusions compared to state-of-the-art.

## 🗃️ Dataset

We used:

* **SpineDepth Dataset**: 9 ex vivo spinal surgery cases with RGB-D recordings and paired CT ground-truths
* **CTSpine1K (synthetic)**: For ablation studies on noise and input-to-output ratios

## 🧪 Ablation Studies

* Evaluated on noise levels (low, medium, high)
* Varying visibility of vertebral anatomy (10% to 40%)
* Demonstrated robustness to partial input and sensor noise

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/SurgPointTransformer.git
cd SurgPointTransformer
pip install -r requirements.txt
```

## 🚀 Training & Inference

```bash
# Train YOLOv8 for localization
python train_yolo.py \
  --model yolov10n.yaml \
  --data dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --name spine_detection_v10n


# visualize YOLO results with ground-truth stls
python visualize_yolo_stl.py \
  --model ./weights/best.pt \
  --specimen 2 \
  --rgb_dir ./data/fold_2/test/images \
  --depth_root ./data/fold_2/test/depth  \
  --stl_root ./data/fold_2/test/stls \
  --calib_dir ./data/fold_2/calib/SN10027879.conf
  
  
# evaluate YOLO results
python evaluate_yolo.py \
  --model ./weights/best.pt \
  --data dataset.yaml



# Train PointNet++ for segmentation
python train_segmentation.py \
  --dataset ./data/PointNet_data \
  --outf ./checkpoints/ \
  --fold 2 \
  --batchSize 32 \
  --nepoch 25 \
  --channels 3 \
  --feature_transform


# Run inference Example of PointNet++ for segmentation
python evaluate_segmentation.py \
  --dataset_root ./data/PointNet_data \
  --stl_root ./data/stls \
  --checkpoints_dir ./checkpoints \
  --output_dir ./results \
  --channels 3 \
  --folds 2 


# Train SurgPointTransformer
python train_transformer.py

# Run inference
python inference.py --input path/to/rgbd
```

## 📁 Project Structure

```
├── data/                # SpineDepth RGB-D data & GT meshes
├── models/              # YOLOv8, PointNet++, Transformer
├── scripts/             # Training/inference scripts
├── utils/               # Metrics, visualizations, point cloud tools
└── README.md
```



## 📜 Citation

```bibtex
@article{massalimova2025surgpointtransformer,
  title={SurgPointTransformer: transformer-based vertebra shape completion using RGB-D imaging},
  author={Massalimova, Aidana and Liebmann, Florentin and Jecklin, Sascha and Carrillo, Fabio and Farshad, Mazda and Fürnstahl, Philipp},
  journal={Computer Assisted Surgery},
  volume={30},
  number={1},
  pages={2511126},
  year={2025},
  publisher={Taylor & Francis},
  doi={10.1080/24699322.2025.2511126}
}
```

## 🛡️ License

This project is licensed under the **Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)** license.


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

### 🟢 **Train YOLOv8 / v10 for Localization**

```bash
python YOLO/train_yolo.py \
  --model <path/to/yolo_model.yaml> \
  --data <path/to/dataset.yaml> \
  --epochs <num_epochs> \
  --imgsz <image_size> \
  --batch <batch_size> \
  --device <cuda_device_id> \
  --name <run_name>
```

---

### 🟢 **Visualize YOLO Results with STL Meshes**

```bash
python YOLO/visualize_yolo_stl.py \
  --model <path/to/best.pt> \
  --specimen <specimen_id> \
  --rgb_dir <path/to/rgb/images> \
  --depth_root <path/to/depth/maps> \
  --stl_root <path/to/stl/meshes> \
  --calib_dir <path/to/camera_calib.conf>
```

---

### 🟢 **Evaluate YOLO Predictions**

```bash
python YOLO/evaluate_yolo.py \
  --model <path/to/best.pt> \
  --data <path/to/dataset.yaml>
```

---

### 🟢 **Train PointNet++ for Segmentation**

```bash
python PointNet/train_segmentation.py \
  --dataset <path/to/PointNet_data> \
  --outf <output_dir_for_checkpoints> \
  --fold <fold_number> \
  --batchSize <batch_size> \
  --nepoch <num_epochs> \
  --channels <num_input_channels> \
  --feature_transform
```

---

### 🟢 **Evaluate PointNet++ Segmentation**

```bash
python PointNet/evaluate_segmentation.py \
  --dataset_root <path/to/PointNet_data> \
  --stl_root <path/to/stl_root> \
  --checkpoints_dir <path/to/checkpoints_dir> \
  --output_dir <path/to/save_results> \
  --channels <num_input_channels> \
  --folds <fold_number>
```

---

### 🟢 **Test VRCNet**

```bash
python VRCNet/test_vrcnet.py \
  --config <path/to/vrcnet_config.yaml> \
  --fold <fold_number> \
  --model <model_file_name_or_tag> \
  --checkpoints <path/to/ckpt-best.pth> \
  --dataset <path/to/data_dir>
```

---

### 🟢 **Train VRCNet**

```bash
python VRCNet/train_vrcnet.py \
  --config <path/to/train_config.yaml> \
  --dataset <path/to/data_dir> \
  --fold <fold_number>
```

---

### 🟢 **Train SurgPointTransformer**

```bash
python PoinTr/main.py \
  --config <path/to/PointTransformer_config.yaml>
```

---

### 🟢 **Test SurgPointTransformer**

```bash
python PoinTr/main.py \
  --test \
  --config <path/to/PointTransformer_config.yaml> \
  --ckpts <path/to/ckpt-best.pth>
```

---




## 📁 Project Structure

```
├── YOLO/                            # YOLOv8/YOLOv10 training, inference, evaluation
│   ├── train_yolo.py                # Train YOLO for localization
│   ├── visualize_yolo_stl.py       # Visualize YOLO boxes + STL meshes
│   ├── evaluate_yolo.py            # Evaluate detection results
    ├── dataset.yaml                 #config file for training 
│   └── weights/                    # Trained YOLO models (.pt)
│
├── PointNet/                        # PointNet++ segmentation
│   ├── train_segmentation.py       # Train segmentation model
│   ├── evaluate_segmentation.py    # Inference + metric evaluation
│   ├── model.py                    # PointNet++ model definition
│   ├── utils.py                    # utils for loss, metrics, model
│   ├── dataloader.py               # Custom dataset loader
│   └── checkpoints/                # Saved model checkpoints
│
├── VRCNet/                          # VRCNet shape completion baseline
│   ├── test_vrcnet.py              # Evaluate VRCNet
│   ├── train_vrcnet.py             # Train VRCNet
│   ├── dataset.py                  # dataloader
│   └── cfgs/                       # VRCNet YAML configs
│
├── PoinTr/                          # SurgPointTransformer
│   ├── main.py                     # Train/test entrypoint
│   ├── tools/                      # Utils for loss, model, metrics
│   ├── models/                     # AdaPoinTr transformer models
│   ├── cfgs/                       # Training YAML configurations
│   └── experiments/                # Checkpoints and logs
│
├── requirements.txt
├── README.md

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


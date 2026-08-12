# CNN Document Scanning & Enhancement System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete deep learning pipeline for automatic document scanning and enhancement from unconstrained smartphone photos. This system performs intelligent corner detection, dynamic perspective correction, and U-Net-based image enhancement to transform casual, degraded document photos into professional, high-fidelity scans.

## 🌟 Key Features

- **Advanced Corner Detection**: Heatmap regression with Sub-pixel precision (SoftArgmax) and Scheduled Bottleneck Regularization.
- **Smart Ensemble 3.0**: Projective geometry heuristics and physical symmetry rules to prevent reward hacking during corner detection.
- **Out-Of-Distribution (OOD) Gatekeepers**: Statistical border variance and histogram analysis to gracefully handle pre-cropped or pre-scanned inputs.
- **Dynamic Perspective Correction**: GPU-accelerated differentiable warping using Kornia, maintaining natural aspect ratios via Euclidean bounds.
- **Zero-Annotation Synthetic Pipeline**: 10+ geometric and photometric degradations (e.g., 3D paper curl, cylindrical open-book warp, drop shadows) for purely data-driven training.
- **Production UI & PDF Engine**: FastAPI backend with an HTML5 interactive canvas, multi-page PDF reconstruction (`PyMuPDF`), and "Magic Ink Boost" post-processing.

---

## 📋 Table of Contents

1. [Installation](#-installation)
2. [Data Preparation](#-data-preparation)
3. [Usage: Inference & Demo](#-usage-inference--demo)
4. [Usage: Unified Evaluation](#-usage-unified-evaluation)
5. [Usage: Training](#-usage-training)
6. [Interactive Web UI](#-interactive-web-ui)
7. [Model Zoo & Benchmarks](#-model-zoo--benchmarks)
8. [Project Structure](#-project-structure)

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (Highly recommended)

### Step 1: Clone Repository
```bash
git clone [https://github.com/mahajialirezaei/CNN-Applications-Doc-Scanning-And-Enhancement.git](https://github.com/mahajialirezaei/CNN-Applications-Doc-Scanning-And-Enhancement.git)
cd CNN-Applications-Doc-Scanning-And-Enhancement

```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt

```

*(Ensure `kornia`, `fastapi`, `uvicorn`, `PyMuPDF (fitz)`, and `pytesseract` are installed for full functionality).*

---

## 📊 Data Preparation

Organize your data exactly as follows. The training pipeline relies strictly on `clean_scans` and `random_backgrounds` to generate synthetic pairs on-the-fly. The `raw` directory is explicitly reserved for evaluation.

```text
data/
├── clean_scans/             # 50 Clean target scans (1.jpg to 50.jpg)
├── random_backgrounds/      # Background images (tables, carpets, floors)
└── raw/
    ├── real_photos/         # 15 Real smartphone photos
    │   └── _annotations.coco.json  # Ground-truth corners for evaluation
    └── real_photos_scanned/ # 15 Reference CamScanner targets (for OCR benchmarks)

```

---

## 🚀 Usage: Inference & Demo

The `demo.py` script is the primary entry point for testing the pipeline on new, unseen images. It explicitly requires paths to the model checkpoints.

### Single Image Demo (Visualization)

Process a single real-world photo and generate a side-by-side comparison:

```bash
python demo.py \
    -i data/raw/real_photos/1_jpg.rf.40baed1b840a2eacea27fac3c3e26884.jpg \
    -o outputs/results/ \
    --enhancement-model checkpoints/enhancement_regularized_v2/best_model.pth \
    --corner-model checkpoints/corner_heatmap_regularized_v4/best_model.pth \
    --corner-approach heatmap \
    --visualize

```

### Batch Processing

Process an entire directory of photos automatically:

```bash
python demo.py \
    -i data/raw/real_photos/ \
    -o outputs/results/ \
    --enhancement-model checkpoints/enhancement_regularized_v2/best_model.pth \
    --corner-model checkpoints/corner_heatmap_regularized_v4/best_model.pth \
    --corner-approach heatmap \
    --batch

```

---

## 📈 Usage: Unified Evaluation

The `evaluate.py` script is designed to run rigorous benchmarks on all models. You can evaluate models on `synthetic` data (measuring PSNR, SSIM, and geometric localization errors) or `real` data (measuring Optical Character Recognition (OCR) confidence and real-world geometric accuracy).

Below is the complete reference of commands required to evaluate every iteration of the models in the registry.

### 1. Evaluating Pure Enhancement Models (Using Ground Truth Corners)

To isolate the performance of the U-Net models without the influence of corner prediction errors, use the `--use-gt-corners` flag.

**Enhancement Clean Nodropout (v1 & v2):**

```bash
# Synthetic Data (PSNR / SSIM)
python evaluate.py --dataset-type synthetic --use-gt-corners --enhancement-ckpt checkpoints/enhancement_clean_nodropout/best_model.pth
python evaluate.py --dataset-type synthetic --use-gt-corners --enhancement-ckpt checkpoints/enhancement_clean_nodropout_v2/best_model.pth

# Real Data (OCR Confidence)
python evaluate.py --dataset-type real --use-gt-corners --enhancement-ckpt checkpoints/enhancement_clean_nodropout/best_model.pth
python evaluate.py --dataset-type real --use-gt-corners --enhancement-ckpt checkpoints/enhancement_clean_nodropout_v2/best_model.pth

```

**Enhancement Regularized (v1 & v2):**

```bash
# Synthetic Data (PSNR / SSIM)
python evaluate.py --dataset-type synthetic --use-gt-corners --enhancement-ckpt checkpoints/enhancement_regularized/best_model.pth
python evaluate.py --dataset-type synthetic --use-gt-corners --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth

# Real Data (OCR Confidence)
python evaluate.py --dataset-type real --use-gt-corners --enhancement-ckpt checkpoints/enhancement_regularized/best_model.pth
python evaluate.py --dataset-type real --use-gt-corners --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth

```

---

### 2. Evaluating Corner Detection Models (Fixed Enhancement Baseline)

To measure the exact geometric impact of each corner model, we pair them with the Gold Enhancement model (`enhancement_regularized_v2`).

#### A. Corner Heatmap Models (`--task corner_heatmap`)

**Clean Nodropout Iterations (v1 to v4):**

```bash
# Synthetic Data
python evaluate.py --dataset-type synthetic --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_clean_nodropout/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type synthetic --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_clean_nodropout_v2/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type synthetic --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_clean_nodropout_v3/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type synthetic --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_clean_nodropout_v4/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth

# Real Data
python evaluate.py --dataset-type real --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_clean_nodropout/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type real --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_clean_nodropout_v2/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type real --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_clean_nodropout_v3/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type real --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_clean_nodropout_v4/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth

```

**Regularized Iterations (v1 to v4):**

```bash
# Synthetic Data
python evaluate.py --dataset-type synthetic --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_regularized/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type synthetic --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_regularized_v2/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type synthetic --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_regularized_v3/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type synthetic --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_regularized_v4/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth

# Real Data
python evaluate.py --dataset-type real --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_regularized/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type real --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_regularized_v2/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type real --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_regularized_v3/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type real --task corner_heatmap --corner-ckpt checkpoints/corner_heatmap_regularized_v4/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth

```

#### B. Corner Regression Models (`--task corner_regression`)

```bash
# Synthetic Data
python evaluate.py --dataset-type synthetic --task corner_regression --corner-ckpt checkpoints/corner_regression_clean_nodropout/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type synthetic --task corner_regression --corner-ckpt checkpoints/corner_regression_regularized/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth

# Real Data
python evaluate.py --dataset-type real --task corner_regression --corner-ckpt checkpoints/corner_regression_clean_nodropout/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth
python evaluate.py --dataset-type real --task corner_regression --corner-ckpt checkpoints/corner_regression_regularized/best_model.pth --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth

```

---

### 3. Evaluating End-to-End Joint Training Model

The `e2e_finetuned` checkpoint holds combined state dictionaries for both networks. You must supply this singular checkpoint path to both arguments.

```bash
# Synthetic Data
python evaluate.py --dataset-type synthetic --task corner_heatmap --corner-ckpt checkpoints/e2e_finetuned/best_model.pth --enhancement-ckpt checkpoints/e2e_finetuned/best_model.pth

# Real Data
python evaluate.py --dataset-type real --task corner_heatmap --corner-ckpt checkpoints/e2e_finetuned/best_model.pth --enhancement-ckpt checkpoints/e2e_finetuned/best_model.pth

```

---

## 🏋️ Usage: Training

The training module handles on-the-fly dataset generation, learning rate scheduling, and dynamic dropout injection.

**1. Train Enhancement Network (Clean Baseline):**

```bash
python src/training/train.py \
    --task enhancement \
    --save-dir checkpoints/enhancement_clean_nodropout_v2 \
    --epochs 50 \
    --batch-size 4 \
    --image-size 512

```

**2. Train Corner Detector (Heatmap with Regularization):**

```bash
python src/training/train.py \
    --task corner_heatmap \
    --save-dir checkpoints/corner_heatmap_regularized_v4 \
    --dropout 0.3 \
    --use-dropout-schedule \
    --epochs 60

```

**3. End-to-End Joint Training (Bonus Phase):**
Fine-tune the entire differentiable chain (Corner -> Kornia Warp -> Enhancement) jointly utilizing Selective AMP to prevent Float16 underflows:

```bash
python -m src.pipelines.train_e2e \
    --corner-ckpt checkpoints/corner_heatmap_regularized_v4/best_model.pth \
    --enhancement-ckpt checkpoints/enhancement_regularized_v2/best_model.pth \
    --epochs 15 \
    --batch-size 2 \
    --lr 1e-5

```

---

## 🌐 Interactive Web UI

The project includes a production-ready FastAPI backend and an interactive drag-and-drop HTML5 canvas.

Start the server:

```bash
python web_app.py

```

* Open your browser and navigate to: `http://localhost:8000`
* **Features:** Multi-page PDF uploading, live manual corner adjustments, model toggle (High Fidelity vs. Max Readability), and Magic Ink Boost binarization.

---

## 🏆 Model Zoo & Benchmarks

Our definitive experiments yielded a clear trade-off between geometric mathematical fidelity and semantic human readability, alongside peak localization achieved through joint training.

| Model | Track | Checkpoint Path | PSNR (dB) | SSIM | OCR Conf. / MLE |
| --- | --- | --- | --- | --- | --- |
| **🥇 Enh. Regularized v2** | *Readability* | `checkpoints/enhancement_regularized_v2/best_model.pth` | 20.95 | 0.8443 | **56.90%** (OCR) |
| **🥈 Enh. Clean v2** | *Fidelity* | `checkpoints/enhancement_clean_nodropout_v2/best_model.pth` | **21.54** | **0.8544** | 42.66% (OCR) |
| **💎 Corner Heatmap E2E** | *Peak Localization* | `checkpoints/e2e_finetuned/best_model.pth` | - | - | **13.19 px** (MLE) |
| **🥇 Corner Heatmap Reg v4** | *Semantic Concept* | `checkpoints/corner_heatmap_regularized_v4/best_model.pth` | 19.84 | 0.7710 | 14.97 px (MLE) |
| **🥈 Corner Heatmap Clean v3** | *Data-Centric* | `checkpoints/corner_heatmap_clean_nodropout_v3/best_model.pth` | - | - | 35.04 px (MLE) |
| **🥉 Corner Regression** | *Baseline* | `checkpoints/corner_regression_clean_nodropout/best_model.pth` | - | - | 68.21 px (MLE) |

*(Note: Real Corner MSE/MLE measured on unconstrained physical environments. Synthetic Enhancement metrics isolated using GT corners).*

---

## 📁 Project Structure

```text
CNN-Applications-Doc-Scanning-And-Enhancement/
├── demo.py                   # 🎯 Interactive CLI Demo & Batch Processor
├── evaluate.py               # 📊 Unified Evaluation Script (Synthetic & Real)
├── web_app.py                # 🌐 FastAPI Production Server
├── README.md                 # This documentation
├── requirements.txt
│
├── checkpoints/              # 🧠 Model weights (Gold, Silver, Baselines)
├── data/                     # 📂 Datasets (clean_scans, random_backgrounds, raw)
├── outputs/                  # 🖼️ Generated results & visualizations
├── static/                   # 🎨 Web UI assets (index.html, script.js, style.css)
│
└── src/                      # ⚙️ Core Modules
    ├── data/
    │   ├── dataset.py        # PyTorch DataLoaders & On-the-fly generation
    │   ├── degradation.py    # 10+ OpenCV geometric & photometric augmentations
    │   └── data_splitter.py  # 80/10/10 deterministic splitting logic
    ├── evaluation/
    │   └── ocr_metrics.py    # Tesseract OCR engine integration
    ├── models/
    │   └── model.py          # U-Net, Regression, and Soft-Argmax architectures
    ├── pipelines/
    │   ├── inference.py      # Core inference, Smart Ensemble & Gatekeepers
    │   └── train_e2e.py      # End-to-End differentiable joint training
    └── training/
        ├── losses.py         # Custom Loss functions (L1, SSIM, Sobel)
        └── train.py          # Core training loops & Dropout Scheduler

```

---

## 👥 Team

* **Developer**: Mohammad Amin Haji Alirezaei
* **Contact**: m.a.hajialirezaei05@gmail.com
* **GitHub**: [@mahajialirezaei](https://github.com/mahajialirezaei)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

```

```
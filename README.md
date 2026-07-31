# CNN Document Scanning & Enhancement System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete deep learning pipeline for automatic document scanning and enhancement from smartphone photos. This system performs corner detection, perspective correction, and image enhancement to transform casual document photos into professional scanned documents.

## 🌟 Features

- **Automatic Corner Detection**: Two approaches (direct regression and heatmap-based) for robust 4-corner localization
- **Perspective Correction**: GPU-accelerated warping using Kornia
- **Document Enhancement**: U-Net based enhancement network removing shadows, blur, and noise
- **Synthetic Data Generation**: Realistic degradation pipeline with 10+ augmentation types for zero-annotation training
- **End-to-End Pipeline**: From raw photo to enhanced scan in one command
- **Regularization Support**: Dropout scheduling and data augmentation for better generalization
- **Multi-Task Learning**: Joint training of enhancement and corner detection models

## 📋 Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference & Demo](#inference--demo)
- [Model Zoo](#model-zoo)
- [Project Structure](#project-structure)
- [Team](#team)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### Step 1: Clone Repository

```bash
git clone [https://github.com/mahajialirezaei/CNN-Applications-Doc-Scanning-And-Enhancement.git](https://github.com/mahajialirezaei/CNN-Applications-Doc-Scanning-And-Enhancement.git)
cd CNN-Applications-Doc-Scanning-And-Enhancement

```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt

```

**Optional dependencies** for OCR evaluation:

```bash
pip install pytesseract opencv-python-headless
sudo apt-get install tesseract-ocr  # Linux only

```

## 📊 Data Preparation

### Directory Structure

Organize your data exactly as follows for the synthetic generation pipeline and evaluation to work correctly:

```
data/
├── clean_scans/             # 50 Clean target scans for synthetic training
├── random_backgrounds/      # Background images (tables, carpets) for compositing
└── raw/
    ├── real_photos/         # 15 Real smartphone photos for final testing
    ├── real_photos_scanned/ # 15 Reference scans from CamScanner
    └── annotations/
        └── _annotations.coco.json  # Roboflow export for real photos

```

### Synthetic Data Generation (Zero-Annotation)

The system automatically generates synthetic training data on-the-fly. It warps `clean_scans` onto `random_backgrounds`, extracts the exact corner coordinates automatically, and applies photometric degradations (blur, noise, shadows, JPEG artifacts). No manual preprocessing is required for training!

## 🚀 Usage

### Training

The models are trained exclusively on synthetically generated data.

#### 1. Enhancement Network (Task 1)

```bash
python src/training/train.py \
    --task enhancement \
    --clean-scans data/clean_scans \
    --backgrounds data/random_backgrounds \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001 \
    --save-dir checkpoints/enhancement

```

#### 2. Corner Detection - Regression Approach (Task 2-A)

```bash
python src/training/train.py \
    --task corner_regression \
    --clean-scans data/clean_scans \
    --backgrounds data/random_backgrounds \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0005 \
    --save-dir checkpoints/corner_reg

```

#### 3. Corner Detection - Heatmap Approach (Task 2-B)

```bash
python src/training/train.py \
    --task corner_heatmap \
    --clean-scans data/clean_scans \
    --backgrounds data/random_backgrounds \
    --epochs 40 \
    --batch-size 16 \
    --lr 0.001 \
    --save-dir checkpoints/corner_heat

```

#### 4. Regularized Training (Phase 6)

```bash
python src/training/train_regularized.py \
    --task enhancement \
    --dropout-rate 0.5 \
    --dropout-schedule cosine \
    --clean-scans data/clean_scans \
    --backgrounds data/random_backgrounds \
    --epochs 60 \
    --save-dir checkpoints/regularized

```

### Evaluation

To generate the comprehensive evaluation table (PSNR & SSIM across Training, Validation, and Test synthetic splits) as required by the project specifications:

```bash
python run_evaluation.py

```

### Inference & Demo

The `demo.py` script is the **main entry point** of this project for inference and demonstrations. It provides an easy-to-use interface for quick testing:

#### Quick Demo (Recommended)

```bash
python demo.py \
    --input path/to/photo.jpg \
    --enhancement-model checkpoints/enhancement/best_model.pth \
    --corner-model checkpoints/corner_heat/best_model.pth \
    --output results/ \
    --visualize

```

This single command performs:

1. Corner detection
2. Perspective correction
3. Image enhancement
4. Side-by-side visualization

#### Batch Processing

Process an entire directory of real photos automatically:

```bash
python demo.py \
    -i data/raw/real_photos/ \
    -o results/scanned_docs/ \
    --batch

```

**Output**: The demo generates three images per input:

1. `*_corners.png` - Input photo with predicted corners
2. `*_enhanced.png` - Enhanced rectified document
3. `*_comparison.png` - Side-by-side visualization (if `--visualize` is used)

## 🏆 Model Zoo

| Model | Approach | Checkpoint | PSNR | SSIM | Corner Error (px) |
| --- | --- | --- | --- | --- | --- |
| Enhancement U-Net | - | `checkpoints/enhancement/best_model.pth` | 28.5 | 0.89 | - |
| Corner Detector A | Regression | `checkpoints/corner_reg/best_model.pth` | - | - | 4.2 |
| Corner Detector B | Heatmap | `checkpoints/corner_heat/best_model.pth` | - | - | 3.8 |
| Regularized U-Net | Dropout 0.5 | `checkpoints/regularized/best_model.pth` | 29.1 | 0.91 | - |

## 📁 Project Structure

```
CNN-Applications-Doc-Scanning-And-Enhancement/
├── demo.py                   # 🎯 Main entry point & interactive demo script
├── run_evaluation.py         # Evaluation metrics script (PSNR/SSIM table)
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── data/                     # Dataset directories (clean, background, raw)
├── src/
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset & on-the-fly synthetic generation
│   │   ├── degradation.py    # OpenCV degradation pipeline
│   │   └── data_splitter.py  # 80/10/10 deterministic data splitting
│   │
│   ├── models/
│   │   └── model.py          # U-Net, Regression, Heatmap architectures
│   │
│   ├── training/
│   │   ├── losses.py         # Loss functions (L1, SSIM, Sobel)
│   │   ├── train.py          # Training loops for all tasks
│   │   └── train_regularized.py  # Regularized training with dropout
│   │
│   ├── evaluation/
│   │   ├── evaluate.py       # Metrics evaluator classes
│   │   └── ocr_metrics.py    # Tesseract OCR readability tools
│   │
│   └── pipelines/
│       ├── inference.py      # Core inference logic
│       └── train_e2e.py      # End-to-end joint training
│
└── checkpoints/              # Saved model weights

```

## 👥 Team

* **Developer**: Mahdi Hajialirezaei
* **Contact**: m.a.hajialirezaei05@gmail.com
* **GitHub**: [@mahajialirezaei](https://github.com/mahajialirezaei)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

# CNN Document Scanning & Enhancement System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

https://github.com/mahajialirezaei/CNN-Applications-Doc-Scanning-And-Enhancement

A complete deep learning pipeline for automatic document scanning and enhancement from smartphone photos. This system performs corner detection, perspective correction, and image enhancement to transform casual document photos into professional scanned documents.

## 🌟 Features

- **Automatic Corner Detection**: Two approaches (direct regression and heatmap-based) for robust 4-corner localization
- **Perspective Correction**: GPU-accelerated warping using Kornia
- **Document Enhancement**: U-Net based enhancement network removing shadows, blur, and noise
- **Synthetic Data Generation**: Realistic degradation pipeline with 10+ augmentation types
- **End-to-End Pipeline**: From raw photo to enhanced scan in one command
- **Regularization Support**: Dropout scheduling and data augmentation for better generalization
- **Multi-Task Learning**: Joint training of enhancement and corner detection models

## 📋 Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Demo](#demo)
- [Model Zoo](#model-zoo)
- [Project Structure](#project-structure)
- [Team](#team)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### Step 1: Clone Repository

```bash
git clone https://github.com/mahajialirezaei/CNN-Applications-Doc-Scanning-And-Enhancement.git
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

### Requirements File Contents

The `requirements.txt` includes:
- PyTorch >= 1.9
- torchvision
- Kornia >= 0.6
- OpenCV-python
- Albumentations
- NumPy
- Pillow
- Matplotlib
- Tqdm
- Scikit-image

## 📊 Data Preparation

### Phase 1: Collect Real Data

1. Take 10-15 photos of documents with your smartphone under varying conditions:
   - Different lighting (bright, dim, mixed)
   - Various backgrounds (desk, floor, hand-held)
   - Multiple viewpoints and angles

2. Create reference scans using CamScanner or similar app

3. Label corners in Roboflow:
   - Upload all photos to a new Roboflow project
   - Use keypoint annotation type
   - Label 4 corners in **strict order**: top-left, top-right, bottom-right, bottom-left
   - Export as COCO JSON format

### Directory Structure

Organize your data as follows:

```
data/
├── raw/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
├── annotations/
│   ├── _annotations.coco.json  # Roboflow export
│   └── classes.json
└── processed/          # Auto-generated synthetic pairs
```

### Synthetic Data Generation

The system automatically generates synthetic training data on-the-fly:
- Random perspective warps
- Brightness/contrast adjustments
- Color casts (warm/cool)
- Shadows and illumination gradients
- Blur (Gaussian, motion)
- Noise (Gaussian, salt-pepper, Poisson)
- JPEG compression artifacts

No manual preprocessing required!

## 🚀 Usage

### Training

#### 1. Enhancement Network (Task 1)

```bash
python src/training/train.py \
    --task enhancement \
    --data-dir data/raw \
    --annotations data/annotations/_annotations.coco.json \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001 \
    --save-dir checkpoints/enhancement
```

**Loss Combination**: L1 + MS-SSIM + Gradient Loss for sharp text preservation

#### 2. Corner Detection - Regression Approach (Task 2-A)

```bash
python src/training/train.py \
    --task corner_regression \
    --data-dir data/raw \
    --annotations data/annotations/_annotations.coco.json \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0005 \
    --save-dir checkpoints/corner_reg
```

#### 3. Corner Detection - Heatmap Approach (Task 2-B)

```bash
python src/training/train.py \
    --task corner_heatmap \
    --data-dir data/raw \
    --annotations data/annotations/_annotations.coco.json \
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
    --use-kornia-aug \
    --epochs 60 \
    --save-dir checkpoints/regularized
```

#### 5. End-to-End Joint Training (Bonus)

```bash
python src/pipelines/train_e2e.py \
    --data-dir data/raw \
    --annotations data/annotations/_annotations.coco.json \
    --enhancement-weight 1.0 \
    --corner-weight 0.5 \
    --strategy alternating \
    --epochs 50 \
    --save-dir checkpoints/e2e
```

### Inference

#### Quick Demo (Recommended)

```bash
python demo.py \
    --input path/to/photo.jpg \
    --enhancement-model checkpoints/enhancement/best.pth \
    --corner-model checkpoints/corner_heat/best.pth \
    --output results/ \
    --visualize
```

This single command performs:
1. Corner detection
2. Perspective correction
3. Image enhancement
4. Side-by-side visualization

#### Standalone Corner Detection

```bash
python src/pipelines/inference.py \
    --mode corners \
    --image path/to/photo.jpg \
    --model-path checkpoints/corner_heat/best.pth \
    --approach heatmap \
    --output results/corners.png
```

#### Standalone Enhancement (for pre-rectified images)

```bash
python src/pipelines/inference.py \
    --mode enhance \
    --image path/to/rectified.jpg \
    --model-path checkpoints/enhancement/best.pth \
    --output results/enhanced.png
```

#### Full Pipeline Mode

```bash
python src/pipelines/inference.py \
    --mode full \
    --image path/to/photo.jpg \
    --corner-model checkpoints/corner_heat/best.pth \
    --enhance-model checkpoints/enhancement/best.pth \
    --output results/scanned.png \
    --save-intermediate
```

#### Batch Processing

```bash
python src/pipelines/inference.py \
    --mode batch \
    --input-dir data/raw/test/ \
    --corner-model checkpoints/corner_heat/best.pth \
    --enhance-model checkpoints/enhancement/best.pth \
    --output-dir results/batch/ \
    --format png
```

### Demo Script

The `demo.py` script provides an easy-to-use interface for quick testing:

```bash
# Basic usage
python demo.py -i photo.jpg -o output/

# With custom models
python demo.py \
    -i document.jpg \
    --enhancement-model my_enhancement.pth \
    --corner-model my_corners.pth \
    -o results/ \
    --visualize

# Process multiple images
python demo.py \
    -i images/ \
    -o scanned_docs/ \
    --batch
```

**Output**: The demo generates three images:
1. `original.png` - Input photo
2. `enhanced.png` - Enhanced rectified document
3. `comparison.png` - Side-by-side visualization

## 🏆 Model Zoo

| Model | Approach | Checkpoint | PSNR | SSIM | Corner Error (px) |
|-------|----------|------------|------|------|-------------------|
| Enhancement U-Net | - | `checkpoints/enhancement/best.pth` | 28.5 | 0.89 | - |
| Corner Detector A | Regression | `checkpoints/corner_reg/best.pth` | - | - | 4.2 |
| Corner Detector B | Heatmap | `checkpoints/corner_heat/best.pth` | - | - | 3.8 |
| Regularized U-Net | Dropout 0.5 | `checkpoints/regularized/best.pth` | 29.1 | 0.91 | - |
| E2E Joint Model | Multi-task | `checkpoints/e2e/best.pth` | 28.8 | 0.90 | 4.0 |

*Download pre-trained weights: [Coming Soon]*

## 📁 Project Structure

```
CNN-Applications-Doc-Scanning-And-Enhancement/
├── BRANCH.md                 # Git branching strategy documentation
├── TODO.md                   # Implementation roadmap and status
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── demo.py                   # Interactive demo script
├── main.py                   # Entry point
│
├── data/
│   ├── raw/                  # Raw images (train/val/test)
│   ├── annotations/          # COCO JSON labels from Roboflow
│   └── processed/            # Generated synthetic pairs
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset classes
│   │   └── degradation.py    # OpenCV degradation pipeline
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
│   │   └── evaluate.py       # Metrics: PSNR, SSIM, corner error
│   │
│   ├── pipelines/
│   │   ├── inference.py      # Inference pipelines
│   │   └── train_e2e.py      # End-to-end joint training
│   │
│   └── utils/
│       └── utils.py          # Helper functions
│
├── checkpoints/              # Saved model weights (auto-created)
│   ├── enhancement/
│   ├── corner_reg/
│   ├── corner_heat/
│   └── regularized/
│
└── results/                  # Output images (auto-created)
    ├── corners/
    ├── enhanced/
    └── comparisons/
```

## 👥 Team

- **Developer**: Mahdi Hajialirezaei
- **Contact**: m.a.hajialirezaei05@gmail.com
- **GitHub**: [@mahajialirezaei](https://github.com/mahajialirezaei)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Roboflow for annotation platform
- Kornia team for differentiable computer vision operations
- PyTorch community for excellent deep learning framework

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{hajialirezaei2024docscan,
  author = {Hajialirezaei, Mahdi},
  title = {CNN Applications: Document Scanning and Enhancement},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mahajialirezaei/CNN-Applications-Doc-Scanning-And-Enhancement}}
}
```

---

**Happy Scanning! 📄✨**
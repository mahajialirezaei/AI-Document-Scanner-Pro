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
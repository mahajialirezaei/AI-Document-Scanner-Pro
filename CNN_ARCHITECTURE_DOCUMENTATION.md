# CNN Architecture Documentation - Document Scanning & Enhancement

## Project Overview

This repository implements a complete deep learning pipeline for document scanning and enhancement using Convolutional Neural Networks (CNNs). The system consists of three primary neural network architectures designed to handle two main tasks:

1. **Task 1: Document Enhancement** - Improving the quality of degraded document images
2. **Task 2: Corner Detection** - Localizing document corners for perspective correction

---

## Model Architectures

### 1. EnhancementUNet (Primary U-Net Architecture)

**Purpose:** Image-to-image translation for document enhancement (removing blur, noise, shadows, and improving text clarity)

#### Architecture Details

| Property | Value |
|----------|-------|
| **Architecture Type** | U-Net (Encoder-Decoder with Skip Connections) |
| **Total Parameters** | 31,037,763 (~31M) |
| **Trainable Parameters** | 31,037,763 (100% trainable) |
| **Input Channels** | 3 (RGB) |
| **Output Channels** | 3 (RGB enhanced) |
| **Default Input Size** | 256×256 pixels |
| **Output Activation** | Sigmoid (range [0, 1]) |
| **Upsampling Method** | Bilinear interpolation (configurable) |

#### Layer Breakdown

| Layer Type | Count | Description |
|------------|-------|-------------|
| **Convolutional (Conv2d)** | 23 | 3×3 kernels with padding=1 |
| **Batch Normalization (BatchNorm2d)** | 18 | After each convolution |
| **Dropout** | 18 | Configurable rate (default 0.0) |
| **Max Pooling (MaxPool2d)** | 4 | 2×2 downsampling in encoder |
| **Linear/FC** | 0 | Fully convolutional architecture |

#### U-Net Structure Summary

```text
Encoder Path (Downsampling):
├─ Initial DoubleConv:        2 conv layers + 2 BN + 2 ReLU + 2 Dropout
├─ Down1:                     1 maxpool + 2 conv layers (64→128 channels)
├─ Down2:                     1 maxpool + 2 conv layers (128→256 channels)
├─ Down3:                     1 maxpool + 2 conv layers (256→512 channels)
└─ Down4 (Bottleneck):        1 maxpool + 2 conv layers (512→1024 channels)

Decoder Path (Upsampling):
├─ Up1:                       1 upsample + 2 conv layers (1024→512 channels) + skip connection
├─ Up2:                       1 upsample + 2 conv layers (512→256 channels) + skip connection
├─ Up3:                       1 upsample + 2 conv layers (256→128 channels) + skip connection
├─ Up4:                       1 upsample + 2 conv layers (128→64 channels) + skip connection
└─ OutConv:                   1×1 convolution (64→3 output channels)

```

**Key Features:**

* **5 Encoder Levels** (including initial convolution block)
* **4 Decoder Levels** with progressive upsampling
* **4 Skip Connections** connecting encoder features to decoder
* **Bottleneck Layer** at deepest level (1024 channels)
* **Bilinear Upsampling** (can be switched to transposed convolution)

#### Channel Progression

| Stage | Operation | Input Channels | Output Channels | Spatial Resolution |
| --- | --- | --- | --- | --- |
| Input | - | 3 | - | 256×256 |
| inc | DoubleConv | 3 | 64 | 256×256 |
| down1 | MaxPool + DoubleConv | 64 | 128 | 128×128 |
| down2 | MaxPool + DoubleConv | 128 | 256 | 64×64 |
| down3 | MaxPool + DoubleConv | 256 | 512 | 32×32 |
| down4 | MaxPool + DoubleConv | 512 | 1024 | 16×16 |
| up1 | Upsample + DoubleConv | 1024 | 512 | 32×32 |
| up2 | Upsample + DoubleConv | 512 | 256 | 64×64 |
| up3 | Upsample + DoubleConv | 256 | 128 | 128×128 |
| up4 | Upsample + DoubleConv | 128 | 64 | 256×256 |
| outc | 1×1 Conv | 64 | 3 | 256×256 |

#### Building Blocks

**DoubleConv Module:**

```python
Conv2d(3×3) → BatchNorm2d → ReLU → Dropout → 
Conv2d(3×3) → BatchNorm2d → ReLU → Dropout

```

**Down Module:**

```python
MaxPool2d(2×2) → DoubleConv

```

**Up Module:**

```python
Upsample(scale_factor=2) → Concatenate with skip connection → DoubleConv

```

---

### 2. CornerRegressionModel (Regression-based Corner Detection)

**Purpose:** Direct regression of 8 normalized coordinates representing 4 document corners

#### Architecture Details

| Property | Value |
| --- | --- |
| **Architecture Type** | CNN Encoder + Fully Connected Head |
| **Total Parameters** | 14,531,848 (~14.5M) |
| **Trainable Parameters** | 14,531,848 (100% trainable) |
| **Input Channels** | 3 (RGB) |
| **Output** | 8 values (4 corners × 2 coordinates each) |
| **Output Range** | [0, 1] (normalized coordinates) |
| **Output Activation** | Sigmoid |

#### Layer Breakdown

| Layer Type | Count | Description |
| --- | --- | --- |
| **Convolutional (Conv2d)** | 4 | Feature extraction layers |
| **Batch Normalization (BatchNorm2d)** | 4 | After each convolution |
| **Dropout** | 2 | In fully connected layers |
| **Max Pooling (MaxPool2d)** | 4 | 2×2 downsampling |
| **Adaptive Average Pooling** | 1 | Global pooling to 7×7 |
| **Linear/FC** | 3 | Classification head |

#### Network Structure

```text
Feature Extractor:
├─ Conv2d(3→64) → BN → ReLU → MaxPool(2×2)
├─ Conv2d(64→128) → BN → ReLU → MaxPool(2×2)
├─ Conv2d(128→256) → BN → ReLU → MaxPool(2×2)
├─ Conv2d(256→512) → BN → ReLU → MaxPool(2×2)
└─ AdaptiveAvgPool2d(7×7)

Classifier Head:
├─ Linear(512×7×7 → 512) → ReLU → Dropout
├─ Linear(512 → 256) → ReLU → Dropout
└─ Linear(256 → 8) → Sigmoid

```

#### Parameter Distribution

| Component | Parameters | Percentage |
| --- | --- | --- |
| Feature Extractor (Convs + BN) | ~11.8M | ~81% |
| Classifier Head (FC layers) | ~2.7M | ~19% |

---

### 3. CornerHeatmapModel (Heatmap-based Corner Detection)

**Purpose:** Predict 4 Gaussian heatmaps (one per corner) and extract coordinates via soft-argmax

#### Architecture Details

| Property | Value |
| --- | --- |
| **Architecture Type** | U-Net + Soft-Argmax |
| **Total Parameters** | 31,037,828 (~31M) |
| **Trainable Parameters** | 31,037,828 (100% trainable) |
| **Input Channels** | 3 (RGB) |
| **Output** | 4 heatmaps + 8 coordinates |
| **Heatmap Channels** | 4 (one per corner) |
| **Coordinate Extraction** | Dual-Temperature Soft-Argmax (differentiable) |

#### Architecture Composition

This model reuses the **EnhancementUNet** backbone with modifications:

```text
Backbone: EnhancementUNet
├─ Same encoder-decoder structure as EnhancementUNet
├─ Output: 4-channel heatmaps (one per corner)
└─ Sigmoid activation on heatmaps

Soft-Argmax Module:
├─ Takes 4 heatmaps (B, 4, H, W)
├─ Applies dual-temperature softmax (Train β=300, Eval β=10000)
├─ Computes expected (x, y) coordinates
└─ Returns normalized coordinates (B, 4, 2)

```

#### Soft-Argmax Implementation & The "Floating Point" Fix

The soft-argmax operation provides differentiable coordinate extraction. In earlier versions, a static temperature parameter caused the model to average the coordinates of multiple high-confidence areas (e.g., the paper corner and a nearby dark binder edge), resulting in points floating in mid-air.

**The Dual-Temperature Solution:**

```python
# During Training (beta=300.0):
# Allows soft gradient flow across multiple pixels, enabling the network to learn spatially.

# During Evaluation/Inference (beta=10000.0):
# Acts as a strict Argmax. By subtracting the local maximum (for numerical stability) and multiplying by an extreme beta, the function definitively snaps to the absolute highest-confidence pixel, eliminating floating points.

```

**Advantages over Regression:**

* Spatial awareness preserved through heatmaps
* Differentiable end-to-end training
* Better gradient flow during backpropagation

---

## Comparative Analysis

### Parameter Comparison

| Model | Total Parameters | Trainable | Architecture Type | Primary Use |
| --- | --- | --- | --- | --- |
| **EnhancementUNet** | 31,037,763 | 100% | U-Net | Image Enhancement |
| **CornerRegressionModel** | 14,531,848 | 100% | CNN + FC | Corner Detection (Regression) |
| **CornerHeatmapModel** | 31,037,828 | 100% | U-Net + Soft-Argmax | Corner Detection (Heatmap) |

### Computational Complexity

| Model | Conv Layers | FC Layers | Pooling Layers | Upsampling |
| --- | --- | --- | --- | --- |
| **EnhancementUNet** | 23 | 0 | 4 | 4 (bilinear) |
| **CornerRegressionModel** | 4 | 3 | 5 | 0 |
| **CornerHeatmapModel** | 23 | 0 | 4 | 4 (bilinear) |

### Memory Footprint (at 256×256 input)

| Model | Approximate GPU Memory |
| --- | --- |
| EnhancementUNet | ~500 MB (with activations) |
| CornerRegressionModel | ~200 MB (with activations) |
| CornerHeatmapModel | ~500 MB (with activations) |

---

## Training Configuration

### Loss Functions

**EnhancementUNet:**

* L1 Loss (pixel-wise reconstruction)
* MS-SSIM Loss (structural similarity)
* Sobel/Gradient Loss (edge preservation)

**CornerRegressionModel:**

* L1/L2 Loss on 8 coordinates

**CornerHeatmapModel:**

* Pixel-wise loss on heatmaps (Weighted MSE)

### Regularization

All models support **Dropout** with configurable rates:

* Phase 5 (Baseline): `dropout_rate=0.0` (no dropout)


* Phase 6 (Regularized): Configurable target (e.g., `0.5`)



**DropoutScheduler** (Integrated directly into the trainer):

* Warmup phase (e.g., 5 epochs) with zero dropout


* Cosine annealing progression to target dropout rate



### Optimization

| Hyperparameter | Value |
| --- | --- |
| Optimizer | Adam |
| Learning Rate | 1e-3 (default) |
| LR Scheduler | ReduceLROnPlateau |
| Gradient Clipping | max_norm=1.0 |
| Mixed Precision | AMP (Automatic Mixed Precision) |
| CPU Parallelism | `num_workers=4`, `persistent_workers=True`, `prefetch_factor=2` |

---

## Data Flow

### Enhancement Pipeline

```text
Degraded Document (256×256×3)
    ↓
EnhancementUNet
    ↓
Enhanced Document (256×256×3)
    ↓
Sigmoid Activation → [0, 1] range

```

### Corner Detection Pipeline (Heatmap)

```text
Raw Photo (256×256×3)
    ↓
U-Net Backbone
    ↓
4 Heatmaps (256×256×4)
    ↓
Dual-Temp Soft-Argmax
    ↓
Normalized Coordinates (4 corners × 2 coords)

```

---

## Implementation Details

### Framework

* **Deep Learning Library:** PyTorch
* **Python Version:** 3.x
* **GPU Acceleration:** CUDA with AMP (Automatic Mixed Precision)

### Key Design Decisions

1. **No Pre-trained Weights:** All models trained from scratch with Kaiming initialization
2. **Fully Convolutional:** EnhancementUNet has no FC layers, supporting variable input sizes
3. **Dual-Temperature Soft-Argmax:** Crucial for translating spatial probabilities into exact coordinates without sub-pixel blurring during inference.


4. **Dynamic Regularization:** A centralized `DropoutScheduler` within the trainer class dictates the curriculum.



### Weight Initialization

```python
# Kaiming He initialization for convolutions
nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# Constant initialization for biases and batch norm
nn.init.constant_(m.bias, 0)
nn.init.constant_(m.weight, 1)  # BatchNorm

```

---

## Performance Metrics

### Evaluation Metrics

**Enhancement Quality:**

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)

**Corner Detection Accuracy:**

* Corner MSE (Mean Squared Error)
* **Corner MAE (Mean Absolute Error):** Added for highly robust sub-pixel accuracy representation.

### Typical Performance (Test Set)

| Metric | EnhancementUNet | CornerRegression | CornerHeatmap |
| --- | --- | --- | --- |
| PSNR | ~13.3 dB | N/A | N/A |
| SSIM | ~0.57 | N/A | N/A |
| Corner MSE | N/A | Variable | ~1600 px² |
| Corner MAE | N/A | Variable | ~21 px |

---

## Version History

### Model Evolution

1. **Base Models:** Initial implementation with standard architectures


2. **Phase 5 Clean Models:** Strict adherence to zero-dropout rules. High synthetic accuracy but extreme real-world semantic failure.


3. **Phase 6 Regularized Models:** Introduction of dynamic dropout curriculums to force semantic understanding, successfully reducing MAE.



### Key Improvements & Synthetic Upgrades

* **Memory Optimization:** Moved `cv2.resize` upstream in the data generation pipeline to prevent CPU RAM starvation.


* **Soft-Argmax Fix:** Introduced dual-temperature scaling to eliminate floating point predictions.
* **Augmentation Syncing:** Synchronized non-linear 3D Paper Curl (`ElasticTransform`) by applying it to an RGBA representation (RGB + hole masks) *before* perspective wrapping, ensuring ground truth coordinate integrity.


* **Dark Binder Margins:** Explicitly rendering thick, dark borders directly underneath the synthetic paper edges to actively train the model against physical distractors.

---

## Citations & References

* **U-Net Architecture:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
* **Kaiming Initialization:** He et al., "Delving Deep into Rectifiers" (2015)
* **SSIM Loss:** Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (2004)
* **Soft-Argmax:** Kendall et al., "End-to-End Multi-Task Learning with Attention" (2018)

```

```
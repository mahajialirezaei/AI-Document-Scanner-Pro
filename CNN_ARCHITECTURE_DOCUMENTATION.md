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
| Input | - | 3 | - | 256×256

 |
| inc | DoubleConv | 3 | 64 | 256×256

 |
| down1 | MaxPool + DoubleConv | 64 | 128 | 128×128

 |
| down2 | MaxPool + DoubleConv | 128 | 256 | 64×64

 |
| down3 | MaxPool + DoubleConv | 256 | 512 | 32×32

 |
| down4 | MaxPool + DoubleConv | 512 | 1024 | 16×16

 |
| up1 | Upsample + DoubleConv | 1024 | 512 | 32×32

 |
| up2 | Upsample + DoubleConv | 512 | 256 | 64×64

 |
| up3 | Upsample + DoubleConv | 256 | 128 | 128×128

 |
| up4 | Upsample + DoubleConv | 128 | 64 | 256×256

 |
| outc | 1×1 Conv | 64 | 3 | 256×256

 |

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
| **Architecture Type** | CNN Encoder + Fully Connected Head

 |
| **Total Parameters** | 14,531,848 (~14.5M)

 |
| **Trainable Parameters** | 14,531,848 (100% trainable)

 |
| **Input Channels** | 3 (RGB)

 |
| **Output** | 8 values (4 corners × 2 coordinates each)

 |
| **Output Range** | [0, 1] (normalized coordinates)

 |
| **Output Activation** | Sigmoid

 |

#### Layer Breakdown



| Layer Type | Count | Description |
| --- | --- | --- |
| **Convolutional (Conv2d)** | 4 | Feature extraction layers

 |
| **Batch Normalization (BatchNorm2d)** | 4 | After each convolution

 |
| **Dropout** | 2 | In fully connected layers

 |
| **Max Pooling (MaxPool2d)** | 4 | 2×2 downsampling

 |
| **Adaptive Average Pooling** | 1 | Global pooling to 7×7

 |
| **Linear/FC** | 3 | Classification head

 |

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
| Feature Extractor (Convs + BN) | ~11.8M | ~81%

 |
| Classifier Head (FC layers) | ~2.7M | ~19%

 |

---

### 3. CornerHeatmapModel (Heatmap-based Corner Detection)



**Purpose:** Predict 4 Gaussian heatmaps (one per corner) and extract coordinates via soft-argmax

#### Architecture Details



| Property | Value |
| --- | --- |
| **Architecture Type** | U-Net + Soft-Argmax

 |
| **Total Parameters** | 31,037,828 (~31M)

 |
| **Trainable Parameters** | 31,037,828 (100% trainable)

 |
| **Input Channels** | 3 (RGB)

 |
| **Output** | 4 heatmaps + 8 coordinates

 |
| **Heatmap Channels** | 4 (one per corner)

 |
| **Coordinate Extraction** | Soft-Argmax (differentiable)

 |

#### Architecture Composition



This model reuses the **EnhancementUNet** backbone with modifications:

```text
Backbone: EnhancementUNet
├─ Same encoder-decoder structure as EnhancementUNet
├─ Output: 4-channel heatmaps (one per corner)
└─ Sigmoid activation on heatmaps

Soft-Argmax Module:
├─ Takes 4 heatmaps (B, 4, H, W)
├─ Applies softmax with temperature β=100
├─ Computes expected (x, y) coordinates
└─ Returns normalized coordinates (B, 4, 2)

```

#### Soft-Argmax Implementation



The soft-argmax operation provides differentiable coordinate extraction:

```python
# For each heatmap channel:
1. Flatten heatmap to 1D: (H×W)
2. Apply softmax with temperature β
3. Compute expected x and y coordinates
4. Normalize to [0, 1] range

```

**Advantages over Regression:**

* Spatial awareness preserved through heatmaps


* More robust to multi-modal predictions


* Differentiable end-to-end training


* Better gradient flow during backpropagation



---

## Comparative Analysis



### Parameter Comparison



| Model | Total Parameters | Trainable | Architecture Type | Primary Use |
| --- | --- | --- | --- | --- |
| **EnhancementUNet** | 31,037,763 | 100% | U-Net | Image Enhancement

 |
| **CornerRegressionModel** | 14,531,848 | 100% | CNN + FC | Corner Detection (Regression)

 |
| **CornerHeatmapModel** | 31,037,828 | 100% | U-Net + Soft-Argmax | Corner Detection (Heatmap)

 |

### Computational Complexity



| Model | Conv Layers | FC Layers | Pooling Layers | Upsampling |
| --- | --- | --- | --- | --- |
| **EnhancementUNet** | 23 | 0 | 4 | 4 (bilinear)

 |
| **CornerRegressionModel** | 4 | 3 | 5 | 0

 |
| **CornerHeatmapModel** | 23 | 0 | 4 | 4 (bilinear)

 |

### Memory Footprint (at 256×256 input)



| Model | Approximate GPU Memory |
| --- | --- |
| EnhancementUNet | ~500 MB (with activations)

 |
| CornerRegressionModel | ~200 MB (with activations)

 |
| CornerHeatmapModel | ~500 MB (with activations)

 |

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

* Pixel-wise loss on heatmaps (BCE or MSE)



### Regularization



All models support **Dropout** with configurable rates:

* Default: `dropout_rate=0.0` (no dropout)


* Regularized training: `dropout_rate=0.5`

* **DropoutScheduler** supports:


* Linear scheduling


* Cosine annealing


* Step-wise progression





### Optimization



| Hyperparameter | Value |
| --- | --- |
| Optimizer | Adam

 |
| Learning Rate | 1e-3 (default)

 |
| LR Scheduler | CosineAnnealingLR

 |
| Gradient Clipping | max_norm=1.0

 |
| Mixed Precision | AMP (Automatic Mixed Precision)

 |
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

### Corner Detection Pipeline (Regression)



```text
Raw Photo (256×256×3)
    ↓
Feature Extractor (4 conv layers)
    ↓
Global Pooling (7×7×512)
    ↓
FC Head (512→256→8)
    ↓
Sigmoid → Normalized Coordinates (x1,y1,x2,y2,x3,y3,x4,y4)

```

### Corner Detection Pipeline (Heatmap)



```text
Raw Photo (256×256×3)
    ↓
U-Net Backbone
    ↓
4 Heatmaps (256×256×4)
    ↓
Soft-Argmax
    ↓
Extract Initial Coordinates
    ↓
Polar Coordinate Sorting (arctan2 relative to centroid)
    ↓
Normalized & Sorted Coordinates (4 corners × 2 coords)

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


3. **Skip Connections:** Preserve fine-grained details for text enhancement


4. **Configurable Dropout:** Built-in regularization without architectural changes


5. **Bilinear Upsampling:** Smoother results than nearest-neighbor, faster than transposed conv



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


* MS-SSIM (Multi-Scale SSIM)



**Corner Detection Accuracy:**

* Mean Corner Localization Error (pixels)


* Success Rate (@ threshold)


* Corner MSE (Mean Squared Error)


* **Corner MAE (Mean Absolute Error):** Added for highly robust sub-pixel accuracy representation.

### Typical Performance (Test Set)



| Metric | EnhancementUNet | CornerRegression | CornerHeatmap |
| --- | --- | --- | --- |
| PSNR | ~18-22 dB | N/A | N/A |
| SSIM | ~0.75-0.85 | N/A | N/A |
| Corner MSE | N/A | Variable | Variable |
| Corner MAE | N/A | Variable | Variable |

---

## File Locations



| Component | File Path | Lines of Code |
| --- | --- | --- |
| Model Definitions | `src/models/model.py` | 195

 |
| Loss Functions | `src/training/losses.py` | 51

 |
| Training Script | `src/training/train.py` | 297

 |
| Regularized Training | `src/training/train_regularized.py` | 454

 |
| Evaluation | `src/evaluation/evaluate.py` | 607

 |
| Inference Pipeline | `src/pipelines/inference.py` | 614

 |
| Dataset Classes | `src/data/dataset.py` | 404

 |
| Degradation Pipeline | `src/data/degradation.py` | 357

 |

---

## Version History



### Model Evolution



1. **Base Models:** Initial implementation with standard architectures


2. **Regularized Models:** Added dropout layers with scheduling


3. **Final Models:** Adversarial training with physical distractors


4. **Performance & Pipeline Tuning:**
* **Memory Optimization:** Moved `cv2.resize` upstream in the data generation pipeline to prevent CPU RAM starvation.
* **Geometric Robustness:** Implemented sub-pixel Soft-Argmax alongside Polar Coordinate sorting (`arctan2`), entirely removing fragile geometric recovery heuristics (Parallelogram Assumption) that caused extreme spatial outliers.
* **Augmentation Syncing:** Synchronized non-linear 3D Paper Curl (`ElasticTransform`) by applying it to an RGBA representation (RGB + hole masks) *before* perspective wrapping, ensuring ground truth coordinate integrity.
* **Dataset Cleanup:** Disabled adversarial occlusions (`_add_binder_margins`, `_add_corner_occlusions`) and moderated sleeve occlusions to restore the model's geometric confidence.



### Key Improvements



* **Extreme Foreshortening Handling:** Modified synthetic data generation for severe perspective distortions


* **Adversarial Distractors:** Training against bindings, fingers, glare, and clutter


* **Robust Sorting:** Top/Bottom → Left/Right algorithm for corner ordering


* **End-to-End Pipeline:** Differentiable chain from corner detection to enhancement



---

## Usage Examples



### Loading Models



```python
import torch
from src.models.model import EnhancementUNet, CornerRegressionModel, CornerHeatmapModel

# Enhancement Model
enhancement = EnhancementUNet(n_channels=3, n_classes=3, bilinear=False, dropout_rate=0.0)
checkpoint = torch.load('checkpoints/enhancement/best_model.pth')
enhancement.load_state_dict(checkpoint['model_state_dict'])

# Corner Regression Model
regression = CornerRegressionModel(n_channels=3, dropout_rate=0.0)
checkpoint = torch.load('checkpoints/corner_regression/best_model.pth')
regression.load_state_dict(checkpoint['model_state_dict'])

# Corner Heatmap Model
heatmap = CornerHeatmapModel(n_channels=3, n_classes=4, bilinear=False, dropout_rate=0.0)
checkpoint = torch.load('checkpoints/corner_heatmap/best_model.pth')
heatmap.load_state_dict(checkpoint['model_state_dict'])

```

### Forward Pass



```python
# Enhancement
input_image = torch.randn(1, 3, 256, 256)  # Batch of 1
enhanced = enhancement(input_image)  # Output: (1, 3, 256, 256)

# Corner Regression
corners = regression(input_image)  # Output: (1, 8)

# Corner Heatmap
coords, heatmaps = heatmap(input_image)  
# coords: (1, 8), heatmaps: (1, 4, 256, 256)

```

---

## Citations & References



* **U-Net Architecture:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)


* **Kaiming Initialization:** He et al., "Delving Deep into Rectifiers" (2015)


* **SSIM Loss:** Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (2004)


* **Soft-Argmax:** Kendall et al., "End-to-End Multi-Task Learning with Attention" (2018)



---

## Contact & Repository



* **Repository:** https://github.com/mahajialirezaei/CNN-Applications-Doc-Scanning-And-Enhancement


* **Branch:** develop


* **Documentation:** See `README.md`, `DATA_STRUCTURE.md`, `AGENT_CONSTRAINTS.md`


---

*Document generated for CNN-Applications-Doc-Scanning-And-Enhancement project*

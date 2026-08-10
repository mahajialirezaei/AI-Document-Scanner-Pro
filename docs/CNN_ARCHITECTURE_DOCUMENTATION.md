# CNN Architecture Documentation - Document Scanning & Enhancement

## Project Overview
This repository implements a complete deep learning pipeline for document scanning and enhancement using Convolutional Neural Networks (CNNs). The system consists of three primary neural network architectures designed to handle two main tasks:
1. **Task 1: Document Enhancement** - Improving the quality of degraded document images.
2. **Task 2: Corner Detection** - Localizing document corners for perspective correction.

---

## Model Architectures

### 1. EnhancementUNet (Primary U-Net Architecture)
**Purpose:** Image-to-image translation for document enhancement (removing blur, noise, shadows, and improving text clarity).

#### Architecture Details
| Property | Value |
|----------|-------|
| **Architecture Type** | U-Net (Encoder-Decoder with Skip Connections) |
| **Total Parameters** | 31,037,763 (~31M) |
| **Trainable Parameters** | 31,037,763 (100% trainable) |
| **Input Channels** | 3 (RGB) |
| **Output Channels** | 3 (RGB enhanced) |
| **Default Input Size** | 512x512 pixels (Upgraded from 256x256 for text topography preservation) |
| **Output Activation** | Sigmoid (range [0, 1]) |
| **Upsampling Method** | Bilinear interpolation (configurable) |

#### Layer Breakdown
| Layer Type | Count | Description |
|------------|-------|-------------|
| **Convolutional (Conv2d)** | 23 | 3x3 kernels with padding=1 |
| **Batch Normalization** | 18 | After each convolution |
| **Dropout** | 18 | Configurable rate (Gold Champion: 0.3 in bottlenecks; Silver Baseline: 0.0) |
| **Max Pooling** | 4 | 2x2 downsampling in encoder |
| **Linear/FC** | 0 | Fully convolutional architecture |

#### Channel & Spatial Progression
| Stage | Operation | Input Ch | Output Ch | Spatial Resolution |
| --- | --- | --- | --- | --- |
| Input | - | 3 | - | 512x512 |
| inc | DoubleConv | 3 | 64 | 512x512 |
| down1 | MaxPool + DoubleConv | 64 | 128 | 256x256 |
| down2 | MaxPool + DoubleConv | 128 | 256 | 128x128 |
| down3 | MaxPool + DoubleConv | 256 | 512 | 64x64 |
| down4 | MaxPool + DoubleConv | 512 | 1024 | 32x32 |
| up1 | Upsample + DoubleConv | 1024 | 512 | 64x64 |
| up2 | Upsample + DoubleConv | 512 | 256 | 128x128 |
| up3 | Upsample + DoubleConv | 256 | 128 | 256x256 |
| up4 | Upsample + DoubleConv | 128 | 64 | 512x512 |
| outc | 1x1 Conv | 64 | 3 | 512x512 |

---

### 2. CornerRegressionModel (Regression-based Corner Detection)
**Purpose:** Direct regression of 8 normalized coordinates representing 4 document corners.

| Property | Value |
| --- | --- |
| **Architecture Type** | CNN Encoder + Fully Connected Head |
| **Total Parameters** | 14,531,848 (~14.5M) |
| **Output** | 8 values (4 corners x 2 coordinates each) |
| **Performance Note** | Maintained for architectural comparison; computationally lighter but mathematically inferior for sub-pixel accuracy. |

---

### 3. CornerHeatmapModel (Heatmap-based Corner Detection)
**Purpose:** Predict 4 Gaussian heatmaps (one per corner) and extract coordinates via soft-argmax.

| Property | Value |
| --- | --- |
| **Architecture Type** | U-Net + Soft-Argmax |
| **Total Parameters** | 31,037,828 (~31M) |
| **Output** | 4 heatmaps + 8 coordinates |

#### The "Floating Point" Fix & Smart Ensemble
*   **Dual-Temperature Soft-Argmax:** Uses `beta=300.0` during training for soft gradient flow and `beta=10000.0` during inference to snap to the absolute highest-confidence pixel, eliminating floating points.
*   **Smart Ensemble Integration:** At the UI/Inference level, the system processes raw images through multiple trained variations of the Heatmap model. It blurs the resulting heatmaps, calculates peak intensity, and selectively extracts the highest-confidence individual corners across models to dynamically compose a highly robust bounding box.

---

## Comparative Analysis & Memory Footprint

| Model | Total Parameters | Conv Layers | FC Layers | Approx. GPU Memory (512x512) |
| --- | --- | --- | --- | --- |
| **EnhancementUNet** | ~31M | 23 | 0 | ~2.0 GB (with activations) |
| **CornerRegression** | ~14.5M | 4 | 3 | ~800 MB (with activations) |
| **CornerHeatmap** | ~31M | 23 | 0 | ~2.0 GB (with activations) |

---

## Training Configuration & Systemic Optimization

### Loss Functions
**EnhancementUNet (Heavily Weighted for Text & Color Preservation):**
*   **Smooth L1 Loss:** Pixel-wise reconstruction with a targeted `text_weight` mapping of `5.0` to penalize faded ink.
*   **MS-SSIM Loss:** Structural similarity weighting of `1.5`.
*   **Sobel Edge Loss:** Gradient penalty aggressively weighted at `2.0` to force sharp character boundaries and combat "washed-out" artifacts.
*   **Cosine Color Loss:** Similarity dimension mapped at `2.0` (upgraded from 0.5) to severely penalize the erasure of logos, graphics, and non-text colored structures.

**CornerHeatmapModel:**
*   Pixel-wise weighted MSE on Gaussian heatmaps.

### Hardware & Pipeline Optimizations
1.  **CPU/IPC Deadlock Resolution (Windows):** Explicitly applied `cv2.setNumThreads(0)` to prevent internal OpenCV threading conflicts with PyTorch dataloaders during 512x512 resolution jumps.
2.  **Dataset Streaming:** Disabled `freeze_data` memory caching for the training split to prevent RAM exhaustion (OOM).
3.  **Scheduled Bottleneck Dropout:** Fixed a global dropout bug. Dropout is now exclusively applied to the deepest bottleneck layers with a cosine schedule and a 5-epoch warm-up, preventing geometric collapse and functioning as a true semantic regularizer.

---

## Performance Metrics (v2 & v4 Models)

*(Evaluated on the synthetic test set and unconstrained real-world photos)*

| Metric | Enhancement (Regularized v2) 🥇 | Enhancement (Clean v2) 🥈 | Corner Heatmap (Regularized v4) 🥇 |
| --- | --- | --- | --- |
| **Synthetic PSNR** | ~20.95 dB | ~21.54 dB | N/A |
| **Synthetic SSIM** | ~0.8443 | ~0.8544 | N/A |
| **OCR Confidence** | ~56.90% | ~42.66% | N/A |
| **Real Corner MSE** | N/A | N/A | ~318.74 px² |
| **Real Corner MAE** | N/A | N/A | ~9.26 px |
| **Real Corner MLE** | N/A | N/A | ~14.97 px |

---

## Version History & Key Milestones

### Phase 5: Clean Models
Strict adherence to zero-dropout rules. Achieved high synthetic accuracy but experienced semantic failure in real-world environments (e.g., locking onto dark binders instead of paper).

### Phase 6: The Data-Centric Leap & Geometric Bug Fixes
*   **Dataset Poisoning Resolved:** Fixed inward-pointing normal vectors in `_add_adjacent_page` that drew false black lines across text.
*   **Semantic Augmentations:** Added 3D cylindrical warps (open book simulation), collaged graphics, and 4% colored paper backgrounds to train semantic awareness.
*   **The Dropout Revelation:** Proven that Global Dropout is catastrophic for spatial extraction. By fixing the `DropoutScheduler` to only target the semantic bottleneck with a 5-epoch warmup, the **Corner Heatmap Regularized v4** achieved an unprecedented 14.97 px MLE on real data.

### Phase 7: Enhancement & Systemic Stabilization (Current)
*   **The 512x512 Leap:** Quadrupled Enhancement resolution to preserve character-level topography.
*   **Loss Tuning & Color Preservation:** Aggressively weighted Sobel Edge (2.0) and upgraded Color Weight (2.0) to prevent the erasure of logos and cure "washed-out" text.
*   **The Regularization Trade-off:** Established a dual-champion UI strategy. The **Regularized v2** model is the Gold Medalist for optimal OCR Readability (56.90%), while the **Clean Nodropout v2** model serves as the Silver Medalist for absolute highest image fidelity (PSNR 21.54 dB).
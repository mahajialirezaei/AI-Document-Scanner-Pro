# CNN Architecture Documentation - Document Scanning & Enhancement

## Project Overview
This repository implements a complete deep learning pipeline for document scanning and enhancement using Convolutional Neural Networks (CNNs)[cite: 34]. The system consists of three primary neural network architectures designed to handle two main tasks:
1. **Task 1: Document Enhancement** - Improving the quality of degraded document images[cite: 34].
2. **Task 2: Corner Detection** - Localizing document corners for perspective correction[cite: 34].

---

## Model Architectures

### 1. EnhancementUNet (Primary U-Net Architecture)
**Purpose:** Image-to-image translation for document enhancement (removing blur, noise, shadows, and improving text clarity)[cite: 34].

#### Architecture Details
| Property | Value |
|----------|-------|
| **Architecture Type** | U-Net (Encoder-Decoder with Skip Connections)[cite: 34] |
| **Total Parameters** | 31,037,763 (~31M)[cite: 34] |
| **Trainable Parameters** | 31,037,763 (100% trainable)[cite: 34] |
| **Input Channels** | 3 (RGB)[cite: 34] |
| **Output Channels** | 3 (RGB enhanced)[cite: 34] |
| **Default Input Size** | 512×512 pixels (Upgraded from 256×256 for text topography preservation) |
| **Output Activation** | Sigmoid (range [0, 1])[cite: 34] |
| **Upsampling Method** | Bilinear interpolation (configurable)[cite: 34] |

#### Layer Breakdown
| Layer Type | Count | Description |
|------------|-------|-------------|
| **Convolutional (Conv2d)** | 23 | 3×3 kernels with padding=1[cite: 34] |
| **Batch Normalization** | 18 | After each convolution[cite: 34] |
| **Dropout** | 18 | Configurable rate (Default: 0.0 for champion model) |
| **Max Pooling** | 4 | 2×2 downsampling in encoder[cite: 34] |
| **Linear/FC** | 0 | Fully convolutional architecture[cite: 34] |

#### Channel & Spatial Progression
| Stage | Operation | Input Ch | Output Ch | Spatial Resolution |
| --- | --- | --- | --- | --- |
| Input | - | 3 | - | 512×512 |
| inc | DoubleConv | 3 | 64 | 512×512 |
| down1 | MaxPool + DoubleConv | 64 | 128 | 256×256 |
| down2 | MaxPool + DoubleConv | 128 | 256 | 128×128 |
| down3 | MaxPool + DoubleConv | 256 | 512 | 64×64 |
| down4 | MaxPool + DoubleConv | 512 | 1024 | 32×32 |
| up1 | Upsample + DoubleConv | 1024 | 512 | 64×64 |
| up2 | Upsample + DoubleConv | 512 | 256 | 128×128 |
| up3 | Upsample + DoubleConv | 256 | 128 | 256×256 |
| up4 | Upsample + DoubleConv | 128 | 64 | 512×512 |
| outc | 1×1 Conv | 64 | 3 | 512×512 |

---

### 2. CornerRegressionModel (Regression-based Corner Detection)
**Purpose:** Direct regression of 8 normalized coordinates representing 4 document corners[cite: 34].

| Property | Value |
| --- | --- |
| **Architecture Type** | CNN Encoder + Fully Connected Head[cite: 34] |
| **Total Parameters** | 14,531,848 (~14.5M)[cite: 34] |
| **Output** | 8 values (4 corners × 2 coordinates each)[cite: 34] |
| **Performance Note** | Maintained for architectural comparison; computationally lighter but mathematically inferior for sub-pixel accuracy. |

---

### 3. CornerHeatmapModel (Heatmap-based Corner Detection)
**Purpose:** Predict 4 Gaussian heatmaps (one per corner) and extract coordinates via soft-argmax[cite: 34].

| Property | Value |
| --- | --- |
| **Architecture Type** | U-Net + Soft-Argmax[cite: 34] |
| **Total Parameters** | 31,037,828 (~31M)[cite: 34] |
| **Output** | 4 heatmaps + 8 coordinates[cite: 34] |

#### The "Floating Point" Fix & Smart Ensemble
*   **Dual-Temperature Soft-Argmax:** Uses `beta=300.0` during training for soft gradient flow and `beta=10000.0` during inference to snap to the absolute highest-confidence pixel, eliminating floating points[cite: 34].
*   **Smart Ensemble Integration:** At the UI/Inference level, the system processes raw images through multiple trained variations of the Heatmap model (e.g., v2 and v3). It blurs the resulting heatmaps, calculates peak intensity, and selectively extracts the highest-confidence individual corners across models to dynamically compose a highly robust bounding box.

---

## Comparative Analysis & Memory Footprint

| Model | Total Parameters | Conv Layers | FC Layers | Approx. GPU Memory (512×512) |
| --- | --- | --- | --- | --- |
| **EnhancementUNet** | ~31M | 23 | 0 | ~2.0 GB (with activations) |
| **CornerRegression** | ~14.5M | 4 | 3 | ~800 MB (with activations) |
| **CornerHeatmap** | ~31M | 23 | 0 | ~2.0 GB (with activations) |

---

## Training Configuration & Systemic Optimization

### Loss Functions
**EnhancementUNet (Heavily Weighted for Text):**
*   **Smooth L1 Loss:** Pixel-wise reconstruction with a targeted text_weight mapping of `5.0` to penalize faded ink.
*   **MS-SSIM Loss:** Structural similarity weighting of `1.5`.
*   **Sobel Edge Loss:** Gradient penalty aggressively weighted at `2.0` to force sharp character boundaries and combat "washed-out" artifacts.
*   **Cosine Color Loss:** Similiarity dimension mapped at `0.5`.

**CornerHeatmapModel:**
*   Pixel-wise weighted MSE on Gaussian heatmaps[cite: 34].

### CPU/IPC Deadlock Resolution (Windows)
Upgrading the Enhancement training to 512×512 resolution introduced severe Inter-Process Communication (IPC) deadlocks on Windows environments. To stabilize the pipeline:
1.  **OpenCV Threading Disabled:** Explicitly applied `cv2.setNumThreads(0)` to prevent internal threading conflicts with PyTorch dataloaders.
2.  **Dataset Streaming:** Disabled `freeze_data` memory caching for the training split to prevent RAM exhaustion (OOM).
3.  **Batch & Worker Optimization:** Batch size optimized to `4-8` with `num_workers=0` (or `2` conditionally) to maintain a steady GPU feed without starving CPU resources.

---

## Performance Metrics

*(Evaluated on real-world unconstrained test set against Ground Truth corners to bypass spatial shift penalizations)*

| Metric | EnhancementUNet (No Dropout) | CornerHeatmap | CornerRegression |
| --- | --- | --- | --- |
| **PSNR** | ~13.66 dB | N/A | N/A |
| **SSIM** | ~0.5896 | N/A | N/A |
| **OCR Confidence** | ~31.56% | N/A | N/A |
| **Corner MSE** | N/A | ~1532.8 px² | > 5000 px² |
| **Corner MAE** | N/A | ~21.4 px | Variable |
| **Corner MLE** | N/A | ~35.0 px | ~68.2 px |

---

## Version History

### Phase 5: Clean Models
Strict adherence to zero-dropout rules. High synthetic accuracy but experienced semantic failure in real-world environments[cite: 34].

### Phase 6: Regularized Models & Architectural Benchmarks
Introduction of dynamic dropout curriculums to force semantic understanding, successfully reducing MAE[cite: 34]. Proven that Dropout is catastrophic for spatial coordinate extraction (Corner models) but structurally useful in standard tasks[cite: 34].

### Phase 7: Enhancement & Systemic Stabilization (Current)
*   **The 512×512 Leap:** Quadrupled Enhancement resolution to preserve character-level topography.
*   **IPC Deadlock Resolution:** Implemented zero-thread constraints and pipeline streaming to bypass Windows resource starvation.
*   **Loss Tuning:** Aggressively weighted Sobel Edge (2.0) and Text (5.0) to cure "washed-out" artifacts.
*   **Smart Ensemble Deployment:** Added dynamic UI-level multi-model confidence selection for robust corner prediction.
*   **Post-Processing Paradox:** Tested Adaptive Sauvola Binarization; discovered it dropped PSNR by ~2.5 dB and slightly reduced OCR by fragmenting ink strokes. Relegated Binarization to an optional UI toggle. The raw, Baseline U-Net (No Dropout) remains the champion model.
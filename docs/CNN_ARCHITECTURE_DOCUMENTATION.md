# CNN Architecture Documentation - Document Scanning & Enhancement

## Project Overview

This repository implements a complete deep learning pipeline for document scanning and enhancement using Convolutional Neural Networks (CNNs). The system consists of three primary neural network architectures designed to handle two main tasks:

1. **Task 1: Document Enhancement** - Image-to-image translation for removing blur, noise, shadows, and improving text clarity.


2. **Task 2: Corner Detection** - Localizing document corners for perspective correction.



---

## Model Architectures

### 1. EnhancementUNet (Primary U-Net Architecture)

**Purpose:** Image-to-image translation for document enhancement.

#### Architecture Details

| Property | Value |
| --- | --- |
| **Architecture Type** | U-Net (Encoder-Decoder with Skip Connections)

 |
| **Total Parameters** | 31,037,763 (~31M)

 |
| **Trainable Parameters** | 31,037,763 (100% trainable)

 |
| **Input Channels** | 3 (RGB)

 |
| **Output Channels** | 3 (RGB enhanced)

 |
| **Default Input Size** | 512x512 pixels (Upgraded from 256x256 for character-level topography preservation)

 |
| **Output Activation** | Sigmoid (range [0, 1])

 |
| **Upsampling Method** | Bilinear interpolation

 |

#### Layer Breakdown

| Layer Type | Count | Description |
| --- | --- | --- |
| **Convolutional (Conv2d)** | 23 | 3x3 kernels with padding=1

 |
| **Batch Normalization** | 18 | After each convolution

 |
| **Dropout** | 18 | Configurable rate (Gold Champion: 0.3 in deep bottlenecks; Silver/Baseline: 0.0)

 |
| **Max Pooling** | 4 | 2x2 downsampling in encoder

 |
| **Linear/FC** | 0 | Fully convolutional architecture

 |

#### Channel & Spatial Progression

| Stage | Operation | Input Ch | Output Ch | Spatial Resolution |
| --- | --- | --- | --- | --- |
| Input | - | 3 | - | 512x512

 |
| inc | DoubleConv | 3 | 64 | 512x512

 |
| down1 | MaxPool + DoubleConv | 64 | 128 | 256x256

 |
| down2 | MaxPool + DoubleConv | 128 | 256 | 128x128

 |
| down3 | MaxPool + DoubleConv | 256 | 512 | 64x64

 |
| down4 | MaxPool + DoubleConv | 512 | 1024 | 32x32

 |
| up1 | Upsample + DoubleConv | 1024 | 512 | 64x64

 |
| up2 | Upsample + DoubleConv | 512 | 256 | 128x128

 |
| up3 | Upsample + DoubleConv | 256 | 128 | 256x256

 |
| up4 | Upsample + DoubleConv | 128 | 64 | 512x512

 |
| outc | 1x1 Conv | 64 | 3 | 512x512

 |

---

### 2. CornerRegressionModel (Approach A - Direct Regression)

**Purpose:** Direct coordinate regression of 8 normalized values representing 4 document corners.

| Property | Value |
| --- | --- |
| **Architecture Type** | CNN Encoder + Fully Connected Head

 |
| **Total Parameters** | 14,531,848 (~14.5M)

 |
| **Output** | 8 values (4 corners x 2 coordinates each)

 |
| **Performance Note** | Maintained for baseline comparison; `Flatten` layer destroys 2D spatial topography, resulting in higher error margins (MLE ~68.21 px).

 |

---

### 3. CornerHeatmapModel (Approach B - Heatmap Regression)

**Purpose:** Predict 4 Gaussian heatmaps (one per corner) and extract coordinates via soft-argmax.

| Property | Value |
| --- | --- |
| **Architecture Type** | U-Net + Soft-Argmax

 |
| **Total Parameters** | 31,037,828 (~31M)

 |
| **Output** | 4 heatmaps + 8 coordinates

 |

#### Sub-Pixel Precision & Smart Ensemble 3.0

* **Dual-Temperature Soft-Argmax:** Utilizes `beta=300.0` during training for differentiable gradient flow, and `beta=10000.0` during evaluation/inference to snap to absolute peak probabilities, completely eliminating floating-point localization drift.


* **Smart Ensemble 3.0 (Projective Geometry Rules):** Combines Gold (`heatmap_v4_reg`), Silver (`heatmap_v3`), and Bronze (`heatmap_v2`) models across 81 potential 4-corner permutations. Candidate quadrilaterals are filtered via strict physical heuristics:


1. **Internal Angle Range:** Constrained to $[55^\circ, 125^\circ]$.


2. **Opposite Edge Ratio:** Opposite edge lengths cannot diverge beyond $2.2\times$.


3. **Perspective Symmetry Rule:** If two adjacent angles are near $90^\circ$ ($\pm 15^\circ$), opposite angles cannot diverge from each other by more than $25^\circ$.


4. **Weighted Scoring:** $85\%$ confidence weight (giving a $1.2\times$ trust boost to Gold) + $15\%$ normalized polygon area tie-breaker.





---

## Comparative Analysis & Memory Footprint

| Model | Total Parameters | Conv Layers | FC Layers | Approx. GPU Memory (512x512) |
| --- | --- | --- | --- | --- |
| **EnhancementUNet** | ~31M | 23 | 0 | ~2.0 GB (with activations)

 |
| **CornerRegression** | ~14.5M | 4 | 3 | ~800 MB (with activations)

 |
| **CornerHeatmap** | ~31M | 23 | 0 | ~2.0 GB (with activations)

 |

---

## Training Configuration & Systemic Optimization

### Loss Functions

**EnhancementUNet (Text & Color Preservation Focus):**

* **Smooth L1 Loss:** Pixel-wise reconstruction error weighted by a text-stroke weight map (`text_weight=5.0`) to penalize faded or erased ink.


* **MS-SSIM Loss:** Structural similarity weighting mapped at `1.5`.


* **Sobel Edge Loss:** Gradient penalty aggressively weighted at `2.0` to force sharp character boundaries and cure "washed-out" text.


* **Cosine Color Loss:** Color similarity dimension weighted at `2.0` to prevent the erasure of logos, graphics, and non-text structures.



**CornerHeatmapModel:**

* Pixel-wise weighted MSE on Gaussian heatmaps.



### Hardware & Pipeline Optimizations

1. **Windows IPC Deadlock Resolution:** Applied `cv2.setNumThreads(0)` to prevent OpenCV internal threading from clashing with PyTorch DataLoader multiprocessing during high-resolution ($512\times512$) streaming.


2. **Dataset Streaming:** Disabled `freeze_data` memory caching during training to eliminate Out-Of-Memory (OOM) RAM crashes.


3. **Scheduled Bottleneck Dropout:** Fixed a global dropout bug. Dropout is applied exclusively to the deepest semantic bottleneck layers with a 5-epoch warm-up and cosine decay, preventing spatial collapse while forcing high-level semantic generalization.



---

## Performance Metrics (v2 & v4 Medalists)

*(Evaluated on synthetic test sets with GT corners and unconstrained real-world test photos)*

| Metric | Enhancement (Regularized v2) 🥇 | Enhancement (Clean v2) 🥈 | Corner Heatmap (Regularized v4) 🥇 | Corner Heatmap (Clean v3) 🥈 | Corner Regression (Baseline) |
| --- | --- | --- | --- | --- | --- |
| **Synthetic PSNR** | ~20.95 dB | **~21.54 dB** | N/A | N/A | N/A

 |
| **Synthetic SSIM** | ~0.8443 | **~0.8544** | N/A | N/A | N/A

 |
| **OCR Confidence** | **~56.90%** | ~42.66% | N/A | N/A | N/A

 |
| **Real Corner MSE** | N/A | N/A | **~318.74 px²** | ~1532.80 px² | ~3301.81 px²

 |
| **Real Corner MAE** | N/A | N/A | **~9.26 px** | ~21.42 px | ~43.22 px

 |
| **Real Corner MLE** | N/A | N/A | **~14.97 px** | ~35.04 px | ~68.21 px

 |

---

## Inference Pipeline, Gatekeepers & Post-Processing Filters

```text
Raw Input (Image/PDF) ──> [Gatekeeper A v2] ──(Not Cropped)──> [Smart Ensemble 3.0] ──> [Dynamic Perspective Warp]
                              │                                                                  │
                         (Is Cropped)                                                            │
                              └──────────────────────────────────────────────────────────────────┘
                                                              │
                                                              v
                                                     [Gatekeeper C] ──(Is Enhanced)──> Bypass U-Net
                                                              │
                                                        (Not Enhanced)
                                                              │
                                                              v
                                                    [Enhancement U-Net]
                                                              │
                                                              v
                                               [Post-Processing Filters]
                                               ├─ Magic Ink Boost (Gamma + Sharpen)
                                               └─ Adaptive Sauvola Binarization

```

### 1. Out-Of-Distribution (OOD) Gatekeepers

* **Gatekeeper A v2 (`is_already_cropped`):** Evaluates the 3% outer border of each edge (top, bottom, left, right) independently. If at least 3 edges exhibit low variance ($\text{variance} < 800$) and high brightness ($\text{white\_ratio} > 0.70$), corner detection is bypassed.


* **Gatekeeper C (`is_already_enhanced`):** Evaluates grayscale histogram distribution. If more than 40% of pixels exceed a brightness value of 240 (Right-Tail Mass), U-Net enhancement is bypassed to prevent ink fading.



### 2. Dynamic Aspect Ratio Transformation

The `apply_perspective_transform` function calculates dynamic output bounds using Euclidean distance:


$$\text{Width} = \max\left(\sqrt{(x_{br}-x_{bl})^2 + (y_{br}-y_{bl})^2},\ \sqrt{(x_{tr}-x_{tl})^2 + (y_{tr}-y_{tl})^2}\right)$$

$$\text{Height} = \max\left(\sqrt{(x_{tr}-x_{br})^2 + (y_{tr}-y_{br})^2},\ \sqrt{(x_{tl}-x_{bl})^2 + (y_{tl}-y_{bl})^2}\right)$$


This maps the predicted quadrilateral onto a dynamic $(\text{Width}, \text{Height})$ grid, maintaining the natural aspect ratio of documents (e.g., A4).

### 3. Magic Ink Boost Filter (`apply_ink_boost_filter`)

Applies a non-destructive post-processing enhancement:

1. **Luminance Gamma Correction:** Converts image to YCrCb space and applies gamma correction ($\gamma = 0.82$) strictly to the $Y$ (Luminance) channel via Look-Up Table (LUT), darkening pen strokes without distorting color hues (e.g., preserving blue ink).


2. **Unsharp Masking:** Blends the result with a Gaussian-blurred variant using `cv2.addWeighted` (sharpness $1.35$), sharpening character boundaries without threshold fragmentation.



---

## End-to-End API & UI Ecosystem

* **Backend Router (`web_app.py`):** FastAPI application serving asynchronous single-shot processing (`/scan`), page-by-page PDF reconstruction using `PyMuPDF` (`fitz`), and isolated interactive endpoints (`/interactive-detect` and `/interactive-enhance`).


* **Interactive Canvas Editor:** HTML5 Canvas frontend allowing manual drag-and-drop corner adjustment. Coordinates are normalized and scaled using dynamic bounding rect ratios ($\text{scaleX} = \text{canvas.width} / \text{rect.width}$, $\text{scaleY} = \text{canvas.height} / \text{rect.height}$) to eliminate CSS misalignment bugs.



---

## Version History & Key Milestones

### Phase 5: Clean Models Baseline

Zero-dropout training baseline. Achieved strong synthetic numbers but failed in real-world environments due to semantic traps (e.g., snapping to dark binder borders).

### Phase 6: The Data-Centric Leap & Geometric Bug Fixes

* **Dataset Poisoning Resolved:** Fixed inward-pointing normal vectors in `_add_adjacent_page` that drew false black lines across clean text targets.


* **Semantic Augmentations:** Added 3D cylindrical warps (open-book curvature), collaged graphics, and 4% colored paper backgrounds.


* **The Dropout Revelation:** Proved Global Dropout causes spatial collapse in coordinate extraction. Fixing `DropoutScheduler` to target deep bottlenecks with a 5-epoch warmup allowed **Corner Heatmap Regularized v4** to achieve a **14.97 px MLE** on real data.



### Phase 7: Enhancement Resolution & Loss Tuning

* Quadrupled resolution to $512\times512$.


* Tuned Sobel Edge (`2.0`), Text Weight (`5.0`), and Color Weight (`2.0`) to cure washed-out artifacts and preserve graphics/logos.


* Established **Regularized v2** as Gold Medalist for OCR Readability (56.90%) and **Clean Nodropout v2** as Silver Medalist for Image Fidelity (21.54 dB PSNR).



### Phase 8: Smart Ensemble 3.0 & Projective Geometry Heuristics

* Implemented the Perspective Symmetry Rule, angle boundaries $[55^\circ, 125^\circ]$, and opposite edge ratios ($2.2\times$) to eliminate ensemble reward hacking.



### Phase 9: Gatekeepers, Magic Ink Boost & Production UI

* Added OOD Gatekeepers (Border Variance & Histogram Tail-Mass Analysis) to handle pre-cropped and pre-scanned documents.


* Replaced square warp constraints with Dynamic Aspect Ratio Homography.


* Added Magic Ink Boost Filter (YCrCb Gamma + Unsharp Masking) to enhance pen stroke density.


* Finalized Interactive Canvas UI, Multi-Page PDF Engine, and API function shadowing bug fixes.
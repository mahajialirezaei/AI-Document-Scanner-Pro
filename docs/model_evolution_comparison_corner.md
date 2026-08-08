# Model Evolution & Comparison Log

**Objective:** Documenting the evolutionary path, data augmentation strategies, and evaluation metrics from the base model to the Phase-compliant regularized versions. The primary focus is tracking the transition from geometric optimization to semantic robustness, comparing Direct Regression vs. Heatmap approaches, and analyzing the impact of dropout regularization as mandated by the project guidelines[cite: 22].

## Evolution Tree

```text
[Legacy Branch] Unintentional Static Regularization Era
├── corner_heatmap -> coordconv -> robust -> 3d_camo -> extreme
└── corner_heatmap_robust_extreme_v2 (Peak Geometric Accuracy, Failed Semantic Generalization)

[Official Pipeline] Approach B: Heatmap Regression
├── [Phase 5] corner_heatmap_clean_nodropout_v2 (Baseline, Failed Semantic Trap)
├── [Phase 5] corner_heatmap_clean_nodropout_v3 (Optimized Baseline -> OVERALL CHAMPION)
├── [Phase 6] corner_heatmap_regularized_v2 (Global Dropout 0.5 -> Total Geometric Collapse)
└── [Phase 6] corner_heatmap_regularized_v3 (Bottleneck Dropout 0.3 -> Partial Geometric Collapse)

[Official Pipeline] Approach A: Direct Regression
├── [Phase 5] corner_regression_clean_nodropout (Regression Baseline -> REGRESSION CHAMPION)
└── [Phase 6] corner_regression_regularized (Global Dropout 0.5 -> Increased Instability)

```

---

## 1. Approach B: The Heatmap Iterations

### 1.1 `corner_heatmap_clean_nodropout_v2` (Phase 5)

* **Architecture:** Standard U-Net Backbone (`dropout=0.0`).
* **Additions:** Dual-Temperature SoftArgmax introduced to fix sub-pixel floating points. "Dark Binder Margins" added to synthetic training.
* **Real-World Performance:**
* Corner MAE: 21.75 px
* Corner MSE: 1285.53 px²


* **Analysis:** Extremely confident geometric extraction, but fell into the "Semantic Trap." The network confidently snapped its predictions to the external edges of dark binders instead of the actual document paper.



### 1.2 `corner_heatmap_clean_nodropout_v3` 🏆 (THE OVERALL CHAMPION)

* **Architecture:** Refined U-Net architecture (`dropout=0.0`).
* **Real-World Performance:**
* Corner MAE: **21.42 px**
* Corner MLE (Euclidean): **35.04 px**
* Corner MSE: **1532.80 px²**


* **Analysis:** The pinnacle of the Heatmap approach. By relying entirely on heavy synthetic data augmentations (Dark Binder Margins, 3D Drop Shadows, etc.) instead of artificial dropout, the network learned the true semantic boundary of the paper. It successfully bypassed dark binders without losing its precise geometric vision.



### 1.3 `corner_heatmap_regularized_v2` (Phase 6 - Global Dropout Failure)

* **Architecture:** Global `dropout=0.5` applied across all layers.
* **Real-World Performance:**
* Corner MAE: 84.86 px
* Corner MLE: 145.05 px
* Corner MSE: 45565.57 px²


* **Analysis (Total Geometric Collapse):** Randomly deactivating 50% of the neurons in early layers blinded the network to continuous lines. The network predicted random, disconnected blobs, resulting in intersecting lines (the "Butterfly Effect") when sorted via polar coordinates.



### 1.4 `corner_heatmap_regularized_v3` (Phase 6 - Selective Dropout Failure)

* **Architecture:** Selective Bottleneck Dropout (`dropout=0.3`) restricted to deepest semantic layers.
* **Real-World Performance:**
* Corner MAE: 75.18 px
* Corner MLE: 124.22 px
* Corner MSE: 42015.77 px²


* **Analysis:** Deactivating neurons in the semantic bottleneck destroyed the network's spatial coherency—its ability to relate the four corners to one another.



---

## 2. Approach A: The Direct Regression Iterations

**Overview:** Testing the direct coordinate regression approach (CNN + Fully Connected layers) to observe if a mathematically explicit output head performs better than spatial probability maps.

### 2.1 `corner_regression_clean_nodropout` (Phase 5) 🥇 (REGRESSION CHAMPION)

* **Architecture:** CNN Feature Extractor + FC Head (`dropout=0.0`).
* **Real-World Performance:**
* Corner MAE: **43.22 px**
* Corner MLE: **68.21 px**
* Corner MSE: **3301.81 px²**


* **Analysis:** Functioned as a stable but inherently less accurate baseline. The Flatten operation completely destroys the 2D spatial context of the image, forcing the network to blindly guess coordinate locations rather than maintaining spatial awareness like the U-Net.

### 2.2 `corner_regression_regularized` (Phase 6)

* **Architecture:** CNN Feature Extractor + FC Head (`dropout=0.5` in FC layers).
* **Real-World Performance:**
* Corner MAE: 49.18 px
* Corner MLE: 78.99 px
* Corner MSE: 4063.06 px²


* **Analysis:** Regression models are highly sensitive and brittle. Introducing 50% dropout in the fully connected layers caused the network to lose the fine-grained precision required to output eight exact floating-point numbers, leading to increased error margins and higher instability across real-world photos.

---

## 3. Final Conclusion & Project Reporting

As requested by the project guidelines to "observe the difference in performance" and "report the impact on both models":

**I. Approach A (Regression) vs. Approach B (Heatmap):**
The **Heatmap approach is the definitive winner**. The best Heatmap model achieved an MLE of 35.04 px compared to Regression's best MLE of 68.21 px. Regression fundamentally fails because the `Flatten` layer destroys spatial topography, whereas Heatmap's fully convolutional U-Net preserves geometry from pixels to output probabilities.

**II. The Impact of Dropout on Spatial Extraction:**
We conclusively report that **Dropout regularization acts as destructive spatial noise rather than a semantic regularizer in coordinate extraction architectures**.

* In Heatmaps, it shatters the spatial probability distribution and causes "Geometric Collapse".


* In Regression, it destabilizes the sensitive fully connected weights calculating precise decimals.

**The Ultimate Solution:**
The most robust and accurate model is the **Phase 5 Baseline Heatmap (`nodropout_v3`)**. The "Semantic Trap" (overfitting to background distractors like binders) was solved not through network regularization, but through **Data-Centric Regularization**—specifically, by actively teaching the model to ignore thick structural edges via synthetic augmentations.

```

```
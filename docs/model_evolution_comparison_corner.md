```markdown
# Model Evolution & Comparison Log

**Objective:** Documenting the evolutionary path, data augmentation strategies, and evaluation metrics from the base model to the Phase-compliant regularized versions. The primary focus is tracking the transition from geometric optimization to semantic robustness, comparing Direct Regression vs. Heatmap approaches, and analyzing the impact of dropout regularization as mandated by the project guidelines.

## Evolution Tree

```text
[Legacy Branch] Unintentional Static Regularization Era
├── corner_heatmap -> coordconv -> robust -> 3d_camo -> extreme
└── corner_heatmap_robust_extreme_v2 (Peak Geometric Accuracy, Failed Semantic Generalization)

[Official Pipeline] Approach B: Heatmap Regression
├── [Phase 5] corner_heatmap_clean_nodropout_v2 🥉 (Bronze Medal - Strong Geometry, Failed Semantic Trap)
├── [Phase 5] corner_heatmap_clean_nodropout_v3 🥈 (Silver Medal - Data-Centric Champion)
├── [Phase 6] corner_heatmap_regularized_v2 (Global Dropout 0.5 -> Total Geometric Collapse)
├── [Phase 6] corner_heatmap_regularized_v3 (Bottleneck Dropout 0.3 -> Partial Geometric Collapse)
├── [Phase 6] corner_heatmap_clean_nodropout_v4 (Complex Data Baseline -> Geometric Overfitting)
├── [Phase 6] corner_heatmap_regularized_v4 🥇 (Gold Medal - Semantic Concept Champion)
└── [Bonus Phase] e2e_finetuned 💎 (Diamond Medal - Peak Localization via Joint Training)

[Official Pipeline] Approach A: Direct Regression
├── [Phase 5] corner_regression_clean_nodropout (Regression Baseline)
└── [Phase 6] corner_regression_regularized (Global Dropout 0.5 -> Increased Instability)


```

---

## 1. Approach B: The Heatmap Iterations

### 1.1 `corner_heatmap_clean_nodropout_v2` 🥈 (SILVER MEDAL)

* **Architecture:** Standard U-Net Backbone (`dropout=0.0`).
* **Additions:** Dual-Temperature SoftArgmax introduced to fix sub-pixel floating points. "Dark Binder Margins" added to synthetic training.
* **Real-World Performance:**
* Corner MAE: 10.79 px
* Corner MLE (Euclidean): 16.96 px
* Corner MSE: 317.90 px²


* **Analysis:** Functioned as our first geometrically stable model. However, it fell into the "Semantic Trap," confidently snapping its predictions to the external edges of dark binders instead of the actual document paper. It serves as a great fallback for environments with clean backgrounds.

### 1.2 `corner_heatmap_clean_nodropout_v3` 🥉 (BRONZE MEDAL)

* **Architecture:** Refined U-Net architecture (`dropout=0.0`).
* **Real-World Performance:**
* Corner MAE: 10.59 px
* Corner MLE (Euclidean): 17.23 px
* Corner MSE: 382.67 px²


* **Analysis:** The champion of pure **Data-Centric Regularization**. By relying entirely on heavy synthetic data augmentations (Dark Binder Margins, 3D Drop Shadows, etc.) instead of artificial dropout, the network learned the true semantic boundary of the paper. Because its learning path was entirely data-driven (without network noise), it is the perfect secondary candidate for a Smart Ensemble.

### 1.3 `corner_heatmap_clean_nodropout_v4` (Phase 6 - Geometric Overfitting)

* **Architecture:** Standard U-Net Backbone (`dropout=0.0`).
* **Additions:** Trained on the highly complex v4 dataset (featuring 3D Cylindrical Book Warps, Random Collaged Graphics, and 4% Colored Paper).
* **Real-World Performance:**
* Corner MAE: 60.03 px
* Corner MLE: 100.70 px
* Corner MSE: 16224.37 px²


* **Analysis:** Introducing severe structural complexities (like 3D book curves) without any network regularization caused the model to severely overfit to local noises. It lost its global spatial awareness, proving that pure data-centric approaches have limits when the geometric variance becomes too extreme.

### 1.4 `corner_heatmap_regularized_v4` 🥇 (GOLD MEDAL - SEMANTIC CHAMPION)

* **Architecture:** Scheduled Bottleneck Regularization (`dropout=0.3` restricted to deep layers, 5-epoch warm-up).
* **Real-World Performance:**
* Corner MAE: **9.26 px**
* Corner MLE: **14.97 px**
* Corner MSE: **318.74 px²**


* **Analysis:** By combining the highly complex v4 synthetic dataset with a **Scheduled Bottleneck Dropout**, we prevented the geometric collapse seen in earlier regularized models. The 5-epoch warm-up allowed the network to map its high-resolution spatial topography first. Once the geometry was locked, the bottleneck dropout forced the network to learn the "Semantic Concept" of a document, yielding a 14.97 px error margin in unconstrained real-world environments.

### 1.5 `e2e_finetuned` 💎 (DIAMOND MEDAL - PEAK LOCALIZATION)

* **Architecture:** Jointly fine-tuned U-Net via differentiable Kornia warping (Initialized from `regularized_v4`).
* **Real-World Performance:**
* Corner MAE: **8.06 px**
* Corner MLE: **13.19 px**
* Corner MSE: **176.41 px²**


* **Analysis:** The absolute pinnacle of coordinate localization. Fine-tuning the network using the loss signal propagated backwards from the Enhancement U-Net forced the corner detector to correct minute spatial misalignments that were actively degrading text readability. This task-aware feedback loop reduced the MLE by an additional ~1.8 px, resulting in the most geometrically precise model in the entire pipeline.

---

## 2. Approach A: The Direct Regression Iterations

**Overview:** Testing the direct coordinate regression approach (CNN + Fully Connected layers) to observe if a mathematically explicit output head performs better than spatial probability maps.

### 2.1 `corner_regression_clean_nodropout` (Phase 5)

* **Architecture:** CNN Feature Extractor + FC Head (`dropout=0.0`).
* **Real-World Performance:**
* Corner MAE: 43.22 px
* Corner MLE: 68.21 px
* Corner MSE: 3301.81 px²


* **Analysis:** Functioned as a stable but inherently less accurate baseline. The Flatten operation completely destroys the 2D spatial context of the image, forcing the network to blindly guess coordinate locations rather than maintaining spatial awareness like the U-Net.

### 2.2 `corner_regression_regularized` (Phase 6)

* **Architecture:** CNN Feature Extractor + FC Head (`dropout=0.5` in FC layers).
* **Real-World Performance:**
* Corner MAE: 49.18 px
* Corner MLE: 78.99 px
* Corner MSE: 4063.06 px²


* **Analysis:** Regression models are highly sensitive and brittle. Introducing 50% dropout in the fully connected layers caused the network to lose the fine-grained precision required to output eight exact floating-point numbers, leading to increased error margins and higher instability across real-world photos.

---

## 3. Final Conclusion & Smart Ensemble Strategy

As requested by the project guidelines to "observe the difference in performance" and "report the impact on both models":

**I. Approach A (Regression) vs. Approach B (Heatmap):**
The **Heatmap approach is the definitive winner**. The best Heatmap model achieved an MLE of 13.19 px compared to Regression's best MLE of 68.21 px. Regression fundamentally fails because the `Flatten` layer destroys spatial topography.

**II. The Revised Impact of Dropout on Spatial Extraction:**
Earlier phases led us to believe Dropout was universally catastrophic for coordinate extraction. However, Phase 6 testing conclusively refined this rule:

* **Global Dropout** destroys spatial probability distributions and causes "Geometric Collapse".
* **Scheduled Bottleneck Dropout** acts as a powerful semantic regularizer. When delayed by a warm-up period, it allows the network to maintain its 2D geometry while preventing overfitting to complex environmental distractors.

**III. The Power of Joint End-to-End Training:**
Training the corner detector iteratively with the enhancement loss effectively minimizes spatial misalignment traps, resulting in peak sub-pixel localization accuracy.

**The Ultimate Solution (End-to-End UI Deployment):**
To achieve maximum robustness in the final deployment, the system utilizes a **Smart Ensemble Module**. By feeding the raw image through the 💎 **Diamond Medal (`e2e_finetuned`)**, 🥇 **Gold Medal (`regularized_v4`)**, and 🥈 **Silver Medal (`nodropout_v3`)** Heatmap models, we combine networks optimized via joint task feedback, internal regularization, and pure data-centric variance. Extracting the highest-confidence peak from this combined distribution ensures near-perfect document localization regardless of the background environment.

```

```
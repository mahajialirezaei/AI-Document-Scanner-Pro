# Model Evolution & Comparison Log

**Objective:** Documenting the evolutionary path, data augmentation strategies, and evaluation metrics from the base model to the strict Phase-compliant regularized versions, specifically addressing the transition from geometric optimization to semantic robustness[cite: 14].

## Evolution Tree

```text
[Legacy Branch] Unintentional Static Regularization Era
├── corner_heatmap -> coordconv -> robust -> 3d_camo -> extreme
└── corner_heatmap_robust_extreme_v2 (Peak Geometric Accuracy, Failed Semantic Generalization)

[Official Pipeline] Strict Document Phase Compliance
├── [Phase 5] corner_heatmap_clean_nodropout_v2/v3 (Zero Regularization Baseline)
├── [Phase 6 - Failed] corner_heatmap_regularized_v2 (Global Dropout 0.5 -> Geometric Collapse)
└── [Phase 6 - Final] corner_heatmap_regularized_v3 (Selective Bottleneck Dropout 0.3)

```

## 1. `corner_heatmap_robust_extreme_v2` (The Legacy Optimizer)

**Overview:** The culmination of our data pipeline engineering. This model resolved the geometric desynchronization of physical paper curl by applying `ElasticTransform` on a 4-channel matrix (RGB + hole mask) *before* perspective warping, and utilized persistent CPU workers to drastically reduce training bottlenecks.

* **Dropout Status:** Unintentionally trained with a static `dropout=0.2`.


* **Performance / Corner MAE:** **25.57 px** (End-to-End Real Data).


* **Key Finding (The Semantic Trap):** While geometrically sound, the model exhibited severe semantic confusion. It learned to hunt for the highest contrast 90-degree angles in the image, successfully latching onto physical distractors like dark binders and desk edges instead of the true paper corners. Furthermore, the `SoftArgmax2D` layer often predicted coordinates floating in mid-air (averaging multiple high-confidence background spots). This proved the necessity for formal regularization.



---

## 2. `corner_heatmap_clean_nodropout_v2` (Phase 5: The Pure Baseline)

**Overview:** Developed in strict compliance with Phase 5 of the project guidelines, which mandates zero regularization and no dropout layers for the initial architecture.

* **Additions & Fixes:**
* Forced `dropout=0.0` across all UNet and Corner Regression blocks.


* Re-introduced "Dark Binder Margins" to the synthetic dataset to explicitly test the model's robustness against strong background edges.
* Upgraded `SoftArgmax2D` with Dual-Temperature scaling (using `beta=10000.0` during evaluation) to forcefully eliminate mid-air "floating point" predictions.




* **Real-World Performance (Corner MAE):** **21.75 px**
* **Known Issues (The Overfitting Reality):** As expected from a 0.0 dropout model, it became highly overfitted to the superficial features of the synthetic dataset. Visual evaluations confirmed that the Dual-Temperature SoftArgmax successfully eliminated floating points, but the predictions snapped with extreme confidence to the external edges of dark binders instead of the white paper. The model entirely ignored document textures and text boundaries.



---

## 3. The `v2` Global Dropout Failure (Catastrophic Geometric Collapse)

**Overview:** Our initial attempt to regularize the network (Phase 6) involved applying a global dropout rate of `0.5` across all encoder and decoder layers using a Cosine Annealing scheduler.

* **Performance / Corner MAE:** Spiked to **84.86 px**
* **Performance / Corner MLE (Euclidean):** Spiked to **145.05 px**
* **Analysis:** The network suffered a complete **Geometric Collapse**. By randomly deactivating 50% of the neurons in the *early* feature-extraction layers (which are responsible for detecting low-level lines and simple contrasts), the network effectively went "blind".
* **The "Butterfly Effect":** Because the network lost its foundational ability to perceive a quadrilateral, it predicted four random, disconnected points in space. When the polar coordinate sorting algorithm (`order_points`) attempted to connect these logically disjointed points, it generated intersecting lines (bowtie/butterfly shapes) and placed points hundreds of pixels away from any object.

---

## 4. `corner_heatmap_regularized_v3` (Phase 6: The Final Robust Curriculum)

**Overview:** The architectural correction to the geometric collapse. This version intelligently restricts regularization to the deeper, semantic layers of the network, preventing geometric amnesia while forcing the model to solve the Semantic Trap.

* **Architectural Refinements:**
* **Selective Bottleneck Dropout:** Modified the `EnhancementUNet` backbone so that the initial feature extraction blocks (`inc`, `down1`, `down2`) operate with `dropout=0.0`. Dropout is exclusively applied to the deepest layers (`down3`, `down4`, `up1`) where semantic decisions (e.g., "Is this a paper edge or a binder edge?") are processed.
* **Moderated Regularization Rate:** Reduced the target dropout rate from `0.5` to `0.3` to maintain stability against the highly complex 3D distractors now present in the dataset.
* **Curriculum Learning:** The model starts with a 5-epoch warmup at `dropout=0.0` to establish basic spatial awareness of the document geometry. Over the remaining epochs, the dropout gradually scales to `0.3`.




* **Expected Outcome:** By keeping the network's "eyes" (early layers) open but restricting its "brain" (deep layers), the model can clearly see the geometry of both the paper and the binder, but is forced to use auxiliary features (like text alignment and paper color) to determine which quadrilateral is the true document.

```
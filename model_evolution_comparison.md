# Model Evolution & Comparison Log

**Objective:** Documenting the evolutionary path, data augmentation strategies, and evaluation metrics from the base model to the strict Phase-compliant regularized versions, specifically addressing the transition from geometric optimization to semantic robustness.

## Evolution Tree

```text
[Legacy Branch] Unintentional Static Regularization Era
├── corner_heatmap -> coordconv -> robust -> 3d_camo -> extreme
└── corner_heatmap_robust_extreme_v2 (Peak Geometric Accuracy, Failed Semantic Generalization)

[Official Pipeline] Strict Document Phase Compliance
├── [Phase 5] corner_heatmap_clean_nodropout (Zero Regularization Baseline)
└── [Phase 6] corner_heatmap_regularized (Dynamic Dropout Curriculum)

```

## 1. `corner_heatmap_robust_extreme_v2` (The Legacy Optimizer)

**Overview:** The culmination of our data pipeline engineering. This model resolved the geometric desynchronization of physical paper curl by applying `ElasticTransform` on a 4-channel matrix (RGB + hole mask) *before* perspective warping, and utilized persistent CPU workers to drastically reduce training bottlenecks.

* **Dropout Status:** Unintentionally trained with a static `dropout=0.2`.
* **Performance / Corner MAE:** **25.57 px** (End-to-End Real Data).
* **Key Finding (The Semantic Trap):** While geometrically sound, the model exhibited severe semantic confusion. It learned to hunt for the highest contrast 90-degree angles in the image, successfully latching onto physical distractors like dark binders and desk edges instead of the true paper corners. Furthermore, the `SoftArgmax2D` layer often predicted coordinates floating in mid-air (averaging multiple high-confidence background spots). This proved the necessity for formal regularization.

---

## 2. `corner_heatmap_clean_nodropout` (Phase 5: The Pure Baseline)

**Overview:** Developed in strict compliance with Phase 5 of the project guidelines, which mandates zero regularization and no dropout layers for the initial architecture.

* **Additions & Fixes:**
* Forced `dropout=0.0` across all UNet and Corner Regression blocks.
* Removed artificial black border drawing and direct corner occlusion patches to test pure optical understanding.


* **Synthetic Performance:** Achieved an astonishingly low Validation Loss of **0.0092**.
* **Real-World Performance (Corner MAE):** Spiked to **31.98 px**.
* **Known Issues:** This model is the textbook definition of synthetic domain overfitting. By removing all dropout constraints, the network relied entirely on superficial features (high-contrast edges). When evaluated on real data (e.g., a white paper on a black binder), it completely ignored paper texture and text alignment, snapping its predictions to the binder's external edges with extreme confidence.

---

## 3. `corner_heatmap_regularized` (Phase 6: The Robust Curriculum)

**Overview:** The final evolution, addressing Phase 6 of the project. This model targets the "Semantic Trap" by intentionally degrading the network's capacity during training, forcing it to look beyond superficial high-contrast edges and understand the actual content (text lines, paper texture, and margins).

* **Additions & Fixes:**
* **Dynamic Dropout Scheduler:** Implemented a Cosine Annealing scheduler for the dropout layers.
* **Curriculum Learning:** The model starts with a 5-epoch warmup at `dropout=0.0` to establish basic spatial awareness of the document geometry. Over the remaining 25 epochs, the dropout gradually increases to `0.5`.


* **Expected Outcome:** By randomly blinding half of the network's neurons in later epochs, the model can no longer rely solely on a single prominent background edge (like a black folder). It is mathematically forced to cross-reference multiple features (the corner itself + text layout + paper color) to survive the loss function, bridging the synthetic-to-real gap.
# Model Evolution & Comparison Log

**Objective:** Documenting the evolutionary path, data augmentation strategies, and evaluation metrics from the base model to the strict Phase-compliant regularized versions, specifically addressing the transition from geometric optimization to semantic robustness[cite: 16].

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




* **Actual Outcome & Performance:**
* **Real-World Performance (Corner MAE):** Dropped significantly to **21.01 px**.
* **Analysis:** The dropout scheduler successfully prevented severe overfitting. By randomly blinding half of the network's neurons in later epochs, the model was forced to cross-reference multiple structural features (paper texture + text layout) rather than solely relying on high-contrast external edges.


* **Remaining Bottleneck & Final Pipeline Upgrades:**
* **The "Floating Points" Issue:** While the MAE improved by ~11 pixels compared to the pure baseline, visual evaluations revealed that predictions occasionally hovered in mid-air between the paper corner and nearby dark binder edges.
* **Applied Fix 1 (SoftArgmax Temperature):** Identified that standard `SoftArgmax` averages probabilities between multiple hotspots. Modified `SoftArgmax2D` to use a high temperature parameter (`beta=10000.0`) during inference, forcing sharp, decisive predictions rather than weighted spatial averages.
* **Applied Fix 2 (Targeted Distractors):** Updated the data generator to explicitly render "Dark Binder Margins" near synthetic paper edges, actively teaching the model to reject these specific high-contrast physical distractors during the regularization phase.



```

```
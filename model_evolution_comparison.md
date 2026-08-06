# Model Evolution & Comparison Log

**Objective:** Documenting the evolutionary path, data augmentation strategies, and evaluation metrics from the base model to the upcoming extreme-robustness version.

## Evolution Tree

```text
corner_heatmap (Base Optical Model)
└── corner_heatmap_coordconv (Architectural Upgrade)
    └── corner_heatmap_robust (Structural Distractors)
        └── corner_heatmap_robust_3d_camo (Advanced 3D & Camouflage)
            ├── corner_heatmap_robust_extreme (Over-Degraded Branch)
            └── [Upcoming] corner_heatmap_robust_extreme_v2 (Optimized Pipeline & Refined Degradation)

```

## 1. `corner_heatmap` (Base Model)



**Overview:** The foundational training phase focused on basic digital and optical degradations (blur, noise, simple perspective).

* **Additions:** Basic synthetic dataset, optical camera flaws.


* **Performance:** Suffered from coordinate sorting logic errors during early evaluation; struggled with real-world edge clutter.



## 2. `corner_heatmap_coordconv` (Architectural Upgrade)



**Overview:** Addressed the "floating-point" sub-pixel localization issue by modifying the network architecture.

* **Additions:** Implemented `CoordConv` layers and upgraded to `SoftArgmax2D` with a high `beta` parameter for sharp activations.


* **Performance / Corner MSE:** **~4660 px²**

* **Known Issues:** Overfitted to the synthetic domain. The high confidence caused the model to lock onto high-contrast physical distractors (binders, open books) instead of the actual paper.



## 3. `corner_heatmap_robust` (Structural Distractors)



**Overview:** The first major phase of targeted adversarial training to force the model to learn semantic document structures rather than raw contrast.

* **Additions:**

* Synthetic binder margins (thick textured borders).


* Adjacent page rendering (open book simulation with dark spine).


* Color temperature shifts (warm/cool lighting).


* Targeted corner occlusion masks.




* **Performance / Corner MSE:** **1512.87 px²** (Massive ~67% improvement).


* **Known Issues:** Failed against 3D objects with drop shadows (e.g., pencil cases) and white objects causing color camouflage (e.g., white mouse on white paper).



## 4. `corner_heatmap_robust_3d_camo` (Advanced 3D & Camouflage)



**Overview:** Fine-tuned to overcome the "Natural Adversarial Patch" effect caused by 3D depth and color blending.

* **Additions:**

* 3D Drop Shadow Distractors (simulating physical depth).


* Camouflage Polygons (forcing reliance on structural edges).


* Severe corner occlusion by dark objects.




* **Performance / Corner MSE:** **1289.51 px²** (Further ~15% improvement).


* **Known Issues:** Still vulnerable to massive structured occlusions (like sleeves/arms covering entire corners), true spiral cutouts (where the background shows through the paper), and physical paper curl near bindings.



## 5. `corner_heatmap_robust_extreme` (Over-Degraded Branch)

**Overview:** The previous fine-tuning phase targeting complex occlusions and curl, which unexpectedly backfired due to excessive degradation.

* **Additions:** Included `_add_clothing_occlusions` (simulating sleeves), upgraded binding artifacts, and `ElasticTransform` for 3D paper curl.
* **Performance / Corner MSE:** Spiked to **~3500-4900 px²**.
* **Known Issues:** The model suffered from severe geometric confusion. By painting thick black borders and directly occluding corners with dark polygons, the model's ability to trust spatial features was destroyed. Furthermore, applying `ElasticTransform` *after* perspective warping caused a fatal spatial desynchronization between the physical paper curl and the flat ground-truth coordinates.

## 6. [Upcoming] `corner_heatmap_robust_extreme_v2` (Optimized Pipeline & Refined Degradation)

**Overview:** The new, highly-optimized fine-tuning phase. This version acts as a direct correction to the `extreme` branch, focusing on structural precision, sub-pixel accuracy, and eliminating systemic bottlenecks.

* **Planned Additions & Fixes:**
* **Pruned Augmentations:** Completely disabled manual binder borders and direct corner occlusions to restore geometric confidence. Sleeve occlusions are moderated and pushed further from the true corners.
* **RGBA Curl Synchronization:** `ElasticTransform` is now applied to a 4-channel matrix (RGB + hole mask) *before* perspective wrapping, ensuring ground truth coordinates perfectly track the curved paper.
* **Processing Optimization:** Resolved CPU/RAM starvation by executing `cv2.resize` upstream and employing multi-worker dataloaders (`persistent_workers=True`), drastically reducing epoch times.
* **Evaluation Hardening:** Evaluation purely relies on `SoftArgmax2D` output with Polar Coordinate (`arctan2`) sorting, fully abandoning fragile geometric heuristics (like the parallelogram assumption) and introducing Mean Absolute Error (MAE) for clear sub-pixel accuracy tracking.



```

```
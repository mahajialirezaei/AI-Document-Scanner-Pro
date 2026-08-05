# Model Evolution & Comparison Log

**Objective:** Documenting the evolutionary path, data augmentation strategies, and evaluation metrics from the base model to the upcoming extreme-robustness version.

## Evolution Tree

```text
corner_heatmap (Base Optical Model)
└── corner_heatmap_coordconv (Architectural Upgrade)
    └── corner_heatmap_robust (Structural Distractors)
        └── corner_heatmap_robust_3d_camo (Advanced 3D & Camouflage)
            └── [Upcoming] corner_heatmap_robust_extreme (Complex Occlusions & Curl)

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

## 5. [Upcoming] `corner_heatmap_robust_extreme` (Complex Occlusions & Curl)

**Overview:** The upcoming fine-tuning phase targeting the final edge-cases observed in real-world extreme scanning.

* **Planned Additions:**
* `_add_clothing_occlusions`: Large multi-colored polygons (yellow, blue, grey) simulating sleeves/arms (Strictly 10-15% probability).
* Upgraded `_add_binding_artifacts`: True spiral cutouts using alpha transparency to reveal the background through the paper.
* `ElasticTransform`: Non-linear grid distortions in `degradation.py` to simulate 3D paper curl.



```
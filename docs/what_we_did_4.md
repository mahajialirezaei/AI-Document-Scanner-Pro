## Overcoming the Synthetic Domain Gap and Adversarial Real-World Distractors

### 1. The "Sharpness" Dilemma
We successfully eliminated the "floating point" coordinate issue by increasing the `beta` hyperparameter in the `SoftArgmax2D` layer. However, this high confidence introduced a new problem: **Overfitting to the synthetic domain**. Because synthetic data lacked complex real-world clutter, the model started confidently locking onto the wrong high-contrast edges instead of the subtle document boundaries, causing the Corner MSE to jump to ~4660 px².

### 2. Identifying Structural Failure Modes
By analyzing the evaluation script outputs (`eval_01` through `eval_23`), we categorized the failures into specific environmental traps:
*   **The Binder/Folder Trap:** High-contrast plastic binder edges were predicted as the document boundary (e.g., `eval_06`, `eval_15`, `eval_22`, `eval_23`).
*   **The Adjacent Page Trap:** On open books, the model bounded the opposite page instead of isolating the target page (e.g., `eval_07`).
*   **The Spiral & Lighting Trap:** Deformed edges near notebook spirals (`eval_20`) and varying room lighting threw off the edge detection.

### 3. Structural Degradations (Phase 1 Fixes)
To force the model to rely on semantic document structure rather than just raw contrast, we heavily upgraded `dataset.py` and `degradation.py`:
*   **Synthetic Binder Margins:** Added thick, textured borders to simulate physical folders.
*   **Adjacent Page Simulation:** Drawn secondary pages with a dark spine crease.
*   **Color Temperature Shift:** Fast R/B channel scaling to simulate warm (tungsten) and cool (shade) lighting.
*   **Targeted Distractors:** Injected fingers and random polygons directly occluding the corners.

### 4. Intermediate Results & The 3D Camouflage Challenge
Training the `corner_heatmap_robust` model (Resumed from `coordconv` weights) for 60 epochs yielded massive improvements. **Average Corner MSE dropped from 4660 to 1512 (a ~67% reduction).** The model successfully learned to ignore binders, open books, and human hands.

However, a new class of **Natural Adversarial Distractors** was discovered:
*   **Camouflage (e.g., `eval_01`):** A white mouse resting on a white paper boundary blended the edges, causing the model to extend the bounding box over the mouse.
*   **3D Drop Shadows (e.g., `eval_17`):** A dark pencil case casting a soft, 3D shadow over the corner acted as an adversarial patch, pulling the prediction away from the paper.

### 5. Next Steps for Phase 2
To resolve the 3D and camouflage traps without altering the neural network architecture, we have designed three advanced synthetic augmentations to be added to `dataset.py`:
1.  **3D Drop Shadow Distractors:** Rendering objects with offset dark gradients to simulate physical depth.
2.  **Camouflage Polygons:** Injecting white/off-white shapes tangent to the document edges.
3.  **Corner Occlusion by Dark Objects:** Placing soft-edged dark rectangles that completely sever a document corner.

The model will then undergo further fine-tuning (Resuming from the current 60-epoch checkpoint) to build immunity against these final real-world illusions.
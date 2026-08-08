## Phase 3: Resolving Spatial Desynchronization, Geometric Fallacies, and Processing Bottlenecks

### 1. The "Extreme Distractor" Backfire (Pivot from Phase 2)
In Phase 2 (what_we_did_4), we planned to introduce severe 3D camouflage and corner-severing occlusions to harden the model. However, training the `corner_heatmap_robust_extreme` model with these extreme distractors caused the Average Corner MSE to spike back up to ~3500-4900 px². 
**What Changed:** We realized we were over-degrading the images. By painting thick black borders (`_add_binder_margins`) and dark polygons (`_add_corner_occlusions`) directly *over* the ground truth points, we destroyed the model's geometric confidence. 
**The Fix:** We pruned the adversarial augmentations. We disabled the explicit corner/binder occlusions and pushed the clothing/sleeve occlusions further away from the absolute corners, forcing the network to learn real document boundaries rather than chasing artificial black patches.

### 2. Fixing Spatial Desynchronization (The Elastic Curl Bug)
A critical logic flaw was discovered in the degradation pipeline. The 3D paper curl (`ElasticTransform`) was being applied *after* the perspective warp and ground-truth corner assignment. The model was looking at curved paper, but the loss function was penalizing it based on straight, flat coordinates.
**The Fix:** We moved the `ElasticTransform` upstream into `dataset.py`. The clean scan and the hole mask are now combined into a 4-channel (RGBA) matrix and curled *before* the perspective transformation. This guarantees that the geometric ground truth perfectly synchronizes with the physical warping of the paper.

### 3. Eliminating Evaluation Bugs (The Parallelogram Fallacy & Butterfly Effect)
The evaluation script (`evaluate_real_data.py`) was artificially inflating our error metrics (causing extreme 20,000+ px² outliers) due to two major mathematical flaws:
*   **The Sorting Paradox:** The basic top/bottom sorting algorithm failed when predictions were slightly rotated. **Fix:** Implemented a robust Polar Coordinate sorting algorithm using `arctan2` relative to the predicted centroid, ensuring a perfect clockwise order.
*   **The Recovery Bug:** Low-confidence corners triggered a heuristic that tried to calculate the missing point using vector math (assuming the paper was a perfect parallelogram). Under real-world perspective distortion, this linearly projected the points far out of bounds. **Fix:** Removed the geometric recovery block entirely, trusting the sub-pixel precision of the `SoftArgmax2D` layer.
*   **Metric Addition:** Introduced **Mean Absolute Error (MAE)** alongside MSE to provide a more interpretable, linear measurement of sub-pixel accuracy.

### 4. Overcoming CPU/RAM Starvation and Training Bottlenecks
The data generation pipeline was causing the system to freeze during the `Data Split` phase and pushing epoch times to ~300 seconds. The CPU was starved because it was applying heavy non-linear matrix operations on full-resolution (12-megapixel) clean scans.
**The Fix:** 
*   Moved `cv2.resize` to the absolute beginning of the pipeline (immediately post-imread) to shrink the computational payload.
*   Optimized PyTorch's `DataLoader` in `train.py` by enabling multiprocessing (`num_workers=4`, `persistent_workers=True`, and `prefetch_factor=2`).
*   **Result:** The pipeline freezing was eliminated, and epoch training time was slashed by ~50% (down to ~147 seconds), allowing the GPU to operate efficiently.
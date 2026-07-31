# Project Correction Directives (FIXING-REQ.md)
**Target Audience:** AI Development Agent
**Objective:** Refactor and fix critical compliance flaws and architectural deviations identified in the codebase. The implementation must strictly adhere to the requirements of the "Document Scanning Enhancement" project.

---

## Task 1: Implement True Synthetic Background Compositing
**Target Files:** `src/data/dataset.py`, `src/data/degradation.py`
**Context & Rationale:** The project fundamentally relies on "zero-annotation" synthetic data generation. The current implementation mistakenly relies on manual annotations (`_annotations.coco.json`) for training data, which violates the core logic of Section 1.3.
**Current Flaw:** `DocumentScanningDataset` reads JSON labels. No background compositing is implemented.
**Actionable Directives:**
1. **Refactor `SyntheticDocumentDataset`:** Modify the dataset class to accept two directories: `clean_scans` and `random_backgrounds`. Remove the dependency on the JSON annotation file for training.
2. **Generate Random Points:** For each sample, select a random background image and a clean scan. Generate 4 random, logically placed coordinates on the background image (representing the target corners).
3. **Compute Homography:** Use `cv2.getPerspectiveTransform` to calculate the transformation matrix from the flat scan's corners to the 4 random background points.
4. **Warp and Composite:** Use `cv2.warpPerspective` to project the clean scan onto the background. 
5. **Set Labels:** The 4 random background points generated in Step 2 MUST be returned directly as the ground-truth corner labels (`corners`) for the Corner Detection Network.

---

## Task 2: Implement "Warp-Back" for Enhancement Network Inputs
**Target Files:** `src/data/dataset.py`, `src/data/degradation.py`
**Context & Rationale:** The Enhancement Network MUST operate on "rectified crops" (flat images), not raw, angled photos. To compute pixel-wise loss, the degraded input must perfectly align with the clean target.
**Current Flaw:** `apply_perspective_distortion` is defined but never called in the degradation pipeline. The model is bypassing the perspective alignment challenge.
**Actionable Directives:**
1. **Integrate Degradation:** After compositing the image in Task 1, apply the photometric degradations (blur, noise, shadows, etc.) to the composite image.
2. **Calculate Inverse Homography:** Calculate the inverse transformation matrix (`H_inv`) to map the 4 random background points back to a flat, rectangular shape matching the original clean scan's dimensions.
3. **Warp-Back (Rectification):** Apply `cv2.warpPerspective` using `H_inv` on the *degraded composite image*.
4. **Assign Targets:** The result of Step 3 must be returned as `rectified_input` (input for the U-Net), and the original unmodified clean scan must be returned as `clean_target`. 
5. **Cleanup:** Remove the unused `apply_perspective_distortion` function from `degradation.py`, as perspective is now handled inherently by the compositing and warp-back process.

---

## Task 3: Refactor End-to-End (E2E) Training to a Differentiable Sequential Chain
**Target Files:** `src/pipelines/train_e2e.py`
**Context & Rationale:** Section 7 requires a sequential, differentiable pipeline where the output of the corner detector is used to crop the image *during training*, and the error flows backward from the Enhancement network to the Corner network.
**Current Flaw:** The code implements Multi-Task Learning with a Shared Backbone, feeding the raw image to both networks simultaneously. `kornia` is completely ignored.
**Actionable Directives:**
1. **Remove Multi-Task Logic:** Completely remove `SharedBackboneNetwork`, `train_step_alternating`, and the independent loss combinations (Multi-Task Loss).
2. **Build Sequential Forward Pass:** In `train_step_joint`, implement the following exact chain:
    * Pass the raw image tensor to the Corner Detector to get predicted corners.
    * Use `kornia.geometry.transform.get_perspective_transform` to compute the homography from the predicted corners to a flat rectangle.
    * Use `kornia.geometry.transform.warp_perspective` to extract the rectified crop from the raw image.
    * Pass this differentiable rectified crop to the Enhancement Network.
3. **Fine-Tuning:** Calculate the loss ONLY using the Enhancement Loss (comparing the final output to the clean target). Backpropagate this single loss through the entire chain so the Corner Detector learns to predict corners that minimize enhancement artifacts.

---

## Task 4: Freeze Validation and Test Sets
**Target Files:** `src/data/dataset.py`, `src/training/train.py`
**Context & Rationale:** Because synthetic data is generated on-the-fly, the validation and test sets will change every epoch (especially with `num_workers > 0`), destroying the reliability of the validation metrics.
**Current Flaw:** Randomness is applied dynamically in `__getitem__`. The fixed Seed in `__init__` does not prevent different augmentations across epochs when multi-processing is used.
**Actionable Directives:**
1. **Implement Caching Mechanism:** Add a `freeze_data` boolean parameter to the synthetic dataset class.
2. **Pre-generate Validation/Test Data:** If `freeze_data=True`, the dataset must generate the degraded image, the rectified input, and the corners for all samples exactly *once* (either storing them in RAM inside a list/dict during `__init__`, or saving them to a temporary directory on the disk).
3. **Deterministic Fetching:** When `__getitem__` is called on a frozen dataset, it must retrieve the exact same pre-generated sample every time, bypassing the random generation functions entirely.
4. **Instantiate Correctly:** Ensure `train.py` initializes the Validation and Test datasets with `freeze_data=True`, while the Training dataset keeps `freeze_data=False`.
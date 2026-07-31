# Project Correction Directives (FIXING.md)

**Objective:** Resolve critical compliance flaws and logic inconsistencies identified during the Phase 7 code review before the final presentation. All fixes must be implemented strictly using OpenCV and PyTorch native functions.

## Git Branching Strategy

Create a single, unified bugfix branch from `develop` to address all compliance issues simultaneously. 

*   **Source Branch:** `develop`
*   **Target Branch:** `bugfix/compliance-and-metrics`
*   **Command:** `git checkout -b bugfix/compliance-and-metrics develop`

---

## Task 1: Eliminate Third-Party Augmentation Libraries

**Target File:** `src/data/dataset.py`

**Description:** The project explicitly forbids third-party libraries for transformations. The current implementation uses `albumentations`, which is a critical rule violation.

**Directives:**
1. Remove `import albumentations as A` and `from albumentations.pytorch import ToTensorV2`.
2. Delete the `get_train_transform` and `get_val_transform` functions.
3. Rewrite the transformation logic inside the `DocumentScanningDataset` or `degradation.py` using strictly `cv2` (OpenCV) functions.
4. Replace `A.RandomRotate90` with `cv2.rotate`.
5. Replace `A.RandomBrightnessContrast` with OpenCV matrix scaling operations.
6. Replace `ToTensorV2()` with standard PyTorch tensor conversions (`torch.from_numpy().permute(2, 0, 1)`).

---

## Task 2: Implement Distance Simulation Degradation

**Target File:** `src/data/degradation.py`

**Description:** The mandatory requirement to simulate camera distance (resolution loss) via consecutive downscaling and upscaling is missing from the degradation pipeline.

**Directives:**
1. Create a new function named `apply_resolution_loss` inside the `DegradationPipeline` class.
2. The function must accept an `image` and a random scale factor between 2 and 4.
3. Use `cv2.resize` with `cv2.INTER_LINEAR` or `cv2.INTER_AREA` to downscale the image by the random factor.
4. Immediately use `cv2.resize` to upscale the image back to its original dimensions.
5. Add this new function to the list of augmentations in `apply_random_degradation`.

---

## Task 3: Resolve Loss Function Inconsistency

**Target File:** `src/training/losses.py`

**Description:** The documentation claims the use of MS-SSIM for the Enhancement Network, but the actual codebase only implements L1 and Sobel Loss. 

**Directives:**
1. Import the necessary PyTorch functional operations to compute structural similarity, or implement a native SSIM/MS-SSIM function. 
2. Update the `EnhancementLoss` class initialization to accept an `ssim_weight` parameter.
3. Integrate the SSIM calculation into the `forward` pass of `EnhancementLoss`.
4. Ensure the final loss formula correctly balances L1, Sobel (edge), and SSIM components to optimize text legibility.
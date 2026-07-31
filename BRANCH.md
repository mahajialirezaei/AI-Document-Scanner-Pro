### Main and Base (Permanent) Branches

https://github.com/mahajialirezaei/CNN-Applications-Doc-Scanning-And-Enhancement


* **`main` (or `master`)**: This branch only contains the final, tested, and ready-to-deliver code. You never commit directly to this branch.
* **`develop`**: This branch is the beating heart of your project development. All features are merged into this branch after they are completed. When this branch reaches full stability, it is merged into `main`.

---

### Feature Development Branches

These branches should all be built from the `develop` branch and merged back into `develop` when finished. I have set the names exactly according to the phases of the project:

*
**`feature/data-engineering`**: Specific to the implementation of Phase 2. Creating the Dataset class, reading the JSON labels files, and implementing all the Degradation functions with OpenCV.

*
**`feature/enhancement-network`**: Specific to the implementation of Phases 3 and 4 (Task 1). Designing the U-Net architecture, writing the Loss functions (combination of L1, MS-SSIM, and Sobel), and writing the Training Loop.

*
**`feature/corner-detection`**: Specific to the implementation of Phases 3 and 4 (Task 2). Design of both direct regression (Approach A) and heatmap (Approach B) approaches along with their respective training loops.

*
**`feature/evaluation-metrics`**: Specific to the implementation of Phase 5 (Evaluation Metrics). Writing evaluation scripts, calculating PSNR and SSIM, checking corner detection error, and integrating the OCR engine (Tesseract) to calculate readability.

*
**`feature/inference-scripts`**: ✅ **COMPLETED** - For creating final, clean scripts for a TA to run the model on a raw image. 
    - **Location**: `src/pipelines/inference.py` (614 lines)
    - **Contents**: 
        - Model loading utilities for enhancement and corner detection models
        - Preprocessing functions (image normalization, tensor conversion)
        - Enhancement inference: `enhance_document()` for rectified document enhancement
        - Corner detection: `detect_corners_regression()` (Approach A) and `detect_corners_heatmap()` (Approach B with soft-argmax)
        - Utility functions: corner ordering, perspective transformation, visualization with overlays
        - `DocumentScanningPipeline` class: End-to-end processing pipeline
        - Batch processing: Multi-image processing with automatic saving
        - CLI interface: Three operational modes (corner detection, enhancement, full pipeline)
    - **Status**: Merged into `develop` branch, ready for TA evaluation

*
**`feature/regularization-e2e`**: ✅ **COMPLETED** - Phase 6 implementation with regularization strategies and end-to-end joint training.
    - **Locations**: 
        - `src/training/train_regularized.py` (454 lines) - Regularization techniques
        - `src/pipelines/train_e2e.py` (585 lines) - End-to-end multi-task learning
    - **Contents of train_regularized.py**:
        - `DropoutScheduler`: Dynamic dropout rate adjustment with linear, cosine, and step strategies
        - `DataAugmentationTrainer`: Kornia-based on-the-fly geometric and photometric augmentations
        - `create_robust_loss()`: Huber, Smooth L1, L1, MSE loss functions
        - `RegularizedTrainingPipeline`: Complete pipeline combining dropout scheduling, augmentation, and robust losses
    - **Contents of train_e2e.py**:
        - `MultiTaskLoss`: Weighted combination of enhancement and corner losses with optional uncertainty-based weighting
        - `SharedBackboneNetwork`: Multi-task architecture with shared backbone and task-specific heads
        - `JointTrainer`: Supports alternating and simultaneous gradient updates
        - `EndToEndPipeline`: Complete end-to-end training with progressive unfreezing strategy
    - **Status**: Merged into `develop` branch, Phase 6 complete

---

### Summary of Completed Phases

| Phase | Branch | Status | Key Deliverables |
|-------|--------|--------|------------------|
| Phase 2 | `feature/data-engineering` | ✅ Merged | Dataset classes, degradation pipeline (OpenCV) |
| Phase 3 | `feature/enhancement-network` | ✅ Merged | U-Net, regression & heatmap models |
| Phase 4 | `feature/enhancement-network`, `feature/corner-detection` | ✅ Merged | Training loops, loss functions |
| Phase 5 | `feature/inference-scripts` | ✅ Merged | Inference pipelines, evaluation metrics |
| Phase 6 | `feature/regularization-e2e` | ✅ Merged | Dropout scheduling, Kornia augmentation, E2E training |
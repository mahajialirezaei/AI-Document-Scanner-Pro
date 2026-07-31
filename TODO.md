# TODO: Convolutional Neural Network Applications - Document Scanning & Enhancement

## Phase 1: Real-world Dataset Collection (TA Preparation)

* Collect 10 to 15 real smartphone photos of your own documents, varying lighting, viewpoints, and backgrounds.
* Produce a reference scan for each document using a scanning app like CamScanner at the exact time of photographing.
* Label the four corners of every real photo in Roboflow using a strict, consistent order: top-left, top-right, bottom-right, bottom-left.
* Submit the public Roboflow project link to the TAs via the designated Google Sheets.

## Phase 2: Data Engineering & Synthetic Pipeline

* Parse the Roboflow COCO keypoint JSON to extract the 4 corners as an ordered array of shape (4, 2).
* Implement a PyTorch `Dataset` or `tf.data.Dataset` to generate synthetic pairs on the fly without writing images to disk.
* Build a degradation pipeline using exclusively OpenCV, explicitly avoiding third-party transformation libraries.
* Apply random perspective warp to a clean scan over a random background and record the resulting four corners.
* Apply a random downscale/upscale by a factor of 2 to 4.
* Apply random brightness, contrast, and warm/cool color cast adjustments.
* Multiply by a random illumination gradient and composite soft random shadows.
* Apply Gaussian/motion blur followed by Gaussian noise.
* Apply JPEG re-encoding at a random quality between 30 and 80.
* **Explicitly exclude flipping or mirroring operations, as restoring mirrored text is irrelevant for document scanning.**
* Standardize input dimensions (e.g., 256x256) while identically scaling the corresponding corner coordinates.
* Normalize image pixel values and corner coordinates to a range of [0, 1].
* Ensure the 80/10/10 split is done strictly by source scan, meaning two degradations of the same page must never appear across different splits.
* Generate and freeze the Validation and Test sets using a fixed random seed to ensure consistent evaluation across epochs.
* Prepare real evaluation photos by rectifying them using Roboflow labels, then resizing and normalizing them exactly like synthetic inputs for the enhancement task.
* **Visually inspect a batch of generated samples by overlaying the corner labels on the composited photos to ensure proper alignment**.
* **Verify that warping the degraded photo back using the recorded homography aligns pixel-perfectly with the clean target**.

## Phase 3: Model Architectures (Bonus-Ready)

* Create `model.py` to house all network architectures without using pre-trained weights.
* Include Dropout layers in all architectures now, but initialize with `dropout_rate=0.0` for early phases.
* Design Task 1 (Enhancement): Construct a U-Net encoder-decoder architecture utilizing skip connections to preserve fine text details.
* Design Task 2 (Corner Approach A - Regression): Construct a CNN encoder with a fully connected head outputting 8 normalized coordinates.
* Design Task 2 (Corner Approach B - Heatmap): Construct a U-Net predicting 4 Gaussian heatmaps, extracting coordinates via soft-argmax.

## Phase 4: Training Pipelines ⚠️ PARTIALLY COMPLETED

* ✅ Create `losses.py` with loss functions (L1, SSIM, MS-SSIM, Sobel/Gradient loss) - **IMPLEMENTED (51 lines)**
* ❌ Create `train.py` to manage datasets, models, and optimization loops - **EMPTY FILE (0 lines)**
* ❌ Train Task 1 (Enhancement) leveraging a combination of L1 loss, MS-SSIM, and gradient-based loss like Sobel edge maps to prevent blur.
* ❌ Train Task 2 (Approach A) utilizing an L1 or L2 loss function on the 8 coordinates.
* ❌ Train Task 2 (Approach B) utilizing a pixel-wise loss function on the heatmaps.
* ❌ Plot training and validation loss curves across epochs for all models.
* ❌ **Experiment with the dataset generation ratio to see if the model benefits more from seeing many different degradations of few scans, or few degradations of many scans**[cite: 156, 157].

## Phase 5: Independent Inference & Evaluation ⚠️ PARTIALLY COMPLETED

* ✅ Create `evaluate.py` to assess model performance - **IMPLEMENTED (607 lines)**
* ✅ Compute PSNR and SSIM for the Enhancement network across Training, Validation, and Test splits.
* ✅ Compute baseline PSNR and SSIM on the unenhanced, degraded test inputs.
* ✅ Evaluate both Corner Detectors using mean corner localization error and a tight-threshold success rate.
* ✅ Compare Approach A and Approach B visually and numerically to determine the superior corner detection model.
* ✅ **Compare different loss function combinations for the Enhancement network using qualitative and quantitative analysis.**
* ✅ **Visualize intermediate and final outputs, clearly showing the degraded input, enhanced output, and clean target on synthetic data.**
* ❌ Perform qualitative checks by generating triplets (input, your model output, CamScanner reference) for the real photos.
* ❌ Run an OCR engine like Tesseract on the real photo inputs, your enhanced outputs, and the reference scans to calculate readability improvements.
* ❌ Implement an independent inference pipeline for Task 1 that takes a rectified document, applies preprocessing, infers the enhanced image, and post-processes it for visualization - **`inference.py` EMPTY (0 lines)**
* ❌ Implement an independent inference pipeline for Task 2 that takes a raw photo, predicts corner coordinates, maps them back to the original resolution, and overlays them on the input.
* ❌ Document and discuss model limitations (e.g., curled pages, extreme shadows, and the synthetic-to-real performance gap).

## Phase 6: Regularization & End-to-End Bonus ❌ NOT STARTED

* ❌ Increase the `dropout_rate` in your models (e.g., to 0.5) and retrain to observe the impact on the synthetic-to-real performance gap.
* ❌ Build a unified inference pipeline chaining the raw photo input, the best corner detector, the warping step, and the enhancement network.
* ❌ Implement the warping step using `kornia.geometry.transform.warp_perspective` to ensure the operation remains differentiable.
* ❌ Evaluate the complete automatic pipeline on real test photos and compare the OCR results against the manual-corner baseline.
* ❌ Fine-tune the entire end-to-end system jointly using the Enhancement loss.

## Phase 7: Final Delivery & Presentation Readiness ❌ NOT READY

* ❌ **Ensure the entire codebase is well-documented, modular, and easily executable.**
* ❌ **Prepare to explain the architecture and dynamically modify code during the presentation (e.g., adjusting hyperparameters or adding a new degradation instantly).**

---

## 📊 Implementation Status Summary

| Phase | Status | Key Files | Progress |
|-------|--------|-----------|----------|
| **Phase 1**: Dataset Collection | ⚠️ User Responsibility | N/A | Manual task - collect photos & label in Roboflow |
| **Phase 2**: Data Engineering | ❌ NOT DONE | `dataset.py` (0 lines), `degradation.py` (0 lines) | 0% - Critical foundation missing |
| **Phase 3**: Model Architectures | ✅ COMPLETED | `model.py` (184 lines) | 100% - All 3 architectures implemented with Dropout |
| **Phase 4**: Training Pipelines | ⚠️ PARTIAL | `losses.py` ✅ (51 lines), `train.py` ❌ (0 lines) | ~20% - Losses ready, training loops missing |
| **Phase 5**: Evaluation | ⚠️ PARTIAL | `evaluate.py` ✅ (607 lines), `inference.py` ❌ (0 lines) | ~60% - Metrics done, inference scripts missing |
| **Phase 6**: Regularization & E2E | ❌ NOT STARTED | Not implemented | 0% - Bonus features pending |
| **Phase 7**: Final Delivery | ❌ NOT READY | Documentation needed | 0% - Codebase needs completion first |

### 🔑 Priority Recommendations
1. **Phase 2** (Data Pipeline) - Implement `dataset.py` and `degradation.py` first (foundational)
2. **Phase 4** (Training) - Complete `train.py` with training loops
3. **Phase 5** (Inference) - Fill in `inference.py` with clean inference pipelines
4. **Phase 6** (Bonus) - Add dropout experiments and Kornia integration
5. **Phase 7** (Delivery) - Document and prepare for presentation
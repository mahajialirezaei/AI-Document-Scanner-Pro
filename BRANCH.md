### Main and Base (Permanent) Branches

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
**`feature/evaluation-metrics`**: Specific to the implementation of Phase 5. Writing evaluation scripts, calculating PSNR and SSIM, checking corner detection error, and integrating the OCR engine (Tserket) to calculate readability.

*
**`feature/regularization-dropout`**: Specific to the implementation of Phase 6. Enabling and adjusting Dropout layers in the network architecture and retraining to check the effect of regularization.

*
**`feature/end-to-end-pipeline`**: Specific to the final scoring part (Phase 7). Combining the corner detection network with the Kornia library's `warp_perspective` derivative function, connecting it to the Enhancement network, and fine-tuning the entire system.

*
**`feature/inference-scripts`**: For creating final, clean scripts for a TA to run the model on a raw image.

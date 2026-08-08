# Document Enhancement: Model Evolution and Benchmarks

This document outlines the evolutionary phases of the Enhancement U-Net model, detailing the transition to high-resolution training, overcoming hardware bottlenecks, and the definitive comparative analysis of regularization and post-processing techniques.

## Phase 1: Overcoming the Resolution and Loss Bottlenecks
Initial evaluations of the Enhancement model at a $256 \times 256$ resolution yielded highly "washed-out" and blurry results. The model struggled to reconstruct text because the extreme downscaling destroyed character-level topography. 

**Architectural & Pipeline Upgrades:**
1.  **Resolution Upscaling:** The training resolution was quadrupled from $256 \times 256$ to $512 \times 512$, providing the U-Net with sufficient pixel density to reconstruct readable text.
2.  **Targeted Degradation Moderation:** The degradation pipeline (`degradation.py`) was re-balanced. Extreme destructive forces (like heavy Salt & Pepper noise and severe resolution loss) were moderated. The focus shifted from "destroying pixels" to "distorting light and geometry" to preserve ink structure.
3.  **Loss Function Restructuring (`losses.py`):** 
    *   **Sobel Edge Weight** was aggressively increased (from $0.5$ to $2.0$) to penalize blurry edges and force sharp character boundaries.
    *   **Text Weight** was increased (from $3.0$ to $5.0$) to heavily penalize errors on dark pixels (ink), effectively combating the "washed-out" artifact.

## Phase 2: Resolving IPC Deadlocks on Windows
The jump to $512 \times 512$ resolution introduced a critical systemic failure: a **Thread Deadlock** causing silent terminal crashes during the dataset generation phase.
*   **The Cause:** The combination of PyTorch's multiprocessing (`num_workers=4`) and OpenCV's internal multi-threading (`cv2`) created an Inter-Process Communication (IPC) bottleneck in Windows. The massive data payload of $512 \times 512$ arrays completely starved the CPU.
*   **The Fix:** We completely disabled OpenCV's internal threading (`cv2.setNumThreads(0)`) and initially reduced `num_workers` to `0` to stabilize the pipeline. Once stabilized, we optimized training speed by reducing the `train_samples_per_epoch` to $2000$ and safely operating with `num_workers=2` and a `batch_size` of $4$ to prevent CUDA Out-of-Memory (OOM) exceptions.

## Phase 3: Benchmarking and the Ground Truth (GT) Isolation
To accurately measure the true capacity of the Enhancement U-Net, we bypassed the Corner Detection model during evaluation (`--use-gt-corners`). 
*   **Why?** Predicted corners inherently carry a slight spatial error (a Mean Localization Error of ~35 px). This causes the rectified image to be slightly shifted. Pixel-wise metrics like PSNR heavily penalize spatial shifts, masking the actual image restoration quality.
*   **The Anomaly:** Tesseract OCR exhibited a known anomaly on heavily degraded images, confidently outputting 95%-100% for a single visible character while ignoring the rest of the dark page. By using GT corners, we stabilized the Degraded Confidence baseline to an average of **40.17%**.

## Phase 4: Regularization vs. Baseline (The Dropout Verdict)
We evaluated two variations of the $512 \times 512$ Enhancement model using Ground Truth corners to determine the impact of network regularization in an image-to-image translation task.

| Model Variant | PSNR (dB) | SSIM | OCR Confidence |
| :--- | :---: | :---: | :---: |
| **Baseline (No Dropout)** | **13.66** | **0.5896** | **31.56%** |
| **Regularized (Dropout 0.3)** | 13.60 | 0.5824 | 31.52% |

*   **Conclusion:** Unlike coordinate regression, injecting spatial noise via Dropout in the bottleneck layers of a U-Net slightly degraded both the mathematical metrics (PSNR/SSIM) and the functional OCR readability. The **Baseline (No Dropout)** model proved superior.

## Phase 5: The Post-Processing Paradox (Adaptive Binarization)
To bridge the gap between our enhanced output (31.56%) and the clean target scan (37.53%), we implemented **Adaptive Gaussian Binarization** as a post-processing filter (`--apply-binarization`). The theoretical goal was to create absolute black-and-white contrast for Tesseract.

| Model Variant (No Dropout) | PSNR (dB) | SSIM | OCR Confidence |
| :--- | :---: | :---: | :---: |
| **Raw U-Net Output** | **13.66** | 0.5896 | **31.56%** |
| **With Adaptive Binarization** | 11.12 | **0.6032** | 30.62% |

*   **The PSNR Crash:** Binarization forced soft gray edge pixels into absolute black (0) or absolute white (255). Compared to the smooth gradients of the original scanned target, this caused a massive Mean Squared Error spike, dropping PSNR by ~2.5 dB.
*   **The OCR Drop:** Binarization fragmented continuous handwritten strokes, causing Tesseract to misinterpret characters, leading to a ~1% drop in overall confidence.

## Final Champion Selection
Based on rigorous empirical testing, the **Baseline Enhancement U-Net (No Dropout)**, operating purely on its raw output without algorithmic binarization, stands as the champion model. It delivers the most natural visual restoration and the highest text readability within the defined architectural constraints.
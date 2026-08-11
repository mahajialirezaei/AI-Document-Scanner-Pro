# Document Scanning & Enhancement: Comprehensive Degradation & Augmentation Log

This document outlines all the synthetic degradations, physical distractors, geometric augmentations, and post-processing filters implemented across the entire document scanning pipeline to simulate real-world, unconstrained smartphone photography and optimize downstream OCR/readability.

---

## 1. Geometric & Perspective Distortions

These augmentations simulate the physical position of the camera relative to the document and physical paper deformations.

* **Extreme Foreshortening (Perspective Distortion):** Generates severe acute and obtuse trapezoidal shapes by heavily compressing either the top or bottom edges of the synthetic document, mimicking close-up smartphone photography.
* **3D Paper Curl & Grid Distortion (ElasticTransform):** Simulates non-linear physical paper bending and local warping by applying elastic deformation grids. *Note: Integrated upstream in the pipeline by applying transformations to a 4-channel BGRA matrix (RGB + hole masks) prior to perspective warping to ensure ground-truth coordinate synchronization.*
* **3D Cylindrical Warp (Open Book Curvature):** Applies a non-linear sine-wave displacement map (`_apply_cylindrical_warp`) to the clean scan prior to perspective warping, simulating the physical curvature of open book pages near the spine.
* **Dynamic Aspect Ratio Rectification:** In the inference phase (`apply_perspective_transform`), predicted corners are mapped onto dynamic Euclidean destination bounds ($\text{maxWidth}, \text{maxHeight}$) rather than a square grid, preserving native paper proportions (e.g., A4 ratio).

---

## 2. Physical Artifacts & Environmental Distractors

These augmentations inject physical objects, background clutter, and environmental occlusions into synthetic training pairs.

* **Spiral Bindings & Binding Artifacts:** Simulates notebook coils, punched hole masks, and wire lines along the left or right edges of the document (`_add_binding_artifacts`).
* **Gutter Shadows (Thick Spines):** Applies thick, dark, blurred gradients across one edge to simulate deep shadows found in book gutters or dark binder margins.
* **Human Fingers:** Renders simulated skin-toned circles overlapping document edges.
* **Plastic Cover Glare:** Generates random white, alpha-blended polygons that mask parts of the document, mimicking reflections on plastic sleeves.
* **Background Clutter (Wires/Lines):** Injects random thick lines and colored shapes across the background canvas to simulate messy desks or bedsheets.
* **Adjacent Pages (Open Book Simulation):** Renders a secondary, slightly off-white page next to the target document (`_add_adjacent_page`), separated by a dark spine crease with an enforced outward normal vector to prevent dataset poisoning.
* **Dark Binder Margins (Optimized Reintroduction):** Renders thick, dark, textured borders directly underneath or adjacent to paper edges to combat the "Semantic Trap" by teaching the network to reject high-contrast background distractors.
* **3D Drop Shadow Distractors:** Renders overlapping objects (`_add_3d_shadow_distractors`) with offset dark shadows beneath them to simulate 3D depth from objects resting on the paper or desk.
* **Camouflage Polygons:** Injects white or off-white polygons (`_add_camouflage_polygons`) tangent to document boundaries to force reliance on physical edge structures rather than simple color contrast.
* **Corner Occlusion by Clothing/Sleeves:** Moderated sleeve occlusions (`_add_clothing_occlusions`) injected with constrained probabilities and wide spatial margins from true corners to prevent total geometric confidence loss.
* **Non-Text Collaged Elements & Logos:** Collages random solid rectangles, bone-colormap gradients, and noisy patches (`_add_non_text_elements`) onto clean scans to teach the U-Net to preserve non-text visual structures.

---

## 3. Optical & Sensor Degradations

These functions simulate the physical limitations and flaws of smartphone camera sensors and lenses.

* **Motion Blur:** Applies a directional linear blur (`apply_motion_blur`) with random angles to simulate camera shake or hand movement during capture.
* **Gaussian Blur:** Simulates out-of-focus captures and lens softness (`apply_gaussian_blur`).
* **Gaussian Noise:** Adds standard normal distribution noise (`apply_gaussian_noise`) to simulate high-ISO sensor noise in low-light environments.
* **Poisson Noise (Shot Noise):** Simulates photon shot noise (`apply_poisson_noise`) commonly found in digital image sensors.
* **Salt-and-Pepper Noise:** Injects extreme high (white) and low (black) pixel values (`apply_salt_pepper_noise`) with moderated probabilities to mimic dead sensor pixels or dust on the lens.
* **Resolution Loss (Downscale/Upscale):** Artificially degrades image resolution (`apply_resolution_loss`) by downscaling by a factor between 1.2x and 2.5x and scaling back up to simulate distant photos.
* **JPEG Compression Artifacts:** Re-encodes images (`apply_jpeg_compression`) at random JPEG quality factors (30–90) using `cv2.imencode` to introduce realistic blocky compression artifacts.

---

## 4. Lighting, Color & Ink Variations

These modifications alter the photometric properties of the document, paper substrate, and pen ink.

* **Soft Uneven Shadows:** Generates random dark polygons with heavy Gaussian blur (`apply_shadow`) to cast uneven, soft lighting conditions across the document surface.
* **Brightness & Contrast Shifts:** Randomly scales overall illumination (`apply_brightness_change`) and contrast bounds (`apply_contrast_change`).
* **Ink Simulation (Fading & Color Shifting):** Targets dark pixels ($<130$ intensity) via `apply_ink_simulation` and alters BGR channels to simulate ballpoint blue, light blue, faded grey/black, navy, or purple pens, combined with a transparency fade factor.
* **Color Temperature Shift:** Modifies R and B color channels (`apply_color_temperature`) to simulate warm (tungsten/yellow) indoor lighting and cool (shade/blue) lighting conditions.
* **Tinted Paper Simulation:** Multiplies clean scans by pastel RGB values with a 4% probability to simulate non-white physical paper.

---

## 5. Inference Post-Processing & Gatekeepers

Advanced algorithmic filters and statistical logic gates applied at inference time to handle real-world edge cases.

* **Gatekeeper A v2 (Independent Edge Border Analysis):** Evaluates the 3% outer border of each edge (top, bottom, left, right) independently (`is_already_cropped`). If at least 3 out of 4 edges exhibit low variance ($\text{variance} < 800$) and high brightness ($\text{white\_ratio} > 0.70$), corner detection is bypassed to prevent false crops on pre-scanned images.
* **Gatekeeper C (Histogram Tail-Mass Analysis):** Evaluates grayscale histogram distributions (`is_already_enhanced`). If more than 40% of pixels exceed a brightness value of 240 (Right-Tail Mass), U-Net enhancement is bypassed to prevent catastrophic ink fading on pre-enhanced scans.
* **Magic Ink Boost Filter (`apply_ink_boost_filter`):** An optional non-destructive post-processing filter that darkens pen strokes without threshold fragmentation:
  1. *Luminance Gamma Correction:* Converts image to YCrCb space and applies gamma correction ($\gamma = 0.82$) strictly to the $Y$ (Luminance) channel via Look-Up Table (LUT), darkening blue/black ink strokes while preserving color hues.
  2. *Unsharp Masking:* Blends the result with a Gaussian-blurred variant using `cv2.addWeighted` (sharpness factor 1.35) to sharpen character boundaries.
* **Adaptive Sauvola Binarization (`apply_adaptive_binarization`):** An optional post-processing toggle applying median blur followed by Gaussian adaptive thresholding to produce absolute high-contrast black-and-white outputs.
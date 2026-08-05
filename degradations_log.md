# Document Scanning & Enhancement: Comprehensive Degradation Log

This document outlines all the synthetic degradations, physical distractors, and geometric augmentations implemented in the document scanning pipeline to simulate real-world, unconstrained smartphone photography.

## 1. Geometric & Perspective Distortions
These augmentations simulate the physical position of the camera relative to the document.

*   **Extreme Foreshortening (Perspective Distortion):** Generates severe acute and obtuse trapezoidal shapes by heavily compressing either the top or bottom edges of the synthetic document, mimicking close-up smartphone photography.

## 2. Physical Artifacts & Environmental Distractors
These augmentations inject physical objects and environmental occlusions into the images.

*   **Spiral Bindings:** Simulates notebook coils and punched holes along the left or right edges of the document.
*   **Gutter Shadows (Thick Spines):** Applies thick, dark, blurred gradients across one edge to simulate the deep shadows found in book gutters or dark binders.
*   **Human Fingers:** Renders simulated skin-toned circles overlapping the edges of the document.
*   **Plastic Cover Glare:** Generates random white, alpha-blended polygons that mask parts of the document, mimicking reflections on plastic sleeves.
*   **Background Clutter (Wires/Lines):** Injects random thick lines and colored shapes across the background canvas to simulate cluttered environments like messy desks or bedsheets.
*   **Adjacent Pages (Open Book Simulation):** Renders a secondary, slightly off-white page next to the target document, separated by a dark spine crease, to teach the model to focus only on the target page.
*   **Synthetic Binder Margins:** Generates thick, textured margins in dark or vibrant colors outside the document edges to simulate physical folder covers.
*   **3D Drop Shadow Distractors (NEW):** Renders overlapping objects (distractors) with offset dark shadows beneath them to simulate the 3D depth of objects resting on the paper or desk.
*   **Camouflage Polygons (NEW):** Injects white or off-white polygons (mimicking white mice, cables, or papers) tangent to the document boundaries to force reliance on physical edge structures rather than simple color contrast.
*   **Corner Occlusion by Dark Objects (NEW):** Places soft-edged, dark rectangular shapes completely cutting off one of the document's corners to simulate severe occlusions like pencil cases or electronics.

## 3. Optical & Sensor Degradations
These functions simulate the limitations and flaws of smartphone camera hardware.

*   **Motion Blur:** Applies a directional linear blur to simulate camera shake or hand movement during capture.
*   **Gaussian Blur:** Simulates out-of-focus captures and lens softness.
*   **Gaussian Noise:** Adds standard normal distribution noise to simulate high-ISO sensor noise in low light.
*   **Poisson Noise (Shot Noise):** Simulates photon shot noise commonly found in digital image sensors.
*   **Salt-and-Pepper Noise:** Injects extreme high (white) and low (black) pixel values to mimic dead sensor pixels or dust on the lens.
*   **Resolution Loss (Downscale/Upscale):** Artificially degrades the image by shrinking it and scaling it back up using interpolation, simulating documents photographed from a distance.
*   **JPEG Compression Artifacts:** Compresses the image to a low quality factor to introduce realistic blocking artifacts.

## 4. Lighting, Color & Ink Variations
These modifications alter the photometric properties of the document and the ink on the paper.

*   **Soft Uneven Shadows:** Generates random dark polygons with heavy Gaussian blur to cast uneven, soft lighting conditions across the document surface.
*   **Brightness & Contrast Shifts:** Randomly scales the overall illumination and contrast bounds of the image.
*   **Ink Simulation (Fading & Color Shifting):** Targets dark pixels (text/ink) and alters their BGR values to simulate varying real-world ink types, such as standard blue ballpoint, light blue, faded black/grey, or purple pens, combined with a transparency fade factor.
*   **Color Temperature Shift:** Modifies the R and B color channels to simulate warm (tungsten/yellow) indoor lighting and cool (shade/blue) lighting conditions.
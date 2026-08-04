# Document Scanning & Enhancement: Comprehensive Degradation Log

This document outlines all the synthetic degradations, physical distractors, and geometric augmentations implemented in the document scanning pipeline to simulate real-world, unconstrained smartphone photography.

## 1. Geometric & Perspective Distortions
These augmentations simulate the physical position of the camera relative to the document[cite: 14].

*   **Extreme Foreshortening (Perspective Distortion):** Generates severe acute and obtuse trapezoidal shapes by heavily compressing either the top or bottom edges of the synthetic document, mimicking close-up smartphone photography[cite: 2, 13, 14].

## 2. Physical Artifacts & Environmental Distractors
These augmentations inject physical objects and environmental occlusions into the images[cite: 2, 13, 14].

*   **Spiral Bindings:** Simulates notebook coils and punched holes along the left or right edges of the document[cite: 2, 13, 14].
*   **Gutter Shadows (Thick Spines):** Applies thick, dark, blurred gradients across one edge to simulate the deep shadows found in book gutters or dark binders[cite: 2, 13, 14].
*   **Human Fingers:** Renders simulated skin-toned circles overlapping the edges of the document[cite: 2, 13, 14].
*   **Plastic Cover Glare:** Generates random white, alpha-blended polygons that mask parts of the document, mimicking reflections on plastic sleeves[cite: 2, 13, 14].
*   **Dark Binder Borders:** Draws random thick, dark lines around or near the document boundaries to act as false edges[cite: 2, 13, 14].
*   **Background Clutter (Wires/Lines):** Injects random thick lines and colored shapes across the background canvas to simulate cluttered environments like messy desks or bedsheets[cite: 2, 13, 14].

## 3. Optical & Sensor Degradations
These functions simulate the limitations and flaws of smartphone camera hardware[cite: 3].

*   **Motion Blur:** Applies a directional linear blur to simulate camera shake or hand movement during capture[cite: 3].
*   **Gaussian Blur:** Simulates out-of-focus captures and lens softness[cite: 3].
*   **Gaussian Noise:** Adds standard normal distribution noise to simulate high-ISO sensor noise in low light[cite: 3].
*   **Poisson Noise (Shot Noise):** Simulates photon shot noise commonly found in digital image sensors[cite: 3].
*   **Salt-and-Pepper Noise:** Injects extreme high (white) and low (black) pixel values to mimic dead sensor pixels or dust on the lens[cite: 3].
*   **Resolution Loss (Downscale/Upscale):** Artificially degrades the image by shrinking it and scaling it back up using interpolation, simulating documents photographed from a distance[cite: 3].
*   **JPEG Compression Artifacts:** Compresses the image to a low quality factor to introduce realistic blocking artifacts[cite: 3].

## 4. Lighting, Color & Ink Variations
These modifications alter the photometric properties of the document and the ink on the paper[cite: 3].

*   **Soft Uneven Shadows:** Generates random dark polygons with heavy Gaussian blur to cast uneven, soft lighting conditions across the document surface[cite: 3].
*   **Brightness & Contrast Shifts:** Randomly scales the overall illumination and contrast bounds of the image[cite: 3].
*   **Ink Simulation (Fading & Color Shifting):** Targets dark pixels (text/ink) and alters their BGR values to simulate varying real-world ink types, such as standard blue ballpoint, light blue, faded black/grey, or purple pens, combined with a transparency fade factor[cite: 3].
## Phase 9: Production Hardening, Advanced Heuristics, and Interactive UI Deployment

Following the successful identification of the Gold and Silver models, rigorous testing in simulated production environments revealed critical edge cases. The Smart Ensemble was vulnerable to geometric "reward hacking," and the pipeline struggled with Out-Of-Distribution (OOD) inputs (e.g., images already cropped or scanned). This phase details the final systemic logic gates and UI upgrades implemented to make the application production-ready.

### 1. Defeating Ensemble Reward Hacking & The Perspective Symmetry Rule

In the previous phase, we introduced an "Area Maximization" heuristic to prevent the ensemble from snapping to small inner bounding boxes. However, this introduced a new flaw: weaker models would occasionally lock onto deep shadows at the bottom of a photo, creating a massively distorted, physically impossible polygon that artificially inflated its "Area" score and hijacked the ensemble's decision.

*   **The Projective Geometry Crisis:** Real-world documents are rectangles. Even under extreme camera perspective projection, the resulting 2D quadrilateral obeys strict physical rules. The ensemble was choosing skewed trapezoids (e.g., one 110° angle and one 65° angle) that could never exist in a real photograph of a flat paper.
*   **The Perspective Symmetry Fix (Ensemble 3.0):** We implemented strict physical heuristics to instantly invalidate impossible shapes:
    1.  **Absolute Angle Limits:** Any polygon containing an internal angle outside the [55°, 125°] range is immediately discarded.
    2.  **Edge Distortion Ratio:** Opposite edges cannot exceed a 2.2x length ratio (a paper's top edge cannot appear 3 times larger than its bottom edge under normal smartphone photography distances).
    3.  **The Perspective Symmetry Rule:** If two adjacent angles (e.g., the top two corners) are near 90° (between 75° and 105°), it indicates the camera sensor is roughly parallel to that edge. Under projective geometry, the opposing two angles *must* also be relatively symmetric. If the opposing angles diverge by more than 25°, the shape is deemed physically impossible and rejected.
    4.  **Area Nerf:** The Shoelace area calculation weight was reduced to 15%, serving strictly as a tie-breaker rather than the primary driving metric.

### 2. Out-Of-Distribution (OOD) Gatekeepers

A classic issue in commercial scanning apps is handling user inputs that do not match the training domain (e.g., uploading an already-scanned PDF or a perfectly cropped digital image). Passing these through the standard pipeline resulted in the Corner detector guessing wildly and the Enhancement network "washing out" already-white pixels.

We implemented statistical "Gatekeepers" to intelligently bypass heavy networks when unnecessary:
*   **Gatekeeper A (Corner Bypass):** The `is_already_cropped` function analyzes the outer 2% perimeter of the image. If the color variance is extremely low and the pixels are overwhelmingly bright, the system deduces it is a digital/cropped document. Corner detection is bypassed, and the image's bounding box is returned as `[0,0]` to `[W,H]`.
*   **Gatekeeper C (Enhancement Bypass):** The `is_already_enhanced` function analyzes the image's grayscale histogram. Smartphone photos naturally have a distributed histogram. If the "Right-Tail Mass" (pixels with absolute brightness > 240) exceeds 40% of the image, the system deduces the image has already undergone digital contrast stretching/binarization. The U-Net enhancement is bypassed to prevent catastrophic ink-fading.

### 3. Interactive UI & PDF Support

To elevate the application from an API backend to a full-fledged commercial tool, the frontend was completely overhauled:
*   **Auto vs. Interactive Modes:** Users can now toggle between an automated pipeline and an "Interactive Editor." 
*   **Draggable Canvas Editor:** In Interactive Mode, the raw image is drawn onto an HTML5 Canvas, allowing users to manually drag and refine the AI-predicted corner points before confirming the crop. A CSS-scaling bug that previously misaligned mouse coordinates with the canvas internal resolution was resolved by dynamically calculating the `getBoundingClientRect()` scale ratios (`scaleX`, `scaleY`).
*   **Multi-Page PDF Processing:** The backend was upgraded with `PyMuPDF` (`fitz`). Users can now upload a PDF directly. The pipeline extracts each page, processes it through the enhancement network, and recombines the results into a downloadable, fully enhanced PDF document.
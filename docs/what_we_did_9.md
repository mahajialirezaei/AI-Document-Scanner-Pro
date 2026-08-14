## Phase 9: Production Hardening, Advanced Heuristics, and Interactive UI Deployment

Following the successful identification of the Gold and Silver models, rigorous testing in simulated production environments revealed critical edge cases. The Smart Ensemble was vulnerable to geometric "reward hacking," and the pipeline struggled with Out-Of-Distribution (OOD) inputs (e.g., images already cropped or scanned). This phase details the final systemic logic gates, dynamic geometric corrections, post-processing enhancements, and UI upgrades implemented to make the application production-ready.

---

### 1. Defeating Ensemble Reward Hacking & The Perspective Symmetry Rule

In the previous phase, we introduced an "Area Maximization" heuristic to prevent the ensemble from snapping to small inner bounding boxes. However, this introduced a new flaw: weaker models would occasionally lock onto deep shadows at the bottom of a photo, creating a massively distorted, physically impossible polygon that artificially inflated its "Area" score and hijacked the ensemble's decision.

* **The Projective Geometry Crisis:** Real-world documents are rectangles. Even under extreme camera perspective projection, the resulting 2D quadrilateral obeys strict physical rules. The ensemble was choosing skewed trapezoids (e.g., one 110° angle and one 65° angle) that could never exist in a real photograph of a flat paper.


* **The Perspective Symmetry Fix (Ensemble 3.0):** We implemented strict physical heuristics to instantly invalidate impossible shapes:


1. **Absolute Angle Limits:** Any polygon containing an internal angle outside the [55°, 125°] range is immediately discarded.


2. **Edge Distortion Ratio:** Opposite edges cannot exceed a 2.2x length ratio (a paper's top edge cannot appear 3 times larger than its bottom edge under normal smartphone photography distances).


3. **The Perspective Symmetry Rule:** If two adjacent angles (e.g., the top two corners) are near 90° (between 75° and 105°), it indicates the camera sensor is roughly parallel to that edge. Under projective geometry, the opposing two angles *must* also be relatively symmetric. If the opposing angles diverge by more than 25°, the shape is deemed physically impossible and rejected.


4. **Area Nerf:** The Shoelace area calculation weight was reduced to 15%, serving strictly as a tie-breaker rather than the primary driving metric.





---

### 2. Out-Of-Distribution (OOD) Gatekeepers & Independent Border Analysis

A classic issue in commercial scanning apps is handling user inputs that do not match the training domain (e.g., uploading an already-scanned PDF or a perfectly cropped digital image). Passing these through the standard pipeline resulted in the Corner detector guessing wildly and the Enhancement network "washing out" already-white pixels.

We implemented statistical "Gatekeepers" to intelligently bypass heavy networks when unnecessary:

* **Gatekeeper A v2 (Independent Edge Border Analysis):** Initial implementations concatenated all outer perimeter pixels into a single array, causing false negatives when an image contained localized artifacts (e.g., black notebook spiral binding on one edge). The updated `is_already_cropped` function evaluates the 3% outer border of each edge (top, bottom, left, right) independently. If at least 3 out of 4 edges exhibit low variance ($\text{variance} < 800$) and high brightness ($\text{white\_ratio} > 0.70$), the image is recognized as pre-cropped. Corner detection is bypassed, preserving the full un-warped image.


* **Gatekeeper C (Enhancement Bypass):** The `is_already_enhanced` function analyzes the image's grayscale histogram. Smartphone photos naturally have a distributed histogram. If the "Right-Tail Mass" (pixels with absolute brightness > 240) exceeds 40% of the image, the system deduces the image has already undergone digital contrast stretching/binarization. The U-Net enhancement is bypassed to prevent catastrophic ink-fading.



---

### 3. Dynamic Aspect Ratio Perspective Transformation

During initial end-to-end testing, rectified crops produced "squished" or distorted outputs because the perspective transform function (`apply_perspective_transform`) hardcoded target dimensions to a square $1024 \times 1024$ canvas.

* **The Euclidean Fix:** We refactored `apply_perspective_transform` to compute dynamic, physical dimensions using Euclidean distance between corner pairs:

$$\text{Width} = \max\left(\sqrt{(x_{br}-x_{bl})^2 + (y_{br}-y_{bl})^2},\ \sqrt{(x_{tr}-x_{tl})^2 + (y_{tr}-y_{tl})^2}\right)$$


$$\text{Height} = \max\left(\sqrt{(x_{tr}-x_{br})^2 + (y_{tr}-y_{br})^2},\ \sqrt{(x_{tl}-x_{bl})^2 + (y_{tl}-y_{bl})^2}\right)$$


* **Natural Page Preservation:** The homography matrix (`cv2.findHomography`) maps predicted corners onto the dynamically calculated $(\text{Width}, \text{Height})$ destination grid. This preserves the document's native aspect ratio (e.g., standard A4 $1 : 1.414$ ratio) before passing it to the Enhancement U-Net.



---

### 4. Post-Processing & Ink Contrast Optimization (Magic Ink Boost Filter)

While the U-Net excels at background whitening, pixel-wise regression losses tend to slightly smooth out fine pen strokes, resulting in pale or "washed-out" ink. Standard Adaptive Binarization (Sauvola) fragments continuous handwriting into pixelated dots and ruins visual aesthetics.

To solve this without ruining stroke continuity, we introduced the **Magic Ink Boost Filter** (`apply_ink_boost_filter`) as an optional post-processing toggle:

1. **Luminance Gamma Correction in YCrCb Space:** The image is converted from BGR to YCrCb space. A Look-Up Table (LUT) applies gamma correction ($\gamma = 0.82$) strictly to the $Y$ (Luminance) channel, darkening blue/black ink strokes without altering the $Cr, Cb$ color channels or distorting blue ink hues.


2. **Unsharp Masking:** The image is converted back to BGR and blended with a Gaussian-blurred variant (`cv2.addWeighted` with weight $1.35$), sharpening character boundaries and local stroke contrast.



---

### 5. Interactive UI, PDF Support, and API Systemic Hardening

The web application was fully upgraded across backend routing (`web_app.py`), frontend logic (`script.js`), and user controls (`index.html`):

* **Auto vs. Interactive Modes:** Users can toggle between automated execution (`/scan`) and an "Interactive Editor" (`/interactive-detect` and `/interactive-enhance`). Both pipelines invoke OOD Gatekeepers.


* **Draggable Canvas Editor with CSS Scaling:** In Interactive Mode, raw images are rendered on an HTML5 Canvas where users can manually adjust AI-predicted corner points. Coordinate offsets caused by CSS responsiveness (`max-width: 100%`) were resolved by computing dynamic scaling factors ($\text{scaleX} = \text{canvas.width} / \text{rect.width}$, $\text{scaleY} = \text{canvas.height} / \text{rect.height}$) from `getBoundingClientRect()`.


* **Multi-Page PDF Processing:** Integrated `PyMuPDF` (`fitz`) into the backend. Uploaded multi-page PDFs are rendered page-by-page, processed through the enhancement pipeline, and re-assembled into a single downloadable PDF document.


* **Variable Shadowing Resolution:** Renamed the post-processing filter to `apply_ink_boost_filter` to eliminate Python namespace collisions (`'bool' object is not callable` and `'str' object is not callable`) caused when form parameters shared identical names with imported functions.
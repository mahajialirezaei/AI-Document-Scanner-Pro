# Document Scanning & Enhancement Pipeline - Debugging & Improvement Log
**File:** `what_we_did_3.md`
**Objective:** Comprehensive log of challenges, root causes, and implemented solutions for the CNN Document Scanning project.

---

## 1. Challenge: The "Butterfly" Effect & Static High MSE
**Symptoms:** 
* The perspective warp produced self-intersecting, twisted images (the "butterfly" effect).
* During evaluation (`evaluate_real_data.py`), the Corner MSE metric was extraordinarily high (e.g., exactly 14260.1 or 5622.4) and remained completely static even after applying point-ordering algorithms to the model's predictions.

**Root Cause (Systemic Bug):**
The Ground Truth (GT) coordinates loaded from the Roboflow JSON annotations in `RealDocumentDataset` were entirely unordered. While the predicted points were being sorted, they were being evaluated against scrambled GT points. This caused the Euclidean distance (MSE) to be calculated across diagonal or completely mismatched points. 

**Implemented Solutions:**
* **`src/data/dataset.py`:** Injected a sorting algorithm directly into the `RealDocumentDataset.__getitem__` method to enforce standard ordering (TL, TR, BR, BL) on the GT points the moment they are loaded from the JSON.
* **`evaluate_real_data.py`:** Added a strict geometric sorting step immediately before the `(pred - gt)**2` calculation to ensure a true 1-to-1 point comparison.

---

## 2. Challenge: The "Grey Screen" Bug & Collapsed Quadrilaterals
**Symptoms:**
* For certain highly challenging images (e.g., `eval_21.jpg` featuring a black binder), the enhanced output document rendered as a completely solid grey or black image.

**Root Cause:**
When the model failed to find four distinct corners, it predicted overlapping points (essentially forming a triangle instead of a quadrilateral). The previously implemented "Polar Sorting" (`arctan2` around a centroid) failed to properly order overlapping or collinear points. Passing a degenerate polygon to `cv2.getPerspectiveTransform` resulted in a singular matrix (determinant of zero), breaking the warp entirely.

**Implemented Solutions:**
* **`src/pipelines/inference.py` & `evaluate_real_data.py`:** Replaced the Polar Sorting algorithm with a highly robust, deterministic `Top/Bottom -> Left/Right` sorting algorithm (`order_points`). This algorithm sorts by the Y-axis to split the top two and bottom two points, then sorts each pair by the X-axis, guaranteeing a logical point allocation even if predictions are heavily distorted.

---

## 3. Challenge: Extreme Foreshortening (Perspective Distortion)
**Symptoms:**
* The model failed drastically on images where the camera was held very close to the top or bottom edge of the document (e.g., `eval_14_2.jpg`, `eval_16_2.jpg`, `eval_19_3.jpg`), resulting in massive MSE spikes (e.g., 30,000+ pixels squared). 

**Root Cause:**
The synthetic data generator (`SyntheticDocumentDataset`) utilized a 3D rotation matrix (Pitch, Yaw, Roll) that assumed the camera was relatively far from the document. The neural network had never been trained on acute/obtuse trapezoidal shapes representing severe smartphone camera foreshortening.

**Implemented Solutions:**
* **`src/data/dataset.py`:** Completely rewrote the `_generate_random_corners` function. Replaced the 3D projection math with a direct 2D coordinate generation system that explicitly simulates severe acute and obtuse trapezoids. The algorithm now heavily compresses either the top or bottom edges of the synthetic document to mimic close-up smartphone photography.

---

## 4. Challenge: Adversarial Backgrounds & Physical Distractors
**Symptoms:**
* The model's corner detection was easily hijacked by environmental features that resembled paper edges.
* Specific failures included:
    * **Plastic Glare (`eval_03_4.jpg`):** Reflections on plastic sleeves erased paper edges.
    * **Thick Book Spines (`eval_07_5.jpg`):** Deep shadows in the gutter of a book tricked the edge detector.
    * **Cluttered Backgrounds (`eval_11_2.jpg`):** High-contrast background patterns (e.g., bedsheets, marble tables) acted as false edges.
    * **Black Binders (`eval_21.jpg`):** The model locked onto the black binder edge instead of the white paper.

**Root Cause:**
The training dataset was too "clean" regarding edge boundaries, failing to simulate complex, real-world physical obstructions and lighting artifacts.

**Implemented Solutions:**
* **`src/data/dataset.py`:** Greatly expanded the `_add_distractors` function to randomly inject specific, adversarial physical conditions into the synthetic training pipeline. Added probabilities to render:
    1. **Spiral Bindings:** Simulated coils along document edges.
    2. **Dark Binder Borders:** Thick, dark lines drawn around the document to simulate folders or binders.
    3. **Plastic Cover Glare:** Random white, alpha-blended polygons masking parts of the document.
    4. **Book Spine Shadows:** Dark, thick, blurred gradient lines across one edge.
    5. **Human Fingers:** Simulated skin-toned circles overlapping the edges of the document.
    6. **Background Wires/Lines:** Random thick lines drawn across the background canvas.
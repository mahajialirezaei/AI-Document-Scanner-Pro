# Data Structure Documentation

https://github.com/mahajialirezaei/CNN-Applications-Doc-Scanning-And-Enhancement

This document outlines the exact directory structure and contents of the datasets used in the Document Scanning & Enhancement project. The dataset is strictly divided into two logical segments: Synthetic Data Generation (Training/Validation) and Real-World Evaluation (Testing).

## 1. Synthetic Data Generation Sources (Training & Validation)
These directories contain the raw materials used to generate the synthetic training dataset on-the-fly.

### 1.1. Random Backgrounds
**Path:** `data/random_backgrounds/`
**Description:** Contains generic background images (tables, carpets) used as the canvas to warp clean scans onto. 
**Contents:**
* `carpet1.jpg`
* `carpet2.jpg`
* `table1.jpg`
* `table2.jpg`

### 1.2. Clean Scans
**Path:** `data/clean_scans/`
**Description:** Contains 50 high-resolution, perfectly flat, clean document scans. These serve as the "Clean Targets" for the Enhancement Network and are warped onto the random backgrounds to create degraded inputs.
**Contents:**
* `1.jpg` to `50.jpg` (50 images total)

---

## 2. Real-World Evaluation Dataset (Testing Only)
These files are strictly reserved for final model evaluation and must **never** be used during the training phase.

### 2.1. Real Photos & Annotations
**Path:** `data/raw/real_photos/`
**Description:** Contains 15 real-world smartphone photos of documents under various lighting and perspective conditions, along with their manual corner annotations generated via Roboflow.
**Contents:**
* `1_jpg.rf.[hash].jpg` to `15_jpg.rf.[hash].jpg` (15 real smartphone photos)
* `_annotations.coco.json` (The official COCO format JSON file containing the manual 4-corner keypoint labels for the 15 photos)
* `README.dataset.txt`
* `README.roboflow.txt`

### 2.2. Real Photos Scanned (Reference Scans)
**Path:** `data/raw/real_photos_scanned/`
**Description:** Contains the 15 corresponding clean reference scans (produced via a commercial scanner app) for the real smartphone photos. These serve exclusively as a commercial baseline for qualitative evaluation and OCR readability comparisons.
**Contents:**
* `1.jpg` to `15.jpg` (15 reference scan images)
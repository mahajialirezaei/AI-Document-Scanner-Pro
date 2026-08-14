## Phase 4: Architectural Benchmarks, The Dropout Fallacy, and the Heatmap Triumph

### 1. Approach A (Direct Regression) vs. Approach B (Heatmap)

In accordance with the project guidelines, we trained and evaluated two distinct corner detection architectures to see which approach inherently handled spatial coordinate extraction better.

* **The Findings:** The Heatmap approach emerged as the definitive winner. The optimal Heatmap model achieved a Mean Localization Error (MLE) of 35.04 px, completely crushing the Direct Regression model, which peaked at an MLE of 68.21 px.


* **The Architectural Reason:** Direct Regression fundamentally fails to achieve sub-pixel precision because its `Flatten` layer completely destroys the 2D spatial topography of the image. It forces the network to blindly guess numerical coordinates, whereas the Heatmap's fully convolutional U-Net preserves geometry from the pixel level all the way to output probabilities.



### 2. Phase 6 and The "Dropout Fallacy" (Geometric Collapse)

We introduced Dropout layers across both models and utilized a Cosine Annealing scheduler to observe its impact as a regularizer. The empirical observation yielded a critical engineering insight: **Dropout is catastrophic for spatial coordinate extraction**. We conclusively found that Dropout acts as destructive spatial noise rather than a semantic regularizer.

* **Heatmap Geometric Collapse:** Applying a global dropout of 0.5 blinded the network's early feature extraction layers, preventing it from seeing continuous lines. This resulted in random, disconnected blob predictions that caused intersecting lines ("Butterfly Effect") and predictions floating hundreds of pixels out of bounds. Even limiting dropout to the semantic bottleneck (`0.3`) destroyed the network's spatial coherency.


* **Regression Instability:** Introducing dropout to the fully connected layers of the Regression model destabilized the sensitive weights tasked with calculating precise decimals, leading to increased error margins and brittle predictions.



### 3. Conquering the Semantic Trap via Data-Centric Regularization

Our Phase 5 Baseline Heatmap model (`corner_heatmap_clean_nodropout_v2`) initially fell into a "Semantic Trap," confidently snapping its predictions to the high-contrast edges of dark binders instead of the actual document paper.

* **The Fix:** Instead of relying on network regularization (which destroyed geometry), we utilized **Data-Centric Regularization**. We optimized the baseline (`nodropout_v3`) by actively teaching the model to ignore thick structural edges via heavily augmented synthetic training data (e.g., explicit "Dark Binder Margins" and 3D Drop Shadows).



### 4. The Final Undisputed Champion

By relying entirely on synthetic data augmentations instead of artificial dropout, the network learned the true semantic boundary of the paper without losing its precise geometric vision. The **Phase 5 Baseline Heatmap (`nodropout_v3`)** stands as our ultimate champion, achieving an MAE of 21.42 px and an MSE of 1532.80 px², effectively finalizing the corner detection pipeline.
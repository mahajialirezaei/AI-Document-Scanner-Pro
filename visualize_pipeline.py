import os
import cv2
import torch
import matplotlib.pyplot as plt
from src.pipelines.inference import DocumentScanningPipeline

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    pipeline = DocumentScanningPipeline(
        corner_model_path='checkpoints/corner_heatmap/best_model.pth',
        enhancement_model_path='checkpoints/enhancement/best_model.pth',
        corner_approach='heatmap',
        device=device
    )

    output_dir = 'data/visualizations'
    os.makedirs(output_dir, exist_ok=True)

    real_images_dir = 'data/raw/real_photos'
    image_files = [f for f in os.listdir(real_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:10] # ۱۰ نمونه اول

    print(f"Generating visualizations for {len(image_files)} samples...\n")

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(real_images_dir, img_name)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        results = pipeline.process(image_bgr, return_intermediate=True)

        corners_img_rgb = cv2.cvtColor(results['corners_image'], cv2.COLOR_BGR2RGB)
        rectified_rgb = cv2.cvtColor(results['rectified'], cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.cvtColor(results['enhanced'], cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(corners_img_rgb)
        axes[0].set_title("1. Detected Corners & Quad", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(rectified_rgb)
        axes[1].set_title("2. Perspective Warped (Rectified)", fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(enhanced_rgb)
        axes[2].set_title("3. Final Enhanced Document", fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"vis_{idx+1}_{img_name}")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"[{idx+1}/{len(image_files)}] Saved: {save_path}")

    print(f"\nAll visualizations successfully saved in folder: '{output_dir}'")

if __name__ == "__main__":
    main()
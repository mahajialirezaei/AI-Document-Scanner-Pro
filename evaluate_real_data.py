import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.data.dataset import RealDocumentDataset
from src.models.model import EnhancementUNet # [TEMPORARY WORKAROUND] اضافه شدن برای لود مستقیم مدل

def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """تبدیل تنسور پای‌تورچ به آرایه نامپای برای نمایش و پردازش تصویر"""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    real_photos_dir = 'data/raw/real_photos'
    scanned_photos_dir = 'data/raw/real_photos_scanned'
    annotation_file = os.path.join(real_photos_dir, '_annotations.coco.json')
    
    output_dir = 'data/eval_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Real Document Dataset...")
    dataset = RealDocumentDataset(
        real_photos_dir=real_photos_dir,
        scanned_photos_dir=scanned_photos_dir,
        annotation_file=annotation_file,
        image_size=(1024, 1024) 
    )
    
    total_samples = len(dataset)
    if total_samples == 0:
        print("Error: No valid data pairs found! Check your JSON mapping and filenames.")
        return
        
    print(f"Total valid image-scan pairs found: {total_samples}\n")

    # =====================================================================
    # [TEMPORARY WORKAROUND START]
    # به جای پایپ‌لاین کامل، فقط مدل Enhancement را برای تست ایزوله لود می‌کنیم
    # =====================================================================
    print("Loading ONLY Enhancement Model for isolated testing...")
    enhancement_model = EnhancementUNet(dropout_rate=0.2).to(device)
    checkpoint = torch.load('checkpoints/regularized/best_model.pth', map_location=device)
    enhancement_model.load_state_dict(checkpoint['model_state_dict'])
    enhancement_model.eval()
    # =====================================================================
    # [TEMPORARY WORKAROUND END]
    # =====================================================================

    metrics = {'psnr': [], 'ssim': [], 'corner_mse': []}

    for idx in range(total_samples):
        sample = dataset[idx]
        raw_tensor = sample['raw_photo']
        clean_tensor = sample['clean_target']
        gt_corners = sample['corners'].numpy() 

        raw_rgb = tensor_to_rgb(raw_tensor)
        raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        clean_rgb = tensor_to_rgb(clean_tensor)
        
        h, w = raw_rgb.shape[:2]

        # =====================================================================
        # [TEMPORARY WORKAROUND START]
        # استفاده مستقیم از گوشه‌های Ground Truth برای برش تصویر (WarpPerspective)
        # =====================================================================
        gt_corners_pixel = (gt_corners * [w, h]).astype(np.float32)
        
        dst_pts = np.array([
            [0, 0],
            [1024 - 1, 0],
            [1024 - 1, 1024 - 1],
            [0, 1024 - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(gt_corners_pixel, dst_pts)
        rectified_bgr = cv2.warpPerspective(raw_bgr, matrix, (1024, 1024))
        rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)

        # تبدیل به تنسور و تغذیه به شبکه Enhancement
        rectified_tensor = TF.to_tensor(rectified_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            enhanced_tensor = enhancement_model(rectified_tensor)
            
        enhanced_rgb = tensor_to_rgb(enhanced_tensor.squeeze(0))
        pred_corners_pixel = gt_corners_pixel # موقتا برای پلات کردن خطای گوشه نداریم
        metrics['corner_mse'].append(0.0)
        # =====================================================================
        # [TEMPORARY WORKAROUND END]
        # =====================================================================

        clean_rgb_resized = cv2.resize(clean_rgb, (enhanced_rgb.shape[1], enhanced_rgb.shape[0]))

        val_psnr = psnr(clean_rgb_resized, enhanced_rgb)
        val_ssim = ssim(clean_rgb_resized, enhanced_rgb, channel_axis=2, data_range=255)
        
        metrics['psnr'].append(val_psnr)
        metrics['ssim'].append(val_ssim)

        vis_raw = raw_rgb.copy()
        for i in range(4):
            pt1 = tuple((gt_corners[i] * [w, h]).astype(int))
            pt2 = tuple((gt_corners[(i+1)%4] * [w, h]).astype(int))
            cv2.line(vis_raw, pt1, pt2, (255, 0, 0), 3)
            cv2.circle(vis_raw, pt1, 8, (255, 0, 0), -1)

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        axes[0].imshow(vis_raw)
        axes[0].set_title(f"Corners (GT Only - Temp)\nMSE: 0.0", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(rectified_rgb)
        axes[1].set_title("Perspective Warped (via GT)", fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(enhanced_rgb)
        axes[2].set_title(f"Enhanced Document\nPSNR: {val_psnr:.2f} | SSIM: {val_ssim:.2f}", fontsize=12)
        axes[2].axis('off')

        axes[3].imshow(clean_rgb_resized)
        axes[3].set_title("Ground Truth Scan", fontsize=12)
        axes[3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"eval_{idx+1:02d}.jpg")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"[{idx+1:02d}/{total_samples:02d}] Saved: {save_path} | PSNR: {val_psnr:.2f}")

    print("\n" + "="*40)
    print("=== FINAL EVALUATION SUMMARY (TEMP ISOLATED ENHANCEMENT) ===")
    print("="*40)
    print(f"Total Images Evaluated : {total_samples}")
    print(f"Average Enhancement PSNR: {np.mean(metrics['psnr']):.2f} dB")
    print(f"Average Enhancement SSIM: {np.mean(metrics['ssim']):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
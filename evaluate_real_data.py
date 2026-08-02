import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.data.dataset import RealDocumentDataset
from src.pipelines.inference import DocumentScanningPipeline

def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """تبدیل تنسور پای‌تورچ به آرایه نامپای برای نمایش و پردازش تصویر"""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ۱. مسیرهای فایل‌های جدید
    real_photos_dir = 'data/raw/real_photos'
    scanned_photos_dir = 'data/raw/real_photos_scanned'
    annotation_file = os.path.join(real_photos_dir, '_annotations.coco.json')
    
    output_dir = 'data/eval_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # ۲. بارگذاری دیتاست
    print("Loading Real Document Dataset...")
    dataset = RealDocumentDataset(
        real_photos_dir=real_photos_dir,
        scanned_photos_dir=scanned_photos_dir,
        annotation_file=annotation_file,
        image_size=(1024, 1024) # سایزی که مدل با آن کار می‌کند
    )
    
    total_samples = len(dataset)
    if total_samples == 0:
        print("Error: No valid data pairs found! Check your JSON mapping and filenames.")
        return
        
    print(f"Total valid image-scan pairs found: {total_samples}\n")

    # ۳. راه‌اندازی پایپ‌لاین
    pipeline = DocumentScanningPipeline(
        corner_model_path='checkpoints/corner_heatmap/best_model.pth',
        enhancement_model_path='checkpoints/enhancement/best_model.pth',
        corner_approach='heatmap',
        device=device
    )

    # دیکشنری ذخیره متریک‌ها
    metrics = {'psnr': [], 'ssim': [], 'corner_mse': []}

    for idx in range(total_samples):
        # دریافت داده از دیتاست جدید
        sample = dataset[idx]
        raw_tensor = sample['raw_photo']
        clean_tensor = sample['clean_target']
        gt_corners = sample['corners'].numpy() # Ground-truth (4, 2) [0, 1]

        # تبدیل به فرمت مناسب برای پایپ‌لاین (BGR numpy array)
        raw_rgb = tensor_to_rgb(raw_tensor)
        raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        clean_rgb = tensor_to_rgb(clean_tensor)
        
        h, w = raw_rgb.shape[:2]

        # ۴. اجرای پایپ‌لاین روی عکس خام
        results = pipeline.process(raw_bgr, return_intermediate=True)

        # ---------------- ارزیابی گوشه‌ها ----------------
        pred_corners = results['corners'] # مختصات پیش‌بینی شده در اسکیل تصویر (1024)
        
        # نرمالایز کردن پیش‌بینی‌ها برای مقایسه با Ground Truth
        pred_corners_norm = pred_corners.copy()
        pred_corners_norm[:, 0] /= w
        pred_corners_norm[:, 1] /= h
        
        corner_mse = np.mean((pred_corners_norm - gt_corners) ** 2)
        metrics['corner_mse'].append(corner_mse)

        # ---------------- ارزیابی ارتقاء ----------------
        enhanced_bgr = results['enhanced']
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        # هم‌سایز کردن اسکن مرجع با خروجی شبکه (در صورت تفاوت)
        clean_rgb_resized = cv2.resize(clean_rgb, (enhanced_rgb.shape[1], enhanced_rgb.shape[0]))

        # محاسبه PSNR و SSIM (هرچه PSNR/SSIM بیشتر باشد، بهتر است)
        val_psnr = psnr(clean_rgb_resized, enhanced_rgb)
        val_ssim = ssim(clean_rgb_resized, enhanced_rgb, channel_axis=2, data_range=255)
        
        metrics['psnr'].append(val_psnr)
        metrics['ssim'].append(val_ssim)

        # ---------------- تصویرسازی مقایسه‌ای ----------------
        vis_raw = raw_rgb.copy()
        
        # رسم گوشه‌های هدف (قرمز)
        for i in range(4):
            pt1 = tuple((gt_corners[i] * [w, h]).astype(int))
            pt2 = tuple((gt_corners[(i+1)%4] * [w, h]).astype(int))
            cv2.line(vis_raw, pt1, pt2, (255, 0, 0), 3)
            cv2.circle(vis_raw, pt1, 8, (255, 0, 0), -1)
            
        # رسم گوشه‌های پیش‌بینی شده مدل (سبز)
        for i in range(4):
            pt1 = tuple(pred_corners[i].astype(int))
            pt2 = tuple(pred_corners[(i+1)%4].astype(int))
            cv2.line(vis_raw, pt1, pt2, (0, 255, 0), 2)
            cv2.circle(vis_raw, pt1, 6, (0, 255, 0), -1)

        rectified_rgb = cv2.cvtColor(results['rectified'], cv2.COLOR_BGR2RGB)

        # ساخت پلات 1 در 4
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        axes[0].imshow(vis_raw)
        axes[0].set_title(f"Corners (Red=GT, Green=Pred)\nMSE: {corner_mse:.5f}", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(rectified_rgb)
        axes[1].set_title("Perspective Warped", fontsize=12)
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

    # ---------------- خلاصه نهایی ----------------
    print("\n" + "="*40)
    print("=== FINAL EVALUATION SUMMARY ===")
    print("="*40)
    print(f"Total Images Evaluated : {total_samples}")
    print(f"Average Corner MSE     : {np.mean(metrics['corner_mse']):.5f}")
    print(f"Average Enhancement PSNR: {np.mean(metrics['psnr']):.2f} dB")
    print(f"Average Enhancement SSIM: {np.mean(metrics['ssim']):.4f}")
    print("="*40)
    print(f"All visualizations saved in: {output_dir}")

if __name__ == "__main__":
    main()
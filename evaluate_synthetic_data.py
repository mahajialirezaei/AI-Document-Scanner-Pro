import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.data.data_splitter import get_synthetic_splits
from src.models.model import EnhancementUNet, CornerHeatmapModel, CornerRegressionModel
from src.pipelines.inference import apply_adaptive_binarization

def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img

def order_points(pts: np.ndarray) -> np.ndarray:
    if len(pts) != 4:
        return pts
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    sorted_pts = pts[sorted_idx]
    tl_idx = np.argmin(sorted_pts.sum(axis=1))
    return np.roll(sorted_pts, -tl_idx, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate End-to-End Pipeline on Synthetic Test Data")
    parser.add_argument("--task", type=str, default="corner_heatmap", choices=["corner_regression", "corner_heatmap"])
    parser.add_argument("--corner-ckpt", type=str, default="", help="Path to corner model checkpoint")
    parser.add_argument("--enhancement-ckpt", type=str, required=True, help="Path to enhancement model checkpoint")
    parser.add_argument("--clean-scans", type=str, default="data/clean_scans", help="Path to clean scans directory")
    parser.add_argument("--backgrounds", type=str, default="data/random_backgrounds", help="Path to backgrounds directory")
    parser.add_argument("--image-size", type=int, default=512, help="Resolution for evaluation")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of synthetic test samples to evaluate")
    parser.add_argument("--use-gt-corners", action="store_true", help="Use Ground Truth corners (bypass corner model)")
    parser.add_argument("--apply-binarization", action="store_true", help="Apply adaptive binarization to output")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = 'data/eval_synthetic_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Synthetic Document Dataset (Test Split)...")
    _, _, dataset = get_synthetic_splits(
        clean_scans_dir=args.clean_scans,
        backgrounds_dir=args.backgrounds,
        image_size=(args.image_size, args.image_size),
        seed=42,
        num_eval_samples=args.num_samples,
        train_samples_per_epoch=10 
    )
    total_samples = len(dataset)
    
    print(f"Loading Enhancement Model from: {args.enhancement_ckpt}")
    enhancement_model = EnhancementUNet(dropout_rate=0.0).to(device)
    enhancement_ckpt = torch.load(args.enhancement_ckpt, map_location=device, weights_only=True)
    enhancement_model.load_state_dict(enhancement_ckpt['model_state_dict'])
    enhancement_model.eval()

    if not args.use_gt_corners:
        if not args.corner_ckpt:
            raise ValueError("--corner-ckpt is required unless --use-gt-corners is specified.")
        print(f"Loading Corner Model ({args.task}) from: {args.corner_ckpt}")
        is_regression = (args.task == 'corner_regression')
        if is_regression:
            corner_model = CornerRegressionModel(dropout_rate=0.0).to(device)
        else:
            corner_model = CornerHeatmapModel(dropout_rate=0.0).to(device)
        corner_ckpt = torch.load(args.corner_ckpt, map_location=device, weights_only=True)
        state_dict = corner_ckpt.get('model_state_dict', corner_ckpt)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        corner_model.load_state_dict(state_dict)
        corner_model.eval()
    else:
        print("Using Ground Truth Corners (Corner Model Bypassed).")

    metrics = {'psnr': [], 'ssim': [], 'corner_mse': [], 'corner_mae': [], 'corner_mle': []}

    for idx in range(total_samples):
        sample = dataset[idx]
        raw_tensor = sample['raw_photo']
        clean_tensor = sample['clean_target']
        gt_corners = sample['corners'].numpy()

        raw_rgb = tensor_to_rgb(raw_tensor)
        raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        clean_rgb = tensor_to_rgb(clean_tensor)
        
        h, w = raw_rgb.shape[:2]
        gt_corners_pixel = order_points((gt_corners * [w, h]).astype(np.float32))

        if args.use_gt_corners:
            pred_corners_pixel = gt_corners_pixel.copy()
        else:
            raw_tensor_resized = TF.resize(raw_tensor, [256, 256]).unsqueeze(0).to(device)
            with torch.no_grad():
                if is_regression:
                    pred_coords = corner_model(raw_tensor_resized)
                    pred_corners = pred_coords.squeeze(0).cpu().numpy().reshape(4, 2)
                else:
                    _, pred_heatmaps = corner_model(raw_tensor_resized)
                    heatmaps = pred_heatmaps.squeeze(0).cpu().numpy()
                    pred_corners = []
                    for i in range(4):
                        hm = cv2.GaussianBlur(heatmaps[i], (5, 5), 0)
                        y, x = np.unravel_index(np.argmax(hm), hm.shape)
                        pred_corners.append([x / hm.shape[1], y / hm.shape[0]])
                    pred_corners = np.array(pred_corners, dtype=np.float32)
            pred_corners_pixel = order_points((pred_corners * [w, h]).astype(np.float32))

        corner_mse = np.mean((pred_corners_pixel - gt_corners_pixel) ** 2)
        corner_mae = np.mean(np.abs(pred_corners_pixel - gt_corners_pixel))
        corner_mle = np.mean(np.sqrt(np.sum((pred_corners_pixel - gt_corners_pixel) ** 2, axis=1)))
        
        metrics['corner_mse'].append(corner_mse)
        metrics['corner_mae'].append(corner_mae)
        metrics['corner_mle'].append(corner_mle)

        dst_pts = np.array([[0, 0], [args.image_size - 1, 0], [args.image_size - 1, args.image_size - 1], [0, args.image_size - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(pred_corners_pixel, dst_pts)
        rectified_bgr = cv2.warpPerspective(raw_bgr, matrix, (args.image_size, args.image_size))
        rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            enhanced_tensor = enhancement_model(TF.to_tensor(rectified_rgb).unsqueeze(0).to(device))
            
        enhanced_rgb = tensor_to_rgb(enhanced_tensor.squeeze(0))
        
        if args.apply_binarization:
            enhanced_rgb = apply_adaptive_binarization(enhanced_rgb)

        val_psnr = psnr(clean_rgb, enhanced_rgb)
        val_ssim = ssim(clean_rgb, enhanced_rgb, channel_axis=2, data_range=255)
        
        metrics['psnr'].append(val_psnr)
        metrics['ssim'].append(val_ssim)

        vis_raw = raw_rgb.copy()
        for i in range(4):
            pt1_gt = tuple((gt_corners_pixel[i]).astype(int))
            pt2_gt = tuple((gt_corners_pixel[(i+1)%4]).astype(int))
            cv2.line(vis_raw, pt1_gt, pt2_gt, (0, 255, 0), 2)
            cv2.circle(vis_raw, pt1_gt, 6, (0, 255, 0), -1)

            if not args.use_gt_corners:
                pt1_pred = tuple(pred_corners_pixel[i].astype(int))
                pt2_pred = tuple(pred_corners_pixel[(i+1)%4].astype(int))
                cv2.line(vis_raw, pt1_pred, pt2_pred, (255, 0, 0), 3)
                cv2.circle(vis_raw, pt1_pred, 8, (255, 0, 0), -1)

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        axes[0].imshow(vis_raw)
        title_str = "Corners (GT: Green)" if args.use_gt_corners else f"Corners (GT: Green, Pred: Red)\nMLE: {corner_mle:.1f} px"
        axes[0].set_title(title_str, fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(rectified_rgb)
        axes[1].set_title("Perspective Warped", fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(enhanced_rgb)
        axes[2].set_title(f"Enhanced Document\nPSNR: {val_psnr:.2f} | SSIM: {val_ssim:.2f}", fontsize=12)
        axes[2].axis('off')

        axes[3].imshow(clean_rgb)
        axes[3].set_title("Ground Truth Scan", fontsize=12)
        axes[3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"eval_{idx+1:02d}.jpg")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"[{idx+1:02d}/{total_samples:02d}] Saved: {save_path} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")

    print("\n" + "="*45)
    print("=== SYNTHETIC TEST SET EVALUATION SUMMARY ===")
    print(f"Total Images Evaluated   : {total_samples}")
    if not args.use_gt_corners:
        print(f"Average Corner MSE       : {np.mean(metrics['corner_mse']):.2f} pixels^2")
        print(f"Average Corner MAE       : {np.mean(metrics['corner_mae']):.2f} pixels")
        print(f"Average Corner MLE       : {np.mean(metrics['corner_mle']):.2f} pixels")
    else:
        print("Corner Metrics           : Bypassed (Using GT)")
    print(f"Average Enhancement PSNR : {np.mean(metrics['psnr']):.2f} dB")
    print(f"Average Enhancement SSIM : {np.mean(metrics['ssim']):.4f}")
    print("="*45)

if __name__ == "__main__":
    main()
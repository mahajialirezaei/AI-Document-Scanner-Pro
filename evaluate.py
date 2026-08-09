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
from src.data.dataset import RealDocumentDataset
from src.models.model import EnhancementUNet, CornerHeatmapModel, CornerRegressionModel
from src.evaluation.ocr_metrics import compare_readability
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
    parser = argparse.ArgumentParser(description="Unified Evaluation Script for Document Scanning Pipeline")
    parser.add_argument("--dataset-type", type=str, required=True, choices=["synthetic", "real"], help="Evaluate on 10% Synthetic Test Set or 15 Real Photos")
    parser.add_argument("--task", type=str, default="corner_heatmap", choices=["corner_regression", "corner_heatmap"])
    parser.add_argument("--corner-ckpt", type=str, default="", help="Path to corner model checkpoint")
    parser.add_argument("--enhancement-ckpt", type=str, required=True, help="Path to enhancement model checkpoint")
    
    # Data Paths
    parser.add_argument("--clean-scans", type=str, default="data/clean_scans")
    parser.add_argument("--backgrounds", type=str, default="data/random_backgrounds")
    parser.add_argument("--real-photos-dir", type=str, default="data/raw/real_photos")
    parser.add_argument("--scanned-photos-dir", type=str, default="data/raw/real_photos_scanned")
    
    parser.add_argument("--image-size", type=int, default=512, help="Resolution for evaluation")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of synthetic samples (if synthetic)")
    parser.add_argument("--use-gt-corners", action="store_true", help="Use Ground Truth corners (bypass corner model)")
    parser.add_argument("--apply-binarization", action="store_true", help="Apply adaptive binarization to output")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{args.dataset_type.upper()} EVALUATION] Using device: {device}")

    output_dir = f'data/eval_{args.dataset_type}_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset Loading
    if args.dataset_type == "synthetic":
        _, _, dataset = get_synthetic_splits(
            clean_scans_dir=args.clean_scans,
            backgrounds_dir=args.backgrounds,
            image_size=(args.image_size, args.image_size),
            seed=42, 
            num_eval_samples=args.num_samples,
            train_samples_per_epoch=10 
        )
    else:
        dataset = RealDocumentDataset(
            real_photos_dir=args.real_photos_dir,
            scanned_photos_dir=args.scanned_photos_dir,
            annotation_file=os.path.join(args.real_photos_dir, '_annotations.coco.json'),
            image_size=(args.image_size, args.image_size) 
        )
        
    total_samples = len(dataset)
    print(f"Dataset loaded. Total samples: {total_samples}")
    
    # 2. Model Loading
    print(f"Loading Enhancement Model...")
    enhancement_model = EnhancementUNet(dropout_rate=0.0).to(device)
    enhancement_ckpt = torch.load(args.enhancement_ckpt, map_location=device, weights_only=True)
    
    enhancement_state = enhancement_ckpt.get('model_state_dict', enhancement_ckpt)
    if list(enhancement_state.keys())[0].startswith('module.'):
        enhancement_state = {k.replace('module.', ''): v for k, v in enhancement_state.items()}
    enhancement_model.load_state_dict(enhancement_state)
    enhancement_model.eval()

    if not args.use_gt_corners:
        if not args.corner_ckpt:
            raise ValueError("--corner-ckpt is required unless --use-gt-corners is specified.")
        print(f"Loading Corner Model ({args.task})...")
        is_regression = (args.task == 'corner_regression')
        if is_regression:
            corner_model = CornerRegressionModel(dropout_rate=0.0).to(device)
        else:
            corner_model = CornerHeatmapModel(dropout_rate=0.0).to(device)
            
        corner_ckpt = torch.load(args.corner_ckpt, map_location=device, weights_only=True)
        corner_state = corner_ckpt.get('model_state_dict', corner_ckpt)
        if list(corner_state.keys())[0].startswith('module.'):
            corner_state = {k.replace('module.', ''): v for k, v in corner_state.items()}
        corner_model.load_state_dict(corner_state)
        corner_model.eval()
    else:
        print("Using Ground Truth Corners (Corner Model Bypassed).")

    # 3. Metrics Tracking
    metrics = {
        'psnr': [], 'ssim': [], 'corner_mse': [], 'corner_mae': [], 'corner_mle': [],
        'ocr_deg': [], 'ocr_enh': [], 'ocr_tgt': []
    }

    # 4. Evaluation Loop
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

        # Corner Extraction
        if args.use_gt_corners:
            pred_corners_pixel = gt_corners_pixel.copy()
            if args.dataset_type == "synthetic" and 'rectified_input' in sample:
                rectified_rgb = tensor_to_rgb(sample['rectified_input'])
                rectified_tensor = sample['rectified_input'].unsqueeze(0).to(device)
            else:
                dst_pts = np.array([[0, 0], [args.image_size - 1, 0], [args.image_size - 1, args.image_size - 1], [0, args.image_size - 1]], dtype=np.float32)
                matrix = cv2.getPerspectiveTransform(pred_corners_pixel, dst_pts)
                rectified_bgr = cv2.warpPerspective(raw_bgr, matrix, (args.image_size, args.image_size))
                rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
                rectified_tensor = TF.to_tensor(rectified_rgb).unsqueeze(0).to(device)
        else:
            raw_tensor_resized = TF.resize(raw_tensor, [256, 256]).unsqueeze(0).to(device)
            with torch.no_grad():
                if is_regression:
                    pred_coords = corner_model(raw_tensor_resized)
                else:
                    pred_coords, _ = corner_model(raw_tensor_resized) # Uses SoftArgmax directly
                    
            pred_corners = pred_coords.squeeze(0).cpu().numpy().reshape(4, 2)
            pred_corners_pixel = order_points((pred_corners * [w, h]).astype(np.float32))

            dst_pts = np.array([[0, 0], [args.image_size - 1, 0], [args.image_size - 1, args.image_size - 1], [0, args.image_size - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(pred_corners_pixel, dst_pts)
            rectified_bgr = cv2.warpPerspective(raw_bgr, matrix, (args.image_size, args.image_size))
            rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
            rectified_tensor = TF.to_tensor(rectified_rgb).unsqueeze(0).to(device)

        # Corner Metrics Update
        if not args.use_gt_corners:
            corner_mse = np.mean((pred_corners_pixel - gt_corners_pixel) ** 2)
            corner_mle = np.mean(np.sqrt(np.sum((pred_corners_pixel - gt_corners_pixel) ** 2, axis=1)))
            metrics['corner_mse'].append(corner_mse)
            metrics['corner_mae'].append(np.mean(np.abs(pred_corners_pixel - gt_corners_pixel)))
            metrics['corner_mle'].append(corner_mle)

        # Enhancement Execution
        with torch.no_grad():
            enhanced_tensor = enhancement_model(rectified_tensor)
        enhanced_rgb = tensor_to_rgb(enhanced_tensor.squeeze(0))
        
        if args.apply_binarization:
            enhanced_rgb = apply_adaptive_binarization(enhanced_rgb)

        clean_rgb_resized = cv2.resize(clean_rgb, (args.image_size, args.image_size))

        # Mathematical Metrics (Strictly Synthetic Only)
        val_psnr, val_ssim = 0.0, 0.0
        if args.dataset_type == "synthetic":
            val_psnr = psnr(clean_rgb_resized, enhanced_rgb)
            val_ssim = ssim(clean_rgb_resized, enhanced_rgb, channel_axis=2, data_range=255)
            metrics['psnr'].append(val_psnr)
            metrics['ssim'].append(val_ssim)

        # OCR Functional Metrics
        ocr_res = compare_readability(rectified_rgb, enhanced_rgb, clean_rgb_resized)
        metrics['ocr_deg'].append(ocr_res['degraded_confidence'])
        metrics['ocr_enh'].append(ocr_res['enhanced_confidence'])
        if 'reference_confidence' in ocr_res:
            metrics['ocr_tgt'].append(ocr_res['reference_confidence'])

        # Visualization Plotting
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
        axes[1].set_title(f"Perspective Warped\nOCR Conf: {ocr_res['degraded_confidence']:.1f}%", fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(enhanced_rgb)
        enh_title = f"Enhanced Document\nOCR Conf: {ocr_res['enhanced_confidence']:.1f}%"
        if args.dataset_type == "synthetic":
            enh_title += f"\nPSNR: {val_psnr:.2f} | SSIM: {val_ssim:.2f}"
        axes[2].set_title(enh_title, fontsize=12)
        axes[2].axis('off')

        axes[3].imshow(clean_rgb_resized)
        tgt_label = "Ground Truth Scan" if args.dataset_type == "synthetic" else "CamScanner Ref."
        axes[3].set_title(f"{tgt_label}\nOCR Conf: {ocr_res.get('reference_confidence', 0):.1f}%", fontsize=12)
        axes[3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"eval_{idx+1:02d}.jpg")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"[{idx+1:02d}/{total_samples:02d}] Saved: {save_path} | OCR (Raw -> Enh): {ocr_res['degraded_confidence']:.1f}% -> {ocr_res['enhanced_confidence']:.1f}%")

    # Final Aggregated Report
    print("\n" + "="*50)
    print(f"=== {args.dataset_type.upper()} EVALUATION SUMMARY ===")
    print(f"Total Images Evaluated     : {total_samples}")
    
    print("\n--- Geometric Metrics ---")
    if not args.use_gt_corners:
        print(f"Average Corner MSE         : {np.mean(metrics['corner_mse']):.2f} px^2")
        print(f"Average Corner MAE         : {np.mean(metrics['corner_mae']):.2f} px")
        print(f"Average Corner MLE         : {np.mean(metrics['corner_mle']):.2f} px")
    else:
        print("Corner Metrics             : Bypassed (Using GT)")

    print("\n--- Mathematical Restoration Metrics ---")
    if args.dataset_type == "synthetic":
        print(f"Average Enhancement PSNR   : {np.mean(metrics['psnr']):.2f} dB")
        print(f"Average Enhancement SSIM   : {np.mean(metrics['ssim']):.4f}")
    else:
        print("PSNR / SSIM                : Skipped for Real Data (No Ground Truth)")

    print("\n--- OCR Functional Metrics ---")
    print(f"Average Raw OCR Confidence : {np.mean(metrics['ocr_deg']):.2f}%")
    print(f"Average Enh OCR Confidence : {np.mean(metrics['ocr_enh']):.2f}%")
    if metrics['ocr_tgt']:
        print(f"Average Target Confidence  : {np.mean(metrics['ocr_tgt']):.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
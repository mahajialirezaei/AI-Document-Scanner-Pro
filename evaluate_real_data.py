import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.data.dataset import RealDocumentDataset
from src.models.model import EnhancementUNet, CornerHeatmapModel

def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sorts corners in clockwise order (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    using polar coordinates (arctan2) relative to the centroid.
    This eliminates the "Butterfly Effect" sorting bug completely.
    """
    if len(pts) != 4:
        return pts
        
    centroid = np.mean(pts, axis=0)
    
    # Calculate angles using arctan2 relative to the centroid
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    
    # Sort points by angle
    sorted_idx = np.argsort(angles)
    sorted_pts = pts[sorted_idx]
    
    # Find the top-left (min sum of x+y) in the sorted array to set the anchor
    tl_idx = np.argmin(sorted_pts.sum(axis=1))
    
    # Roll the array so that Top-Left is at index 0
    return np.roll(sorted_pts, -tl_idx, axis=0)

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
    
    enhancement_model = EnhancementUNet(dropout_rate=0.2).to(device)
    enhancement_ckpt = torch.load('checkpoints/enhancement/best_model.pth', map_location=device, weights_only=True)
    enhancement_model.load_state_dict(enhancement_ckpt['model_state_dict'])
    enhancement_model.eval()

    corner_model = CornerHeatmapModel(dropout_rate=0.2).to(device)
    # Update this path based on which model you want to evaluate (e.g., v2 regularized)
    corner_ckpt = torch.load('checkpoints/corner_heatmap_regularized_v2/best_model.pth', map_location=device, weights_only=True)
    
    state_dict = corner_ckpt.get('model_state_dict', corner_ckpt)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    corner_model.load_state_dict(state_dict)
    corner_model.eval()

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

        raw_tensor_resized = TF.resize(raw_tensor, [256, 256]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, pred_heatmaps = corner_model(raw_tensor_resized)
            
        heatmaps = pred_heatmaps.squeeze(0).cpu().numpy()
        h_hm, w_hm = heatmaps.shape[1], heatmaps.shape[2]
        
        pred_corners = []
        
        for i in range(4):
            hm = heatmaps[i]
            hm_blurred = cv2.GaussianBlur(hm, (5, 5), 0)
            y, x = np.unravel_index(np.argmax(hm_blurred), hm_blurred.shape)
            pred_corners.append([x / w_hm, y / h_hm])
            
        pred_corners = np.array(pred_corners, dtype=np.float32)

        pred_corners_pixel = (pred_corners * [w, h]).astype(np.float32)
        gt_corners_pixel = (gt_corners * [w, h]).astype(np.float32)

        pred_corners_pixel = order_points(pred_corners_pixel)
        gt_corners_pixel = order_points(gt_corners_pixel)

        corner_mse = np.mean((pred_corners_pixel - gt_corners_pixel) ** 2)
        metrics['corner_mse'].append(corner_mse)
        
        corner_mae = np.mean(np.abs(pred_corners_pixel - gt_corners_pixel))
        metrics['corner_mae'].append(corner_mae)
        
        distances = np.sqrt(np.sum((pred_corners_pixel - gt_corners_pixel) ** 2, axis=1))
        corner_mle = np.mean(distances)
        metrics['corner_mle'].append(corner_mle)

        dst_pts = np.array([
            [0, 0],
            [1024 - 1, 0],
            [1024 - 1, 1024 - 1],
            [0, 1024 - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(pred_corners_pixel, dst_pts)
        rectified_bgr = cv2.warpPerspective(raw_bgr, matrix, (1024, 1024))
        rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)

        rectified_tensor = TF.to_tensor(rectified_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced_tensor = enhancement_model(rectified_tensor)
            
        enhanced_rgb = tensor_to_rgb(enhanced_tensor.squeeze(0))
        clean_rgb_resized = cv2.resize(clean_rgb, (enhanced_rgb.shape[1], enhanced_rgb.shape[0]))

        val_psnr = psnr(clean_rgb_resized, enhanced_rgb)
        val_ssim = ssim(clean_rgb_resized, enhanced_rgb, channel_axis=2, data_range=255)
        
        metrics['psnr'].append(val_psnr)
        metrics['ssim'].append(val_ssim)

        vis_raw = raw_rgb.copy()
        for i in range(4):
            pt1_gt = tuple((gt_corners_pixel[i]).astype(int))
            pt2_gt = tuple((gt_corners_pixel[(i+1)%4]).astype(int))
            cv2.line(vis_raw, pt1_gt, pt2_gt, (0, 255, 0), 2)
            cv2.circle(vis_raw, pt1_gt, 6, (0, 255, 0), -1)

            pt1_pred = tuple(pred_corners_pixel[i].astype(int))
            pt2_pred = tuple(pred_corners_pixel[(i+1)%4].astype(int))
            cv2.line(vis_raw, pt1_pred, pt2_pred, (255, 0, 0), 3)
            cv2.circle(vis_raw, pt1_pred, 8, (255, 0, 0), -1)

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        axes[0].imshow(vis_raw)
        axes[0].set_title(f"Corners (GT: Green, Pred: Red)\nMLE: {corner_mle:.1f} px | MAE: {corner_mae:.1f} px", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(rectified_rgb)
        axes[1].set_title("Perspective Warped (via PRED)", fontsize=12)
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

        print(f"[{idx+1:02d}/{total_samples:02d}] Saved: {save_path} | PSNR: {val_psnr:.2f} | MLE: {corner_mle:.1f} | MAE: {corner_mae:.1f}")

    print("\n" + "="*45)
    print("=== FINAL END-TO-END EVALUATION SUMMARY ===")
    print(f"Total Images Evaluated   : {total_samples}")
    print(f"Average Corner MSE       : {np.mean(metrics['corner_mse']):.2f} pixels^2")
    print(f"Average Corner MAE       : {np.mean(metrics['corner_mae']):.2f} pixels")
    print(f"Average Corner MLE       : {np.mean(metrics['corner_mle']):.2f} pixels")
    print(f"Average Enhancement PSNR : {np.mean(metrics['psnr']):.2f} dB")
    print(f"Average Enhancement SSIM : {np.mean(metrics['ssim']):.4f}")
    print("="*45)

if __name__ == "__main__":
    main()
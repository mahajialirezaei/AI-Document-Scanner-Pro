import os
import cv2
import torch
import argparse
import numpy as np
import torchvision.transforms.functional as TF

from src.data.dataset import RealDocumentDataset
from src.models.model import EnhancementUNet, CornerHeatmapModel
from src.evaluation.ocr_metrics import compare_readability 
from src.pipelines.inference import apply_adaptive_binarization

def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img

def order_points(pts: np.ndarray) -> np.ndarray:
    if len(pts) != 4: return pts
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_pts = pts[np.argsort(angles)]
    tl_idx = np.argmin(sorted_pts.sum(axis=1))
    return np.roll(sorted_pts, -tl_idx, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR Readability")
    parser.add_argument("--corner-ckpt", type=str, required=True)
    parser.add_argument("--enhancement-ckpt", type=str, required=True)
    parser.add_argument("--apply-binarization", action="store_true", help="Apply adaptive binarization to enhance OCR")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dataset = RealDocumentDataset(
        real_photos_dir='data/raw/real_photos',
        scanned_photos_dir='data/raw/real_photos_scanned',
        annotation_file='data/raw/real_photos/_annotations.coco.json',
        image_size=(1024, 1024)
    )

    enhancement_model = EnhancementUNet(dropout_rate=0.0).to(device)
    enhancement_ckpt = torch.load(args.enhancement_ckpt, map_location=device, weights_only=True)
    enhancement_model.load_state_dict(enhancement_ckpt['model_state_dict'])
    enhancement_model.eval()

    corner_model = CornerHeatmapModel(dropout_rate=0.0).to(device)
    corner_ckpt = torch.load(args.corner_ckpt, map_location=device, weights_only=True)
    state_dict = corner_ckpt.get('model_state_dict', corner_ckpt)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    corner_model.load_state_dict(state_dict)
    corner_model.eval()

    avg_metrics = {'deg_conf': [], 'enh_conf': [], 'ref_conf': []}

    print("Starting OCR Evaluation...")
    for idx in range(len(dataset)):
        sample = dataset[idx]
        raw_bgr = cv2.cvtColor(tensor_to_rgb(sample['raw_photo']), cv2.COLOR_RGB2BGR)
        clean_rgb = tensor_to_rgb(sample['clean_target'])
        
        raw_tensor_resized = TF.resize(sample['raw_photo'], [256, 256]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, pred_heatmaps = corner_model(raw_tensor_resized)
            heatmaps = pred_heatmaps.squeeze(0).cpu().numpy()
            pred_corners = []
            for i in range(4):
                hm = cv2.GaussianBlur(heatmaps[i], (5, 5), 0)
                y, x = np.unravel_index(np.argmax(hm), hm.shape)
                pred_corners.append([x / hm.shape[1], y / hm.shape[0]])
            pred_corners = np.array(pred_corners, dtype=np.float32)

        pred_corners_pixel = order_points((pred_corners * [raw_bgr.shape[1], raw_bgr.shape[0]]).astype(np.float32))
        
        dst_pts = np.array([[0, 0], [1024 - 1, 0], [1024 - 1, 1024 - 1], [0, 1024 - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(pred_corners_pixel, dst_pts)
        rectified_bgr = cv2.warpPerspective(raw_bgr, matrix, (1024, 1024))
        rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            enhanced_tensor = enhancement_model(TF.to_tensor(rectified_rgb).unsqueeze(0).to(device))
            
        enhanced_rgb = tensor_to_rgb(enhanced_tensor.squeeze(0))
        
        # Apply Adaptive Binarization if flag is provided
        if args.apply_binarization:
            enhanced_rgb = apply_adaptive_binarization(enhanced_rgb)
            
        clean_rgb_resized = cv2.resize(clean_rgb, (1024, 1024))

        # OCR Evaluation
        ocr_res = compare_readability(rectified_rgb, enhanced_rgb, clean_rgb_resized)
        
        avg_metrics['deg_conf'].append(ocr_res['degraded_confidence'])
        avg_metrics['enh_conf'].append(ocr_res['enhanced_confidence'])
        if 'reference_confidence' in ocr_res:
            avg_metrics['ref_conf'].append(ocr_res['reference_confidence'])
            
        print(f"Image {idx+1:02d} | Degraded: {ocr_res['degraded_confidence']:5.1f}% | Enhanced: {ocr_res['enhanced_confidence']:5.1f}% | Target: {ocr_res.get('reference_confidence', 0):5.1f}%")

    print("\n" + "="*45)
    print("=== FINAL OCR EVALUATION SUMMARY ===")
    print(f"Average Degraded Confidence : {np.mean(avg_metrics['deg_conf']):.2f}%")
    print(f"Average Enhanced Confidence : {np.mean(avg_metrics['enh_conf']):.2f}%")
    if avg_metrics['ref_conf']:
        print(f"Average Target Confidence   : {np.mean(avg_metrics['ref_conf']):.2f}%")
    print("="*45)

if __name__ == '__main__':
    main()
"""
Inference pipelines for document scanning and enhancement.
"""

import cv2
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List, Union
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.models.model import EnhancementUNet, CornerRegressionModel, CornerHeatmapModel
try:
    from src.evaluation.ocr_metrics import compute_ocr_metrics
except ImportError:
    compute_ocr_metrics = None

def load_model(model_type: str, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", dropout_rate: float = 0.0) -> torch.nn.Module:
    if model_type == "enhancement":
        model = EnhancementUNet(dropout_rate=dropout_rate)
    elif model_type == "corner_regression":
        model = CornerRegressionModel(dropout_rate=dropout_rate)
    elif model_type == "corner_heatmap":
        model = CornerHeatmapModel(dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image: np.ndarray, input_size: int = 256) -> Tuple[torch.Tensor, Dict]:
    original_shape = image.shape[:2]
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
    elif resized.shape[2] == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, {'original_shape': original_shape, 'input_size': input_size}

def enhance_document(model: torch.nn.Module, rectified_image: np.ndarray, device: str) -> np.ndarray:
    input_tensor, _ = preprocess_image(rectified_image, input_size=1024)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    output_np = torch.clamp(output_tensor, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_bgr = cv2.cvtColor((output_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    if output_bgr.shape[:2] != rectified_image.shape[:2]:
        output_bgr = cv2.resize(output_bgr, (rectified_image.shape[1], rectified_image.shape[0]))
        
    return output_bgr

def apply_adaptive_binarization(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 and image.shape[2] == 3 else image
    blurred = cv2.medianBlur(gray, 3)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def detect_corners_regression(model: torch.nn.Module, raw_image: np.ndarray, device: str = None) -> Tuple[np.ndarray, float]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    input_tensor, metadata = preprocess_image(raw_image)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        
    corners_norm = output.squeeze(0).cpu().numpy()
    corners = corners_norm.reshape(4, 2)
    corners[:, 0] *= metadata['original_shape'][1]
    corners[:, 1] *= metadata['original_shape'][0]
    
    corners = order_corners(corners)
    h, w = metadata['original_shape']
    valid = ((corners[:, 0] >= 0) & (corners[:, 0] <= w) & (corners[:, 1] >= 0) & (corners[:, 1] <= h))
    confidence = float(valid.all())
    return corners, confidence

def detect_corners_heatmap(model: torch.nn.Module, raw_image: np.ndarray, device: str) -> Tuple[np.ndarray, List[float], np.ndarray]:
    input_tensor, metadata = preprocess_image(raw_image)
    with torch.no_grad():
        output = model(input_tensor.to(device))
        
    heatmaps_np = np.squeeze((output[1] if isinstance(output, tuple) else output).detach().cpu().numpy())
    if heatmaps_np.ndim == 3 and heatmaps_np.shape[-1] == 4 and heatmaps_np.shape[0] != 4:
        heatmaps_np = np.transpose(heatmaps_np, (2, 0, 1))

    corners, confidences = [], []
    for i in range(4):
        hm_blurred = cv2.GaussianBlur(heatmaps_np[i], (5, 5), 0)
        y, x = np.unravel_index(np.argmax(hm_blurred), hm_blurred.shape)
        corners.append([float(x) / hm_blurred.shape[1], float(y) / hm_blurred.shape[0]])
        confidences.append(float(np.max(hm_blurred)))
        
    corners = np.array(corners, dtype=np.float32)
    corners[:, 0] *= metadata['original_shape'][1]
    corners[:, 1] *= metadata['original_shape'][0]
    return corners, confidences, heatmaps_np

def detect_corners_ensemble(models: List[torch.nn.Module], raw_image: np.ndarray, device: str) -> np.ndarray:
    """Smart Ensemble: Takes max confidence per individual corner across multiple models."""
    best_corners = np.zeros((4, 2), dtype=np.float32)
    best_confs = [-1.0] * 4
    
    for model in models:
        corners, confs, _ = detect_corners_heatmap(model, raw_image, device)
        for i in range(4):
            if confs[i] > best_confs[i]:
                best_confs[i] = confs[i]
                best_corners[i] = corners[i]
                
    return best_corners

def order_corners(pts: np.ndarray) -> np.ndarray:
    if len(pts) != 4: return pts
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_pts = pts[np.argsort(angles)]
    return np.roll(sorted_pts, -np.argmin(sorted_pts.sum(axis=1)), axis=0)

def apply_perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    output_size = (1024, 1024)
    dst = np.array([[0, 0], [1023, 0], [1023, 1023], [0, 1023]], dtype=np.float32)
    H, _ = cv2.findHomography(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, H, output_size)

def draw_corners_on_image(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    output = image.copy()
    color = (0, 255, 0)
    for i in range(4):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i + 1) % 4].astype(int))
        cv2.line(output, pt1, pt2, color, 3)
        cv2.circle(output, pt1, 8, color, -1)
    return output

class DocumentScanningPipeline:
    def __init__(self, models_registry: Dict[str, torch.nn.Module], device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = models_registry
        
    def process(self, raw_image: np.ndarray, corner_method: str, enhancement_method: str, apply_binarization: bool, reference_img: Optional[np.ndarray] = None) -> Dict:
        results = {}
        
        # 1. Corner Detection
        if corner_method == "ensemble":
            models_to_ensemble = [self.models["heatmap_v2"], self.models["heatmap_v3"]]
            corners = detect_corners_ensemble(models_to_ensemble, raw_image, self.device)
        elif corner_method == "regression":
            corners, _ = detect_corners_regression(self.models[corner_method], raw_image, device=self.device)
        else:
            corners, _, _ = detect_corners_heatmap(self.models[corner_method], raw_image, self.device)
            
        corners = order_corners(corners)
        results['corners_image'] = draw_corners_on_image(raw_image, corners)
        
        # 2. Rectification & Enhancement
        rectified = apply_perspective_transform(raw_image, corners)
        enhanced = enhance_document(self.models[enhancement_method], rectified, self.device)
        
        if apply_binarization:
            enhanced = apply_adaptive_binarization(enhanced)
            
        results['enhanced'] = enhanced
        
        # 3. Metrics Calculation
        metrics = {}
        if compute_ocr_metrics:
            raw_ocr = compute_ocr_metrics(rectified)
            enh_ocr = compute_ocr_metrics(enhanced)
            metrics['ocr_raw'] = raw_ocr['confidence']
            metrics['ocr_enhanced'] = enh_ocr['confidence']
            
            if reference_img is not None:
                ref_ocr = compute_ocr_metrics(reference_img)
                metrics['ocr_target'] = ref_ocr['confidence']
                
        if reference_img is not None:
            ref_resized = cv2.resize(reference_img, (enhanced.shape[1], enhanced.shape[0]))
            metrics['psnr'] = float(psnr(ref_resized, enhanced))
            metrics['ssim'] = float(ssim(ref_resized, enhanced, channel_axis=2, data_range=255))
            
        results['metrics'] = metrics
        return results
"""
Inference pipelines for document scanning and enhancement.
Includes Smart Ensemble geometry constraints and statistical bypass logic (Gatekeepers).
"""

import cv2
import numpy as np
import torch
import itertools
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

def polygon_area(corners: np.ndarray) -> float:
    """Calculates the area of a polygon using the Shoelace formula."""
    x = corners[:, 0]
    y = corners[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_internal_angles(quad: np.ndarray) -> List[float]:
    """Calculates the 4 internal angles of a quadrilateral in degrees."""
    angles = []
    for i in range(4):
        p1 = quad[(i-1)%4]
        p2 = quad[i]
        p3 = quad[(i+1)%4]
        v1 = p1 - p2
        v2 = p3 - p2
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        angles.append(angle)
    return angles

def detect_corners_ensemble(models: List[torch.nn.Module], raw_image: np.ndarray, device: str) -> np.ndarray:
    """
    Advanced Smart Ensemble 3.0: 
    Implements strict projective geometry heuristics and perspective symmetry rules.
    """
    all_corners = []
    all_confs = []
    
    for model in models:
        corners, confs, _ = detect_corners_heatmap(model, raw_image, device)
        
        centroid = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        
        tl_idx = np.argmin(corners[sorted_idx].sum(axis=1))
        final_idx = np.roll(sorted_idx, -tl_idx)
        
        all_corners.append(corners[final_idx])
        all_confs.append(np.array(confs)[final_idx])
        
    best_score = -float('inf')
    best_quad = None
    
    h, w = raw_image.shape[:2]
    image_area = h * w
    min_edge_dist = min(h, w) * 0.15 
    
    model_indices = range(len(models))
    for indices in itertools.product(model_indices, repeat=4):
        quad = np.array([
            all_corners[indices[0]][0], 
            all_corners[indices[1]][1], 
            all_corners[indices[2]][2], 
            all_corners[indices[3]][3]  
        ])
        
        weighted_conf = 0.0
        for pos_idx, m_idx in enumerate(indices):
            conf = all_confs[m_idx][pos_idx]
            if m_idx == 0:  
                conf *= 1.2 
            weighted_conf += conf
        avg_conf = weighted_conf / 4.0
        
        valid = True
        
        len_top = np.linalg.norm(quad[0] - quad[1])
        len_right = np.linalg.norm(quad[1] - quad[2])
        len_bottom = np.linalg.norm(quad[2] - quad[3])
        len_left = np.linalg.norm(quad[3] - quad[0])
        
        edges = [len_top, len_right, len_bottom, len_left]
        if min(edges) < min_edge_dist:
            valid = False
            
        if valid:
            if (len_top / (len_bottom + 1e-5) > 2.2) or (len_bottom / (len_top + 1e-5) > 2.2):
                valid = False
            if (len_left / (len_right + 1e-5) > 2.2) or (len_right / (len_left + 1e-5) > 2.2):
                valid = False
            
        if valid:
            internal_angles = get_internal_angles(quad)
            if min(internal_angles) < 55 or max(internal_angles) > 125:
                valid = False
                
            if valid:
                for i in range(4):
                    a1, a2 = internal_angles[i], internal_angles[(i+1)%4]
                    a3, a4 = internal_angles[(i+2)%4], internal_angles[(i+3)%4]
                    if abs(a1 - 90) < 15 and abs(a2 - 90) < 15:
                        if abs(a3 - a4) > 25:
                            valid = False
                            break
            
        if valid:
            cross_products = []
            for i in range(4):
                p0, p1, p2 = quad[i], quad[(i+1)%4], quad[(i+2)%4]
                v1 = p1 - p0
                v2 = p2 - p1
                cross_products.append(v1[0]*v2[1] - v1[1]*v2[0])
                
            signs = np.sign(cross_products)
            if not np.all(signs == signs[0]) or np.any(signs == 0):
                valid = False
            
        if valid:
            quad_area = polygon_area(quad)
            normalized_area = quad_area / image_area
            final_score = (0.85 * avg_conf) + (0.15 * normalized_area)
            
            if final_score > best_score:
                best_score = final_score
                best_quad = quad
            
    if best_quad is None:
        best_quad = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            max_model_idx = np.argmax([all_confs[m][i] for m in range(len(models))])
            best_quad[i] = all_corners[max_model_idx][i]
            
    return best_quad

def order_corners(pts: np.ndarray) -> np.ndarray:
    if len(pts) != 4: return pts
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_pts = pts[np.argsort(angles)]
    return np.roll(sorted_pts, -np.argmin(sorted_pts.sum(axis=1)), axis=0)

def apply_perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = corners

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, H, (maxWidth, maxHeight))

    return warped

def draw_corners_on_image(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    output = image.copy()
    color = (0, 255, 0)
    for i in range(4):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i + 1) % 4].astype(int))
        cv2.line(output, pt1, pt2, color, 3)
        cv2.circle(output, pt1, 8, color, -1)
    return output

# --- GATEKEEPER FUNCTIONS ---

def is_already_cropped(image: np.ndarray, variance_thresh: float = 600.0, white_thresh: int = 200, white_ratio: float = 0.80) -> bool:
    """
    Idea A: Statistically determines if an image has already been cropped
    by analyzing the homogeneity and brightness of its outer borders.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    
    border_y = max(1, int(h * 0.02))
    border_x = max(1, int(w * 0.02))
    
    top = gray[:border_y, :]
    bottom = gray[h-border_y:, :]
    left = gray[border_y:h-border_y, :border_x]
    right = gray[border_y:h-border_y, w-border_x:]
    
    border_pixels = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])
    
    variance = np.var(border_pixels)
    bright_ratio = np.mean(border_pixels > white_thresh)
    
    # If the border is highly uniform (low variance) AND overwhelmingly bright,
    # it is assumed to be an already cropped scan, bypassing corner detection.
    return bool(variance < variance_thresh and bright_ratio > white_ratio)

def is_already_enhanced(image: np.ndarray, white_thresh: int = 240, mass_threshold: float = 0.40) -> bool:
    """
    Idea C: Statistically determines if an image has already been enhanced/scanned
    by checking the right-tail mass (pure white pixels) of its grayscale histogram.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    white_mass_ratio = np.sum(gray > white_thresh) / gray.size
    
    # Raw photos rarely have 40% of their pixels at absolute white (>240).
    return bool(white_mass_ratio > mass_threshold)

class DocumentScanningPipeline:
    def __init__(self, models_registry: Dict[str, torch.nn.Module], device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = models_registry
        
    def process(self, raw_image: np.ndarray, corner_method: str, enhancement_method: str, apply_binarization: bool, reference_img: Optional[np.ndarray] = None) -> Dict:
        results = {}
        
        # --- 1. Check Idea A: Already Cropped? ---
        if is_already_cropped(raw_image):
            h, w = raw_image.shape[:2]
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            results['corners_image'] = raw_image.copy() # No need to draw green lines on already clean borders
            rectified = raw_image.copy() # Preserve original aspect ratio and resolution
        else:
            # Standard Corner Detection
            if corner_method == "ensemble":
                models_to_ensemble = [
                    self.models["heatmap_v4_reg"], # 🥇 Gold
                    self.models["heatmap_v3"],     # 🥈 Silver
                    self.models["heatmap_v2"]      # 🥉 Bronze
                ]
                corners = detect_corners_ensemble(models_to_ensemble, raw_image, self.device)
            elif corner_method == "regression":
                corners, _ = detect_corners_regression(self.models[corner_method], raw_image, device=self.device)
            else:
                corners, _, _ = detect_corners_heatmap(self.models[corner_method], raw_image, self.device)
                
            corners = order_corners(corners)
            results['corners_image'] = draw_corners_on_image(raw_image, corners)
            rectified = apply_perspective_transform(raw_image, corners)
        
        # --- 2. Check Idea C: Already Enhanced? ---
        if is_already_enhanced(rectified):
            enhanced = rectified.copy() # Bypass Enhancement Network to prevent washed-out text
        else:
            enhanced = enhance_document(self.models[enhancement_method], rectified, self.device)
            
        if apply_binarization:
            enhanced = apply_adaptive_binarization(enhanced)
            
        results['enhanced'] = enhanced
        
        # --- 3. Metrics Calculation ---
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
            # Resize reference to match enhanced image's dynamic aspect ratio
            ref_resized = cv2.resize(reference_img, (enhanced.shape[1], enhanced.shape[0]))
            metrics['psnr'] = float(psnr(ref_resized, enhanced))
            metrics['ssim'] = float(ssim(ref_resized, enhanced, channel_axis=2, data_range=255))
            
        results['metrics'] = metrics
        return results
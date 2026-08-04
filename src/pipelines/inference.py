"""
Inference pipelines for document scanning and enhancement.
"""

import cv2
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path

from src.models.model import EnhancementUNet, CornerRegressionModel, CornerHeatmapModel

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
    
    metadata = {
        'original_shape': original_shape,
        'input_size': input_size,
        'scale_h': original_shape[0] / input_size,
        'scale_w': original_shape[1] / input_size
    }
    return tensor, metadata

def enhance_document(model: torch.nn.Module, rectified_image: np.ndarray, device: str = None, image_size: int = 1024) -> Tuple[np.ndarray, torch.Tensor]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    input_tensor, metadata = preprocess_image(rectified_image, input_size=image_size)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_np = (output_np * 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    if output_bgr.shape[:2] != rectified_image.shape[:2]:
        output_bgr = cv2.resize(output_bgr, (rectified_image.shape[1], rectified_image.shape[0]))
        
    return output_bgr, output_tensor

def detect_corners_regression(model: torch.nn.Module, raw_image: np.ndarray, confidence_threshold: float = 0.5, device: str = None) -> Tuple[np.ndarray, float]:
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

def detect_corners_heatmap(model: torch.nn.Module, raw_image: np.ndarray, heatmap_threshold: float = 0.15, device: str = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    input_tensor, metadata = preprocess_image(raw_image)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        
    if isinstance(output, tuple):
        heatmaps = output[1] if len(output) > 1 else output[0]
    else:
        heatmaps = output
        
    heatmaps_np = heatmaps.detach().cpu().numpy()
    heatmaps_np = np.squeeze(heatmaps_np)
    
    if heatmaps_np.ndim == 3:
        if heatmaps_np.shape[-1] == 4 and heatmaps_np.shape[0] != 4:
            heatmaps_np = np.transpose(heatmaps_np, (2, 0, 1))
    elif heatmaps_np.ndim == 2:
        heatmaps_np = np.stack([heatmaps_np] * 4, axis=0)
    elif heatmaps_np.ndim < 2:
        heatmaps_np = np.zeros((4, 64, 64), dtype=np.float32)

    corners = []
    confidences = []
    
    for i in range(4):
        c_idx = i if i < heatmaps_np.shape[0] else 0
        heatmap = heatmaps_np[c_idx]
        if heatmap.ndim > 2: heatmap = np.squeeze(heatmap)
        if heatmap.ndim < 2: heatmap = np.zeros((64, 64), dtype=np.float32)
        
        h_map, w_map = heatmap.shape[:2]
        hm_blurred = cv2.GaussianBlur(heatmap, (5, 5), 0)
        max_conf = float(np.max(hm_blurred))
        
        flat_idx = np.argmax(hm_blurred)
        y, x = np.unravel_index(flat_idx, (h_map, w_map))
        
        x_norm = float(x) / float(w_map)
        y_norm = float(y) / float(h_map)
        corners.append([x_norm, y_norm])
        confidences.append(max_conf)
        
    corners = np.array(corners, dtype=np.float32)
    
    valid_mask = np.array(confidences) > heatmap_threshold
    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(corners[i] - corners[j])
            if dist < 0.1:
                weaker = i if confidences[i] < confidences[j] else j
                valid_mask[weaker] = False
                
    for i in range(4):
        if not valid_mask[i]:
            opp = (i + 2) % 4
            adj1 = (i + 1) % 4
            adj2 = (i - 1) % 4
            if valid_mask[opp] and valid_mask[adj1] and valid_mask[adj2]:
                corners[i] = corners[adj1] + corners[adj2] - corners[opp]
                corners[i] = np.clip(corners[i], 0.0, 1.0)

    corners[:, 0] *= metadata['original_shape'][1]
    corners[:, 1] *= metadata['original_shape'][0]
    
    corners = order_corners(corners)
    return corners, heatmaps_np

def order_corners(pts: np.ndarray) -> np.ndarray:
    if len(pts) != 4:
        return pts
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_half = y_sorted[:2, :]
    bottom_half = y_sorted[2:, :]
    
    tl, tr = top_half[np.argsort(top_half[:, 0])]
    bl, br = bottom_half[np.argsort(bottom_half[:, 0])]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def apply_perspective_transform(image: np.ndarray, corners: np.ndarray, output_size: Tuple[int, int] = None) -> np.ndarray:
    if output_size is None:
        top_width = np.linalg.norm(corners[0] - corners[1])
        bottom_width = np.linalg.norm(corners[3] - corners[2])
        max_width = int(max(top_width, bottom_width))
        left_height = np.linalg.norm(corners[0] - corners[3])
        right_height = np.linalg.norm(corners[1] - corners[2])
        max_height = int(max(left_height, right_height))
        output_size = (max_width, max_height)
        
    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)
    
    H, _ = cv2.findHomography(corners.astype(np.float32), dst)
    rectified = cv2.warpPerspective(image, H, output_size)
    return rectified

def draw_corners_on_image(image: np.ndarray, corners: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3, circle_radius: int = 8) -> np.ndarray:
    output = image.copy()
    for i in range(4):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i + 1) % 4].astype(int))
        cv2.line(output, pt1, pt2, color, thickness)
        
    labels = ['TL', 'TR', 'BR', 'BL']
    for i, corner in enumerate(corners):
        pt = tuple(corner.astype(int))
        cv2.circle(output, pt, circle_radius, color, -1)
        cv2.putText(output, labels[i], (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return output

class DocumentScanningPipeline:
    def __init__(self, corner_model_path: str, enhancement_model_path: str, corner_approach: str = 'heatmap', dropout_rate: float = 0.0, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.corner_approach = corner_approach
        
        if corner_approach == 'regression':
            self.corner_model = load_model('corner_regression', corner_model_path, device, dropout_rate)
        else:
            self.corner_model = load_model('corner_heatmap', corner_model_path, device, dropout_rate)
            
        self.enhancement_model = load_model('enhancement', enhancement_model_path, device, dropout_rate)
        
    def process(self, raw_image: np.ndarray, return_intermediate: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        results = {}
        if self.corner_approach == 'regression':
            corners, confidence = detect_corners_regression(self.corner_model, raw_image, device=self.device)
        else:
            corners, heatmaps = detect_corners_heatmap(self.corner_model, raw_image, device=self.device)
            results['heatmaps'] = heatmaps
            
        results['corners'] = corners
        results['corners_image'] = draw_corners_on_image(raw_image, corners)
        
        rectified = apply_perspective_transform(raw_image, corners)
        results['rectified'] = rectified
        
        enhanced, _ = enhance_document(self.enhancement_model, rectified, device=self.device, image_size=1024)
        results['enhanced'] = enhanced
        
        if return_intermediate:
            return results
        else:
            return enhanced
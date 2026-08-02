"""
Phase 5: Independent Inference & Evaluation

This module provides comprehensive evaluation metrics and inference pipelines
for document enhancement and corner detection models.

Features:
- PSNR and SSIM computation for enhancement quality
- Corner localization error metrics
- OCR-based readability assessment
- Visualization utilities
- Independent inference pipelines
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image


class EnhancementEvaluator:
    """
    Evaluator for Task 1: Document Enhancement Network
    
    Computes PSNR, SSIM, and provides visualization capabilities
    for enhancement model performance assessment.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
    def compute_psnr(self, pred: np.ndarray, target: np.ndarray, 
                     data_range: float = 1.0) -> float:
        """
        Compute Peak Signal-to-Noise Ratio between prediction and target.
        
        Args:
            pred: Predicted image (H, W, C) or (H, W) in range [0, 1]
            target: Target image (H, W, C) or (H, W) in range [0, 1]
            data_range: Maximum value range of the images
            
        Returns:
            PSNR value in dB
        """
        return psnr(target, pred, data_range=data_range)
    
    def compute_ssim(self, pred: np.ndarray, target: np.ndarray,
                     data_range: float = 1.0, channel_axis: int = -1) -> float:
        """
        Compute Structural Similarity Index between prediction and target.
        
        Args:
            pred: Predicted image (H, W, C) or (H, W) in range [0, 1]
            target: Target image (H, W, C) or (H, W) in range [0, 1]
            data_range: Maximum value range of the images
            channel_axis: Axis of the channel dimension
            
        Returns:
            SSIM value in range [-1, 1]
        """
        if len(pred.shape) == 3:
            return ssim(target, pred, data_range=data_range, channel_axis=channel_axis)
        else:
            return ssim(target, pred, data_range=data_range)
    
    def evaluate_batch(self, predictions: torch.Tensor, targets: torch.Tensor,
                       degraded: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate enhancement quality on a batch of images.
        
        Args:
            predictions: Enhanced images tensor (B, C, H, W) in range [0, 1]
            targets: Clean target images tensor (B, C, H, W) in range [0, 1]
            degraded: Degraded input images tensor (B, C, H, W) in range [0, 1]
            
        Returns:
            Dictionary with mean PSNR, SSIM, and baseline metrics
        """
        # Convert to numpy (B, C, H, W) -> (B, H, W, C)
        preds_np = predictions.cpu().permute(0, 2, 3, 1).numpy()
        targets_np = targets.cpu().permute(0, 2, 3, 1).numpy()
        
        psnr_values = []
        ssim_values = []
        
        for i in range(predictions.size(0)):
            pred_img = preds_np[i]
            target_img = targets_np[i]
            
            psnr_val = self.compute_psnr(pred_img, target_img)
            ssim_val = self.compute_ssim(pred_img, target_img)
            
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
        
        results = {
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'ssim_mean': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values)
        }
        
        # Compute baseline metrics on degraded inputs if provided
        if degraded is not None:
            degraded_np = degraded.cpu().permute(0, 2, 3, 1).numpy()
            baseline_psnr = []
            baseline_ssim = []
            
            for i in range(degraded.size(0)):
                deg_img = degraded_np[i]
                target_img = targets_np[i]
                
                baseline_psnr.append(self.compute_psnr(deg_img, target_img))
                baseline_ssim.append(self.compute_ssim(deg_img, target_img))
            
            results['baseline_psnr_mean'] = np.mean(baseline_psnr)
            results['baseline_ssim_mean'] = np.mean(baseline_ssim)
            results['psnr_improvement'] = results['psnr_mean'] - results['baseline_psnr_mean']
            results['ssim_improvement'] = results['ssim_mean'] - results['baseline_ssim_mean']
        
        return results
    
    def create_visualization_triplet(self, degraded: np.ndarray, 
                                      enhanced: np.ndarray,
                                      target: np.ndarray) -> np.ndarray:
        """
        Create a side-by-side visualization triplet.
        
        Args:
            degraded: Degraded input image (H, W, C)
            enhanced: Enhanced output image (H, W, C)
            target: Clean target image (H, W, C)
            
        Returns:
            Combined image (H, 3*W, C) for visualization
        """
        # Ensure all images are in displayable format
        degraded_vis = np.clip(degraded * 255, 0, 255).astype(np.uint8)
        enhanced_vis = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        target_vis = np.clip(target * 255, 0, 255).astype(np.uint8)
        
        # Concatenate horizontally
        triplet = np.hstack([degraded_vis, enhanced_vis, target_vis])
        
        return triplet


class CornerDetectionEvaluator:
    """
    Evaluator for Task 2: Corner Detection Models
    
    Computes localization errors and success rates for both
    regression (Approach A) and heatmap (Approach B) methods.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256),
                 success_threshold: float = 5.0):
        """
        Args:
            image_size: Size of input images (H, W)
            success_threshold: Pixel threshold for successful detection
        """
        self.image_size = image_size
        self.success_threshold = success_threshold
        
    def compute_localization_error(self, predicted_corners: np.ndarray,
                                    ground_truth_corners: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance between predicted and ground truth corners.
        
        Args:
            predicted_corners: Predicted corners (N, 4, 2) or (N, 8)
            ground_truth_corners: Ground truth corners (N, 4, 2) or (N, 8)
            
        Returns:
            Per-corner errors (N, 4)
        """
        # Reshape if flattened
        if predicted_corners.ndim == 2 and predicted_corners.shape[1] == 8:
            predicted_corners = predicted_corners.reshape(-1, 4, 2)
        if ground_truth_corners.ndim == 2 and ground_truth_corners.shape[1] == 8:
            ground_truth_corners = ground_truth_corners.reshape(-1, 4, 2)
        
        # Convert from normalized [0, 1] to pixel coordinates
        h, w = self.image_size
        pred_pixels = predicted_corners * np.array([w - 1, h - 1])
        gt_pixels = ground_truth_corners * np.array([w - 1, h - 1])
        
        # Compute Euclidean distance for each corner
        errors = np.sqrt(np.sum((pred_pixels - gt_pixels) ** 2, axis=-1))
        
        return errors
    
    def compute_success_rate(self, errors: np.ndarray) -> float:
        """
        Compute the percentage of corners detected within threshold.
        
        Args:
            errors: Per-corner localization errors (N, 4)
            
        Returns:
            Success rate as a fraction [0, 1]
        """
        total_corners = errors.size
        successful_detections = np.sum(errors < self.success_threshold)
        
        return successful_detections / total_corners
    
    def evaluate_batch(self, predicted_corners: torch.Tensor,
                       ground_truth_corners: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate corner detection performance on a batch.
        
        Args:
            predicted_corners: Predicted corners (B, 8) or (B, 4, 2)
            ground_truth_corners: Ground truth corners (B, 8) or (B, 4, 2)
            
        Returns:
            Dictionary with mean error, per-corner errors, and success rate
        """
        pred_np = predicted_corners.cpu().numpy()
        gt_np = ground_truth_corners.cpu().numpy()
        
        # Compute per-corner errors
        errors = self.compute_localization_error(pred_np, gt_np)
        
        # Mean error per corner position
        mean_error_per_corner = np.mean(errors, axis=0)  # Shape: (4,)
        
        # Overall statistics
        overall_mean_error = np.mean(errors)
        overall_std_error = np.std(errors)
        success_rate = self.compute_success_rate(errors)
        
        # Corner names for clarity
        corner_names = ['top-left', 'top-right', 'bottom-right', 'bottom-left']
        
        return {
            'mean_localization_error': overall_mean_error,
            'std_localization_error': overall_std_error,
            'success_rate': success_rate,
            'per_corner_errors': dict(zip(corner_names, mean_error_per_corner)),
            'all_errors': errors
        }
    
    def compare_approaches(self, approach_a_results: Dict,
                           approach_b_results: Dict) -> Dict[str, any]:
        """
        Compare Approach A (Regression) vs Approach B (Heatmap).
        
        Args:
            approach_a_results: Results from regression model evaluation
            approach_b_results: Results from heatmap model evaluation
            
        Returns:
            Comparison dictionary
        """
        error_a = approach_a_results['mean_localization_error']
        error_b = approach_b_results['mean_localization_error']
        
        success_a = approach_a_results['success_rate']
        success_b = approach_b_results['success_rate']
        
        better_approach = 'A' if error_a < error_b else 'B'
        
        return {
            'approach_a_error': error_a,
            'approach_b_error': error_b,
            'approach_a_success_rate': success_a,
            'approach_b_success_rate': success_b,
            'better_approach': better_approach,
            'error_difference': abs(error_a - error_b),
            'success_rate_difference': abs(success_a - success_b)
        }


class OCREvaluator:
    """
    OCR-based readability assessment using Tesseract.
    
    Evaluates text readability by comparing OCR confidence scores
    and character accuracy between enhanced and degraded images.
    """
    
    def __init__(self, lang: str = 'eng'):
        """
        Args:
            lang: Tesseract language code
        """
        try:
            import pytesseract
            
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
            
            self.pytesseract = pytesseract
            self.available = True
        except (ImportError, Exception):
            self.available = False
            print("Warning: pytesseract or Tesseract engine not available.")
            
        self.lang = lang
    
    def extract_text_with_confidence(self, image: np.ndarray) -> Dict[str, any]:
        """
        Extract text and confidence scores from an image.
        
        Args:
            image: Input image (H, W, C) in range [0, 255]
            
        Returns:
            Dictionary with text, confidence, and detailed info
        """
        if not self.available:
            return {'text': '', 'confidence': 0.0, 'available': False}
        
        # Ensure image is in proper format
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        # Get detailed OCR data
        data = self.pytesseract.image_to_data(image, lang=self.lang, 
                                               output_type=self.pytesseract.Output.DICT)
        
        # Filter out empty confidences
        confidences = [c for c in data['conf'] if c > -1]
        
        if confidences:
            avg_confidence = np.mean(confidences)
        else:
            avg_confidence = 0.0
        
        text = self.pytesseract.image_to_string(image, lang=self.lang)
        
        return {
            'text': text,
            'confidence': avg_confidence,
            'word_count': len([w for w in data['text'] if w.strip()]),
            'available': True
        }
    
    def evaluate_readability(self, degraded_image: np.ndarray,
                             enhanced_image: np.ndarray,
                             reference_image: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate readability improvement through OCR metrics.
        
        Args:
            degraded_image: Original degraded image
            enhanced_image: Enhanced image from model
            reference_image: Optional clean reference scan
            
        Returns:
            Dictionary with readability metrics
        """
        degraded_result = self.extract_text_with_confidence(degraded_image)
        enhanced_result = self.extract_text_with_confidence(enhanced_image)
        
        results = {
            'degraded_confidence': degraded_result['confidence'],
            'enhanced_confidence': enhanced_result['confidence'],
            'confidence_improvement': enhanced_result['confidence'] - degraded_result['confidence'],
            'degraded_word_count': degraded_result['word_count'],
            'enhanced_word_count': enhanced_result['word_count'],
        }
        
        # Compare with reference if available
        if reference_image is not None and self.available:
            ref_result = self.extract_text_with_confidence(reference_image)
            results['reference_confidence'] = ref_result['confidence']
            results['reference_word_count'] = ref_result['word_count']
            
            # Character-level comparison (simple Levenshtein-like)
            ref_text = ref_result['text'].lower().strip()
            enhanced_text = enhanced_result['text'].lower().strip()
            
            if ref_text:
                # Simple character accuracy
                matches = sum(a == b for a, b in zip(ref_text, enhanced_text))
                max_len = max(len(ref_text), len(enhanced_text))
                char_accuracy = matches / max_len if max_len > 0 else 0
                results['character_accuracy'] = char_accuracy
        
        return results


class InferencePipeline:
    """
    Independent inference pipelines for deployment.
    
    Provides clean, standalone inference functions that can be
    used without the training infrastructure.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def enhance_document(self, model: torch.nn.Module,
                         input_image: np.ndarray,
                         target_size: Tuple[int, int] = (256, 256)) -> Dict[str, np.ndarray]:
        """
        Task 1 Inference: Enhance a rectified document image.
        
        Args:
            model: Trained EnhancementUNet model
            input_image: Input image (H, W, C) in any range
            target_size: Size to resize input to
            
        Returns:
            Dictionary with enhanced image and metadata
        """
        model.eval()
        model.to(self.device)
        
        # Preprocessing
        if input_image.dtype != np.float32:
            input_image = input_image.astype(np.float32) / 255.0
        
        # Resize to model input size
        resized = cv2.resize(input_image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize and convert to tensor
        tensor_input = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        tensor_input = tensor_input.to(self.device)
        
        # Inference
        with torch.no_grad():
            enhanced_tensor = model(tensor_input)
        
        # Post-processing
        enhanced_np = enhanced_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        enhanced_np = np.clip(enhanced_np, 0, 1)
        
        # Resize back to original size if needed
        if input_image.shape[:2] != target_size:
            enhanced_full = cv2.resize(enhanced_np, 
                                        (input_image.shape[1], input_image.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
        else:
            enhanced_full = enhanced_np
        
        return {
            'enhanced': enhanced_full,
            'enhanced_resized': enhanced_np,
            'input_normalized': resized
        }
    
    def detect_corners(self, model: torch.nn.Module,
                       input_image: np.ndarray,
                       original_size: Optional[Tuple[int, int]] = None,
                       approach: str = 'regression') -> Dict[str, any]:
        """
        Task 2 Inference: Detect document corners in a raw photo.
        
        Args:
            model: Trained corner detection model (Regression or Heatmap)
            input_image: Raw input image (H, W, C)
            original_size: Original image size for coordinate mapping
            approach: 'regression' or 'heatmap'
            
        Returns:
            Dictionary with corner coordinates and visualization
        """
        model.eval()
        model.to(self.device)
        
        input_size = (256, 256)  # Model input size
        
        # Preprocessing
        if input_image.dtype != np.float32:
            input_image = input_image.astype(np.float32) / 255.0
        
        # Resize for model
        resized = cv2.resize(input_image, input_size, interpolation=cv2.INTER_LINEAR)
        tensor_input = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        tensor_input = tensor_input.to(self.device)
        
        # Inference
        with torch.no_grad():
            if approach == 'heatmap':
                coords, heatmaps = model(tensor_input)
            else:
                coords = model(tensor_input)
                heatmaps = None
        
        # Convert to numpy
        corners_normalized = coords.cpu().squeeze(0).numpy()
        
        # Reshape to (4, 2)
        if corners_normalized.shape[0] == 8:
            corners = corners_normalized.reshape(4, 2)
        else:
            corners = corners_normalized
        
        # Map back to original image coordinates
        if original_size is None:
            original_size = (input_image.shape[0], input_image.shape[1])
        
        h_orig, w_orig = original_size
        corners_original = corners * np.array([w_orig - 1, h_orig - 1])
        
        # Create visualization
        vis_image = (input_image * 255).astype(np.uint8) if input_image.max() <= 1.0 else input_image.astype(np.uint8)
        vis_image = cv2.resize(vis_image, (w_orig, h_orig))
        
        # Draw corners
        corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        corner_names = ['TL', 'TR', 'BR', 'BL']
        
        for i, (corner, color, name) in enumerate(zip(corners_original, corner_colors, corner_names)):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(vis_image, (x, y), 8, color, -1)
            cv2.putText(vis_image, name, (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw quadrilateral
        pts = corners_original.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_image, [pts], True, (0, 255, 255), 2)
        
        return {
            'corners_normalized': corners,
            'corners_original': corners_original,
            'visualization': vis_image,
            'heatmaps': heatmaps.cpu().numpy() if heatmaps is not None else None
        }


def generate_gaussian_heatmaps(corners: np.ndarray, image_size: Tuple[int, int],
                                sigma: float = 5.0) -> np.ndarray:
    """
    Generate Gaussian heatmaps from corner coordinates for training Approach B.
    
    Args:
        corners: Corner coordinates (4, 2) in normalized [0, 1] range
        image_size: Output heatmap size (H, W)
        sigma: Standard deviation of Gaussian
        
    Returns:
        Heatmaps tensor (4, H, W)
    """
    h, w = image_size
    heatmaps = np.zeros((4, h, w), dtype=np.float32)
    
    # Convert to pixel coordinates
    corners_pixels = corners * np.array([w - 1, h - 1])
    
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    for i, (cx, cy) in enumerate(corners_pixels):
        # 2D Gaussian
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
        heatmaps[i] = heatmap
    
    return heatmaps


def compare_loss_functions(model: torch.nn.Module,
                           test_loader: torch.utils.data.DataLoader,
                           loss_functions: Dict[str, torch.nn.Module],
                           device: str = 'cpu') -> Dict[str, Dict]:
    """
    Compare different loss function combinations for enhancement network.
    
    Args:
        model: Trained enhancement model
        test_loader: DataLoader with test data
        loss_functions: Dictionary of named loss functions
        device: Computing device
        
    Returns:
        Comparison results for each loss function
    """
    model.eval()
    model.to(device)
    
    results = {}
    
    for loss_name, loss_fn in loss_functions.items():
        total_loss = 0.0
        psnr_values = []
        ssim_values = []
        
        evaluator = EnhancementEvaluator(device)
        
        for batch in test_loader:
            degraded = batch['degraded'].to(device)
            target = batch['target'].to(device)
            
            with torch.no_grad():
                output = model(degraded)
            
            loss = loss_fn(output, target)
            total_loss += loss.item()
            
            # Compute metrics
            metrics = evaluator.evaluate_batch(output, target)
            psnr_values.append(metrics['psnr_mean'])
            ssim_values.append(metrics['ssim_mean'])
        
        results[loss_name] = {
            'mean_loss': total_loss / len(test_loader),
            'mean_psnr': np.mean(psnr_values),
            'mean_ssim': np.mean(ssim_values)
        }
    
    return results
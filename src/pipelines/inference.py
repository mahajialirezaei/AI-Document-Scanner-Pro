"""
Inference pipelines for document scanning and enhancement.

This module provides end-to-end inference capabilities for:
1. Document Enhancement: Rectified document → Enhanced scan
2. Corner Detection: Raw photo → Corner coordinates overlay
3. Full Pipeline: Raw photo → Corners → Warp → Enhance → Final scan
"""

import cv2
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path

from src.models.model import EnhancementUNet, CornerRegressionModel, CornerHeatmapModel
from src.data.degradation import create_degradation_pipeline


def load_model(
    model_type: str,
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dropout_rate: float = 0.0
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_type: One of 'enhancement', 'corner_regression', 'corner_heatmap'
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model on ('cuda' or 'cpu')
        dropout_rate: Dropout rate for the model
        
    Returns:
        Loaded model in eval mode
    """
    if model_type == "enhancement":
        model = EnhancementUNet(dropout_rate=dropout_rate)
    elif model_type == "corner_regression":
        model = CornerRegressionModel(dropout_rate=dropout_rate)
    elif model_type == "corner_heatmap":
        model = CornerHeatmapModel(dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(
    image: np.ndarray,
    input_size: int = 256
) -> Tuple[torch.Tensor, Dict]:
    """
    Preprocess image for model inference.
    
    Args:
        image: Input image (H, W, C) in BGR format
        input_size: Target size for resizing
        
    Returns:
        Preprocessed tensor (1, C, H, W) and metadata dict
    """
    original_shape = image.shape[:2]
    
    # Resize to input size
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB and normalize to [0, 1]
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
    elif resized.shape[2] == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] and convert to tensor
    normalized = resized.astype(np.float32) / 255.0
    
    # Convert to CHW format and add batch dimension
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    
    metadata = {
        'original_shape': original_shape,
        'input_size': input_size,
        'scale_h': original_shape[0] / input_size,
        'scale_w': original_shape[1] / input_size
    }
    
    return tensor, metadata


def enhance_document(
    model: torch.nn.Module,
    rectified_image: np.ndarray,
    device: str = None,
    image_size: int = 1024
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Enhance a rectified document image using the enhancement network.
    
    Args:
        model: Trained EnhancementUNet model
        rectified_image: Rectified document image (already cropped and warped)
        device: Device for inference
        
    Returns:
        Enhanced image (uint8, same shape as input) and output tensor
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Preprocess
    input_tensor, metadata = preprocess_image(rectified_image, input_size=image_size)
    input_tensor = input_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Post-process: clamp to [0, 1] and convert to uint8
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_np = (output_np * 255).astype(np.uint8)
    
    # Convert RGB back to BGR for OpenCV
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    # Resize back to original dimensions if needed
    if output_bgr.shape[:2] != rectified_image.shape[:2]:
        output_bgr = cv2.resize(output_bgr, (rectified_image.shape[1], rectified_image.shape[0]))
    
    return output_bgr, output_tensor


def detect_corners_regression(
    model: torch.nn.Module,
    raw_image: np.ndarray,
    confidence_threshold: float = 0.5,
    device: str = None
) -> Tuple[np.ndarray, float]:
    """
    Detect document corners using regression approach (Approach A).
    
    Args:
        model: Trained CornerRegressionModel
        raw_image: Raw input photo
        confidence_threshold: Threshold for corner validity
        device: Device for inference
        
    Returns:
        Corner coordinates (4, 2) in original image scale and confidence score
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Preprocess
    input_tensor, metadata = preprocess_image(raw_image)
    input_tensor = input_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Output is 8 normalized coordinates (x1, y1, x2, y2, x3, y3, x4, x4)
    corners_norm = output.squeeze(0).cpu().numpy()
    
    # Reshape to (4, 2) - format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    corners = corners_norm.reshape(4, 2)
    
    # Scale back to original image dimensions
    corners[:, 0] *= metadata['original_shape'][1]  # x * width
    corners[:, 1] *= metadata['original_shape'][0]  # y * height
    
    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = order_corners(corners)
    
    # Simple confidence based on coordinate validity (all within image bounds)
    h, w = metadata['original_shape']
    valid = ((corners[:, 0] >= 0) & (corners[:, 0] <= w) & 
             (corners[:, 1] >= 0) & (corners[:, 1] <= h))
    confidence = float(valid.all())
    
    return corners, confidence


def detect_corners_heatmap(
    model: torch.nn.Module,
    raw_image: np.ndarray,
    heatmap_threshold: float = 0.3,
    device: str = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Detect document corners using heatmap approach (Approach B).
    
    Args:
        model: Trained CornerHeatmapModel
        raw_image: Raw input photo
        heatmap_threshold: Threshold for heatmap peak detection
        device: Device for inference
        
    Returns:
        Corner coordinates (4, 2) in original image scale and heatmaps
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Preprocess
    input_tensor, metadata = preprocess_image(raw_image)
    input_tensor = input_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        heatmaps = model(input_tensor)
    
    # Heatmaps shape: (1, 4, H, W) - 4 channels for 4 corners
    heatmaps_np = heatmaps.squeeze(0).cpu().numpy()
    
    # Extract corner coordinates from heatmaps using argmax
    corners = []
    for i in range(4):  # 4 corners
        heatmap = heatmaps_np[i]
        
        # Find peak location
        peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y, x = peak_idx
        
        # Get heatmap dimensions
        h_map, w_map = heatmap.shape
        
        # Normalize coordinates
        x_norm = x / w_map
        y_norm = y / h_map
        
        corners.append([x_norm, y_norm])
    
    corners = np.array(corners)
    
    # Scale back to original image dimensions
    corners[:, 0] *= metadata['original_shape'][1]
    corners[:, 1] *= metadata['original_shape'][0]
    
    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = order_corners(corners)
    
    return corners, heatmaps_np


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners consistently: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        corners: Unordered corner coordinates (4, 2)
        
    Returns:
        Ordered corners (4, 2)
    """
    # Sort by y-coordinate to separate top and bottom
    y_sorted = corners[np.argsort(corners[:, 1])]
    
    # Top two corners (smaller y values)
    top_two = y_sorted[:2]
    # Bottom two corners (larger y values)
    bottom_two = y_sorted[2:]
    
    # Sort top two by x-coordinate (left to right)
    top_left = top_two[np.argmin(top_two[:, 0])]
    top_right = top_two[np.argmax(top_two[:, 0])]
    
    # Sort bottom two by x-coordinate (left to right)
    bottom_left = bottom_two[np.argmin(bottom_two[:, 0])]
    bottom_right = bottom_two[np.argmax(bottom_two[:, 0])]
    
    ordered = np.array([top_left, top_right, bottom_right, bottom_left])
    return ordered


def apply_perspective_transform(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: Tuple[int, int] = None
) -> np.ndarray:
    """
    Apply perspective transform to rectify document based on detected corners.
    
    Args:
        image: Input image with detected corners
        corners: Ordered corners (TL, TR, BR, BL) in format [[x,y], ...]
        output_size: Desired output size (width, height). If None, estimated from corners.
        
    Returns:
        Rectified document image
    """
    # Order corners: TL, TR, BR, BL
    corners = order_corners(corners)
    
    # Calculate output dimensions if not provided
    if output_size is None:
        # Width: max of top and bottom edge lengths
        top_width = np.linalg.norm(corners[0] - corners[1])
        bottom_width = np.linalg.norm(corners[3] - corners[2])
        max_width = int(max(top_width, bottom_width))
        
        # Height: max of left and right edge lengths
        left_height = np.linalg.norm(corners[0] - corners[3])
        right_height = np.linalg.norm(corners[1] - corners[2])
        max_height = int(max(left_height, right_height))
        
        output_size = (max_width, max_height)
    
    # Define destination points
    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)
    
    # Compute homography matrix
    H, _ = cv2.findHomography(corners.astype(np.float32), dst)
    
    # Apply perspective transform
    rectified = cv2.warpPerspective(image, H, output_size)
    
    return rectified


def draw_corners_on_image(
    image: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    circle_radius: int = 8
) -> np.ndarray:
    """
    Draw detected corners on image for visualization.
    
    Args:
        image: Input image
        corners: Corner coordinates (4, 2)
        color: Line color in BGR format
        thickness: Line thickness
        circle_radius: Radius of circles at corner points
        
    Returns:
        Image with corners drawn
    """
    output = image.copy()
    
    # Draw lines connecting corners (quadrilateral)
    for i in range(4):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i + 1) % 4].astype(int))
        cv2.line(output, pt1, pt2, color, thickness)
    
    # Draw circles at each corner
    labels = ['TL', 'TR', 'BR', 'BL']
    for i, corner in enumerate(corners):
        pt = tuple(corner.astype(int))
        cv2.circle(output, pt, circle_radius, color, -1)
        
        # Add label
        cv2.putText(output, labels[i], (pt[0] + 10, pt[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return output


class DocumentScanningPipeline:
    """
    End-to-end document scanning pipeline combining corner detection,
    perspective warping, and document enhancement.
    """
    
    def __init__(
        self,
        corner_model_path: str,
        enhancement_model_path: str,
        corner_approach: str = 'heatmap',
        dropout_rate: float = 0.0,
        device: str = None
    ):
        """
        Initialize the full pipeline.
        
        Args:
            corner_model_path: Path to corner detection model checkpoint
            enhancement_model_path: Path to enhancement model checkpoint
            corner_approach: 'regression' or 'heatmap'
            dropout_rate: Dropout rate for models
            device: Device for inference
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.corner_approach = corner_approach
        
        # Load corner detection model
        if corner_approach == 'regression':
            self.corner_model = load_model(
                'corner_regression', corner_model_path, device, dropout_rate
            )
        else:
            self.corner_model = load_model(
                'corner_heatmap', corner_model_path, device, dropout_rate
            )
        
        # Load enhancement model
        self.enhancement_model = load_model(
            'enhancement', enhancement_model_path, device, dropout_rate
        )
    
    def process(
        self,
        raw_image: np.ndarray,
        return_intermediate: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Process a raw document photo through the complete pipeline.
        
        Args:
            raw_image: Raw input photo (BGR format)
            return_intermediate: If True, return all intermediate results
            
        Returns:
            Final enhanced scan, or dict with all intermediate results
        """
        results = {}
        
        # Step 1: Detect corners
        if self.corner_approach == 'regression':
            corners, confidence = detect_corners_regression(
                self.corner_model, raw_image, device=self.device
            )
        else:
            corners, heatmaps = detect_corners_heatmap(
                self.corner_model, raw_image, device=self.device
            )
            results['heatmaps'] = heatmaps
        
        results['corners'] = corners
        results['corners_image'] = draw_corners_on_image(raw_image, corners)
        
        # Step 2: Apply perspective transform
        rectified = apply_perspective_transform(raw_image, corners)
        results['rectified'] = rectified
        
        # Step 3: Enhance document
        enhanced, _ = enhance_document(
            self.enhancement_model, rectified, device=self.device, image_size=1024
        )
        results['enhanced'] = enhanced
        
        if return_intermediate:
            return results
        else:
            return enhanced
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Process multiple images.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Optional directory to save results
            
        Returns:
            List of result dictionaries
        """
        results_list = []
        
        for img_path in image_paths:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            # Process
            results = self.process(image, return_intermediate=True)
            results['input_path'] = str(img_path)
            results_list.append(results)
            
            # Save if output directory specified
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                stem = Path(img_path).stem
                
                cv2.imwrite(str(output_dir / f"{stem}_corners.png"), results['corners_image'])
                cv2.imwrite(str(output_dir / f"{stem}_rectified.png"), results['rectified'])
                cv2.imwrite(str(output_dir / f"{stem}_enhanced.png"), results['enhanced'])
        
        return results_list


# Convenience functions for quick inference
def run_enhancement_inference(
    model_path: str,
    input_image_path: str,
    output_image_path: str,
    device: str = None
) -> None:
    """
    Run enhancement inference on a single rectified document image.
    
    Args:
        model_path: Path to enhancement model checkpoint
        input_image_path: Path to input rectified document
        output_image_path: Path to save enhanced output
        device: Device for inference
    """
    model = load_model('enhancement', model_path, device)
    image = cv2.imread(input_image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image: {input_image_path}")
    
    enhanced, _ = enhance_document(model, image, device)
    cv2.imwrite(output_image_path, enhanced)
    print(f"Enhanced image saved to: {output_image_path}")


def run_corner_detection_inference(
    model_path: str,
    input_image_path: str,
    output_image_path: str,
    approach: str = 'heatmap',
    device: str = None
) -> None:
    """
    Run corner detection inference on a raw document photo.
    
    Args:
        model_path: Path to corner detection model checkpoint
        input_image_path: Path to input raw photo
        output_image_path: Path to save image with corners drawn
        approach: 'regression' or 'heatmap'
        device: Device for inference
    """
    if approach == 'regression':
        model = load_model('corner_regression', model_path, device)
    else:
        model = load_model('corner_heatmap', model_path, device)
    
    image = cv2.imread(input_image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image: {input_image_path}")
    
    if approach == 'regression':
        corners, _ = detect_corners_regression(model, image, device)
    else:
        corners, _ = detect_corners_heatmap(model, image, device)
    
    output_image = draw_corners_on_image(image, corners)
    cv2.imwrite(output_image_path, output_image)
    print(f"Corner detection result saved to: {output_image_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Scanning Inference Pipeline")
    parser.add_argument("--mode", choices=["enhance", "corners", "full"], required=True,
                       help="Inference mode")
    parser.add_argument("--corner-model", type=str, help="Path to corner detection model")
    parser.add_argument("--enhance-model", type=str, help="Path to enhancement model")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--approach", choices=["regression", "heatmap"], default="heatmap",
                       help="Corner detection approach")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.mode == "enhance":
        run_enhancement_inference(args.enhance_model, args.input, args.output, args.device)
    
    elif args.mode == "corners":
        run_corner_detection_inference(
            args.corner_model, args.input, args.output, args.approach, args.device
        )
    
    elif args.mode == "full":
        if not args.corner_model or not args.enhance_model:
            raise ValueError("Full pipeline requires both --corner-model and --enhance-model")
        
        pipeline = DocumentScanningPipeline(
            args.corner_model, args.enhance_model, args.approach, device=args.device
        )
        
        image = cv2.imread(args.input)
        results = pipeline.process(image, return_intermediate=True)
        
        # Save all outputs
        cv2.imwrite(args.output.replace(".png", "_corners.png"), results['corners_image'])
        cv2.imwrite(args.output.replace(".png", "_rectified.png"), results['rectified'])
        cv2.imwrite(args.output, results['enhanced'])
        
        print(f"Full pipeline results saved to: {args.output}")
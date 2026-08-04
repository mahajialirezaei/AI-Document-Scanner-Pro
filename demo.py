#!/usr/bin/env python3
"""
Demo script for CNN Document Scanning & Enhancement System.

This script provides an easy-to-use interface for running the complete
document scanning pipeline on single images or batches.

Usage:
    python demo.py -i input.jpg -o output/ --visualize
    python demo.py -i images/ -o results/ --batch
    python demo.py --help
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipelines.inference import (
    load_model,
    preprocess_image,
    detect_corners_regression,
    detect_corners_heatmap,
    enhance_document,
    order_corners,
    apply_perspective_transform,
    draw_corners_on_image,
    DocumentScanningPipeline,
)
from src.models.model import EnhancementUNet, CornerRegressionModel, CornerHeatmapModel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CNN Document Scanning & Enhancement Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image with default models
  python demo.py -i document.jpg -o results/
  
  # Process with custom model paths
  python demo.py -i photo.jpg --enhancement-model my_unet.pth --corner-model my_corners.pth -o output/
  
  # Enable visualization
  python demo.py -i doc.jpg -o results/ --visualize
  
  # Batch process a directory
  python demo.py -i images/ -o scanned/ --batch
        """,
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input image path or directory for batch processing",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--enhancement-model",
        type=str,
        default=None,
        help="Path to enhancement model checkpoint (default: checkpoints/enhancement/best.pth)",
    )
    parser.add_argument(
        "--corner-model",
        type=str,
        default=None,
        help="Path to corner detection model checkpoint (default: checkpoints/corner_heat/best.pth)",
    )
    parser.add_argument(
        "--corner-approach",
        type=str,
        choices=["regression", "heatmap"],
        default="heatmap",
        help="Corner detection approach (default: heatmap)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate side-by-side comparison visualization",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch processing mode for directories",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cuda', 'cpu', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Input image size for processing (default: 512)",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results (corners, warped image)",
    )
    
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Determine the device to use for inference."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def find_default_model(model_type: str) -> Optional[str]:
    """Find default model checkpoint path."""
    default_paths = {
        "enhancement": [
            "checkpoints/enhancement/best.pth",
            "checkpoints/enhancement/latest.pth",
        ],
        "corner_heat": [
            "checkpoints/corner_heat/best.pth",
            "checkpoints/corner_heat/latest.pth",
        ],
        "corner_reg": [
            "checkpoints/corner_reg/best.pth",
            "checkpoints/corner_reg/latest.pth",
        ],
    }
    
    for path in default_paths.get(model_type, []):
        if os.path.exists(path):
            return path
    
    return None


def load_enhancement_model(
    model_path: Optional[str],
    device: torch.device,
    image_size: int = 512,
) -> torch.nn.Module:
    """Load enhancement model from checkpoint or default location."""
    if model_path is None:
        model_path = find_default_model("enhancement")
        if model_path is None:
            raise FileNotFoundError(
                "Enhancement model not found. Please provide --enhancement-model path "
                "or place model at checkpoints/enhancement/best.pth"
            )
    
    print(f"Loading enhancement model from: {model_path}")
    model = load_model(
        "enhancement",
        model_path,
        device=str(device),
    )
    model.eval()
    return model


def load_corner_model(
    model_path: Optional[str],
    approach: str,
    device: torch.device,
    image_size: int = 512,
) -> torch.nn.Module:
    """Load corner detection model from checkpoint or default location."""
    if model_path is None:
        model_type = "corner_heat" if approach == "heatmap" else "corner_reg"
        model_path = find_default_model(model_type)
        if model_path is None:
            raise FileNotFoundError(
                f"Corner model not found. Please provide --corner-model path "
                f"or place model at checkpoints/{model_type}/best.pth"
            )
    
    print(f"Loading corner model ({approach}) from: {model_path}")
    
    model_type = "corner_heatmap" if approach == "heatmap" else "corner_regression"
    model = load_model(
        model_type,
        model_path,
        device=str(device),
    )
    model.eval()
    return model


def process_single_image(
    image_path: str,
    enhancement_model: torch.nn.Module,
    corner_model: torch.nn.Module,
    corner_approach: str,
    output_dir: str,
    device: torch.device,
    image_size: int = 512,
    visualize: bool = False,
    save_intermediate: bool = False,
) -> dict:
    """
    Process a single image through the complete pipeline.
    
    Returns:
        Dictionary with paths to output files
    """
    # Load image - Keep in BGR format for all processing steps
    print(f"Processing: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Create output filename
    base_name = Path(image_path).stem
    output_paths = {}
    
    # Step 1: Detect corners (Pass corner_model FIRST, then image_bgr)
    print("  Step 1: Detecting corners...")
    if corner_approach == "heatmap":
        corners_px, heatmaps_np = detect_corners_heatmap(
            corner_model,
            image_bgr,
            device=str(device),
        )
        
        # --- بخش جدید: ذخیره نقشه‌های حرارتی خام ---
        if heatmaps_np is not None:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            titles = ['Channel 0', 'Channel 1', 'Channel 2', 'Channel 3']
            for i, ax in enumerate(axes.flat):
                if i < len(heatmaps_np):
                    # استفاده از colormap حرارتی برای دید بهتر
                    ax.imshow(heatmaps_np[i], cmap='jet')
                    ax.set_title(f"Heatmap {titles[i]}")
                ax.axis('off')
            plt.tight_layout()
            heatmap_viz_path = os.path.join(output_dir, f"{base_name}_raw_heatmaps.png")
            plt.savefig(heatmap_viz_path, dpi=150)
            plt.close()
            output_paths["heatmaps"] = heatmap_viz_path
            print(f"  Saved raw heatmaps: {heatmap_viz_path}")
        # --------------------------------------------

    else:
        corners_px, _ = detect_corners_regression(
            corner_model,
            image_bgr,
            device=str(device),
        )
    
    corners_ordered = order_corners(corners_px)
    print(f"  Corners detected: {corners_ordered.tolist()}")
    
    # Save corner visualization if requested
    if save_intermediate or visualize:
        image_rgb_for_viz = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        corner_viz = draw_corners_on_image(image_rgb_for_viz, corners_ordered)
        corner_path = os.path.join(output_dir, f"{base_name}_corners.png")
        cv2.imwrite(corner_path, cv2.cvtColor(corner_viz, cv2.COLOR_RGB2BGR))
        output_paths["corners"] = corner_path
        print(f"  Saved corner visualization: {corner_path}")
    
    # Step 2: Apply perspective transform
    print("  Step 2: Applying perspective transform...")
    warped_bgr = apply_perspective_transform(
        image_bgr,
        corners_ordered,
        output_size=(image_size, image_size),
    )
    
    if save_intermediate:
        warped_path = os.path.join(output_dir, f"{base_name}_warped.png")
        cv2.imwrite(warped_path, warped_bgr)
        output_paths["warped"] = warped_path
        print(f"  Saved warped image: {warped_path}")
    
    # Step 3: Enhance document
    print("  Step 3: Enhancing document...")
    enhanced_bgr, _ = enhance_document(
        enhancement_model,
        warped_bgr,
        device=str(device),
        image_size=image_size,
    )
    
    # Save enhanced result
    enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
    cv2.imwrite(enhanced_path, enhanced_bgr)
    output_paths["enhanced"] = enhanced_path
    print(f"  Saved enhanced image: {enhanced_path}")
    
    # Step 4: Create visualization if requested
    if visualize:
        print("  Step 4: Creating visualization...")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_RGB2BGR)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original with corners
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Photo")
        axes[0].axis('off')
        
        # Warped
        axes[1].imshow(warped_rgb)
        axes[1].set_title("Perspective Corrected")
        axes[1].axis('off')
        
        # Enhanced
        axes[2].imshow(enhanced_rgb)
        axes[2].set_title("Enhanced Scan")
        axes[2].axis('off')
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_paths["comparison"] = viz_path
        print(f"  Saved comparison: {viz_path}")
    
    return output_paths


def batch_process(
    input_dir: str,
    enhancement_model: torch.nn.Module,
    corner_model: torch.nn.Module,
    corner_approach: str,
    output_dir: str,
    device: torch.device,
    image_size: int = 512,
    visualize: bool = False,
    save_intermediate: bool = False,
) -> List[dict]:
    """Process all images in a directory."""
    input_path = Path(input_dir)
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in supported_extensions
    ]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return []
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    for idx, img_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing {img_file.name}")
        try:
            result = process_single_image(
                image_path=str(img_file),
                enhancement_model=enhancement_model,
                corner_model=corner_model,
                corner_approach=corner_approach,
                output_dir=output_dir,
                device=device,
                image_size=image_size,
                visualize=visualize,
                save_intermediate=save_intermediate,
            )
            result["input"] = str(img_file)
            results.append(result)
        except Exception as e:
            print(f"  Error processing {img_file.name}: {str(e)}")
            continue
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Check if batch mode
    is_batch = args.batch or Path(args.input).is_dir()
    
    if is_batch:
        print("Mode: Batch processing")
    else:
        print("Mode: Single image")
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)
    
    # Load models
    print("\nLoading models...")
    try:
        enhancement_model = load_enhancement_model(
            model_path=args.enhancement_model,
            device=device,
            image_size=args.image_size,
        )
        
        corner_model = load_corner_model(
            model_path=args.corner_model,
            approach=args.corner_approach,
            device=device,
            image_size=args.image_size,
        )
        print("Models loaded successfully!\n")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo fix this, either:")
        print("  1. Train models and place them in checkpoints/")
        print("  2. Provide model paths with --enhancement-model and --corner-model")
        sys.exit(1)
    
    # Process images
    print("=" * 60)
    if is_batch:
        results = batch_process(
            input_dir=args.input,
            enhancement_model=enhancement_model,
            corner_model=corner_model,
            corner_approach=args.corner_approach,
            output_dir=args.output,
            device=device,
            image_size=args.image_size,
            visualize=args.visualize,
            save_intermediate=args.save_intermediate,
        )
        
        print("\n" + "=" * 60)
        print(f"Batch processing complete!")
        print(f"Successfully processed: {len(results)} images")
    else:
        result = process_single_image(
            image_path=args.input,
            enhancement_model=enhancement_model,
            corner_model=corner_model,
            corner_approach=args.corner_approach,
            output_dir=args.output,
            device=device,
            image_size=args.image_size,
            visualize=args.visualize,
            save_intermediate=args.save_intermediate,
        )
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        print(f"Enhanced image saved to: {result['enhanced']}")
        if args.visualize and 'comparison' in result:
            print(f"Comparison saved to: {result['comparison']}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
OCR Evaluation Script for Document Enhancement.

This script evaluates the readability of real photos using Tesseract OCR
to compare three states:
1. The raw rectified crop (Baseline)
2. The model's enhanced output
3. A commercial reference scan

Usage:
    python run_ocr_eval.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import torch
from tabulate import tabulate

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.model import EnhancementUNet
from src.data.dataset import RealEvaluationDataset
from src.evaluation.evaluate import OCREvaluator


def load_enhancement_model(
    checkpoint_path: str,
    device: str = None
) -> torch.nn.Module:
    """Load enhancement model from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = EnhancementUNet(dropout_rate=0.0)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW float tensor [0,1] to HWC uint8 array [0,255]."""
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to numpy (CHW -> HWC)
    np_array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Scale to [0, 255] and convert to uint8
    return (np_array * 255).astype(np.uint8)


def evaluate_dataset(
    dataset: RealEvaluationDataset,
    enhancement_model: torch.nn.Module,
    ocr_evaluator: OCREvaluator,
    scanned_photos_dir: str,
    device: str = None
) -> Dict[str, List[Dict]]:
    """
    Evaluate entire dataset with OCR metrics.
    
    Returns:
        Dictionary with OCR results for baseline, model output, and reference
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    import cv2
    
    results = {
        'baseline': [],      # Raw rectified crop
        'enhanced': [],      # Model output
        'reference': []      # Commercial scan
    }
    
    print(f"Evaluating {len(dataset)} samples...\n")
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        filename = sample['filename']
        
        print(f"[{idx + 1}/{len(dataset)}] Processing: {filename}")
        
        # Get rectified input and pass through enhancement model
        rectified_input = sample['rectified_input'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            enhanced_output_tensor = enhancement_model(rectified_input)
        
        # Convert tensors to uint8 arrays for OCR
        # Baseline: raw rectified crop
        baseline_img = tensor_to_uint8(sample['rectified_input'].unsqueeze(0))
        
        # Enhanced: model output
        enhanced_img = tensor_to_uint8(enhanced_output_tensor)
        
        # Reference: load commercial scan
        ref_filename = filename  # Assuming same filename
        ref_path = os.path.join(scanned_photos_dir, ref_filename)
        
        if os.path.exists(ref_path):
            ref_img_bgr = cv2.imread(ref_path)
            if ref_img_bgr is not None:
                ref_img_rgb = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
                ref_img = ref_img_rgb
            else:
                print(f"  Warning: Could not load reference image: {ref_path}")
                ref_img = None
        else:
            print(f"  Warning: Reference image not found: {ref_path}")
            ref_img = None
        
        # Run OCR evaluation
        baseline_result = ocr_evaluator.extract_text_with_confidence(baseline_img)
        enhanced_result = ocr_evaluator.extract_text_with_confidence(enhanced_img)
        
        results['baseline'].append({
            'filename': filename,
            'confidence': baseline_result['confidence'],
            'word_count': baseline_result['word_count'],
            'text': baseline_result['text'][:100]  # First 100 chars for logging
        })
        
        results['enhanced'].append({
            'filename': filename,
            'confidence': enhanced_result['confidence'],
            'word_count': enhanced_result['word_count'],
            'text': enhanced_result['text'][:100]
        })
        
        if ref_img is not None:
            ref_result = ocr_evaluator.extract_text_with_confidence(ref_img)
            results['reference'].append({
                'filename': filename,
                'confidence': ref_result['confidence'],
                'word_count': ref_result['word_count'],
                'text': ref_result['text'][:100]
            })
        else:
            results['reference'].append({
                'filename': filename,
                'confidence': 0.0,
                'word_count': 0,
                'text': ''
            })
        
        print(f"  Baseline: conf={baseline_result['confidence']:.2f}, words={baseline_result['word_count']}")
        print(f"  Enhanced: conf={enhanced_result['confidence']:.2f}, words={enhanced_result['word_count']}")
        if ref_img is not None:
            print(f"  Reference: conf={ref_result['confidence']:.2f}, words={ref_result['word_count']}")
        print()
    
    return results


def print_summary_table(results: Dict[str, List[Dict]]):
    """Print formatted summary table of OCR evaluation results."""
    print("\n" + "=" * 80)
    print("OCR EVALUATION SUMMARY")
    print("=" * 80)
    
    # Calculate averages
    baseline_avg_conf = np.mean([r['confidence'] for r in results['baseline']])
    baseline_total_words = sum([r['word_count'] for r in results['baseline']])
    
    enhanced_avg_conf = np.mean([r['confidence'] for r in results['enhanced']])
    enhanced_total_words = sum([r['word_count'] for r in results['enhanced']])
    
    # Filter out invalid references
    valid_refs = [r for r in results['reference'] if r['confidence'] > 0 or r['word_count'] > 0]
    if valid_refs:
        ref_avg_conf = np.mean([r['confidence'] for r in results['reference']])
        ref_total_words = sum([r['word_count'] for r in results['reference']])
    else:
        ref_avg_conf = 0.0
        ref_total_words = 0
    
    # Create table
    table_data = [
        ["Baseline (Raw Rectified)", f"{baseline_avg_conf:.2f}%", baseline_total_words],
        ["Model Output (Enhanced)", f"{enhanced_avg_conf:.2f}%", enhanced_total_words],
        ["Commercial App (Reference)", f"{ref_avg_conf:.2f}%", ref_total_words]
    ]
    
    print("\nAverage OCR Confidence and Total Word Count:\n")
    print(tabulate(
        table_data,
        headers=["Method", "Avg Confidence", "Total Words"],
        tablefmt="grid"
    ))
    
    # Calculate improvement
    conf_improvement = enhanced_avg_conf - baseline_avg_conf
    words_improvement = enhanced_total_words - baseline_total_words
    
    print(f"\nImprovement (Enhanced vs Baseline):")
    print(f"  Confidence: {conf_improvement:+.2f}% ({'+' if conf_improvement > 0 else ''}{(conf_improvement/baseline_avg_conf*100) if baseline_avg_conf > 0 else 0:.1f}%)")
    print(f"  Word Count: {words_improvement:+d}")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    # Configuration
    checkpoint_path = "checkpoints/enhancement/best_model.pth"
    real_photos_dir = "data/raw/real_photos"
    annotation_file = "data/annotations/_annotations.coco.json"
    scanned_photos_dir = "data/raw/real_photos_scanned"
    image_size = (512, 512)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Enhancement model checkpoint not found at: {checkpoint_path}")
        print("Please train the model first or update the checkpoint path.")
        sys.exit(1)
    
    # Check if data directories exist
    if not os.path.exists(real_photos_dir):
        print(f"Error: Real photos directory not found at: {real_photos_dir}")
        sys.exit(1)
    
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found at: {annotation_file}")
        sys.exit(1)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load enhancement model
    print(f"Loading enhancement model from: {checkpoint_path}")
    enhancement_model = load_enhancement_model(checkpoint_path, device)
    print("Model loaded successfully!\n")
    
    # Initialize OCR evaluator
    print("Initializing OCR evaluator (Tesseract)...")
    ocr_evaluator = OCREvaluator(lang='eng')
    if not ocr_evaluator.available:
        print("Warning: Tesseract OCR is not available. Install with: pip install pytesseract")
        print("Continuing without OCR evaluation...\n")
    
    # Load dataset
    print(f"Loading real evaluation dataset from: {real_photos_dir}")
    dataset = RealEvaluationDataset(
        real_photos_dir=real_photos_dir,
        annotation_file=annotation_file,
        image_size=image_size
    )
    print(f"Dataset loaded: {len(dataset)} samples\n")
    
    # Run evaluation
    results = evaluate_dataset(
        dataset=dataset,
        enhancement_model=enhancement_model,
        ocr_evaluator=ocr_evaluator,
        scanned_photos_dir=scanned_photos_dir,
        device=device
    )
    
    # Print summary
    print_summary_table(results)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

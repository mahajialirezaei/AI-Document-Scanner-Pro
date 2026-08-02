#!/usr/bin/env python3
"""
OCR Evaluation Script for Document Enhancement.
This script evaluates the readability of real photos using Tesseract OCR to compare 4 states:
1. The raw rectified crop (Baseline)
2. The model's enhanced output (using Ground-Truth Annotated Corners)
3. The model's enhanced output (using Predicted Corners - End-to-End)
4. A commercial reference scan
"""
import os
import sys
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import torch
from tabulate import tabulate

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.model import EnhancementUNet
from src.data.dataset import RealEvaluationDataset
from src.evaluation.evaluate import OCREvaluator
from src.pipelines.inference import DocumentScanningPipeline

def load_enhancement_model(checkpoint_path: str, device: str = None) -> torch.nn.Module:
    """Load enhancement model from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model = EnhancementUNet(dropout_rate=0.0)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    return model

def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW float tensor [0,1] to HWC uint8 array [0,255]."""
    tensor = torch.clamp(tensor, 0, 1)
    np_array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (np_array * 255).astype(np.uint8)

def evaluate_dataset(
    dataset: RealEvaluationDataset,
    enhancement_model: torch.nn.Module,
    pipeline: DocumentScanningPipeline,
    ocr_evaluator: OCREvaluator,
    scanned_photos_dir: str,
    device: str = None
) -> Dict[str, List[Dict]]:
    """Evaluate entire dataset with OCR metrics for both GT and Predicted corners."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    results = {
        'baseline': [],
        'enhanced_gt': [],
        'enhanced_pred': [],
        'reference': []
    }
    
    print(f"Evaluating {len(dataset)} samples...\n")
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        filename = sample['filename']
        print(f"[{idx + 1}/{len(dataset)}] Processing: {filename}")
        
        # --- 1 & 2. Baseline and Enhanced (Ground Truth Corners) ---
        rectified_input = sample['rectified_input'].unsqueeze(0).to(device)
        with torch.no_grad():
            enhanced_gt_tensor = enhancement_model(rectified_input)
            
        baseline_img = tensor_to_uint8(sample['rectified_input'])
        enhanced_gt_img = tensor_to_uint8(enhanced_gt_tensor)
        
        # --- 3. Enhanced (Predicted Corners - End-to-End via Pipeline) ---
        raw_path = os.path.join(dataset.root_dir, filename)
        raw_bgr = cv2.imread(raw_path)
        
        if raw_bgr is not None:
            # Pipeline expects and returns BGR, we convert to RGB for Tesseract
            enhanced_pred_bgr = pipeline.process(raw_bgr, return_intermediate=False)
            enhanced_pred_img = cv2.cvtColor(enhanced_pred_bgr, cv2.COLOR_BGR2RGB)
        else:
            print(f"  Warning: Could not load raw image for E2E: {raw_path}")
            enhanced_pred_img = baseline_img # Fallback
        
        # --- 4. Reference (Commercial Scan) ---
        ref_path = os.path.join(scanned_photos_dir, filename)
        if os.path.exists(ref_path):
            ref_img_bgr = cv2.imread(ref_path)
            ref_img = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB) if ref_img_bgr is not None else None
        else:
            ref_img = None
            
        # --- Run OCR evaluation ---
        res_base = ocr_evaluator.extract_text_with_confidence(baseline_img)
        res_enh_gt = ocr_evaluator.extract_text_with_confidence(enhanced_gt_img)
        res_enh_pred = ocr_evaluator.extract_text_with_confidence(enhanced_pred_img)
        res_ref = ocr_evaluator.extract_text_with_confidence(ref_img) if ref_img is not None else {'confidence': 0.0, 'word_count': 0}
        
        results['baseline'].append({'filename': filename, 'confidence': res_base['confidence'], 'word_count': res_base['word_count']})
        results['enhanced_gt'].append({'filename': filename, 'confidence': res_enh_gt['confidence'], 'word_count': res_enh_gt['word_count']})
        results['enhanced_pred'].append({'filename': filename, 'confidence': res_enh_pred['confidence'], 'word_count': res_enh_pred['word_count']})
        results['reference'].append({'filename': filename, 'confidence': res_ref['confidence'], 'word_count': res_ref['word_count']})
        
        print(f"  Baseline                : conf={res_base['confidence']:.2f}, words={res_base['word_count']}")
        print(f"  Enhanced (GT Corners)   : conf={res_enh_gt['confidence']:.2f}, words={res_enh_gt['word_count']}")
        print(f"  Enhanced (Pred Corners) : conf={res_enh_pred['confidence']:.2f}, words={res_enh_pred['word_count']}")
        if ref_img is not None:
            print(f"  Reference               : conf={res_ref['confidence']:.2f}, words={res_ref['word_count']}")
        print()
        
    return results

def print_summary_table(results: Dict[str, List[Dict]]):
    """Print formatted summary table of OCR evaluation results."""
    print("\n" + "=" * 80)
    print("OCR EVALUATION SUMMARY")
    print("=" * 80)
    
    # Calculate averages
    base_conf = np.mean([r['confidence'] for r in results['baseline']])
    base_words = sum([r['word_count'] for r in results['baseline']])
    
    enh_gt_conf = np.mean([r['confidence'] for r in results['enhanced_gt']])
    enh_gt_words = sum([r['word_count'] for r in results['enhanced_gt']])
    
    enh_pred_conf = np.mean([r['confidence'] for r in results['enhanced_pred']])
    enh_pred_words = sum([r['word_count'] for r in results['enhanced_pred']])
    
    # Filter out invalid references
    valid_refs = [r for r in results['reference'] if r['confidence'] > 0 or r['word_count'] > 0]
    if valid_refs:
        ref_conf = np.mean([r['confidence'] for r in results['reference']])
        ref_words = sum([r['word_count'] for r in results['reference']])
    else:
        ref_conf = 0.0
        ref_words = 0
        
    # Create table
    table_data = [
        ["1. Baseline (Raw Rectified GT)", f"{base_conf:.2f}%", base_words],
        ["2. Enhanced (GT Corners)", f"{enh_gt_conf:.2f}%", enh_gt_words],
        ["3. Enhanced (Predicted Corners - E2E)", f"{enh_pred_conf:.2f}%", enh_pred_words],
        ["4. Commercial App (Reference)", f"{ref_conf:.2f}%", ref_words]
    ]
    
    print("\nAverage OCR Confidence and Total Word Count:\n")
    print(tabulate(table_data, headers=["Method", "Avg Confidence", "Total Words"], tablefmt="grid"))
    
    # Calculate corner error cost
    cost_conf = enh_gt_conf - enh_pred_conf
    cost_words = enh_gt_words - enh_pred_words
    
    print(f"\nAnalysis:")
    print(f"  Cost of Corner Errors on Enhancement: {cost_conf:.2f}% drop in OCR confidence.")
    print(f"  Cost of Corner Errors on Word Count : {cost_words} fewer words detected.")
    print("\n" + "=" * 80)

def main():
    """Main entry point."""
    # Configuration
    enhancement_ckpt = "checkpoints/enhancement/best_model.pth"
    corner_ckpt = "checkpoints/corner_heatmap/best_model.pth"
    real_photos_dir = "data/raw/real_photos"
    annotation_file = "data/raw/real_photos/_annotations.coco.json"
    scanned_photos_dir = "data/raw/real_photos_scanned"
    image_size = (512, 512)
    
    if not os.path.exists(enhancement_ckpt) or not os.path.exists(corner_ckpt):
        print("Error: Required model checkpoints not found in 'checkpoints/' directory.")
        sys.exit(1)
        
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    print("Loading models and initializing End-to-End Pipeline...")
    enhancement_model = load_enhancement_model(enhancement_ckpt, device)
    
    # Load E2E Pipeline for predicted corners evaluation
    pipeline = DocumentScanningPipeline(
        corner_model_path=corner_ckpt,
        enhancement_model_path=enhancement_ckpt,
        corner_approach="heatmap", # Default to best performing approach
        device=device
    )
    print("Models loaded successfully!\n")
    
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
    
    # Run evaluation
    results = evaluate_dataset(
        dataset=dataset,
        enhancement_model=enhancement_model,
        pipeline=pipeline,
        ocr_evaluator=ocr_evaluator,
        scanned_photos_dir=scanned_photos_dir,
        device=device
    )
    
    # Print summary
    print_summary_table(results)
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.model import CornerRegressionModel, CornerHeatmapModel
from src.data.dataset import RealEvaluationDataset
from src.data.data_splitter import get_synthetic_splits
from src.evaluation.evaluate import CornerDetectionEvaluator

def evaluate_model(model, dataloader, evaluator, device):
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        # For evaluation, we pass the raw_photo tensor
        images = batch['raw_photo'].to(device)
        targets = batch['corners'].to(device)
        
        with torch.no_grad():
            if isinstance(model, CornerHeatmapModel):
                # Heatmap returns (coords, heatmaps)
                preds, _ = model(images)
            else:
                preds = model(images)
                
        all_preds.append(preds)
        all_targets.append(targets)
        
    # Concatenate all batches
    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    
    # Run evaluation
    return evaluator.evaluate_batch(all_preds_tensor, all_targets_tensor)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Configs
    real_photos_dir = "data/raw/real_photos"
    annotation_file = "data/raw/real_photos/_annotations.coco.json"
    clean_dir = "data/clean_scans"
    bg_dir = "data/random_backgrounds"
    image_size = (256, 256)
    
    if not os.path.exists(real_photos_dir):
        print(f"Error: Directory {real_photos_dir} not found.")
        return

    # Load Real Dataset
    print("Loading Real Evaluation Dataset...")
    real_ds = RealEvaluationDataset(real_photos_dir, annotation_file, image_size=image_size)
    real_loader = DataLoader(real_ds, batch_size=8, shuffle=False)
    
    # Load Synthetic Test Dataset
    print("Loading Synthetic Evaluation Dataset...")
    _, _, test_ds = get_synthetic_splits(clean_dir, bg_dir, image_size=image_size, num_eval_samples=100)
    synth_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # Load Models
    print("\nLoading Models...")
    model_reg = CornerRegressionModel(dropout_rate=0.0).to(device)
    model_heat = CornerHeatmapModel(dropout_rate=0.0).to(device)
    
    try:
        model_reg.load_state_dict(torch.load("checkpoints/corner_reg/best_model.pth", map_location=device)['model_state_dict'])
        model_heat.load_state_dict(torch.load("checkpoints/corner_heat/best_model.pth", map_location=device)['model_state_dict'])
    except Exception as e:
        print(f"Could not load model checkpoints. Ensure they exist. Error: {e}")
        return
        
    model_reg.eval()
    model_heat.eval()
    
    # Initialize Evaluator (success threshold: 5 pixels)
    evaluator = CornerDetectionEvaluator(image_size=image_size, success_threshold=5.0)
    
    # Evaluate Real
    print("\nEvaluating Approach A: Direct Regression (Real Photos)...")
    reg_real_metrics = evaluate_model(model_reg, real_loader, evaluator, device)
    print("Evaluating Approach B: Heatmap (Real Photos)...")
    heat_real_metrics = evaluate_model(model_heat, real_loader, evaluator, device)
    
    # Evaluate Synthetic
    print("\nEvaluating Approach A: Direct Regression (Synthetic Test)...")
    reg_synth_metrics = evaluate_model(model_reg, synth_loader, evaluator, device)
    print("Evaluating Approach B: Heatmap (Synthetic Test)...")
    heat_synth_metrics = evaluate_model(model_heat, synth_loader, evaluator, device)

    # Print Comparison Table
    print("\n" + "="*80)
    print("CORNER DETECTION EVALUATION")
    print("="*80)
    print(f"{'Metric':<25} | {'Approach A (Regression)':<25} | {'Approach B (Heatmap)':<25}")
    print("-" * 80)
    print("--- REAL PHOTOS ---")
    print(f"{'Mean Error (Pixels)':<25} | {reg_real_metrics['mean_localization_error']:<25.2f} | {heat_real_metrics['mean_localization_error']:<25.2f}")
    print(f"{'Success Rate (<= 5px)':<25} | {reg_real_metrics['success_rate']*100:<24.1f}% | {heat_real_metrics['success_rate']*100:<24.1f}%")
    print("-" * 80)
    print("--- SYNTHETIC TEST SET ---")
    print(f"{'Mean Error (Pixels)':<25} | {reg_synth_metrics['mean_localization_error']:<25.2f} | {heat_synth_metrics['mean_localization_error']:<25.2f}")
    print(f"{'Success Rate (<= 5px)':<25} | {reg_synth_metrics['success_rate']*100:<24.1f}% | {heat_synth_metrics['success_rate']*100:<24.1f}%")
    print("="*80)

if __name__ == '__main__':
    main()
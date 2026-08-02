import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.model import EnhancementUNet
from src.evaluation.evaluate import EnhancementEvaluator
from src.data.data_splitter import get_synthetic_splits

def evaluate_bucket(model, dataloader, evaluator, device, bucket_name):
    total_psnr = 0.0
    total_ssim = 0.0
    baseline_psnr = 0.0
    baseline_ssim = 0.0
    num_batches = 0
    
    for batch in dataloader:
        degraded = batch['rectified_input'].to(device)
        clean = batch['clean_target'].to(device)
        
        with torch.no_grad():
            output = model(degraded)
            
        metrics = evaluator.evaluate_batch(output, clean, degraded)
        total_psnr += metrics['psnr_mean']
        total_ssim += metrics['ssim_mean']
        
        if 'baseline_psnr_mean' in metrics:
            baseline_psnr += metrics['baseline_psnr_mean']
            baseline_ssim += metrics['baseline_ssim_mean']
            
        num_batches += 1
        
    return {
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches,
        'base_psnr': baseline_psnr / num_batches if num_batches > 0 else 0,
        'base_ssim': baseline_ssim / num_batches if num_batches > 0 else 0
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data Splits
    clean_dir = "data/clean_scans"
    bg_dir = "data/random_backgrounds"
    
    print("Preparing Datasets...")
    train_ds, val_ds, test_ds = get_synthetic_splits(clean_dir, bg_dir, image_size=(256, 256), num_eval_samples=100)
    
    # Create DataLoaders (batch_size=1 for accurate average, or adjust as needed)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # 2. Load Trained Model
    model_path = "checkpoints/enhancement/best_model.pth"
    model = EnhancementUNet(dropout_rate=0.0).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Train the model first.")
        return

    model.eval()
    evaluator = EnhancementEvaluator(device=device)
    
    # 3. Evaluate on all three buckets
    print("\nRunning Evaluation on Synthetic Splits (This may take a minute)...")
    train_metrics = evaluate_bucket(model, train_loader, evaluator, device, "Training")
    val_metrics = evaluate_bucket(model, val_loader, evaluator, device, "Validation")
    test_metrics = evaluate_bucket(model, test_loader, evaluator, device, "Test")
    
    # 4. Print Table (Exact format requested in PDF section 3.3)
    print("\n" + "="*50)
    print(f"{'Split':<15} | {'PSNR':<10} | {'SSIM':<10}")
    print("-" * 50)
    print(f"{'Degraded Input':<15} | {test_metrics['base_psnr']:<10.2f} | {test_metrics['base_ssim']:<10.4f}  <-- (Do nothing baseline)")
    print("-" * 50)
    print(f"{'Training':<15} | {train_metrics['psnr']:<10.2f} | {train_metrics['ssim']:<10.4f}")
    print(f"{'Validation':<15} | {val_metrics['psnr']:<10.2f} | {val_metrics['ssim']:<10.4f}")
    print(f"{'Test':<15} | {test_metrics['psnr']:<10.2f} | {test_metrics['ssim']:<10.4f}")
    print("="*50)

if __name__ == '__main__':
    main()
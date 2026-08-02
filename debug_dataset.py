import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
from src.data.data_splitter import get_synthetic_splits

def save_debug_images(num_samples=5, save_dir="debug_outputs"):
    """
    Extracts samples from the training dataset and saves them side-by-side 
    to visually inspect the augmentations and spatial alignment.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    # Using the same parameters as train.py
    train_ds, _, _ = get_synthetic_splits(
        clean_scans_dir="data/clean_scans",
        backgrounds_dir="data/random_backgrounds",
        image_size=(256, 256),
        seed=42,
        num_eval_samples=100
    )
    
    print(f"Saving {num_samples} debug samples to {save_dir}/ ...")
    
    for i in range(num_samples):
        sample = train_ds[i]
        
        # Tensors are [C, H, W]. Convert to [H, W, C] for Matplotlib
        degraded = sample['rectified_input'].permute(1, 2, 0).cpu().numpy()
        clean = sample['clean_target'].permute(1, 2, 0).cpu().numpy()
        
        # Ensure values are in [0, 1] range for plotting
        degraded = degraded.clip(0, 1)
        clean = clean.clip(0, 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(degraded)
        axes[0].set_title("Degraded (Input to Model)")
        axes[0].axis("off")
        
        axes[1].imshow(clean)
        axes[1].set_title("Clean (Target Output)")
        axes[1].axis("off")
        
        # Overlay to check pixel-perfect spatial alignment
        axes[2].imshow(clean, alpha=0.5)
        axes[2].imshow(degraded, alpha=0.5)
        axes[2].set_title("Overlay (Alignment Check)")
        axes[2].axis("off")
        
        save_path = os.path.join(save_dir, f"sample_{i:03d}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    save_debug_images(num_samples=10)
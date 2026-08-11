"""
Phase 6: End-to-End Joint Training Pipeline

This module implements joint fine-tuning of Enhancement U-Net and Corner Detection models
using a sequential, differentiable pipeline where corner detector output is used to 
rectify the image during training, and error flows backward from Enhancement to Corner network.

Features:
- Sequential forward pass: corner detection -> perspective transform -> enhancement
- Differentiable warping using kornia
- Fine-tuning with enhancement loss only (backpropagated through entire chain)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

try:
    import kornia
    from kornia.geometry.transform import get_perspective_transform, warp_perspective
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Warning: kornia not installed. Install with: pip install kornia")

from src.models.model import EnhancementUNet, CornerRegressionModel, CornerHeatmapModel
from src.training.losses import EnhancementLoss

logger = logging.getLogger(__name__)


class SequentialE2ETrainer:
    """
    Trains corner detector and enhancement network in a sequential differentiable chain.
    
    Forward pass:
    1. Raw image -> Corner Detector -> predicted corners
    2. Predicted corners + target flat corners -> Homography
    3. Raw image + Homography -> warp_perspective -> rectified crop
    4. Rectified crop -> Enhancement Network -> enhanced output
    5. Enhanced output vs clean target -> loss
    6. Backpropagate through entire chain to update both networks
    """
    
    def __init__(
        self,
        enhancement_model: EnhancementUNet,
        corner_model: nn.Module,
        device: torch.device,
        image_size: Tuple[int, int] = (256, 256),
        lr: float = 1e-4
    ):
        if not KORNIA_AVAILABLE:
            raise ImportError("kornia is required for SequentialE2ETrainer. Install with: pip install kornia")
        
        self.enhancement_model = enhancement_model.to(device)
        self.corner_model = corner_model.to(device)
        self.device = device
        self.image_size = image_size
        
        # Target flat rectangle corners (normalized to [0, 1])
        h, w = image_size
        self.register_buffer('flat_corners', torch.tensor([
            [[0.0, 0.0], [w - 1, 0.0], [w - 1, h - 1], [0.0, h - 1]]], dtype=torch.float32).to(device))
        
        # Single enhancement loss for the entire chain
        self.criterion = EnhancementLoss(l1_weight=1.0, edge_weight=0.1).to(device)
        
        # Single optimizer for both models
        self.optimizer = optim.Adam(
            list(self.enhancement_model.parameters()) + list(self.corner_model.parameters()),
            lr=lr
        )
        
        logger.info(f"SequentialE2ETrainer initialized - Image size: {image_size}, LR: {lr}")
    
    def _corners_to_homography(self, pred_corners: torch.Tensor) -> torch.Tensor:
        """
        Convert predicted corners to homography matrix.
        
        Args:
            pred_corners: (B, 8) or (B, 4, 2) tensor of predicted corner coordinates
                         (assumed normalized to [0, 1])
        
        Returns:
            H: (B, 3, 3) homography matrices
        """
        B = pred_corners.shape[0]
        
        # Reshape to (B, 4, 2) if needed
        if pred_corners.dim() == 2:
            pred_corners = pred_corners.view(B, 4, 2)
        
        # Denormalize from [0, 1] to pixel coordinates
        h, w = self.image_size
        pred_corners_denorm = pred_corners.clone()
        pred_corners_denorm[:, :, 0] *= (w - 1)
        pred_corners_denorm[:, :, 1] *= (h - 1)
        
        # Expand flat_corners to batch size
        flat_corners_batch = self.flat_corners.expand(B, -1, -1)
        
        # Compute homography from predicted corners to flat rectangle
        H = get_perspective_transform(pred_corners_denorm, flat_corners_batch)
        
        return H
    
    def train_step_joint(
        self,
        images: torch.Tensor,
        enhancement_targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform one joint training step with sequential forward pass.
        
        Args:
            images: Raw document photos (B, C, H, W)
            enhancement_targets: Clean target images (B, C, H, W)
        
        Returns:
            Loss dictionary
        """
        self.enhancement_model.train()
        self.corner_model.train()
        
        images = images.to(self.device)
        enhancement_targets = enhancement_targets.to(self.device)
        
        # Step 1: Predict corners from raw image
        corner_output = self.corner_model(images)
        
        # Handle both regression (B, 8) and heatmap (tuple) outputs
        if isinstance(corner_output, tuple):
            pred_corners, _ = corner_output
        else:
            pred_corners = corner_output
        
        # Step 2: Compute homography from predicted corners to flat rectangle
        H = self._corners_to_homography(pred_corners)
        
        # Step 3: Warp raw image using predicted homography (differentiable)
        h, w = self.image_size
        rectified_crop = warp_perspective(images, H, (h, w))
        
        # Step 4: Pass rectified crop through enhancement network
        enhanced_output = self.enhancement_model(rectified_crop)
        
        # Step 5: Compute enhancement loss (compare enhanced output to clean target)
        loss = self.criterion(enhanced_output, enhancement_targets)
        
        # Step 6: Backpropagate through entire chain
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'total_loss': loss.item(), 'enhancement_loss': loss.item()}


def train_e2e_pipeline(
    dataloader: DataLoader,
    enhancement_model: EnhancementUNet,
    corner_model: nn.Module,
    device: torch.device,
    epochs: int = 50,
    image_size: Tuple[int, int] = (256, 256),
    lr: float = 1e-4,
    checkpoint_dir: Optional[Path] = None
) -> Dict[str, List[float]]:
    """
    End-to-end training pipeline with sequential differentiable chain.
    
    Args:
        dataloader: Dataset with raw photos and clean targets
        enhancement_model: U-Net for document enhancement
        corner_model: Model for corner detection
        device: CUDA/CPU device
        epochs: Number of training epochs
        image_size: Target image size for warping
        lr: Learning rate
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Training history dictionary
    """
    trainer = SequentialE2ETrainer(
        enhancement_model,
        corner_model,
        device,
        image_size=image_size,
        lr=lr
    )
    
    history = {
        'total_loss': [],
        'enhancement_loss': []
    }
    
    for epoch in range(epochs):
        epoch_total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract data from batch
            images = batch['raw_photo'] if isinstance(batch, dict) else batch[0]
            targets = batch['clean_target'] if isinstance(batch, dict) else batch[1]
            
            losses = trainer.train_step_joint(images, targets)
            
            epoch_total_loss += losses.get('total_loss', 0.0)
            num_batches += 1
            
            if batch_idx % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}: "
                           f"Loss = {losses.get('total_loss', 0.0):.4f}")
        
        # Average losses for epoch
        avg_total_loss = epoch_total_loss / num_batches
        
        history['total_loss'].append(avg_total_loss)
        history['enhancement_loss'].append(avg_total_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} completed - Avg Loss: {avg_total_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f'e2e_checkpoint_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'enhancement_model_state_dict': enhancement_model.state_dict(),
                'corner_model_state_dict': corner_model.state_dict(),
                'history': history
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    return history


if __name__ == '__main__':
    import argparse
    import sys
    import os
    
    from src.models.model import EnhancementUNet, CornerHeatmapModel
    from src.data.data_splitter import get_synthetic_splits
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="End-to-End Joint Fine-tuning (Bonus Phase)")
    parser.add_argument("--clean-scans", type=str, default="data/clean_scans")
    parser.add_argument("--backgrounds", type=str, default="data/random_backgrounds")
    parser.add_argument("--corner-ckpt", type=str, required=True, help="Path to pre-trained corner model (Gold)")
    parser.add_argument("--enhancement-ckpt", type=str, required=True, help="Path to pre-trained enhancement model (Gold)")
    parser.add_argument("--save-dir", type=str, default="checkpoints/e2e_finetuned")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5, help="Low learning rate for fine-tuning")
    parser.add_argument("--image-size", type=int, default=512)
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Dataset
    logger.info("Preparing Synthetic Dataset Splits...")
    train_ds, val_ds, _ = get_synthetic_splits(
        clean_scans_dir=args.clean_scans,
        backgrounds_dir=args.backgrounds,
        image_size=(args.image_size, args.image_size),
        seed=42,
        num_eval_samples=50,
        train_samples_per_epoch=1000  # Smaller epoch for fine-tuning
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 2. Load Gold Models
    logger.info("Loading pre-trained Gold models...")
    
    enhancement_model = EnhancementUNet(dropout_rate=0.3)
    enh_ckpt = torch.load(args.enhancement_ckpt, map_location=device, weights_only=True)
    enh_state = enh_ckpt.get('model_state_dict', enh_ckpt)
    if list(enh_state.keys())[0].startswith('module.'):
        enh_state = {k.replace('module.', ''): v for k, v in enh_state.items()}
    enhancement_model.load_state_dict(enh_state)

    corner_model = CornerHeatmapModel(dropout_rate=0.3)
    corner_ckpt = torch.load(args.corner_ckpt, map_location=device, weights_only=True)
    corner_state = corner_ckpt.get('model_state_dict', corner_ckpt)
    if list(corner_state.keys())[0].startswith('module.'):
        corner_state = {k.replace('module.', ''): v for k, v in corner_state.items()}
    corner_model.load_state_dict(corner_state)

    # 3. Start E2E Training
    logger.info("Starting End-to-End Fine-tuning...")
    history = train_e2e_pipeline(
        dataloader=train_loader,
        enhancement_model=enhancement_model,
        corner_model=corner_model,
        device=device,
        epochs=args.epochs,
        image_size=(args.image_size, args.image_size),
        lr=args.lr,
        checkpoint_dir=Path(args.save_dir)
    )
    
    logger.info("✅ End-to-End Fine-tuning completed!")
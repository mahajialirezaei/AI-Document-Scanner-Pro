"""
Phase 6: End-to-End Joint Training Pipeline

This module implements joint fine-tuning of Enhancement U-Net and Corner Detection models
using a sequential, differentiable pipeline where corner detector output is used to 
rectify the image during training, and error flows backward from Enhancement to Corner network.
"""

import os
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Disable OpenCV internal multithreading to prevent Deadlocks with PyTorch DataLoader
cv2.setNumThreads(0)

try:
    import kornia
    from kornia.geometry.transform import get_perspective_transform, warp_perspective
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Warning: kornia not installed. Install with: pip install kornia")

from src.models.model import EnhancementUNet, CornerHeatmapModel
from src.training.losses import EnhancementLoss
from src.data.data_splitter import get_synthetic_splits

logger = logging.getLogger(__name__)

class SequentialE2ETrainer:
    def __init__(
        self,
        enhancement_model: nn.Module,
        corner_model: nn.Module,
        device: torch.device,
        image_size: Tuple[int, int] = (512, 512),
        lr: float = 1e-5
    ):
        if not KORNIA_AVAILABLE:
            raise ImportError("kornia is required for SequentialE2ETrainer. Install with: pip install kornia")
        
        self.enhancement_model = enhancement_model.to(device)
        self.corner_model = corner_model.to(device)
        self.device = device
        self.image_size = image_size
        
        h, w = image_size
        self.flat_corners = torch.tensor([
            [[0.0, 0.0], [w - 1, 0.0], [w - 1, h - 1], [0.0, h - 1]]
        ], dtype=torch.float32, device=self.device)
        
        self.criterion = EnhancementLoss(l1_weight=1.0, edge_weight=2.0).to(device)
        
        self.optimizer = optim.Adam(
            list(self.enhancement_model.parameters()) + list(self.corner_model.parameters()),
            lr=lr
        )
        
        # AMP Scaler restored for speed
        self.scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        logger.info(f"SequentialE2ETrainer initialized - Image size: {image_size}, LR: {lr} (Selective AMP Mode)")
    
    def _corners_to_homography(self, pred_corners: torch.Tensor) -> torch.Tensor:
        B = pred_corners.shape[0]
        if pred_corners.dim() == 2:
            pred_corners = pred_corners.view(B, 4, 2)
        
        h, w = self.image_size
        pred_corners_denorm = pred_corners.clone()
        pred_corners_denorm[:, :, 0] *= (w - 1)
        pred_corners_denorm[:, :, 1] *= (h - 1)
        
        flat_corners_batch = self.flat_corners.expand(B, -1, -1)
        H = get_perspective_transform(pred_corners_denorm, flat_corners_batch)
        return H
    
    def train_step_joint(self, images: torch.Tensor, enhancement_targets: torch.Tensor) -> float:
        self.enhancement_model.train()
        self.corner_model.train()
        
        images = images.to(self.device, non_blocking=True)
        enhancement_targets = enhancement_targets.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # 1. Corner Detection (AMP Float16 for speed)
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            corner_output = self.corner_model(images)
            pred_corners = corner_output[0] if isinstance(corner_output, tuple) else corner_output
        
        # 2. Kornia Geometry (Forced Float32 to prevent Overflow)
        pred_corners_fp32 = pred_corners.float()
        images_fp32 = images.float()
        
        H = self._corners_to_homography(pred_corners_fp32)
        h, w = self.image_size
        rectified_crop_fp32 = warp_perspective(images_fp32, H, (h, w))
        
        # 3. Enhancement U-Net (AMP Float16 for speed)
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            enhanced_output = self.enhancement_model(rectified_crop_fp32)
            
        # 4. Loss Calculation (Forced Float32 to prevent Sobel 1e-8 underflow -> NaN)
        enhanced_output_fp32 = enhanced_output.float()
        targets_fp32 = enhancement_targets.float()
        loss = self.criterion(enhanced_output_fp32, targets_fp32)
        
        # NaN Guard
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf loss detected. Skipping gradient update for this batch.")
            return 0.0
            
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(list(self.enhancement_model.parameters()) + list(self.corner_model.parameters()), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

    @torch.no_grad()
    def validate_step_joint(self, images: torch.Tensor, enhancement_targets: torch.Tensor) -> float:
        self.enhancement_model.eval()
        self.corner_model.eval()
        
        images = images.to(self.device, non_blocking=True)
        enhancement_targets = enhancement_targets.to(self.device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            corner_output = self.corner_model(images)
            pred_corners = corner_output[0] if isinstance(corner_output, tuple) else corner_output
        
        pred_corners_fp32 = pred_corners.float()
        images_fp32 = images.float()
        
        H = self._corners_to_homography(pred_corners_fp32)
        h, w = self.image_size
        rectified_crop_fp32 = warp_perspective(images_fp32, H, (h, w))
        
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            enhanced_output = self.enhancement_model(rectified_crop_fp32)
            
        enhanced_output_fp32 = enhanced_output.float()
        targets_fp32 = enhancement_targets.float()
        loss = self.criterion(enhanced_output_fp32, targets_fp32)
            
        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0
            
        return loss.item()


def train_e2e_pipeline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    enhancement_model: nn.Module,
    corner_model: nn.Module,
    device: torch.device,
    epochs: int = 15,
    image_size: Tuple[int, int] = (512, 512),
    lr: float = 1e-5,
    checkpoint_dir: Optional[Path] = None
) -> Dict[str, List[float]]:
    
    trainer = SequentialE2ETrainer(enhancement_model, corner_model, device, image_size=image_size, lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    print(f"Starting End-to-End Joint training for {epochs} epochs...")
    print(f"Save directory: {checkpoint_dir}")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Training Phase
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        for batch in train_loader:
            images = batch['raw_photo'] if isinstance(batch, dict) else batch[0]
            targets = batch['clean_target'] if isinstance(batch, dict) else batch[1]
            
            loss = trainer.train_step_joint(images, targets)
            epoch_train_loss += loss
            num_train_batches += 1
            
        train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        
        # Validation Phase
        epoch_val_loss = 0.0
        num_val_batches = 0
        for batch in val_loader:
            images = batch['raw_photo'] if isinstance(batch, dict) else batch[0]
            targets = batch['clean_target'] if isinstance(batch, dict) else batch[1]
            
            loss = trainer.validate_step_joint(images, targets)
            epoch_val_loss += loss
            num_val_batches += 1
            
        val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        
        current_lr = trainer.optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_dir:
                torch.save({
                    'epoch': epoch,
                    'enhancement_model_state_dict': enhancement_model.state_dict(),
                    'corner_model_state_dict': corner_model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_dir / 'best_model.pth')
        
        print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    return history


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.ERROR)
    
    parser = argparse.ArgumentParser(description="End-to-End Joint Fine-tuning (Bonus Phase)")
    parser.add_argument("--clean-scans", type=str, default="data/clean_scans")
    parser.add_argument("--backgrounds", type=str, default="data/random_backgrounds")
    parser.add_argument("--corner-ckpt", type=str, required=True, help="Path to pre-trained corner model (Gold)")
    parser.add_argument("--enhancement-ckpt", type=str, required=True, help="Path to pre-trained enhancement model (Gold)")
    parser.add_argument("--save-dir", type=str, default="checkpoints/e2e_finetuned")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--image-size", type=int, default=512)
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    print("Preparing Synthetic Dataset Splits...")
    train_ds, val_ds, _ = get_synthetic_splits(
        clean_scans_dir=args.clean_scans,
        backgrounds_dir=args.backgrounds,
        image_size=(args.image_size, args.image_size),
        seed=42,
        num_eval_samples=20, 
        train_samples_per_epoch=1000 
    )
    
    # Restored high-speed DataLoaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2,
    )

    print("Loading pre-trained Gold models...")
    
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

    history = train_e2e_pipeline(
        train_loader=train_loader,
        val_loader=val_loader,
        enhancement_model=enhancement_model,
        corner_model=corner_model,
        device=device,
        epochs=args.epochs,
        image_size=(args.image_size, args.image_size),
        lr=args.lr,
        checkpoint_dir=Path(args.save_dir)
    )
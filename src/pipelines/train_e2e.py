"""
Phase 6: End-to-End Joint Training Pipeline

This module implements joint fine-tuning of Enhancement U-Net and Corner Detection models
with multi-task learning strategies for improved document scanning performance.

Features:
- Joint training with combined loss functions
- Alternating optimization strategy
- Shared backbone architecture support
- Gradient balancing between tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from src.models.model import EnhancementUNet, CornerRegressionModel, CornerHeatmapModel
from src.training.losses import EnhancementLoss, CornerLoss

logger = logging.getLogger(__name__)


class JointTrainer:
    """
    Orchestrates simultaneous training of Enhancement and Corner Detection models.
    
    Supports:
    - Multi-task loss combination
    - Alternating batch updates
    - Shared backbone architectures
    - Gradient normalization
    """
    
    def __init__(
        self,
        enhancement_model: EnhancementUNet,
        corner_model: nn.Module,
        device: torch.device,
        enhancement_weight: float = 0.5,
        corner_weight: float = 0.5,
        use_shared_backbone: bool = False,
        gradient_balancing: str = 'none'  # 'none', 'norm', 'uncertainty'
    ):
        self.enhancement_model = enhancement_model.to(device)
        self.corner_model = corner_model.to(device)
        self.device = device
        self.enhancement_weight = enhancement_weight
        self.corner_weight = corner_weight
        self.use_shared_backbone = use_shared_backbone
        self.gradient_balancing = gradient_balancing
        
        # Initialize loss functions
        self.enhancement_criterion = EnhancementLoss(edge_weight=0.1, use_sobel=True)
        self.corner_criterion = CornerLoss(heatmap_weight=0.5, coordinate_weight=0.5)
        
        # Learnable uncertainty weights (if using uncertainty balancing)
        if gradient_balancing == 'uncertainty':
            self.log_var_enhancement = nn.Parameter(torch.zeros(1)).to(device)
            self.log_var_corner = nn.Parameter(torch.zeros(1)).to(device)
            logger.info("Using uncertainty-based gradient balancing")
        
        # Optimizers
        if use_shared_backbone:
            # Share early layers between models
            self._setup_shared_backbone()
            self.optimizer = optim.Adam(
                list(self.enhancement_model.parameters()) + 
                list(self.corner_model.parameters()),
                lr=1e-3
            )
        else:
            self.optimizer_enhancement = optim.Adam(
                self.enhancement_model.parameters(), lr=1e-3
            )
            self.optimizer_corner = optim.Adam(
                self.corner_model.parameters(), lr=1e-3
            )
        
        logger.info(f"JointTrainer initialized - Enhancement weight: {enhancement_weight}, "
                   f"Corner weight: {corner_weight}")
    
    def _setup_shared_backbone(self) -> None:
        """
        Configure models to share early convolutional layers.
        
        This creates a common feature extractor used by both tasks.
        """
        # Share the first 3 encoder blocks
        shared_layers = []
        for name, param in self.enhancement_model.named_parameters():
            if 'encoder' in name and ('block1' in name or 'block2' in name or 'block3' in name):
                shared_layers.append(name)
        
        logger.info(f"Sharing {len(shared_layers)} layers between models")
        
        # Copy weights from enhancement model to corner model
        for name, param in self.enhancement_model.named_parameters():
            if name in shared_layers:
                corner_param_name = name.replace('enhancement', 'corner')
                if hasattr(self.corner_model, 'encoder'):
                    try:
                        getattr(self.corner_model.encoder, name.split('.')[-1]).weight.data.copy_(param.data)
                    except (AttributeError, KeyError):
                        pass
    
    def compute_combined_loss(
        self,
        enhancement_output: torch.Tensor,
        enhancement_target: torch.Tensor,
        corner_output: torch.Tensor,
        corner_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined multi-task loss.
        
        Returns:
            Total loss and individual loss components
        """
        # Individual losses
        loss_enhancement = self.enhancement_criterion(enhancement_output, enhancement_target)
        loss_corner = self.corner_criterion(corner_output, corner_target)
        
        # Apply gradient balancing
        if self.gradient_balancing == 'uncertainty':
            # Uncertainty weighting (Kendall et al., 2018)
            prec_enhancement = torch.exp(-self.log_var_enhancement)
            prec_corner = torch.exp(-self.log_var_corner)
            
            loss_total = (
                prec_enhancement * loss_enhancement / 2 + self.log_var_enhancement / 2 +
                prec_corner * loss_corner / 2 + self.log_var_corner / 2
            )
        elif self.gradient_balancing == 'norm':
            # Gradient normalization
            grad_norm_enh = torch.sqrt(sum(p.grad.data.norm(2)**2 for p in self.enhancement_model.parameters() if p.grad is not None))
            grad_norm_corner = torch.sqrt(sum(p.grad.data.norm(2)**2 for p in self.corner_model.parameters() if p.grad is not None))
            
            # Normalize weights inversely proportional to gradient norms
            total_norm = grad_norm_enh + grad_norm_corner + 1e-8
            norm_enh = grad_norm_enh / total_norm
            norm_corner = grad_norm_corner / total_norm
            
            loss_total = norm_enh * loss_enhancement + norm_corner * loss_corner
        else:
            # Fixed weighting
            loss_total = self.enhancement_weight * loss_enhancement + self.corner_weight * loss_corner
        
        loss_dict = {
            'total': loss_total.item(),
            'enhancement': loss_enhancement.item(),
            'corner': loss_corner.item()
        }
        
        return loss_total, loss_dict
    
    def train_step_joint(
        self,
        images: torch.Tensor,
        enhancement_targets: torch.Tensor,
        corner_targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform one joint training step updating both models simultaneously.
        """
        self.enhancement_model.train()
        self.corner_model.train()
        
        images = images.to(self.device)
        enhancement_targets = enhancement_targets.to(self.device)
        corner_targets = corner_targets.to(self.device)
        
        # Forward pass through both models
        enhancement_output = self.enhancement_model(images)
        corner_output = self.corner_model(images)
        
        # Compute combined loss
        loss_total, loss_dict = self.compute_combined_loss(
            enhancement_output, enhancement_targets,
            corner_output, corner_targets
        )
        
        # Backward pass
        if self.use_shared_backbone:
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
        else:
            # Separate backward passes
            self.optimizer_enhancement.zero_grad()
            (loss_dict['enhancement'] * self.enhancement_weight).backward(retain_graph=True)
            self.optimizer_enhancement.step()
            
            self.optimizer_corner.zero_grad()
            (loss_dict['corner'] * self.corner_weight).backward()
            self.optimizer_corner.step()
        
        return loss_dict
    
    def train_step_alternating(
        self,
        images: torch.Tensor,
        enhancement_targets: torch.Tensor,
        corner_targets: torch.Tensor,
        task: str = 'enhancement'
    ) -> Dict[str, float]:
        """
        Perform alternating training step (update one task at a time).
        
        Args:
            task: 'enhancement' or 'corner' - which task to update
        """
        if task == 'enhancement':
            self.enhancement_model.train()
            images = images.to(self.device)
            enhancement_targets = enhancement_targets.to(self.device)
            
            self.optimizer_enhancement.zero_grad()
            output = self.enhancement_model(images)
            loss = self.enhancement_criterion(output, enhancement_targets)
            loss.backward()
            self.optimizer_enhancement.step()
            
            return {'enhancement': loss.item()}
        
        elif task == 'corner':
            self.corner_model.train()
            images = images.to(self.device)
            corner_targets = corner_targets.to(self.device)
            
            self.optimizer_corner.zero_grad()
            output = self.corner_model(images)
            loss = self.corner_criterion(output, corner_targets)
            loss.backward()
            self.optimizer_corner.step()
            
            return {'corner': loss.item()}
        
        else:
            raise ValueError(f"Unknown task: {task}")


def train_e2e_pipeline(
    dataloader: DataLoader,
    enhancement_model: EnhancementUNet,
    corner_model: nn.Module,
    device: torch.device,
    epochs: int = 50,
    strategy: str = 'joint',  # 'joint' or 'alternating'
    enhancement_weight: float = 0.5,
    corner_weight: float = 0.5,
    checkpoint_dir: Optional[Path] = None
) -> Dict[str, List[float]]:
    """
    End-to-end training pipeline for joint model optimization.
    
    Args:
        dataloader: Combined dataset with both enhancement and corner targets
        enhancement_model: U-Net for document enhancement
        corner_model: Model for corner detection
        device: CUDA/CPU device
        epochs: Number of training epochs
        strategy: 'joint' (simultaneous) or 'alternating' (batch-wise switching)
        enhancement_weight: Weight for enhancement loss
        corner_weight: Weight for corner detection loss
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Training history dictionary
    """
    trainer = JointTrainer(
        enhancement_model,
        corner_model,
        device,
        enhancement_weight=enhancement_weight,
        corner_weight=corner_weight,
        use_shared_backbone=False,
        gradient_balancing='none'
    )
    
    history = {
        'enhancement_loss': [],
        'corner_loss': [],
        'total_loss': []
    }
    
    for epoch in range(epochs):
        epoch_enh_loss = 0.0
        epoch_corner_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Assume batch contains: (images, enhancement_targets, corner_targets)
            if len(batch) == 3:
                images, enh_targets, corner_targets = batch
            else:
                # Fallback for different data formats
                images = batch[0]
                enh_targets = batch[1] if len(batch) > 1 else batch[0]
                corner_targets = batch[2] if len(batch) > 2 else batch[1]
            
            if strategy == 'joint':
                losses = trainer.train_step_joint(images, enh_targets, corner_targets)
            elif strategy == 'alternating':
                # Alternate between tasks each batch
                task = 'enhancement' if batch_idx % 2 == 0 else 'corner'
                losses = trainer.train_step_alternating(images, enh_targets, corner_targets, task)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            epoch_enh_loss += losses.get('enhancement', 0.0)
            epoch_corner_loss += losses.get('corner', 0.0)
            epoch_total_loss += losses.get('total', losses.get('enhancement', 0.0) + losses.get('corner', 0.0))
            num_batches += 1
            
            if batch_idx % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}: "
                           f"Enh Loss = {losses.get('enhancement', 0.0):.4f}, "
                           f"Corner Loss = {losses.get('corner', 0.0):.4f}")
        
        # Average losses for epoch
        avg_enh_loss = epoch_enh_loss / num_batches
        avg_corner_loss = epoch_corner_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        
        history['enhancement_loss'].append(avg_enh_loss)
        history['corner_loss'].append(avg_corner_loss)
        history['total_loss'].append(avg_total_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} completed - "
                   f"Avg Enh Loss: {avg_enh_loss:.4f}, "
                   f"Avg Corner Loss: {avg_corner_loss:.4f}, "
                   f"Avg Total Loss: {avg_total_loss:.4f}")
        
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
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample models
    enhancement_model = EnhancementUNet(dropout_rate=0.1)
    corner_model = CornerRegressionModel(dropout_rate=0.1)
    
    print("Creating JointTrainer...")
    trainer = JointTrainer(
        enhancement_model,
        corner_model,
        device,
        enhancement_weight=0.5,
        corner_weight=0.5
    )
    
    # Test with dummy data
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    enh_targets = torch.randn(batch_size, 3, 256, 256).to(device)
    corner_targets = torch.randn(batch_size, 4, 2).to(device)  # 4 corners, (x, y)
    
    print("\nTesting joint training step...")
    losses = trainer.train_step_joint(images, enh_targets, corner_targets)
    print(f"Losses: {losses}")
    
    print("\nTesting alternating training step (enhancement)...")
    losses = trainer.train_step_alternating(images, enh_targets, corner_targets, 'enhancement')
    print(f"Losses: {losses}")
    
    print("\nTesting alternating training step (corner)...")
    losses = trainer.train_step_alternating(images, enh_targets, corner_targets, 'corner')
    print(f"Losses: {losses}")
    
    print("\n✅ End-to-End pipeline ready for Phase 6!")

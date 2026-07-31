"""
Phase 6: End-to-End Joint Training Pipeline
Combines Document Enhancement and Corner Detection in a unified multi-task learning framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Combines multiple loss functions for joint training.
    
    Supports weighted combination of enhancement and corner detection losses.
    Implements uncertainty-based weighting (Kendall et al., 2018) optionally.
    """
    
    def __init__(
        self,
        enhancement_loss: nn.Module,
        corner_loss: nn.Module,
        enhancement_weight: float = 1.0,
        corner_weight: float = 1.0,
        use_uncertainty_weighting: bool = False
    ):
        super().__init__()
        self.enhancement_loss = enhancement_loss
        self.corner_loss = corner_loss
        self.enhancement_weight = enhancement_weight
        self.corner_weight = corner_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if use_uncertainty_weighting:
            # Learnable log variance parameters
            self.log_var_enhancement = nn.Parameter(torch.zeros(1))
            self.log_var_corner = nn.Parameter(torch.zeros(1))
            logger.info("Using uncertainty-based loss weighting")
        else:
            logger.info(f"Using fixed weights: enhancement={enhancement_weight}, corner={corner_weight}")
    
    def forward(
        self,
        enhanced_output: torch.Tensor,
        enhanced_target: torch.Tensor,
        corner_output: torch.Tensor,
        corner_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined multi-task loss.
        
        Args:
            enhanced_output: Output from enhancement network
            enhanced_target: Ground truth enhanced images
            corner_output: Output from corner detection network
            corner_target: Ground truth corner coordinates/heatmaps
        
        Returns:
            Combined loss and individual loss components
        """
        # Compute individual losses
        loss_enh = self.enhancement_loss(enhanced_output, enhanced_target)
        loss_corner = self.corner_loss(corner_output, corner_target)
        
        if self.use_uncertainty_weighting:
            # Uncertainty-based weighting
            prec_enh = torch.exp(-self.log_var_enhancement)
            prec_corner = torch.exp(-self.log_var_corner)
            
            loss_enh_weighted = prec_enh * loss_enh + self.log_var_enhancement
            loss_corner_weighted = prec_corner * loss_corner + self.log_var_corner
            
            total_loss = loss_enh_weighted + loss_corner_weighted
        else:
            # Fixed weighting
            total_loss = (
                self.enhancement_weight * loss_enh +
                self.corner_weight * loss_corner
            )
        
        loss_dict = {
            'total': total_loss.item(),
            'enhancement': loss_enh.item(),
            'corner': loss_corner.item()
        }
        
        return total_loss, loss_dict


class SharedBackboneNetwork(nn.Module):
    """
    Neural network with shared backbone for multi-task learning.
    
    Early layers are shared between enhancement and corner detection tasks,
    while task-specific heads process the features separately.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        enhancement_head: nn.Module,
        corner_head: nn.Module,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.backbone = backbone
        self.enhancement_head = enhancement_head
        self.corner_head = corner_head
        
        if freeze_backbone:
            self._freeze_backbone()
            logger.info("Backbone frozen for fine-tuning")
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through shared backbone and task-specific heads.
        
        Args:
            x: Input tensor
            return_features: Whether to return intermediate features
        
        Returns:
            Enhanced image, corner predictions, and optionally features
        """
        # Shared backbone
        features = self.backbone(x)
        
        # Task-specific heads
        enhanced = self.enhancement_head(features)
        corners = self.corner_head(features)
        
        if return_features:
            return enhanced, corners, features
        return enhanced, corners


class JointTrainer:
    """
    Trainer for end-to-end joint training of enhancement and corner detection.
    
    Supports:
    - Alternating batch updates
    - Simultaneous gradient updates
    - Progressive unfreezing strategies
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        multi_task_loss: MultiTaskLoss,
        optimizer: optim.Optimizer,
        config: Dict
    ):
        self.model = model.to(device)
        self.device = device
        self.multi_task_loss = multi_task_loss
        self.optimizer = optimizer
        self.config = config
        
        self.grad_accum_steps = config.get('grad_accum_steps', 1)
        self.alternating_strategy = config.get('alternating_strategy', False)
        
        logger.info(f"Joint trainer initialized on {device}")
        logger.info(f"Alternating strategy: {self.alternating_strategy}")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        task_scheduler: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch with joint optimization.
        
        Args:
            dataloader: Data loader with both enhancement and corner targets
            epoch: Current epoch number
            task_scheduler: Strategy for task scheduling ('alternating', 'simultaneous')
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_metrics = {
            'total_loss': 0.0,
            'enhancement_loss': 0.0,
            'corner_loss': 0.0
        }
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract data
            images = batch['image'].to(self.device)
            enhanced_targets = batch['enhanced'].to(self.device)
            corner_targets = batch['corners'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            if self.alternating_strategy:
                # Alternating batch updates
                loss_dict = self._alternating_step(
                    images, enhanced_targets, corner_targets
                )
            else:
                # Simultaneous update
                loss_dict = self._simultaneous_step(
                    images, enhanced_targets, corner_targets
                )
            
            # Accumulate metrics
            for key in total_metrics:
                if key in loss_dict:
                    total_metrics[key] += loss_dict[key]
            
            num_batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Total Loss = {loss_dict.get('total_loss', 0):.6f}"
                )
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= max(num_batches, 1)
        
        logger.info(
            f"Epoch {epoch} completed | "
            f"Total: {total_metrics['total_loss']:.6f} | "
            f"Enhancement: {total_metrics['enhancement_loss']:.6f} | "
            f"Corner: {total_metrics['corner_loss']:.6f}"
        )
        
        return total_metrics
    
    def _simultaneous_step(
        self,
        images: torch.Tensor,
        enhanced_targets: torch.Tensor,
        corner_targets: torch.Tensor
    ) -> Dict[str, float]:
        """Perform simultaneous gradient update for both tasks."""
        # Forward pass
        enhanced_output, corner_output = self.model(images)
        
        # Compute combined loss
        total_loss, loss_dict = self.multi_task_loss(
            enhanced_output, enhanced_targets,
            corner_output, corner_targets
        )
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return loss_dict
    
    def _alternating_step(
        self,
        images: torch.Tensor,
        enhanced_targets: torch.Tensor,
        corner_targets: torch.Tensor
    ) -> Dict[str, float]:
        """Perform alternating updates for each task."""
        # Enhancement step
        enhanced_output, _ = self.model(images)
        loss_enh = self.multi_task_loss.enhancement_loss(
            enhanced_output, enhanced_targets
        )
        loss_enh.backward(retain_graph=True)
        
        # Corner detection step
        _, corner_output = self.model(images)
        loss_corner = self.multi_task_loss.corner_loss(
            corner_output, corner_targets
        )
        loss_corner.backward()
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {
            'total_loss': loss_enh.item() + loss_corner.item(),
            'enhancement_loss': loss_enh.item(),
            'corner_loss': loss_corner.item()
        }
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Run validation without gradient updates."""
        self.model.eval()
        total_metrics = {
            'total_loss': 0.0,
            'enhancement_loss': 0.0,
            'corner_loss': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                enhanced_targets = batch['enhanced'].to(self.device)
                corner_targets = batch['corners'].to(self.device)
                
                enhanced_output, corner_output = self.model(images)
                
                _, loss_dict = self.multi_task_loss(
                    enhanced_output, enhanced_targets,
                    corner_output, corner_targets
                )
                
                for key in total_metrics:
                    if key in loss_dict:
                        total_metrics[key] += loss_dict[key]
                
                num_batches += 1
        
        for key in total_metrics:
            total_metrics[key] /= max(num_batches, 1)
        
        return total_metrics


class EndToEndPipeline:
    """
    Complete end-to-end training pipeline for joint optimization.
    
    Manages:
    - Model initialization
    - Training loop with scheduling
    - Checkpoint saving/loading
    - Progressive unfreezing
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict
    ):
        self.model = model
        self.device = device
        self.config = config
        
        # Initialize losses
        self.enhancement_loss_fn = self._create_enhancement_loss(
            config.get('enhancement_loss_type', 'l1')
        )
        self.corner_loss_fn = self._create_corner_loss(
            config.get('corner_loss_type', 'smooth_l1')
        )
        
        # Multi-task loss
        self.multi_task_loss = MultiTaskLoss(
            enhancement_loss=self.enhancement_loss_fn,
            corner_loss=self.corner_loss_fn,
            enhancement_weight=config.get('enhancement_weight', 1.0),
            corner_weight=config.get('corner_weight', 1.0),
            use_uncertainty_weighting=config.get('use_uncertainty_weighting', False)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('scheduler_T0', 10),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Trainer
        self.trainer = JointTrainer(
            model=model,
            device=device,
            multi_task_loss=self.multi_task_loss,
            optimizer=self.optimizer,
            config=config
        )
        
        logger.info("End-to-End pipeline initialized")
    
    def _create_enhancement_loss(self, loss_type: str) -> nn.Module:
        """Create enhancement loss function."""
        if loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'ssim':
            # SSIM would require additional implementation
            logger.warning("SSIM not implemented, using L1 instead")
            return nn.L1Loss()
        else:
            return nn.L1Loss()
    
    def _create_corner_loss(self, loss_type: str) -> nn.Module:
        """Create corner detection loss function."""
        if loss_type == 'smooth_l1':
            return nn.SmoothL1Loss()
        elif loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'huber':
            return nn.HuberLoss()
        else:
            return nn.SmoothL1Loss()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Full end-to-end training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save checkpoints
        
        Returns:
            Training history
        """
        epochs = self.config.get('epochs', 100)
        history = {
            'train_total': [],
            'train_enh': [],
            'train_corner': [],
            'val_total': [],
            'val_enh': [],
            'val_corner': []
        }
        
        best_val_loss = float('inf')
        
        # Progressive unfreezing schedule
        unfreeze_epoch = self.config.get('unfreeze_epoch', 20)
        if isinstance(self.model, SharedBackboneNetwork):
            logger.info(f"Backbone will be unfrozen at epoch {unfreeze_epoch}")
        
        logger.info(f"Starting end-to-end training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Progressive unfreezing
            if (epoch == unfreeze_epoch and 
                isinstance(self.model, SharedBackboneNetwork)):
                self.model._unfreeze_backbone()
                logger.info(f"Epoch {epoch}: Unfroze backbone for fine-tuning")
            
            # Training
            train_metrics = self.trainer.train_epoch(
                train_loader,
                epoch,
                task_scheduler=self.config.get('task_scheduler')
            )
            
            history['train_total'].append(train_metrics['total_loss'])
            history['train_enh'].append(train_metrics['enhancement_loss'])
            history['train_corner'].append(train_metrics['corner_loss'])
            
            # Validation
            if val_loader is not None:
                val_metrics = self.trainer.validate(val_loader)
                history['val_total'].append(val_metrics['total_loss'])
                history['val_enh'].append(val_metrics['enhancement_loss'])
                history['val_corner'].append(val_metrics['corner_loss'])
                
                # Save best model
                if val_metrics['total_loss'] < best_val_loss and save_path:
                    best_val_loss = val_metrics['total_loss']
                    self._save_checkpoint(save_path, epoch, val_metrics)
                    logger.info(
                        f"Saved new best model at epoch {epoch} | "
                        f"Val Total Loss: {best_val_loss:.6f}"
                    )
            
            # Update scheduler
            self.scheduler.step()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Total: {train_metrics['total_loss']:.6f} | "
                    f"Val Total: {history['val_total'][-1] if val_loader else 'N/A':.6f}"
                )
        
        logger.info("End-to-end training completed!")
        return history
    
    def _save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint


def create_joint_training_pipeline(
    backbone: nn.Module,
    enhancement_head: nn.Module,
    corner_head: nn.Module,
    device: torch.device,
    config: Optional[Dict] = None
) -> EndToEndPipeline:
    """
    Convenience function to create complete joint training pipeline.
    
    Args:
        backbone: Shared backbone network
        enhancement_head: Enhancement task head
        corner_head: Corner detection task head
        device: Training device
        config: Configuration dictionary
    
    Returns:
        Configured EndToEndPipeline
    """
    if config is None:
        config = {
            'epochs': 100,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'enhancement_weight': 1.0,
            'corner_weight': 1.0,
            'enhancement_loss_type': 'l1',
            'corner_loss_type': 'smooth_l1',
            'use_uncertainty_weighting': False,
            'alternating_strategy': False,
            'unfreeze_epoch': 20
        }
    
    model = SharedBackboneNetwork(
        backbone=backbone,
        enhancement_head=enhancement_head,
        corner_head=corner_head,
        freeze_backbone=True  # Start frozen
    )
    
    pipeline = EndToEndPipeline(model, device, config)
    return pipeline

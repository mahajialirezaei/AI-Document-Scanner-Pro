"""
Phase 6: Regularization Strategies for Document Scanning Models

This module implements advanced regularization techniques including:
- Dynamic Dropout Scheduling
- Robust Loss Functions (Huber, Smooth L1)
- Comparison experiments between regularized and baseline models

NOTE: Per PDF section 4.4, third-party libraries like Kornia are prohibited for 
data augmentation in training phases. Only OpenCV functions should be used for 
degradations (handled in degradation.py). Kornia is only allowed in Bonus section
for differentiable warp_perspective operations.
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


class DropoutScheduler:
    """
    Dynamically adjusts dropout rates during training.
    
    Strategies:
    - Linear increase: Gradually increase dropout to prevent overfitting
    - Cosine annealing: Smooth dropout variation
    - Step function: Discrete changes at specific epochs
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategy: str = 'linear',
        initial_dropout: float = 0.0,
        final_dropout: float = 0.5,
        warmup_epochs: int = 5,
        total_epochs: int = 100
    ):
        self.model = model
        self.strategy = strategy
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        # Find all dropout layers in the model
        self.dropout_layers = self._find_dropout_layers()
        logger.info(f"Found {len(self.dropout_layers)} dropout layers")
    
    def _find_dropout_layers(self) -> List[nn.Dropout]:
        """Find all Dropout layers in the model."""
        dropout_layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                dropout_layers.append(module)
        return dropout_layers
    
    def step(self, epoch: int) -> None:
        """Update dropout rates based on current epoch."""
        if epoch < self.warmup_epochs:
            # During warmup, keep dropout at initial value
            current_dropout = self.initial_dropout
        else:
            # Calculate progress through training (after warmup)
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(1.0, max(0.0, progress))
            
            if self.strategy == 'linear':
                current_dropout = self.initial_dropout + progress * (self.final_dropout - self.initial_dropout)
            elif self.strategy == 'cosine':
                import math
                current_dropout = self.initial_dropout + (self.final_dropout - self.initial_dropout) * (1 - math.cos(progress * math.pi)) / 2
            elif self.strategy == 'step':
                # Increase dropout every 25% of training
                if progress < 0.25:
                    current_dropout = self.initial_dropout
                elif progress < 0.5:
                    current_dropout = self.initial_dropout + 0.3 * (self.final_dropout - self.initial_dropout)
                elif progress < 0.75:
                    current_dropout = self.initial_dropout + 0.6 * (self.final_dropout - self.initial_dropout)
                else:
                    current_dropout = self.final_dropout
            else:
                current_dropout = self.initial_dropout
        
        # Apply new dropout rate to all layers
        for dropout_layer in self.dropout_layers:
            dropout_layer.p = current_dropout
        
        if epoch >= self.warmup_epochs:
            logger.info(f"Epoch {epoch}: Dropout rate set to {current_dropout:.4f} ({self.strategy} strategy)")
    
    def set_dropout_for_eval(self) -> None:
        """Set dropout to 0 for evaluation/inference."""
        for dropout_layer in self.dropout_layers:
            dropout_layer.p = 0.0


def create_robust_criterion(
    task_type: str = 'enhancement',
    loss_type: str = 'huber',
    **kwargs
) -> nn.Module:
    """
    Create robust loss functions for better generalization.
    
    Args:
        task_type: 'enhancement' or 'corner_detection'
        loss_type: 'huber', 'smooth_l1', or 'l1'
        **kwargs: Additional arguments for loss initialization
    
    Returns:
        Configured loss function
    """
    if task_type == 'enhancement':
        if loss_type == 'huber':
            return nn.HuberLoss(reduction='mean', delta=kwargs.get('delta', 1.0))
        elif loss_type == 'smooth_l1':
            return nn.SmoothL1Loss(reduction='mean')
        else:
            return EnhancementLoss(
                edge_weight=kwargs.get('edge_weight', 0.1),
                use_sobel=kwargs.get('use_sobel', True)
            )
    
    elif task_type == 'corner_detection':
        if loss_type == 'huber':
            return nn.HuberLoss(reduction='mean', delta=kwargs.get('delta', 0.5))
        elif loss_type == 'smooth_l1':
            return nn.SmoothL1Loss(reduction='mean')
        else:
            return CornerLoss(
                heatmap_weight=kwargs.get('heatmap_weight', 0.5),
                coordinate_weight=kwargs.get('coordinate_weight', 0.5)
            )
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


class RegularizationExperiment:
    """
    Run experiments comparing regularized vs baseline training.
    """
    
    def __init__(
        self,
        model_class: type,
        model_config: Dict,
        device: torch.device,
        results_dir: Path
    ):
        self.model_class = model_class
        self.model_config = model_config
        self.device = device
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_comparison(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 1e-3
    ) -> Dict[str, List[float]]:
        """
        Train baseline and regularized models, compare performance.
        
        Returns:
            Dictionary with training histories
        """
        results = {}
        
        # Train baseline model (no regularization)
        logger.info("=" * 60)
        logger.info("Training BASELINE model (no regularization)")
        logger.info("=" * 60)
        
        baseline_model = self.model_class(**self.model_config).to(self.device)
        baseline_results = self._train_model(
            baseline_model, train_loader, val_loader, epochs, learning_rate,
            use_regularization=False
        )
        results['baseline'] = baseline_results
        
        # Save baseline checkpoint
        torch.save(
            baseline_model.state_dict(),
            self.results_dir / 'baseline_checkpoint.pth'
        )
        
        # Train regularized model
        logger.info("=" * 60)
        logger.info("Training REGULARIZED model")
        logger.info("=" * 60)
        
        reg_model = self.model_class(**self.model_config).to(self.device)
        reg_results = self._train_model(
            reg_model, train_loader, val_loader, epochs, learning_rate,
            use_regularization=True
        )
        results['regularized'] = reg_results
        
        # Save regularized checkpoint
        torch.save(
            reg_model.state_dict(),
            self.results_dir / 'regularized_checkpoint.pth'
        )
        
        return results
    
    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        learning_rate: float,
        use_regularization: bool
    ) -> Dict[str, List[float]]:
        """Internal training loop with optional regularization."""
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Initialize regularization components
        dropout_scheduler = None
        criterion = None
        
        if use_regularization:
            # Dropout scheduling
            dropout_scheduler = DropoutScheduler(
                model,
                strategy='cosine',
                initial_dropout=0.0,
                final_dropout=0.3,
                warmup_epochs=10,
                total_epochs=epochs
            )
            
            # Robust loss function
            if 'UNet' in self.model_class.__name__:
                criterion = create_robust_criterion('enhancement', 'huber', delta=1.0)
            else:
                criterion = create_robust_criterion('corner_detection', 'huber', delta=0.5)
        else:
            # Baseline: simple L1 loss
            if 'UNet' in self.model_class.__name__:
                criterion = nn.L1Loss()
            else:
                criterion = nn.L1Loss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Update dropout schedule if using regularization
            if dropout_scheduler is not None:
                dropout_scheduler.step(epoch)
            
            # Training phase - simple training loop (no Kornia augmentation)
            model.train()
            total_loss = 0.0
            for images, targets in train_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_metrics = {'loss': total_loss / len(train_loader)}
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            train_losses.append(train_metrics['loss'])
            val_losses.append(val_loss / len(val_loader))
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_losses[-1]:.6f}, "
                    f"Val Loss: {val_losses[-1]:.6f}"
                )
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test dropout scheduler
    model = EnhancementUNet(dropout_rate=0.1)
    scheduler = DropoutScheduler(
        model,
        strategy='cosine',
        initial_dropout=0.0,
        final_dropout=0.5,
        warmup_epochs=5,
        total_epochs=50
    )
    
    print("Testing dropout scheduler...")
    for epoch in range(0, 50, 10):
        scheduler.step(epoch)
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                print(f"  Epoch {epoch}: Dropout = {module.p:.4f}")
                break
    
    print("\nRegularization module ready for Phase 6 experiments!")

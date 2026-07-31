"""
Phase 6: Regularization Strategies for Robust Training
Implements Dropout Scheduling, Data Augmentation with Kornia, and Robust Loss Functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import kornia.augmentation as K
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DropoutScheduler:
    """
    Dynamically adjusts dropout rates during training.
    
    Strategies:
    - 'linear': Linearly increase/decrease dropout
    - 'cosine': Cosine annealing of dropout rate
    - 'step': Step-wise changes at specific epochs
    """
    
    def __init__(
        self,
        model: nn.Module,
        initial_dropout: float = 0.1,
        final_dropout: float = 0.5,
        strategy: str = 'linear',
        warmup_epochs: int = 5,
        total_epochs: int = 100
    ):
        self.model = model
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Set initial dropout
        self._set_dropout_rate(initial_dropout)
    
    def _set_dropout_rate(self, rate: float):
        """Apply dropout rate to all Dropout layers in the model."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate
    
    def step(self, epoch: int):
        """Update dropout rate based on current epoch."""
        self.current_epoch = epoch
        
        if self.strategy == 'linear':
            if epoch < self.warmup_epochs:
                # Warmup: gradually increase from 0 to initial_dropout
                rate = (epoch / self.warmup_epochs) * self.initial_dropout
            else:
                # Linear increase to final_dropout
                progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                progress = min(1.0, max(0.0, progress))
                rate = self.initial_dropout + progress * (self.final_dropout - self.initial_dropout)
        
        elif self.strategy == 'cosine':
            import math
            if epoch < self.warmup_epochs:
                rate = (epoch / self.warmup_epochs) * self.initial_dropout
            else:
                progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                progress = min(1.0, max(0.0, progress))
                rate = self.final_dropout - 0.5 * (self.final_dropout - self.initial_dropout) * \
                       (1 + math.cos(math.pi * progress))
        
        elif self.strategy == 'step':
            if epoch < self.warmup_epochs:
                rate = (epoch / self.warmup_epochs) * self.initial_dropout
            elif epoch < self.total_epochs * 0.5:
                rate = self.initial_dropout
            elif epoch < self.total_epochs * 0.75:
                rate = (self.initial_dropout + self.final_dropout) / 2
            else:
                rate = self.final_dropout
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self._set_dropout_rate(rate)
        logger.info(f"Epoch {epoch}: Dropout rate set to {rate:.4f}")
        return rate


class DataAugmentationTrainer:
    """
    Trainer with Kornia-based on-the-fly data augmentation for regularization.
    
    Applies geometric and photometric perturbations during training
    to improve model robustness and generalization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        augment_config: Optional[Dict] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        # Configure Kornia augmentations
        self.augment_config = augment_config or self._default_config()
        self.train_augments = self._build_augmentations()
    
    def _default_config(self) -> Dict:
        """Default augmentation configuration."""
        return {
            'rotation': {'degrees': 15},
            'translation': {'translate': (0.1, 0.1)},
            'scale': {'scale': (0.9, 1.1)},
            'shear': {'degrees': 10},
            'brightness': {'brightness': 0.2},
            'contrast': {'contrast': 0.2},
            'hue': {'hue': 0.05},
            'gaussian_noise': {'std': 0.1},
            'random_erasing': {'p': 0.1, 'scale': (0.02, 0.33)},
        }
    
    def _build_augmentations(self) -> nn.ModuleList:
        """Build Kornia augmentation pipeline."""
        aug_list = nn.ModuleList()
        
        # Geometric augmentations
        if 'rotation' in self.augment_config:
            aug_list.append(K.RandomRotation(**self.augment_config['rotation']))
        
        if 'translation' in self.augment_config:
            aug_list.append(K.RandomTranslation(**self.augment_config['translation']))
        
        if 'scale' in self.augment_config:
            aug_list.append(K.RandomResizedSize(
                size=(512, 512),
                scale=self.augment_config['scale']['scale']
            ))
        
        if 'shear' in self.augment_config:
            aug_list.append(K.RandomShear(**self.augment_config['shear']))
        
        # Photometric augmentations
        if 'brightness' in self.augment_config:
            aug_list.append(K.RandomBrightness(**self.augment_config['brightness']))
        
        if 'contrast' in self.augment_config:
            aug_list.append(K.RandomContrast(**self.augment_config['contrast']))
        
        if 'hue' in self.augment_config:
            aug_list.append(K.RandomHue(**self.augment_config['hue']))
        
        # Noise augmentations
        if 'gaussian_noise' in self.augment_config:
            aug_list.append(K.RandomGaussianNoise(**self.augment_config['gaussian_noise']))
        
        if 'random_erasing' in self.augment_config:
            aug_list.append(K.RandomErasing(**self.augment_config['random_erasing']))
        
        logger.info(f"Built augmentation pipeline with {len(aug_list)} transforms")
        return aug_list
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        apply_augmentation: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch with optional data augmentation.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            apply_augmentation: Whether to apply Kornia augmentations
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Apply augmentations
            if apply_augmentation:
                for augment in self.train_augments:
                    images = augment(images)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.6f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.6f}")
        
        return {'loss': avg_loss}


def create_robust_loss(loss_type: str = 'huber', **kwargs) -> nn.Module:
    """
    Create robust loss functions for training.
    
    Args:
        loss_type: Type of loss ('huber', 'smooth_l1', 'l1', 'mse')
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function module
    """
    if loss_type == 'huber':
        delta = kwargs.get('delta', 1.0)
        logger.info(f"Using Huber Loss with delta={delta}")
        return nn.HuberLoss(delta=delta, reduction='mean')
    
    elif loss_type == 'smooth_l1':
        beta = kwargs.get('beta', 1.0)
        logger.info(f"Using Smooth L1 Loss with beta={beta}")
        return nn.SmoothL1Loss(beta=beta, reduction='mean')
    
    elif loss_type == 'l1':
        logger.info("Using L1 Loss")
        return nn.L1Loss(reduction='mean')
    
    elif loss_type == 'mse':
        logger.info("Using MSE Loss")
        return nn.MSELoss(reduction='mean')
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class RegularizedTrainingPipeline:
    """
    Complete training pipeline with multiple regularization strategies.
    
    Combines:
    - Dropout scheduling
    - Kornia data augmentation
    - Robust loss functions
    - Learning rate scheduling
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
        
        # Initialize components
        self.loss_fn = create_robust_loss(
            config.get('loss_type', 'huber'),
            **config.get('loss_kwargs', {})
        )
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Initialize dropout scheduler
        self.dropout_scheduler = DropoutScheduler(
            model=model,
            initial_dropout=config.get('initial_dropout', 0.1),
            final_dropout=config.get('final_dropout', 0.5),
            strategy=config.get('dropout_strategy', 'cosine'),
            warmup_epochs=config.get('warmup_epochs', 5),
            total_epochs=config.get('epochs', 100)
        )
        
        # Initialize augmentation trainer
        self.aug_trainer = DataAugmentationTrainer(
            model=model,
            device=device,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            augment_config=config.get('augment_config')
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Full training loop with regularization.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            save_path: Path to save best model
        
        Returns:
            Training history dictionary
        """
        epochs = self.config.get('epochs', 100)
        history = {'train_loss': [], 'val_loss': [], 'dropout_rates': []}
        best_val_loss = float('inf')
        
        logger.info(f"Starting regularized training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Loss: {self.config.get('loss_type', 'huber')}")
        
        for epoch in range(epochs):
            # Update dropout rate
            dropout_rate = self.dropout_scheduler.step(epoch)
            history['dropout_rates'].append(dropout_rate)
            
            # Train with augmentation
            train_metrics = self.aug_trainer.train_epoch(
                train_loader, epoch, apply_augmentation=True
            )
            history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    self._save_checkpoint(save_path, epoch, val_loss)
                    logger.info(f"Saved new best model at epoch {epoch} with val_loss={val_loss:.6f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.6f} | "
                    f"Val Loss: {history['val_loss'][-1] if val_loader else 'N/A':.6f} | "
                    f"Dropout: {dropout_rate:.4f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
                )
        
        logger.info("Training completed!")
        return history
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation without augmentation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)


# Convenience function for quick setup
def train_with_regularization(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    config: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> Tuple[nn.Module, Dict]:
    """
    Quick setup for regularized training.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        config: Configuration dictionary
        save_path: Path to save best model
    
    Returns:
        Trained model and training history
    """
    if config is None:
        config = {
            'epochs': 100,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'loss_type': 'huber',
            'initial_dropout': 0.1,
            'final_dropout': 0.5,
            'dropout_strategy': 'cosine',
            'warmup_epochs': 5,
            'augment_config': None
        }
    
    pipeline = RegularizedTrainingPipeline(model, device, config)
    history = pipeline.train(train_loader, val_loader, save_path)
    
    return model, history

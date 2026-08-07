import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from pathlib import Path
import math
from typing import List


class DropoutScheduler:
    """Schedules dropout rates during training to prevent initial shock to the network."""
    def __init__(self, model: nn.Module, strategy: str = 'cosine', initial_dropout: float = 0.0, final_dropout: float = 0.5, warmup_epochs: int = 5, total_epochs: int = 100):
        self.model = model
        self.strategy = strategy
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.dropout_layers = self._find_dropout_layers()
        
        # Initialize all dropout layers to the initial value
        self.step(0)

    def _find_dropout_layers(self) -> List[nn.Dropout]:
        return [module for module in self.model.modules() if isinstance(module, nn.Dropout)]

    def step(self, epoch: int) -> None:
        if not self.dropout_layers:
            return
            
        if epoch < self.warmup_epochs:
            current_dropout = self.initial_dropout
        else:
            progress = (epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
            progress = min(1.0, max(0.0, progress))
            
            if self.strategy == 'linear':
                current_dropout = self.initial_dropout + progress * (self.final_dropout - self.initial_dropout)
            elif self.strategy == 'cosine':
                current_dropout = self.initial_dropout + (self.final_dropout - self.initial_dropout) * (1 - math.cos(progress * math.pi)) / 2
            elif self.strategy == 'step':
                if progress < 0.25: current_dropout = self.initial_dropout
                elif progress < 0.5: current_dropout = self.initial_dropout + 0.3 * (self.final_dropout - self.initial_dropout)
                elif progress < 0.75: current_dropout = self.initial_dropout + 0.6 * (self.final_dropout - self.initial_dropout)
                else: current_dropout = self.final_dropout
            else:
                current_dropout = self.initial_dropout

        for layer in self.dropout_layers:
            layer.p = current_dropout


class BaseTrainer:
    """Base class to handle common trainer operations and Dropout Scheduling."""
    def _init_dropout_scheduler(self, model, num_epochs, use_scheduler, target_dropout):
        if use_scheduler and target_dropout > 0:
            print(f"Enabled DropoutScheduler (Target: {target_dropout}, Strategy: cosine, Warmup: 5 epochs)")
            self.dropout_scheduler = DropoutScheduler(
                model=model,
                strategy='cosine',
                initial_dropout=0.0,
                final_dropout=target_dropout,
                warmup_epochs=5,
                total_epochs=num_epochs
            )
        else:
            self.dropout_scheduler = None
            
    def _step_dropout(self, epoch):
        if self.dropout_scheduler:
            self.dropout_scheduler.step(epoch)


class EnhancementTrainer(BaseTrainer):
    """Training loop for document enhancement U-Net model with AMP and DataParallel."""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir='checkpoints/enhancement',
                 lr=1e-3, l1_weight=1.0, edge_weight=0.1, num_epochs=100, use_dropout_schedule=False, target_dropout=0.5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_dropout_scheduler(self.model, num_epochs, use_dropout_schedule, target_dropout)
        
        from src.training.losses import EnhancementLoss
        self.criterion = EnhancementLoss().to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            degraded = batch['rectified_input'].to(self.device, non_blocking=True)
            clean = batch['clean_target'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                output = self.model(degraded)
                loss = self.criterion(output, clean)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * degraded.size(0)
        
        return total_loss / len(self.train_loader.dataset)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.val_loader:
            degraded = batch['rectified_input'].to(self.device, non_blocking=True)
            clean = batch['clean_target'].to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                output = self.model(degraded)
                loss = self.criterion(output, clean)
                
            total_loss += loss.item() * degraded.size(0)
        
        return total_loss / len(self.val_loader.dataset)

    def train(self):
        print(f"Starting enhancement training for {self.num_epochs} epochs...")
        print(f"Save directory: {self.save_dir}")
        
        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            self._step_dropout(epoch)
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            elapsed = time.time() - start_time
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.save_dir / 'best_model.pth')
            
            print(f"Epoch {epoch:3d}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.6f}")
        return self.history


class CornerRegressionTrainer(BaseTrainer):
    """Training loop for corner detection using direct regression with AMP."""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir='checkpoints/corner_regression',
                 lr=1e-3, loss_type='l1', num_epochs=100, use_dropout_schedule=False, target_dropout=0.5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_dropout_scheduler(self.model, num_epochs, use_dropout_schedule, target_dropout)
        
        from src.training.losses import CornerLoss
        self.criterion = CornerLoss(type=loss_type).to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            image = batch['raw_photo'].to(self.device, non_blocking=True)
            corners = batch['corners'].view(batch['corners'].size(0), -1).to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                pred_corners = self.model(image)
                loss = self.criterion(pred_corners, corners)
                
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * image.size(0)
        
        return total_loss / len(self.train_loader.dataset)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.val_loader:
            image = batch['raw_photo'].to(self.device, non_blocking=True)
            corners = batch['corners'].view(batch['corners'].size(0), -1).to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                pred_corners = self.model(image)
                loss = self.criterion(pred_corners, corners)
                
            total_loss += loss.item() * image.size(0)
        
        return total_loss / len(self.val_loader.dataset)
    
    def train(self):
        print(f"Starting corner regression training for {self.num_epochs} epochs...")
        print(f"Save directory: {self.save_dir}")
        
        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            self._step_dropout(epoch)
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            elapsed = time.time() - start_time
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.save_dir / 'best_model.pth')
            
            print(f"Epoch {epoch:3d}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.6f}")
        return self.history


class CornerHeatmapTrainer(BaseTrainer):
    """Training loop for corner detection using heatmaps with AMP."""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir='checkpoints/corner_heatmap',
                 lr=1e-3, num_epochs=100, use_dropout_schedule=False, target_dropout=0.5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_dropout_scheduler(self.model, num_epochs, use_dropout_schedule, target_dropout)
        
        from src.training.losses import HeatmapLoss
        self.criterion = HeatmapLoss().to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    def _generate_target_heatmaps(self, corners, image_size, sigma=15.0):
        B, num_corners, _ = corners.shape
        H, W = image_size
        
        x_c = (corners[..., 0] * (W - 1)).view(B, num_corners, 1, 1)
        y_c = (corners[..., 1] * (H - 1)).view(B, num_corners, 1, 1)
        
        y = torch.arange(H, device=corners.device, dtype=torch.float32).view(1, 1, H, 1)
        x = torch.arange(W, device=corners.device, dtype=torch.float32).view(1, 1, 1, W)
        
        heatmaps = torch.exp(-((x - x_c)**2 + (y - y_c)**2) / (2 * sigma**2))
        
        return heatmaps

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            image = batch['raw_photo'].to(self.device, non_blocking=True)
            corners = batch['corners'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                heatmaps = self._generate_target_heatmaps(corners, image.shape[2:])
                _, pred_heatmaps = self.model(image)
                loss = self.criterion(pred_heatmaps, heatmaps)
                
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * image.size(0)
        
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.val_loader:
            image = batch['raw_photo'].to(self.device, non_blocking=True)
            corners = batch['corners'].to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                heatmaps = self._generate_target_heatmaps(corners, image.shape[2:])
                _, pred_heatmaps = self.model(image)
                loss = self.criterion(pred_heatmaps, heatmaps)
                
            total_loss += loss.item() * image.size(0)
        
        return total_loss / len(self.val_loader.dataset)
        
    def train(self):
        print(f"Starting corner heatmap training for {self.num_epochs} epochs...")
        print(f"Save directory: {self.save_dir}")
        
        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            self._step_dropout(epoch)
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            elapsed = time.time() - start_time
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.save_dir / 'best_model.pth')
            
            print(f"Epoch {epoch:3d}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.6f}")
        return self.history


def create_trainer(trainer_type, model, train_loader, val_loader, device, **kwargs):
    """Factory function to create appropriate trainer."""
    trainers = {
        'enhancement': EnhancementTrainer,
        'corner_regression': CornerRegressionTrainer,
        'corner_heatmap': CornerHeatmapTrainer,
    }
    
    if trainer_type not in trainers:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {list(trainers.keys())}")
    
    return trainers[trainer_type](model, train_loader, val_loader, device, **kwargs)


if __name__ == '__main__':
    import argparse
    import sys
    import os
    
    from src.models.model import EnhancementUNet, CornerRegressionModel, CornerHeatmapModel, init_weights
    from src.data.data_splitter import get_synthetic_splits
    
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description="Train Document Scanning Models (Phases 3, 4, 5, 6)")
    
    parser.add_argument("--task", type=str, required=True, 
                        choices=["enhancement", "corner_regression", "corner_heatmap"],
                        help="Which task to train")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Unused in training")
    parser.add_argument("--annotations", type=str, default="data/raw/real_photos/_annotations.coco.json", help="Unused")
    parser.add_argument("--clean-scans", type=str, default="data/clean_scans", help="Path to clean scans")
    parser.add_argument("--backgrounds", type=str, default="data/random_backgrounds", help="Path to random backgrounds")
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    
    # Phase 6 Regularization Arguments
    parser.add_argument("--dropout", type=float, default=0.0, help="Target dropout rate for regularization (0.0 for Phase 3/5)")
    parser.add_argument("--use-dropout-schedule", action="store_true", help="Enable dynamic dropout scheduling for Phase 6")
    
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from")

    args = parser.parse_args()

    # Hardware Optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"Number of GPUs available: {torch.cuda.device_count()}")

    print("Preparing Synthetic Dataset Splits...")
    try:
        train_ds, val_ds, test_ds = get_synthetic_splits(
            clean_scans_dir=args.clean_scans,
            backgrounds_dir=args.backgrounds,
            image_size=(args.image_size, args.image_size),
            seed=args.seed,
            num_eval_samples=100,
            train_samples_per_epoch=3500
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Initialize model with maximum possible dropout. 
    # If scheduler is used, it will immediately reset this to 0.0 internally on step 0.
    init_dropout = args.dropout if args.use_dropout_schedule else args.dropout
    print(f"Initializing {args.task} model...")
    
    if args.task == "enhancement":
        model = EnhancementUNet(dropout_rate=init_dropout)
    elif args.task == "corner_regression":
        model = CornerRegressionModel(dropout_rate=init_dropout)
    elif args.task == "corner_heatmap":
        model = CornerHeatmapModel(dropout_rate=init_dropout)
    else:
        raise ValueError("Invalid task")

    model.apply(init_weights)

    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading pre-trained weights from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            if list(state_dict.keys())[0].startswith('module.') and not isinstance(model, nn.DataParallel):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            model.load_state_dict(state_dict)
            print("Weights loaded successfully!")

    if torch.cuda.device_count() > 1:
        print(f"Wrapping model in DataParallel using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    trainer = create_trainer(
        trainer_type=args.task,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        lr=args.lr,
        num_epochs=args.epochs,
        use_dropout_schedule=args.use_dropout_schedule,
        target_dropout=args.dropout
    )

    trainer.train()
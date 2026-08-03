import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from pathlib import Path


class EnhancementTrainer:
    """Training loop for document enhancement U-Net model with AMP and DataParallel."""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir='checkpoints/enhancement',
                 lr=1e-3, l1_weight=1.0, edge_weight=0.1, num_epochs=100):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        from .losses import EnhancementLoss
        self.criterion = EnhancementLoss().to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        
        # Updated to new torch.amp API
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
            
            # Updated to new torch.amp API
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                output = self.model(degraded)
                loss = self.criterion(output, clean)
            
            self.scaler.scale(loss).backward()
            
            # Unscale before Gradient Clipping to prevent Exploding Gradients
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
                # Save unwrapped model state dict if using DataParallel
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
        
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.6f}")
        return self.history


class CornerRegressionTrainer:
    """Training loop for corner detection using direct regression with AMP."""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir='checkpoints/corner_regression',
                 lr=1e-3, loss_type='l1', num_epochs=100):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        from .losses import CornerLoss
        self.criterion = CornerLoss(type=loss_type).to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        # Updated API
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
            
            # Gradient Clipping
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
        
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.6f}")
        return self.history


class CornerHeatmapTrainer:
    """Training loop for corner detection using heatmaps with AMP."""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir='checkpoints/corner_heatmap',
                 lr=1e-3, num_epochs=100):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        from .losses import HeatmapLoss
        self.criterion = HeatmapLoss().to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        # Updated API
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
            
            # Gradient Clipping
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
    parser = argparse.ArgumentParser(description="Train Document Scanning Models (Phases 3 & 4)")
    
    parser.add_argument("--task", type=str, required=True, 
                        choices=["enhancement", "corner_regression", "corner_heatmap"],
                        help="Which task to train")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Unused in training")
    parser.add_argument("--annotations", type=str, default="data/raw/real_photos/_annotations.coco.json", help="Unused")
    parser.add_argument("--clean-scans", type=str, default="data/clean_scans", help="Path to clean scans")
    parser.add_argument("--backgrounds", type=str, default="data/random_backgrounds", help="Path to random backgrounds")
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for regularization")
    
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from")

    args = parser.parse_args()

    # Hardware Optimization: Enable CUDNN Benchmark
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
            train_samples_per_epoch=800
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure 'data/clean_scans' and 'data/random_backgrounds' directories exist and contain images.")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Initializing {args.task} model with Dropout = {args.dropout}...")
    if args.task == "enhancement":
        model = EnhancementUNet(dropout_rate=args.dropout)
    elif args.task == "corner_regression":
        model = CornerRegressionModel(dropout_rate=args.dropout)
    elif args.task == "corner_heatmap":
        model = CornerHeatmapModel(dropout_rate=args.dropout)
    else:
        raise ValueError("Invalid task")

    # Apply Kaiming Normal Weight Initialization
    model.apply(init_weights)

    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading pre-trained weights from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            if list(state_dict.keys())[0].startswith('module.') and not isinstance(model, nn.DataParallel):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            model.load_state_dict(state_dict)
            print("Weights loaded successfully!")
        else:
            print(f"Warning: Checkpoint not found at {args.resume}. Training from scratch.")

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
        num_epochs=args.epochs
    )

    trainer.train()
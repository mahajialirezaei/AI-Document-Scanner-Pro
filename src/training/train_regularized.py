import time
import math
import logging
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.model import EnhancementUNet, init_weights
from src.training.losses import EnhancementLoss
from src.data.data_splitter import get_synthetic_splits

logger = logging.getLogger(__name__)

class DropoutScheduler:
    def __init__(self, model: nn.Module, strategy: str = 'linear', initial_dropout: float = 0.0, final_dropout: float = 0.5, warmup_epochs: int = 5, total_epochs: int = 100):
        self.model = model
        self.strategy = strategy
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.dropout_layers = self._find_dropout_layers()

    def _find_dropout_layers(self) -> List[nn.Dropout]:
        return [module for module in self.model.modules() if isinstance(module, nn.Dropout)]

    def step(self, epoch: int) -> None:
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

class RegularizedEnhancementTrainer:
    def __init__(self, model, train_loader, val_loader, device, save_dir, lr=1e-3, epochs=60, dropout_schedule='cosine', final_dropout=0.5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        
        self.criterion = EnhancementLoss().to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        self.dropout_scheduler = DropoutScheduler(
            model=self.model, 
            strategy=dropout_schedule, 
            initial_dropout=0.0, 
            final_dropout=final_dropout, 
            warmup_epochs=5, 
            total_epochs=epochs
        )
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        self.best_val_loss = float('inf')

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
        print(f"Starting Regularized Enhancement Training for {self.epochs} epochs...")
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            self.dropout_scheduler.step(epoch)
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
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
                
            print(f"Epoch {epoch:3d}/{self.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 6: Regularized Training")
    parser.add_argument("--task", type=str, default="enhancement")
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--dropout-schedule", type=str, choices=['linear', 'cosine', 'step'], default='cosine')
    parser.add_argument("--clean-scans", type=str, required=True)
    parser.add_argument("--backgrounds", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_ds, val_ds, _ = get_synthetic_splits(
        clean_scans_dir=args.clean_scans,
        backgrounds_dir=args.backgrounds,
        image_size=(args.image_size, args.image_size),
        seed=42,
        num_eval_samples=100
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = EnhancementUNet(dropout_rate=0.0) 
    model.apply(init_weights)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    trainer = RegularizedEnhancementTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        lr=args.lr,
        epochs=args.epochs,
        dropout_schedule=args.dropout_schedule,
        final_dropout=args.dropout_rate
    )
    trainer.train()
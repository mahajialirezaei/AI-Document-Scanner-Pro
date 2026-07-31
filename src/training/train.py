import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from pathlib import Path


class EnhancementTrainer:
    """Training loop for document enhancement U-Net model."""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir='checkpoints/enhancement',
                 lr=1e-3, l1_weight=1.0, edge_weight=0.1, num_epochs=100):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        from .losses import EnhancementLoss
        self.criterion = EnhancementLoss(l1_weight=l1_weight, edge_weight=edge_weight).to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for degraded, clean in self.train_loader:
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(degraded)
            loss = self.criterion(output, clean)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * degraded.size(0)
        
        return total_loss / len(self.train_loader.dataset)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        for degraded, clean in self.val_loader:
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
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
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.save_dir / 'best_model.pth')
            
            print(f"Epoch {epoch:3d}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
        
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.6f}")
        return self.history


class CornerRegressionTrainer:
    """Training loop for corner detection using direct regression (Approach A)."""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir='checkpoints/corner_regression',
                 lr=1e-3, loss_type='l1', num_epochs=100):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        from .losses import CornerLoss
        self.criterion = CornerLoss(type=loss_type).to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            image = batch['image'].to(self.device)
            corners = batch['corners'].to(self.device)
            
            self.optimizer.zero_grad()
            pred_corners = self.model(image)
            loss = self.criterion(pred_corners, corners)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * image.size(0)
        
        return total_loss / len(self.train_loader.dataset)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.val_loader:
            image = batch['image'].to(self.device)
            corners = batch['corners'].to(self.device)
            
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
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.save_dir / 'best_model.pth')
            
            print(f"Epoch {epoch:3d}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
        
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.6f}")
        return self.history


class CornerHeatmapTrainer:
    """Training loop for corner detection using heatmaps (Approach B)."""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir='checkpoints/corner_heatmap',
                 lr=1e-3, num_epochs=100):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        from .losses import HeatmapLoss
        self.criterion = HeatmapLoss().to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            image = batch['image'].to(self.device)
            heatmaps = batch['heatmaps'].to(self.device)
            
            self.optimizer.zero_grad()
            pred_heatmaps = self.model(image)
            loss = self.criterion(pred_heatmaps, heatmaps)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * image.size(0)
        
        return total_loss / len(self.train_loader.dataset)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.val_loader:
            image = batch['image'].to(self.device)
            heatmaps = batch['heatmaps'].to(self.device)
            
            pred_heatmaps = self.model(image)
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
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
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

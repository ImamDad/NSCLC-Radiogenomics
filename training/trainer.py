"""
Training utilities for MS-HGNN
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class Trainer:
    """Trainer class for MS-HGNN model"""
    
    def __init__(self, model: nn.Module, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: dict,
                 device: str = 'cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        lr = config['training']['learning_rate']
        wd = config['training']['weight_decay']
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
        # Setup scheduler
        scheduler_type = config['training'].get('scheduler', 'CosineAnnealingLR')
        if scheduler_type == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=config['training']['epochs'],
                eta_min=config['training'].get('scheduler_kwargs', {}).get('eta_min', 1e-6)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Setup logging
        self.log_dir = config['logging']['log_dir']
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0
        self.patience_counter = 0
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_c_index': [],
            'val_auc': [],
            'learning_rates': []
        }
        
        # Move model to device
        self.model = self.model.to(device)
        
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_losses = {'loss_survival': [], 'loss_recurrence': []}
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            x_dict = {k: v.to(self.device) for k, v in batch['x_dict'].items()}
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # Forward pass
            outputs = self.model(x_dict)
            losses = self.model.get_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Log losses
            epoch_loss += losses['total_loss'].item()
            epoch_losses['loss_survival'].append(losses['loss_survival'].item())
            epoch_losses['loss_recurrence'].append(losses['loss_recurrence'].item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': losses['total_loss'].item(),
                's': losses['loss_survival'].item(),
                'r': losses['loss_recurrence'].item()
            })
        
        # Average losses
        avg_loss = epoch_loss / len(self.train_loader)
        avg_survival = np.mean(epoch_losses['loss_survival'])
        avg_recurrence = np.mean(epoch_losses['loss_recurrence'])
        
        return {
            'loss': avg_loss,
            'loss_survival': avg_survival,
            'loss_recurrence': avg_recurrence
        }
    
    def validate(self) -> Dict:
        """Validate the model"""
        from src.evaluation.evaluator import Evaluator
        
        evaluator = Evaluator(self.model, self.device)
        metrics = evaluator.evaluate(self.val_loader)
        
        return metrics
    
    def train(self, epochs: Optional[int] = None) -> Dict:
        """
        Full training loop
        
        Args:
            epochs: number of epochs (defaults to config value)
        Returns:
            history: training history
        """
        if epochs is None:
            epochs = self.config['training']['epochs']
        
        patience = self.config['training']['early_stopping_patience']
        min_delta = self.config['training']['early_stopping_min_delta']
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_c_index'].append(val_metrics.get('c_index', 0))
            self.history['val_auc'].append(val_metrics.get('auc', 0))
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print metrics
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val C-index: {val_metrics.get('c_index', 0):.4f}")
            print(f"  Val AUC: {val_metrics.get('auc', 0):.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            current_metric = val_metrics.get('c_index', 0)
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ New best model! C-index: {current_metric:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            # Save latest checkpoint
            if epoch % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Log to wandb
            if self.config['logging'].get('use_wandb', False):
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_c_index': val_metrics.get('c_index', 0),
                    'val_auc': val_metrics.get('auc', 0),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
        
        # Load best model
        self.load_checkpoint('best_model.pth')
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'history': self.history,
            'config': self.config
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename: str):
        """Load checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_val_metric = checkpoint['best_val_metric']
            self.history = checkpoint['history']
            print(f"  Checkpoint loaded: {path}")
            return True
        else:
            print(f"  Checkpoint not found: {path}")
            return False

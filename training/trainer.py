# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional, Any
import os
import json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt


class Trainer:
    """Trainer for MS-HHGN model"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
        
        # Move model to device
        self.model.to(device)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_survival_loss = 0
        total_recurrence_loss = 0
        all_survival_preds = []
        all_recurrence_preds = []
        all_survival_times = []
        all_events = []
        all_recurrence_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        for batch in pbar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(batch)
            
            # Compute loss
            loss_dict = self.model.compute_loss(outputs, batch)
            loss = loss_dict['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_survival_loss += loss_dict['survival'].item()
            total_recurrence_loss += loss_dict['recurrence'].item()
            
            all_survival_preds.append(outputs['survival_pred'].detach().cpu())
            all_recurrence_preds.append(outputs['recurrence_pred'].detach().cpu())
            all_survival_times.append(batch['survival_time'].cpu())
            all_events.append(batch['event_indicator'].cpu())
            if 'recurrence_label' in batch:
                all_recurrence_labels.append(batch['recurrence_label'].cpu())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'surv_loss': loss_dict['survival'].item(),
                'rec_loss': loss_dict['recurrence'].item()
            })
        
        # Concatenate predictions
        survival_preds = torch.cat(all_survival_preds).numpy().flatten()
        recurrence_preds = torch.cat(all_recurrence_preds).numpy().flatten()
        survival_times = torch.cat(all_survival_times).numpy()
        events = torch.cat(all_events).numpy()
        
        # Compute metrics
        c_index = concordance_index(survival_times, -survival_preds, events)
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'survival_loss': total_survival_loss / len(train_loader),
            'recurrence_loss': total_recurrence_loss / len(train_loader),
            'c_index': c_index
        }
        
        if all_recurrence_labels:
            recurrence_labels = torch.cat(all_recurrence_labels).numpy()
            auc = roc_auc_score(recurrence_labels, recurrence_preds)
            metrics['auc'] = auc
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        total_survival_loss = 0
        total_recurrence_loss = 0
        all_survival_preds = []
        all_recurrence_preds = []
        all_survival_times = []
        all_events = []
        all_recurrence_labels = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {self.current_epoch} [Val]'):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                loss_dict = self.model.compute_loss(outputs, batch)
                loss = loss_dict['total']
                
                # Track metrics
                total_loss += loss.item()
                total_survival_loss += loss_dict['survival'].item()
                total_recurrence_loss += loss_dict['recurrence'].item()
                
                all_survival_preds.append(outputs['survival_pred'].cpu())
                all_recurrence_preds.append(outputs['recurrence_pred'].cpu())
                all_survival_times.append(batch['survival_time'].cpu())
                all_events.append(batch['event_indicator'].cpu())
                all_uncertainties.append(outputs['uncertainty'].cpu())
                
                if 'recurrence_label' in batch:
                    all_recurrence_labels.append(batch['recurrence_label'].cpu())
        
        # Concatenate predictions
        survival_preds = torch.cat(all_survival_preds).numpy().flatten()
        recurrence_preds = torch.cat(all_recurrence_preds).numpy().flatten()
        survival_times = torch.cat(all_survival_times).numpy()
        events = torch.cat(all_events).numpy()
        uncertainties = torch.cat(all_uncertainties).numpy().flatten()
        
        # Compute metrics
        c_index = concordance_index(survival_times, -survival_preds, events)
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'survival_loss': total_survival_loss / len(val_loader),
            'recurrence_loss': total_recurrence_loss / len(val_loader),
            'c_index': c_index,
            'mean_uncertainty': uncertainties.mean()
        }
        
        if all_recurrence_labels:
            recurrence_labels = torch.cat(all_recurrence_labels).numpy()
            auc = roc_auc_score(recurrence_labels, recurrence_preds)
            
            # Compute precision, recall, f1 at optimal threshold
            precisions, recalls, thresholds = precision_recall_fscore_support(
                recurrence_labels, 
                (recurrence_preds > 0.5).astype(int),
                average='binary'
            )[:3]
            
            metrics.update({
                'auc': auc,
                'precision': precisions,
                'recall': recalls,
                'f1': precisions * 2 * recalls / (precisions + recalls + 1e-8)
            })
        
        return metrics
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, 
            callbacks: List = None, use_wandb: bool = False):
        """Train the model"""
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            if use_wandb:
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'train_c_index': train_metrics['c_index'],
                    'val_loss': val_metrics['loss'],
                    'val_c_index': val_metrics['c_index'],
                    'val_auc': val_metrics.get('auc', 0),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Print metrics
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, C-index: {train_metrics['c_index']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, C-index: {val_metrics['c_index']:.4f}, "
                  f"AUC: {val_metrics.get('auc', 0):.4f}")
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss - self.config.early_stopping_min_delta:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
                print(f"  ✓ New best model! Loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement for {self.patience_counter} epochs")
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("\nRestored best model from checkpoint")
    
    def test(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate on test set"""
        self.model.eval()
        
        all_survival_preds = []
        all_recurrence_preds = []
        all_survival_times = []
        all_events = []
        all_recurrence_labels = []
        all_uncertainties = []
        all_patient_ids = []
        all_attention_weights = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass with attention
                outputs = self.model(batch, return_attention=True)
                
                all_survival_preds.append(outputs['survival_pred'].cpu())
                all_recurrence_preds.append(outputs['recurrence_pred'].cpu())
                all_survival_times.append(batch['survival_time'].cpu())
                all_events.append(batch['event_indicator'].cpu())
                all_uncertainties.append(outputs['uncertainty'].cpu())
                all_patient_ids.extend(batch['patient_ids'])
                
                if 'recurrence_label' in batch:
                    all_recurrence_labels.append(batch['recurrence_label'].cpu())
                
                if 'cross_attention' in outputs:
                    all_attention_weights.append(outputs['cross_attention'])
        
        # Concatenate
        survival_preds = torch.cat(all_survival_preds).numpy().flatten()
        recurrence_preds = torch.cat(all_recurrence_preds).numpy().flatten()
        survival_times = torch.cat(all_survival_times).numpy()
        events = torch.cat(all_events).numpy()
        uncertainties = torch.cat(all_uncertainties).numpy().flatten()
        
        # Compute metrics
        c_index = concordance_index(survival_times, -survival_preds, events)
        
        results = {
            'c_index': c_index,
            'survival_preds': survival_preds,
            'recurrence_preds': recurrence_preds,
            'survival_times': survival_times,
            'events': events,
            'uncertainties': uncertainties,
            'patient_ids': all_patient_ids
        }
        
        if all_recurrence_labels:
            recurrence_labels = torch.cat(all_recurrence_labels).numpy()
            auc = roc_auc_score(recurrence_labels, recurrence_preds)
            results['auc'] = auc
            results['recurrence_labels'] = recurrence_labels
        
        if all_attention_weights:
            results['attention_weights'] = all_attention_weights
        
        return results
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device"""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        batch[key][sub_key] = sub_value.to(self.device)
        return batch
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'current_epoch': self.current_epoch,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.current_epoch = checkpoint['current_epoch']
        print(f"Checkpoint loaded from {path}")

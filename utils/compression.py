# utils/compression.py
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional
import copy


class ModelCompressor:
    """Model compression utilities"""
    
    @staticmethod
    def apply_unstructured_pruning(model: nn.Module, amount: float = 0.5):
        """Apply unstructured pruning to remove 50% of weights"""
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        
        # Make pruning permanent
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
        
        return model
    
    @staticmethod
    def quantize_model(model: nn.Module, bits: int = 8):
        """Quantize model to INT8 (using PyTorch quantization)"""
        # This is a simplified version - in practice use torch.quantization
        
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model, inplace=False)
        
        # Calibration would be needed here with representative data
        
        model_quantized = torch.quantization.convert(model_prepared, inplace=False)
        
        return model_quantized
    
    @staticmethod
    def knowledge_distillation(teacher_model: nn.Module, student_model: nn.Module,
                              train_loader, num_epochs: int = 10,
                              temperature: float = 3.0, alpha: float = 0.7):
        """Distill knowledge from teacher to student"""
        device = next(teacher_model.parameters()).device
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        
        teacher_model.eval()
        student_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in train_loader:
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(batch)
                    teacher_logits = teacher_outputs['recurrence_logit'] / temperature
                    teacher_probs = torch.softmax(teacher_logits, dim=-1)
                
                # Student predictions
                student_outputs = student_model(batch)
                student_logits = student_outputs['recurrence_logit'] / temperature
                student_log_probs = torch.log_softmax(student_logits, dim=-1)
                
                # Distillation loss
                loss_distill = criterion_kl(student_log_probs, teacher_probs) * (temperature ** 2)
                
                # Ground truth loss (optional)
                if 'recurrence_label' in batch:
                    loss_ce = nn.BCEWithLogitsLoss()(
                        student_outputs['recurrence_logit'].squeeze(),
                        batch['recurrence_label']
                    )
                    loss = alpha * loss_distill + (1 - alpha) * loss_ce
                else:
                    loss = loss_distill
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Distillation Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        return student_model
    
    @staticmethod
    def create_lite_version(original_model, config, cohort_correlations=None):
        """Create a lightweight version with reduced parameters"""
        from models.ms_hhgn import MSHHGNLite
        
        lite_model = MSHHGNLite(config, cohort_correlations)
        
        # Copy weights where dimensions match
        original_state = original_model.state_dict()
        lite_state = lite_model.state_dict()
        
        for name, param in lite_state.items():
            if name in original_state and param.shape == original_state[name].shape:
                param.data.copy_(original_state[name].data)
        
        return lite_model

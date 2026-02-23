# models/uncertainty.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class UncertaintyAwareFusion(nn.Module):
    """Level 3: Late fusion with uncertainty quantification"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.modalities = config.modalities
        self.hidden_dim = config.total_hidden_dim
        
        # Modality-specific prediction heads
        self.prediction_heads = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(config.mc_dropout_rate),
                nn.Linear(64, 1)  # Survival hazard or recurrence logit
            ) for mod in self.modalities
        })
        
        # Learnable modality importance
        self.modality_importance = nn.ParameterDict({
            mod: nn.Parameter(torch.ones(1)) for mod in self.modalities
        })
        
        # Temperature parameter for softmax weighting
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Monte Carlo dropout
        self.mc_dropout_rate = config.mc_dropout_rate
        self.num_mc_passes = config.num_mc_dropout_passes
        
    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            modality_embeddings: Dictionary mapping modalities to pooled embeddings (B, d)
        Returns:
            final_prediction: Weighted fusion of modality predictions (B, 1)
            uncertainty: Total prediction uncertainty (B, 1)
            modality_stats: Dictionary with per-modality predictions and uncertainties
        """
        batch_size = next(iter(modality_embeddings.values())).size(0)
        device = next(iter(modality_embeddings.values())).device
        
        # Enable dropout for MC sampling
        self.train()  # Set to train mode to enable dropout
        
        # Collect MC samples for each modality
        mc_samples = {mod: [] for mod in self.modalities if mod in modality_embeddings}
        
        for _ in range(self.num_mc_passes):
            for mod, emb in modality_embeddings.items():
                # Apply dropout manually
                dropped = F.dropout(emb, self.mc_dropout_rate, training=True)
                pred = self.prediction_heads[mod](dropped)
                mc_samples[mod].append(pred)
        
        # Compute mean and variance for each modality
        modality_means = {}
        modality_vars = {}
        modality_confs = {}
        
        for mod, samples in mc_samples.items():
            samples_tensor = torch.stack(samples, dim=0)  # (T, B, 1)
            mean = samples_tensor.mean(dim=0)  # (B, 1)
            var = samples_tensor.var(dim=0)  # (B, 1)
            
            modality_means[mod] = mean
            modality_vars[mod] = var
            modality_confs[mod] = torch.exp(-var)  # Confidence score
        
        # Compute fusion weights
        weights = []
        weight_nominator = []
        
        for mod in self.modalities:
            if mod in modality_means:
                importance = torch.exp(self.modality_importance[mod])  # Ensure positivity
                confidence = modality_confs[mod]
                weight = importance * confidence
                weight_nominator.append(weight)
                weights.append(weight)
            else:
                weights.append(torch.zeros(batch_size, 1).to(device))
        
        # Normalize weights with temperature
        weights_tensor = torch.stack(weights, dim=1)  # (B, n_mods, 1)
        temperature = F.softplus(self.temperature)  # Ensure positivity
        weights_tensor = F.softmax(weights_tensor / temperature, dim=1)
        
        # Compute final prediction
        predictions = []
        for i, mod in enumerate(self.modalities):
            if mod in modality_means:
                predictions.append(modality_means[mod])
            else:
                predictions.append(torch.zeros(batch_size, 1).to(device))
        
        predictions_tensor = torch.stack(predictions, dim=1)  # (B, n_mods, 1)
        final_prediction = torch.sum(weights_tensor * predictions_tensor, dim=1)  # (B, 1)
        
        # Compute total uncertainty
        total_variance = torch.zeros(batch_size, 1).to(device)
        for i, mod in enumerate(self.modalities):
            if mod in modality_vars:
                total_variance += weights_tensor[:, i:i+1] ** 2 * modality_vars[mod]
        
        uncertainty = torch.sqrt(total_variance)  # Standard deviation
        
        # Compute 95% confidence intervals
        ci_lower = final_prediction - 1.96 * uncertainty
        ci_upper = final_prediction + 1.96 * uncertainty
        
        modality_stats = {
            'means': modality_means,
            'variances': modality_vars,
            'confidences': modality_confs,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        self.eval()  # Set back to eval mode
        
        return final_prediction, uncertainty, modality_stats


class MultiTaskLoss(nn.Module):
    """Multi-task learning objective with uncertainty weighting"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Learnable task uncertainties
        self.log_survival_var = nn.Parameter(torch.zeros(1))
        self.log_recurrence_var = nn.Parameter(torch.zeros(1))
        self.log_regularization_var = nn.Parameter(torch.zeros(1))
        
    def forward(self, survival_pred: torch.Tensor, recurrence_pred: torch.Tensor,
                survival_time: torch.Tensor, event_indicator: torch.Tensor,
                recurrence_label: torch.Tensor, model_params: List) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with uncertainty weighting
        """
        # Survival loss (negative log partial likelihood)
        survival_loss = self._cox_loss(survival_pred, survival_time, event_indicator)
        
        # Recurrence loss (focal loss)
        recurrence_loss = self._focal_loss(recurrence_pred, recurrence_label)
        
        # Regularization loss
        regularization_loss = self._regularization_loss(model_params)
        
        # Uncertainty weighting
        survival_precision = torch.exp(-self.log_survival_var)
        recurrence_precision = torch.exp(-self.log_recurrence_var)
        regularization_precision = torch.exp(-self.log_regularization_var)
        
        total_loss = (
            survival_precision * survival_loss +
            recurrence_precision * recurrence_loss +
            regularization_precision * regularization_loss +
            self.log_survival_var + self.log_recurrence_var + self.log_regularization_var
        )
        
        return {
            'total': total_loss,
            'survival': survival_loss,
            'recurrence': recurrence_loss,
            'regularization': regularization_loss,
            'survival_weight': survival_precision.detach(),
            'recurrence_weight': recurrence_precision.detach()
        }
    
    def _cox_loss(self, hazard_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """Negative log partial likelihood for Cox PH model"""
        # Sort by time
        idx = time.sort(descending=True)[1]
        hazard_pred = hazard_pred[idx].squeeze()
        event = event[idx]
        time = time[idx]
        
        loss = 0.0
        n_events = 0
        
        for i in range(len(time)):
            if event[i] == 1:
                # Risk set: patients with time >= current time
                risk_set = hazard_pred[i:]  # Since sorted descending, all later times are <=
                
                # Partial likelihood contribution
                loss += hazard_pred[i] - torch.logsumexp(risk_set, dim=0)
                n_events += 1
        
        return -loss / (n_events + 1e-8)
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss for imbalanced classification"""
        pred = pred.squeeze()
        target = target.float()
        
        # Binary cross-entropy
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Focal loss modulation
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.config.focal_loss_gamma
        
        # Class balancing
        alpha = self.config.focal_loss_alpha
        alpha_weight = alpha * target + (1 - alpha) * (1 - target)
        
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def _regularization_loss(self, model_params: List) -> torch.Tensor:
        """L2 regularization with edge dropout consistency"""
        l2_loss = 0.0
        for param in model_params:
            l2_loss += torch.norm(param, p=2) ** 2
        
        return self.config.l2_weight_decay * l2_loss

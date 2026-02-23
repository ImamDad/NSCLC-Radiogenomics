# models/ms_hhgn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .encoders import ModalityEncoders
from .attention import CrossModalAttention
from .graph import HeterogeneousGraphConstructor, GraphFusionNetwork
from .uncertainty import UncertaintyAwareFusion, MultiTaskLoss


class MSHHGN(nn.Module):
    """
    Multi-Scale Hierarchical Heterogeneous Graph Network
    for NSCLC survival prediction and recurrence classification
    """
    
    def __init__(self, config, cohort_correlations: Dict[str, np.ndarray] = None):
        super().__init__()
        
        self.config = config
        
        # Level 1: Modality encoders
        self.encoders = ModalityEncoders(config)
        
        # Level 1: Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(config)
        
        # Level 2: Graph constructor
        self.graph_constructor = HeterogeneousGraphConstructor(config)
        self.cohort_correlations = cohort_correlations or {}
        
        # Level 2: Graph fusion network
        self.graph_fusion = GraphFusionNetwork(config, self.graph_constructor)
        
        # Level 3: Uncertainty-aware fusion
        self.uncertainty_fusion = UncertaintyAwareFusion(config)
        
        # Multi-task loss
        self.loss_fn = MultiTaskLoss(config)
        
        # Output heads
        self.survival_head = nn.Sequential(
            nn.Linear(config.total_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Hazard score
        )
        
        self.recurrence_head = nn.Sequential(
            nn.Linear(config.total_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Recurrence logit
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, batch: Dict[str, Any], return_attention: bool = False) -> Dict[str, Any]:
        """
        Forward pass through MS-HHGN
        
        Args:
            batch: Dictionary containing:
                - modality_data: Raw modality inputs
                - patient_graphs: Pre-computed patient graphs
                - survival_time: Survival times
                - event_indicator: Censoring indicators
                - recurrence_label: Recurrence labels (optional)
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with predictions and optional attention weights
        """
        modality_data = batch['modality_data']
        
        # Apply modality dropout during training
        if self.training and self.config.modality_dropout_prob > 0:
            modality_data = self._apply_modality_dropout(modality_data)
        
        # Level 1: Encode each modality
        embeddings = self.encoders(modality_data)
        
        # Level 1: Cross-modal attention
        early_fused, cross_attention = self.cross_modal_attention(embeddings)
        
        # Level 2: Graph-based fusion
        patient_graphs = batch.get('patient_graphs', [])
        if not patient_graphs:
            # Construct graphs on-the-fly if not provided
            patient_graphs = []
            for i in range(len(batch['patient_ids'])):
                patient_data = {
                    'patient_id': batch['patient_ids'][i],
                    'CT_features': batch.get('ct_features', [None])[i],
                    'PET_features': batch.get('pet_features', [None])[i],
                    'clinical_features': batch.get('clinical_features', [None])[i],
                    'pathway_scores': batch.get('pathway_scores', [None])[i]
                }
                graph = self.graph_constructor.build_graph(patient_data, self.cohort_correlations)
                patient_graphs.append(graph)
        
        graph_embedding, meta_path_weights = self.graph_fusion(patient_graphs, early_fused)
        
        # Level 3: Uncertainty-aware fusion
        # First, get modality-specific pooled embeddings from graph
        modality_pooled = self._pool_modality_embeddings(patient_graphs, graph_embedding)
        
        # Apply uncertainty-aware fusion
        final_pred, uncertainty, modality_stats = self.uncertainty_fusion(modality_pooled)
        
        # Generate survival and recurrence predictions
        survival_pred = self.survival_head(graph_embedding)
        recurrence_logit = self.recurrence_head(graph_embedding)
        recurrence_prob = torch.sigmoid(recurrence_logit)
        
        outputs = {
            'survival_pred': survival_pred,
            'recurrence_pred': recurrence_prob,
            'recurrence_logit': recurrence_logit,
            'graph_embedding': graph_embedding,
            'early_fused': early_fused,
            'uncertainty': uncertainty,
            'final_pred': final_pred,
            'modality_stats': modality_stats
        }
        
        if return_attention:
            outputs['cross_attention'] = cross_attention
            outputs['meta_path_weights'] = meta_path_weights
        
        return outputs
    
    def _apply_modality_dropout(self, modality_data: Dict) -> Dict:
        """Randomly mask out modalities during training"""
        if not self.training:
            return modality_data
        
        masked_data = {}
        for mod, data in modality_data.items():
            if torch.rand(1).item() < self.config.modality_dropout_prob:
                # Zero out this modality
                if data is not None:
                    masked_data[mod] = torch.zeros_like(data)
                else:
                    masked_data[mod] = None
            else:
                masked_data[mod] = data
        
        return masked_data
    
    def _pool_modality_embeddings(self, patient_graphs: List[Dict], 
                                  graph_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract modality-specific pooled embeddings from graphs"""
        batch_size = len(patient_graphs)
        device = graph_embedding.device
        
        pooled = {mod: [] for mod in self.config.modalities}
        
        for i, graph in enumerate(patient_graphs):
            # CT features
            if 'CT_feature' in graph['node_features']:
                ct_feats = graph['node_features']['CT_feature'].to(device)
                pooled['CT'].append(ct_feats.mean(dim=0))
            else:
                pooled['CT'].append(torch.zeros(self.config.total_hidden_dim).to(device))
            
            # PET features
            if 'PET_feature' in graph['node_features']:
                pet_feats = graph['node_features']['PET_feature'].to(device)
                pooled['PET'].append(pet_feats.mean(dim=0))
            else:
                pooled['PET'].append(torch.zeros(self.config.total_hidden_dim).to(device))
            
            # Clinical features
            if 'clinical_feature' in graph['node_features']:
                clinical_feats = graph['node_features']['clinical_feature'].to(device)
                pooled['Clinical'].append(clinical_feats.mean(dim=0))
            else:
                pooled['Clinical'].append(torch.zeros(self.config.total_hidden_dim).to(device))
            
            # Genomic pathways
            if 'genomic_pathway' in graph['node_features']:
                genomic_feats = graph['node_features']['genomic_pathway'].to(device)
                pooled['Genomic'].append(genomic_feats.mean(dim=0))
            else:
                pooled['Genomic'].append(torch.zeros(self.config.total_hidden_dim).to(device))
        
        # Stack across batch
        for mod in pooled:
            pooled[mod] = torch.stack(pooled[mod], dim=0)
        
        return pooled
    
    def compute_loss(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        survival_pred = outputs['survival_pred']
        recurrence_pred = outputs['recurrence_pred']
        
        survival_time = batch['survival_time']
        event_indicator = batch['event_indicator']
        recurrence_label = batch.get('recurrence_label', torch.zeros_like(event_indicator))
        
        # Get model parameters for regularization
        model_params = [p for p in self.parameters() if p.requires_grad]
        
        loss_dict = self.loss_fn(
            survival_pred, recurrence_pred,
            survival_time, event_indicator,
            recurrence_label, model_params
        )
        
        return loss_dict


class MSHHGNLite(MSHHGN):
    """
    Compressed version of MS-HHGN with pruning and quantization
    """
    
    def __init__(self, config, cohort_correlations=None):
        super().__init__(config, cohort_correlations)
        
        # Apply pruning mask (will be set after training)
        self.pruning_mask = None
        
    def apply_pruning(self, pruning_threshold: float = 0.01):
        """Apply unstructured pruning to remove small weights"""
        self.pruning_mask = {}
        
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Create binary mask
                mask = (torch.abs(param) > pruning_threshold).float()
                self.pruning_mask[name] = mask
                
                # Apply mask
                param.data = param.data * mask
    
    def forward(self, batch, return_attention=False):
        """Forward pass with pruning mask applied"""
        if self.pruning_mask is not None:
            # Apply pruning mask during forward pass
            for name, param in self.named_parameters():
                if name in self.pruning_mask:
                    param.data = param.data * self.pruning_mask[name]
        
        return super().forward(batch, return_attention)
    
    def quantize(self, bits: int = 8):
        """Quantize model to INT8 (simulated)"""
        # In practice, use torch.quantization
        # This is a simplified version for demonstration
        
        scale = 2**bits - 1
        
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                # Simulate quantization by scaling and rounding
                if hasattr(module, 'weight'):
                    weight_range = module.weight.data.max() - module.weight.data.min()
                    if weight_range > 0:
                        module.weight.data = torch.round(
                            (module.weight.data - module.weight.data.min()) / weight_range * scale
                        ) / scale * weight_range + module.weight.data.min()
        
        return self

"""
MS-HGNN: Multi-Scale Hierarchical Graph Neural Network
Main model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CrossModalAttention(nn.Module):
    """Level 1: Cross-Modal Attention for early feature fusion"""
    
    def __init__(self, embedding_dim: int = 64, attention_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.dropout = dropout
        
        # Query and key projections
        self.W_q = nn.Linear(embedding_dim, attention_dim)
        self.W_k = nn.Linear(embedding_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)
        self.b = nn.Parameter(torch.zeros(attention_dim))
        
        # Value projections for each modality pair
        self.W_qk = nn.ModuleDict({
            f"{i}_{j}": nn.Linear(embedding_dim, embedding_dim)
            for i in range(4) for j in range(4) if i != j
        })
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: [n_modalities, batch_size, embedding_dim]
        Returns:
            F: [n_modalities, batch_size, embedding_dim]
        """
        n_mod, batch_size, emb_dim = H.shape
        
        # Compute attention scores
        scores = torch.zeros(n_mod, n_mod, batch_size, device=H.device)
        
        for q in range(n_mod):
            for k in range(n_mod):
                if q == k:
                    continue
                # Additive attention
                q_proj = self.W_q(H[q])  # [batch, attention_dim]
                k_proj = self.W_k(H[k])  # [batch, attention_dim]
                combined = torch.tanh(q_proj + k_proj + self.b)
                scores[q, k] = self.v(combined).squeeze(-1)  # [batch]
        
        # Softmax normalization (no self-attention)
        mask = torch.eye(n_mod, device=H.device).bool()
        scores_masked = scores.clone()
        scores_masked[mask] = -1e9
        alpha = F.softmax(scores_masked, dim=0)  # [n_mod, n_mod, batch]
        
        # Apply attention
        F_out = torch.zeros_like(H)
        for q in range(n_mod):
            # Residual connection
            F_out[q] = H[q]
            for k in range(n_mod):
                if q == k:
                    continue
                # Weighted value projection
                key = f"{q}_{k}"
                val = self.W_qk[key](H[k])  # [batch, embedding_dim]
                weight = alpha[q, k].unsqueeze(-1)  # [batch, 1]
                F_out[q] = F_out[q] + weight * val
        
        # Layer norm and dropout
        F_out = self.layer_norm(F_out)
        F_out = self.dropout_layer(F_out)
        
        return F_out


class SemanticAttention(nn.Module):
    """Level 2: Semantic attention for meta-path weighting"""
    
    def __init__(self, embedding_dim: int = 512, semantic_dim: int = 128, 
                 num_meta_paths: int = 3, dropout: float = 0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.semantic_dim = semantic_dim
        self.num_meta_paths = num_meta_paths
        self.dropout = dropout
        
        self.W_s = nn.Linear(embedding_dim, semantic_dim)
        self.q_s = nn.Parameter(torch.randn(semantic_dim))
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, h_graphs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_graphs: [batch_size, num_meta_paths, embedding_dim]
        Returns:
            h_final: [batch_size, embedding_dim]
            beta: [batch_size, num_meta_paths] - attention weights
        """
        # Transform to semantic space
        z = torch.tanh(self.W_s(h_graphs))  # [batch, n_paths, semantic_dim]
        
        # Compute importance scores
        w = torch.einsum('bpd,d->bp', z, self.q_s)  # [batch, n_paths]
        
        # Softmax
        beta = F.softmax(w, dim=1)  # [batch, n_paths]
        
        # Weighted sum
        h_final = torch.einsum('bp,bpd->bd', beta, h_graphs)  # [batch, embedding_dim]
        h_final = self.dropout_layer(h_final)
        
        return h_final, beta


class HeterogeneousGraphLayer(nn.Module):
    """Graph attention layer for heterogeneous graphs"""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8, 
                 dropout: float = 0.3, activation: str = 'relu'):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dropout = dropout
        
        # Multi-head attention
        self.attn_layers = nn.ModuleList([
            GATConv(in_dim, out_dim // num_heads, heads=1, dropout=dropout)
            for _ in range(num_heads)
        ])
        
        # Residual connection
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [n_nodes, in_dim]
            edge_index: [2, n_edges]
            edge_type: [n_edges] - optional, for heterogeneous edges
        Returns:
            x_out: [n_nodes, out_dim]
        """
        # Multi-head attention
        head_outputs = []
        for attn in self.attn_layers:
            h = attn(x, edge_index)
            head_outputs.append(h)
        
        # Concatenate heads
        x_attn = torch.cat(head_outputs, dim=1)
        
        # Residual + norm
        x_res = self.residual(x)
        x_out = self.layer_norm(x_attn + x_res)
        x_out = self.activation(x_out)
        x_out = self.dropout_layer(x_out)
        
        return x_out


class MSHGNN(nn.Module):
    """
    Main MS-HGNN model
    Multi-Scale Hierarchical Graph Neural Network for NSCLC prognosis
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embedding_dim = config['model']['embedding_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.num_heads = config['model']['num_heads']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        self.num_meta_paths = 3  # immune, proliferation, treatment
        
        # Modality encoders (simplified - actual encoders would be more complex)
        self.modality_encoders = nn.ModuleDict({
            'ct': nn.Sequential(
                nn.Linear(131, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.embedding_dim)
            ),
            'pet': nn.Sequential(
                nn.Linear(131, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.embedding_dim)
            ),
            'clinical': nn.Sequential(
                nn.Linear(50, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, self.embedding_dim)
            ),
            'genomic': nn.Sequential(
                nn.Linear(50, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.embedding_dim)
            )
        })
        
        # Level 1: Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            embedding_dim=self.embedding_dim,
            attention_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        # Level 2: Graph layers
        self.graph_layers = nn.ModuleList([
            HeterogeneousGraphLayer(
                in_dim=self.embedding_dim,
                out_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Graph readout
        self.graph_readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
        # Semantic attention
        self.semantic_attention = SemanticAttention(
            embedding_dim=self.hidden_dim,
            semantic_dim=128,
            num_meta_paths=self.num_meta_paths,
            dropout=self.dropout
        )
        
        # Level 3: Task heads
        self.survival_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        self.recurrence_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Uncertainty log variances (for multi-task learning)
        self.log_var_survival = nn.Parameter(torch.zeros(1))
        self.log_var_recurrence = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_modalities(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode each modality to embedding
        Args:
            x_dict: dict with keys 'ct', 'pet', 'clinical', 'genomic'
        Returns:
            H: [4, batch_size, embedding_dim]
        """
        embeddings = []
        modalities = ['ct', 'pet', 'clinical', 'genomic']
        
        for mod in modalities:
            if mod in x_dict and x_dict[mod] is not None:
                h = self.modality_encoders[mod](x_dict[mod])
                embeddings.append(h.unsqueeze(0))
            else:
                # Zero embedding for missing modality
                h = torch.zeros(x_dict[list(x_dict.keys())[0]].shape[0], self.embedding_dim,
                              device=next(self.parameters()).device)
                embeddings.append(h.unsqueeze(0))
        
        return torch.cat(embeddings, dim=0)  # [4, batch, embedding_dim]
    
    def build_graph(self, features: torch.Tensor, meta_paths: List) -> Tuple[Data, List]:
        """
        Build heterogeneous graph for a batch
        Args:
            features: [batch_size, feature_dim]
            meta_paths: list of meta-path definitions
        Returns:
            graph: PyG Data object
            path_embeddings: list of meta-path embeddings
        """
        # Simplified graph construction for demonstration
        # In practice, this would build complex heterogeneous graphs
        batch_size = features.shape[0]
        n_nodes = batch_size + 5  # patient + feature nodes
        
        # Create node features
        node_features = torch.zeros(n_nodes, self.hidden_dim)
        node_features[:batch_size] = features
        
        # Create edges (simplified)
        edge_index = []
        for i in range(batch_size):
            # Connect patient to feature nodes
            for j in range(5):
                edge_index.append([i, batch_size + j])
                edge_index.append([batch_size + j, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create graph
        graph = Data(x=node_features, edge_index=edge_index)
        
        # Meta-path embeddings (simplified)
        path_embeddings = [
            node_features.mean(dim=0, keepdim=True).repeat(batch_size, 1)
            for _ in range(self.num_meta_paths)
        ]
        
        return graph, path_embeddings
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MS-HGNN
        
        Args:
            x_dict: dictionary of modality inputs
            return_attention: whether to return attention weights
            
        Returns:
            outputs: dict containing:
                - survival: [batch_size, 1]
                - recurrence: [batch_size, 1]
                - survival_uncertainty: [batch_size, 1]
                - recurrence_uncertainty: [batch_size, 1]
                - beta: [batch_size, 3] (optional)
                - cross_attention: [4, 4, batch] (optional)
        """
        # Level 1: Encode modalities
        H = self.encode_modalities(x_dict)  # [4, batch, embedding_dim]
        
        # Level 1: Cross-modal attention
        F = self.cross_modal_attention(H)  # [4, batch, embedding_dim]
        
        # Aggregate features
        features = F.mean(dim=0)  # [batch, embedding_dim]
        
        # Level 2: Graph construction and processing
        meta_paths = []  # Define meta-paths here
        graph, path_embeddings = self.build_graph(features, meta_paths)
        
        # Process graph through GNN layers
        x_graph = graph.x
        for layer in self.graph_layers:
            x_graph = layer(x_graph, graph.edge_index)
        
        # Readout (take patient nodes)
        x_patient = x_graph[:features.shape[0]]  # [batch, hidden_dim]
        h_graphs = torch.stack(path_embeddings, dim=1)  # [batch, n_paths, hidden_dim]
        
        # Semantic attention
        h_final, beta = self.semantic_attention(h_graphs)  # [batch, hidden_dim], [batch, n_paths]
        
        # Level 3: Task predictions with uncertainty
        # Add Monte Carlo dropout for uncertainty estimation
        if self.training:
            # During training, use standard forward pass
            survival_logits = self.survival_head(h_final)
            recurrence_logits = self.recurrence_head(h_final)
        else:
            # During inference, use Monte Carlo dropout
            with torch.no_grad():
                n_mc = self.config.get('n_monte_carlo', 50)
                survival_preds = []
                recurrence_preds = []
                
                for _ in range(n_mc):
                    # Enable dropout during inference
                    self.train()
                    s_pred = self.survival_head(h_final)
                    r_pred = self.recurrence_head(h_final)
                    survival_preds.append(s_pred)
                    recurrence_preds.append(r_pred)
                    self.eval()
                
                survival_preds = torch.stack(survival_preds, dim=0)  # [T, batch, 1]
                recurrence_preds = torch.stack(recurrence_preds, dim=0)  # [T, batch, 1]
                
                survival_logits = survival_preds.mean(dim=0)
                recurrence_logits = recurrence_preds.mean(dim=0)
                
                survival_uncertainty = survival_preds.var(dim=0)
                recurrence_uncertainty = recurrence_preds.var(dim=0)
        
        outputs = {
            'survival': torch.sigmoid(survival_logits),
            'recurrence': torch.sigmoid(recurrence_logits),
            'survival_uncertainty': survival_uncertainty if not self.training else torch.zeros_like(survival_logits),
            'recurrence_uncertainty': recurrence_uncertainty if not self.training else torch.zeros_like(recurrence_logits),
        }
        
        if return_attention:
            outputs['beta'] = beta
            outputs['cross_attention'] = self.cross_modal_attention.get_attention_weights()
        
        return outputs
    
    def get_loss(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with uncertainty weighting
        
        Args:
            outputs: model outputs
            targets: ground truth targets
        Returns:
            losses: dict of loss components
        """
        # Survival loss (Cox partial likelihood)
        survival_pred = outputs['survival'].squeeze()
        survival_time = targets['survival_time']
        survival_event = targets['survival_event']
        
        # Sort by time
        sorted_idx = torch.argsort(survival_time, descending=True)
        survival_pred = survival_pred[sorted_idx]
        survival_event = survival_event[sorted_idx]
        
        # Cox partial likelihood
        log_risk = survival_pred
        exp_log_risk = torch.exp(log_risk)
        cumsum_exp = torch.cumsum(exp_log_risk, dim=0)
        partial_ll = torch.sum(log_risk - torch.log(cumsum_exp) * survival_event)
        loss_survival = -partial_ll / survival_event.sum()
        
        # Recurrence loss (focal loss)
        recurrence_pred = outputs['recurrence'].squeeze()
        recurrence_label = targets['recurrence_label']
        
        gamma = 2.0
        alpha = 0.25
        pt = torch.where(recurrence_label == 1, recurrence_pred, 1 - recurrence_pred)
        focal_weight = (1 - pt) ** gamma
        ce_loss = F.binary_cross_entropy(recurrence_pred, recurrence_label, reduction='none')
        loss_recurrence = (alpha * focal_weight * ce_loss).mean()
        
        # Uncertainty-weighted multi-task loss
        log_var_s = self.log_var_survival
        log_var_r = self.log_var_recurrence
        
        loss_s = loss_survival / (2 * torch.exp(log_var_s)) + log_var_s / 2
        loss_r = loss_recurrence / (2 * torch.exp(log_var_r)) + log_var_r / 2
        
        # Regularization
        loss_reg = 0
        for param in self.parameters():
            loss_reg += torch.sum(param ** 2)
        loss_reg *= self.config.get('reg_weight', 1e-4)
        
        total_loss = loss_s + loss_r + loss_reg
        
        return {
            'total_loss': total_loss,
            'loss_survival': loss_survival,
            'loss_recurrence': loss_recurrence,
            'loss_s': loss_s,
            'loss_r': loss_r,
            'loss_reg': loss_reg
        }
    
    def set_dropout_mode(self, mode: str = 'train'):
        """Set dropout mode for training or inference"""
        if mode == 'train':
            self.train()
        else:
            self.eval()
            
    def enable_mc_dropout(self, enable: bool = True):
        """Enable Monte Carlo dropout for uncertainty estimation"""
        self.training = enable


# ============================================================================
# Helper functions for graph construction
# ============================================================================

def build_heterogeneous_graph(patient_data: Dict, 
                             feature_thresholds: Dict[str, float],
                             device: str = 'cpu') -> Data:
    """
    Build heterogeneous graph for a single patient
    
    Args:
        patient_data: dict containing all modality features
        feature_thresholds: dict of correlation thresholds per modality
        device: device to place graph on
    Returns:
        graph: PyG Data object
    """
    # Implementation details...
    pass


def create_meta_paths(graph: Data, path_types: List[str]) -> List[torch.Tensor]:
    """
    Extract meta-path embeddings from graph
    
    Args:
        graph: PyG Data object
        path_types: list of meta-path types
    Returns:
        path_embeddings: list of embeddings per meta-path
    """
    # Implementation details...
    pass

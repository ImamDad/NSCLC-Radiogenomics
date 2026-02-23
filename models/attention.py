# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class CrossModalAttention(nn.Module):
    """Level 1: Early fusion via cross-modal attention"""
    
    def __init__(self, config):
        super().__init__()
        
        self.num_modalities = config.num_modalities
        self.modalities = config.modalities
        self.embedding_dim = config.modality_embedding_dim
        self.attention_dim = config.cross_modal_attention_dim
        
        # Query and key projections for each modality
        self.query_projections = nn.ModuleDict({
            m: nn.Linear(self.embedding_dim, self.attention_dim)
            for m in self.modalities
        })
        self.key_projections = nn.ModuleDict({
            m: nn.Linear(self.embedding_dim, self.attention_dim)
            for m in self.modalities
        })
        
        # Attention score vectors
        self.attention_vectors = nn.ModuleDict({
            m: nn.Linear(self.attention_dim, 1, bias=False)
            for m in self.modalities
        })
        
        # Cross-modal transformation matrices
        self.transformations = nn.ModuleDict()
        for q in self.modalities:
            for k in self.modalities:
                if q != k:
                    self.transformations[f"{q}2{k}"] = nn.Linear(
                        self.embedding_dim, self.embedding_dim
                    )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.embedding_dim * self.num_modalities)
        
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            embeddings: Dictionary of modality embeddings
        Returns:
            fused_representation: Concatenated attended features
            attention_weights: Dictionary of cross-modal attention weights
        """
        batch_size = next(iter(embeddings.values())).size(0)
        device = next(iter(embeddings.values())).device
        
        attention_weights = {}
        attended_features = {}
        
        # Compute attention for each query modality
        for query_mod in self.modalities:
            if query_mod not in embeddings:
                continue
                
            H_q = embeddings[query_mod]  # (B, d_m)
            q = self.query_projections[query_mod](H_q)  # (B, d_a)
            
            # Compute attention scores with all other modalities
            scores = []
            key_mods = []
            
            for key_mod in self.modalities:
                if key_mod == query_mod or key_mod not in embeddings:
                    continue
                    
                H_k = embeddings[key_mod]
                k = self.key_projections[key_mod](H_k)  # (B, d_a)
                
                # Additive attention
                score = self.attention_vectors[query_mod](
                    torch.tanh(q + k)
                )  # (B, 1)
                scores.append(score)
                key_mods.append(key_mod)
            
            if not scores:  # No other modalities available
                attended_features[query_mod] = H_q
                continue
            
            # Normalize attention scores
            scores = torch.cat(scores, dim=1)  # (B, n_keys)
            alpha = F.softmax(scores, dim=1)  # (B, n_keys)
            
            # Store attention weights
            for i, key_mod in enumerate(key_mods):
                attention_weights[f"{query_mod}→{key_mod}"] = alpha[:, i]
            
            # Apply attention-weighted integration
            attended_sum = 0
            for i, key_mod in enumerate(key_mods):
                transform_key = f"{query_mod}2{key_mod}"
                if transform_key in self.transformations:
                    transformed = self.transformations[transform_key](embeddings[key_mod])
                    attended_sum += alpha[:, i:i+1] * transformed
            
            # Residual connection
            attended_features[query_mod] = H_q + attended_sum
        
        # Concatenate all attended features
        concat_features = []
        for mod in self.modalities:
            if mod in attended_features:
                concat_features.append(attended_features[mod])
            else:
                # Zero padding for missing modalities
                concat_features.append(torch.zeros(batch_size, self.embedding_dim).to(device))
        
        fused = torch.cat(concat_features, dim=1)  # (B, d_m * n_modalities)
        fused = self.layer_norm(fused)
        
        return fused, attention_weights


class GraphAttentionLayer(nn.Module):
    """Multi-head graph attention layer for heterogeneous graphs"""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert self.head_dim * num_heads == out_dim, "out_dim must be divisible by num_heads"
        
        # Type-specific projections
        self.type_projections = nn.ModuleDict()
        
        # Attention parameters
        self.attention_vectors = nn.ParameterDict()
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def add_relation(self, rel_type: str, src_type: str, dst_type: str):
        """Add projection matrices for a new relation type"""
        self.type_projections[f"{rel_type}_src_{src_type}"] = nn.Linear(self.in_dim, self.out_dim)
        self.type_projections[f"{rel_type}_dst_{dst_type}"] = nn.Linear(self.in_dim, self.out_dim)
        
        # Relation-specific attention vector for each head
        attn_vec = nn.Parameter(torch.zeros(self.num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(attn_vec)
        self.attention_vectors[rel_type] = attn_vec
        
    def forward(self, x_dict, edge_index_dict, edge_type_dict):
        """
        Args:
            x_dict: Dictionary mapping node types to node features
            edge_index_dict: Dictionary mapping relation types to edge indices
            edge_type_dict: Dictionary mapping relation types to (src_type, dst_type) tuples
        Returns:
            Updated node features dictionary
        """
        device = next(iter(x_dict.values())).device
        
        # Project nodes for each relation
        proj_dict = {}
        for rel_type, (src_type, dst_type) in edge_type_dict.items():
            if rel_type in self.type_projections:
                proj_src = self.type_projections[f"{rel_type}_src_{src_type}"](x_dict[src_type])
                proj_dst = self.type_projections[f"{rel_type}_dst_{dst_type}"](x_dict[dst_type])
                proj_dict[rel_type] = (proj_src, proj_dst)
        
        # Prepare output dictionary
        out_dict = {ntype: [] for ntype in x_dict.keys()}
        
        # Process each node type and relation
        for ntype in x_dict.keys():
            node_features = x_dict[ntype]  # (N, in_dim)
            
            # Collect messages from all relations where this node type is destination
            messages = []
            
            for rel_type, (src_type, dst_type) in edge_type_dict.items():
                if dst_type != ntype or rel_type not in proj_dict:
                    continue
                    
                edge_index = edge_index_dict[rel_type]  # (2, E)
                proj_src, proj_dst = proj_dict[rel_type]
                
                # Reshape for multi-head attention
                src_features = proj_src.view(-1, self.num_heads, self.head_dim)  # (N_src, H, d_h)
                dst_features = proj_dst.view(-1, self.num_heads, self.head_dim)  # (N_dst, H, d_h)
                
                # Compute attention coefficients
                src_expanded = src_features[edge_index[0]]  # (E, H, d_h)
                dst_expanded = dst_features[edge_index[1]]  # (E, H, d_h)
                
                # Concatenate source and destination features
                concat = torch.cat([src_expanded, dst_expanded], dim=-1)  # (E, H, 2*d_h)
                
                # Compute attention scores
                attn_vec = self.attention_vectors[rel_type]  # (H, 2*d_h)
                scores = torch.einsum('ehd,hd->eh', concat, attn_vec)  # (E, H)
                scores = self.leaky_relu(scores)
                
                # Normalize attention scores per destination node
                alpha = self._normalize_attention(scores, edge_index[1], dst_features.size(0))
                alpha = self.dropout(alpha)
                
                # Weighted aggregation
                weighted = alpha.unsqueeze(-1) * src_expanded  # (E, H, d_h)
                
                # Aggregate by destination node
                out = torch.zeros(dst_features.size(0), self.num_heads, self.head_dim).to(device)
                out.index_add_(0, edge_index[1], weighted)  # (N_dst, H, d_h)
                
                messages.append(out)
            
            if messages:
                # Combine messages from all relations
                combined = torch.stack(messages).mean(dim=0)  # (N, H, d_h)
                combined = combined.view(-1, self.out_dim)  # (N, out_dim)
                
                # Residual connection
                out_dict[ntype] = node_features + combined
            else:
                out_dict[ntype] = node_features
        
        return out_dict
    
    def _normalize_attention(self, scores, dst_indices, num_dst):
        """Normalize attention scores per destination node"""
        # Convert to sparse for efficient normalization
        alpha = torch.zeros(num_dst, scores.size(1)).to(scores.device)
        alpha.index_add_(0, dst_indices, torch.exp(scores))
        alpha = alpha[dst_indices]  # (E, H)
        
        # Normalize
        alpha = torch.exp(scores) / (alpha + 1e-15)
        return alpha


class SemanticAttention(nn.Module):
    """Semantic attention mechanism for learning meta-path importance"""
    
    def __init__(self, meta_paths: List[str], feature_dim: int, attention_dim: int = 128):
        super().__init__()
        
        self.meta_paths = meta_paths
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        
        # Shared transformation layer
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh()
        )
        
        # Semantic context vector (query)
        self.context_vector = nn.Parameter(torch.randn(attention_dim))
        nn.init.xavier_uniform_(self.context_vector.unsqueeze(0))
        
    def forward(self, meta_path_embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            meta_path_embeddings: Dictionary mapping meta-path names to embeddings (B, d)
        Returns:
            fused_embedding: Weighted sum of meta-path embeddings (B, d)
            weights: Semantic attention weights (B, n_paths)
        """
        batch_size = next(iter(meta_path_embeddings.values())).size(0)
        device = next(iter(meta_path_embeddings.values())).device
        
        # Transform each meta-path embedding
        transformed = []
        path_names = []
        
        for path_name, embedding in meta_path_embeddings.items():
            transformed.append(self.transform(embedding))  # (B, d_s)
            path_names.append(path_name)
        
        transformed = torch.stack(transformed, dim=1)  # (B, n_paths, d_s)
        
        # Compute attention scores
        scores = torch.einsum('bpd,d->bp', transformed, self.context_vector)  # (B, n_paths)
        weights = F.softmax(scores, dim=1)  # (B, n_paths)
        
        # Weighted fusion
        embeddings_stack = torch.stack(list(meta_path_embeddings.values()), dim=1)  # (B, n_paths, d)
        fused = torch.einsum('bp,bpd->bd', weights, embeddings_stack)  # (B, d)
        
        return fused, weights

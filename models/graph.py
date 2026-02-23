# models/graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .attention import GraphAttentionLayer, SemanticAttention


class HeterogeneousGraphConstructor:
    """Constructs patient-specific heterogeneous graphs"""
    
    def __init__(self, config):
        self.config = config
        
        # Node types
        self.node_types = ['patient', 'CT_feature', 'PET_feature', 'clinical_feature', 'genomic_pathway']
        
        # Relation types
        self.relation_types = ['intra_modal', 'inter_modal', 'patient_to_feature']
        
        # Correlation thresholds
        self.corr_thresholds = {
            'CT': config.corr_threshold_ct,
            'PET': config.corr_threshold_pet,
            'Clinical': config.corr_threshold_clinical,
            'Genomic': config.corr_threshold_genomic
        }
        
        # Meta-paths
        self.meta_paths = {
            'immune': ['CT_feature', 'genomic_pathway', 'patient'],
            'proliferation': ['PET_feature', 'genomic_pathway', 'patient'],
            'treatment': [['CT_feature', 'PET_feature'], 'genomic_pathway', 'patient']
        }
        
    def build_graph(self, patient_data: Dict[str, Any], cohort_correlations: Dict[str, np.ndarray]) -> Dict:
        """
        Build heterogeneous graph for a single patient
        
        Args:
            patient_data: Dictionary with patient's features
            cohort_correlations: Pre-computed feature correlations from training cohort
        Returns:
            Graph dictionary with nodes, edges, and features
        """
        graph = {
            'node_features': {},
            'edge_index': {},
            'edge_type': {},
            'node_mapping': {}
        }
        
        # Add patient node
        patient_id = patient_data['patient_id']
        graph['node_mapping']['patient'] = {patient_id: 0}
        
        # Add feature nodes with their indices
        current_idx = 1
        
        # CT features
        ct_features = patient_data.get('CT_features', [])
        if len(ct_features) > 0:
            n_ct = len(ct_features)
            graph['node_mapping']['CT_feature'] = {f"CT_{i}": current_idx + i for i in range(n_ct)}
            graph['node_features']['CT_feature'] = torch.tensor(ct_features, dtype=torch.float32)
            current_idx += n_ct
        
        # PET features
        pet_features = patient_data.get('PET_features', [])
        if len(pet_features) > 0:
            n_pet = len(pet_features)
            graph['node_mapping']['PET_feature'] = {f"PET_{i}": current_idx + i for i in range(n_pet)}
            graph['node_features']['PET_feature'] = torch.tensor(pet_features, dtype=torch.float32)
            current_idx += n_pet
        
        # Clinical features
        clinical_features = patient_data.get('clinical_features', [])
        if len(clinical_features) > 0:
            n_clinical = len(clinical_features)
            graph['node_mapping']['clinical_feature'] = {f"clinical_{i}": current_idx + i for i in range(n_clinical)}
            graph['node_features']['clinical_feature'] = torch.tensor(clinical_features, dtype=torch.float32)
            current_idx += n_clinical
        
        # Genomic pathways
        pathway_scores = patient_data.get('pathway_scores', [])
        if len(pathway_scores) > 0:
            n_pathways = len(pathway_scores)
            graph['node_mapping']['genomic_pathway'] = {f"pathway_{i}": current_idx + i for i in range(n_pathways)}
            graph['node_features']['genomic_pathway'] = torch.tensor(pathway_scores, dtype=torch.float32)
            current_idx += n_pathways
        
        # Build edges
        graph['edge_index'] = {}
        graph['edge_type'] = {}
        
        # Patient-to-feature edges
        patient_edges = self._build_patient_edges(graph['node_mapping'])
        if len(patient_edges) > 0:
            graph['edge_index']['patient_to_feature'] = patient_edges
            graph['edge_type']['patient_to_feature'] = ('patient', 'clinical_feature')  # placeholder
        
        # Intra-modal edges (based on correlations)
        intra_edges = self._build_intra_modal_edges(
            graph['node_mapping'], 
            graph['node_features'],
            cohort_correlations
        )
        for rel_type, edges in intra_edges.items():
            graph['edge_index'][rel_type] = edges
            graph['edge_type'][rel_type] = ('CT_feature', 'CT_feature')  # will be overridden per modality
        
        # Inter-modal edges (meta-paths)
        inter_edges = self._build_inter_modal_edges(
            graph['node_mapping'],
            patient_data
        )
        for rel_type, edges in inter_edges.items():
            graph['edge_index'][rel_type] = edges
            # Determine source and destination types
            if rel_type.startswith('CT2immune'):
                graph['edge_type'][rel_type] = ('CT_feature', 'genomic_pathway')
            elif rel_type.startswith('PET2prolif'):
                graph['edge_type'][rel_type] = ('PET_feature', 'genomic_pathway')
            elif rel_type.startswith('pathway2patient'):
                graph['edge_type'][rel_type] = ('genomic_pathway', 'patient')
        
        return graph
    
    def _build_patient_edges(self, node_mapping: Dict) -> torch.Tensor:
        """Create edges from patient node to all feature nodes"""
        patient_idx = node_mapping['patient'].get(list(node_mapping['patient'].keys())[0], 0)
        edges = []
        
        for node_type, mapping in node_mapping.items():
            if node_type == 'patient':
                continue
            for node_name, node_idx in mapping.items():
                edges.append([patient_idx, node_idx])
        
        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        
        edges = torch.tensor(edges, dtype=torch.long).t()
        return edges
    
    def _build_intra_modal_edges(self, node_mapping: Dict, node_features: Dict, 
                                 cohort_correlations: Dict) -> Dict:
        """Create intra-modal edges based on correlation thresholds"""
        intra_edges = {}
        
        for modality in ['CT_feature', 'PET_feature', 'clinical_feature', 'genomic_pathway']:
            if modality not in node_mapping or modality not in node_features:
                continue
            
            mod_nodes = list(node_mapping[modality].values())
            mod_features = node_features[modality].numpy()
            
            # Get correlation matrix for this modality
            if modality in cohort_correlations:
                corr_matrix = cohort_correlations[modality]
            else:
                # Compute from features if not provided
                corr_matrix = np.corrcoef(mod_features.T)
            
            # Apply threshold
            threshold = self.corr_thresholds.get(modality.replace('_feature', ''), 0.5)
            edges = []
            
            for i in range(len(mod_nodes)):
                for j in range(i+1, len(mod_nodes)):
                    if abs(corr_matrix[i, j]) > threshold:
                        edges.append([mod_nodes[i], mod_nodes[j]])
                        edges.append([mod_nodes[j], mod_nodes[i]])
            
            if edges:
                rel_type = f"intra_{modality}"
                intra_edges[rel_type] = torch.tensor(edges, dtype=torch.long).t()
        
        return intra_edges
    
    def _build_inter_modal_edges(self, node_mapping: Dict, patient_data: Dict) -> Dict:
        """Create inter-modal edges following meta-paths"""
        inter_edges = {}
        
        # Immune pathway: CT_feature -> genomic_pathway
        if 'CT_feature' in node_mapping and 'genomic_pathway' in node_mapping:
            ct_nodes = list(node_mapping['CT_feature'].values())
            pathway_nodes = list(node_mapping['genomic_pathway'].values())
            
            # Connect each CT feature to immune-related pathways
            # In practice, you would have pre-defined which pathways are immune-related
            immune_pathway_indices = patient_data.get('immune_pathway_indices', [0, 1, 2])  # example
            
            edges = []
            for ct_idx in ct_nodes:
                for pw_idx in [pathway_nodes[i] for i in immune_pathway_indices if i < len(pathway_nodes)]:
                    edges.append([ct_idx, pw_idx])
            
            if edges:
                inter_edges['CT2immune'] = torch.tensor(edges, dtype=torch.long).t()
        
        # Proliferation pathway: PET_feature -> genomic_pathway
        if 'PET_feature' in node_mapping and 'genomic_pathway' in node_mapping:
            pet_nodes = list(node_mapping['PET_feature'].values())
            pathway_nodes = list(node_mapping['genomic_pathway'].values())
            
            proliferation_pathway_indices = patient_data.get('proliferation_pathway_indices', [3, 4, 5])
            
            edges = []
            for pet_idx in pet_nodes:
                for pw_idx in [pathway_nodes[i] for i in proliferation_pathway_indices if i < len(pathway_nodes)]:
                    edges.append([pet_idx, pw_idx])
            
            if edges:
                inter_edges['PET2prolif'] = torch.tensor(edges, dtype=torch.long).t()
        
        # Pathway to patient edges
        if 'genomic_pathway' in node_mapping and 'patient' in node_mapping:
            pathway_nodes = list(node_mapping['genomic_pathway'].values())
            patient_idx = list(node_mapping['patient'].values())[0]
            
            edges = [[pw_idx, patient_idx] for pw_idx in pathway_nodes]
            inter_edges['pathway2patient'] = torch.tensor(edges, dtype=torch.long).t()
        
        return inter_edges


class GraphFusionNetwork(nn.Module):
    """Level 2: Graph-based intermediate fusion"""
    
    def __init__(self, config, graph_constructor):
        super().__init__()
        
        self.config = config
        self.graph_constructor = graph_constructor
        
        # Graph attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(config.num_graph_layers):
            layer = GraphAttentionLayer(
                in_dim=config.modality_embedding_dim,
                out_dim=config.total_hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout
            )
            self.attention_layers.append(layer)
        
        # Add relation types to each layer
        self._setup_relations()
        
        # Modality-specific pooling
        self.modality_pooling = nn.ModuleDict({
            'CT_feature': nn.Linear(config.total_hidden_dim, config.total_hidden_dim),
            'PET_feature': nn.Linear(config.total_hidden_dim, config.total_hidden_dim),
            'clinical_feature': nn.Linear(config.total_hidden_dim, config.total_hidden_dim),
            'genomic_pathway': nn.Linear(config.total_hidden_dim, config.total_hidden_dim)
        })
        
        # Semantic attention for meta-paths
        self.semantic_attention = SemanticAttention(
            meta_paths=config.meta_paths,
            feature_dim=config.total_hidden_dim * 4,  # Concatenated modalities
            attention_dim=config.semantic_attention_dim
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.total_hidden_dim * 4, config.total_hidden_dim)
        
    def _setup_relations(self):
        """Add all relation types to attention layers"""
        relation_types = [
            ('intra_CT_feature', 'CT_feature', 'CT_feature'),
            ('intra_PET_feature', 'PET_feature', 'PET_feature'),
            ('intra_clinical_feature', 'clinical_feature', 'clinical_feature'),
            ('intra_genomic_pathway', 'genomic_pathway', 'genomic_pathway'),
            ('patient_to_feature', 'patient', 'clinical_feature'),
            ('CT2immune', 'CT_feature', 'genomic_pathway'),
            ('PET2prolif', 'PET_feature', 'genomic_pathway'),
            ('pathway2patient', 'genomic_pathway', 'patient')
        ]
        
        for layer in self.attention_layers:
            for rel_type, src_type, dst_type in relation_types:
                layer.add_relation(rel_type, src_type, dst_type)
        
    def forward(self, patient_graphs: List[Dict], early_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patient_graphs: List of graph dictionaries for each patient in batch
            early_features: Early fusion features from Level 1 (B, 256)
        Returns:
            graph_embedding: Final graph-level embedding (B, 128)
            meta_path_weights: Semantic attention weights (B, 3)
        """
        batch_size = len(patient_graphs)
        device = early_features.device
        
        # Process each patient graph
        meta_path_embeddings = {
            'immune': [],
            'proliferation': [],
            'treatment': []
        }
        
        for i, graph in enumerate(patient_graphs):
            # Initialize node features using early_features
            x_dict = self._initialize_node_features(graph, early_features[i], device)
            
            # Apply graph attention layers
            edge_index_dict = graph['edge_index']
            edge_type_dict = graph['edge_type']
            
            for layer in self.attention_layers:
                x_dict = layer(x_dict, edge_index_dict, edge_type_dict)
            
            # Get meta-path specific embeddings
            path_embeddings = self._extract_meta_path_embeddings(x_dict, graph)
            
            for path_name, emb in path_embeddings.items():
                meta_path_embeddings[path_name].append(emb)
        
        # Stack embeddings across batch
        stacked_embeddings = {
            name: torch.stack(embs, dim=0) for name, embs in meta_path_embeddings.items()
        }
        
        # Apply semantic attention
        fused_embedding, meta_path_weights = self.semantic_attention(stacked_embeddings)
        
        return fused_embedding, meta_path_weights
    
    def _initialize_node_features(self, graph: Dict, patient_feature: torch.Tensor, device: torch.device) -> Dict:
        """Initialize node features using early fusion representations"""
        x_dict = {}
        
        # Patient node gets the patient_feature (from early fusion)
        x_dict['patient'] = patient_feature.unsqueeze(0).to(device)
        
        # Feature nodes get their corresponding features from the graph
        if 'CT_feature' in graph['node_features']:
            x_dict['CT_feature'] = graph['node_features']['CT_feature'].to(device)
        if 'PET_feature' in graph['node_features']:
            x_dict['PET_feature'] = graph['node_features']['PET_feature'].to(device)
        if 'clinical_feature' in graph['node_features']:
            x_dict['clinical_feature'] = graph['node_features']['clinical_feature'].to(device)
        if 'genomic_pathway' in graph['node_features']:
            x_dict['genomic_pathway'] = graph['node_features']['genomic_pathway'].to(device)
        
        return x_dict
    
    def _extract_meta_path_embeddings(self, x_dict: Dict, graph: Dict) -> Dict[str, torch.Tensor]:
        """Extract meta-path specific embeddings via pooling"""
        embeddings = {}
        
        # Helper for modality pooling
        def pool_modality(mod_type):
            if mod_type in x_dict and x_dict[mod_type].size(0) > 0:
                # Self-attention pooling
                features = x_dict[mod_type]  # (n_nodes, d)
                query = self.modality_pooling[mod_type](features.mean(dim=0, keepdim=True))
                scores = torch.matmul(features, query.t()).squeeze()
                weights = F.softmax(scores, dim=0)
                pooled = torch.sum(weights.unsqueeze(-1) * features, dim=0)
                return pooled
            return torch.zeros(self.config.total_hidden_dim, device=next(iter(x_dict.values())).device)
        
        # Immune pathway embedding
        ct_pooled = pool_modality('CT_feature')
        immune_pooled = pool_modality('genomic_pathway')
        embeddings['immune'] = torch.cat([ct_pooled, immune_pooled], dim=0)
        
        # Proliferation pathway embedding
        pet_pooled = pool_modality('PET_feature')
        prolif_pooled = pool_modality('genomic_pathway')
        embeddings['proliferation'] = torch.cat([pet_pooled, prolif_pooled], dim=0)
        
        # Treatment pathway embedding
        # Combine CT and PET, then add pathway
        ct_pet_pooled = (ct_pooled + pet_pooled) / 2
        response_pooled = pool_modality('genomic_pathway')
        embeddings['treatment'] = torch.cat([ct_pet_pooled, response_pooled], dim=0)
        
        # Pad to consistent size if needed
        target_dim = self.config.total_hidden_dim * 4
        for name in embeddings:
            current_dim = embeddings[name].size(0)
            if current_dim < target_dim:
                padding = torch.zeros(target_dim - current_dim, device=embeddings[name].device)
                embeddings[name] = torch.cat([embeddings[name], padding], dim=0)
        
        return embeddings

# config.py
import torch
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for MS-HHGN model"""
    
    # Modality configuration
    modalities: List[str] = None  # CT, PET, Clinical, Genomic
    num_modalities: int = 4
    
    # Feature dimensions
    ct_feature_dim: int = 131  # Number of radiomic features from CT
    pet_feature_dim: int = 131  # Number of radiomic features from PET
    clinical_feature_dim: int = 25  # Processed clinical features
    genomic_pathway_dim: int = 50  # Number of hallmark pathways from GSVA
    
    # Encoder dimensions
    modality_embedding_dim: int = 64  # d_m in paper
    cross_modal_attention_dim: int = 128  # d_a in paper
    hidden_dim: int = 128  # d_h per attention head
    total_hidden_dim: int = 128  # d total node representation
    semantic_attention_dim: int = 128  # d_s for semantic attention
    
    # Graph attention configuration
    num_attention_heads: int = 8  # K
    num_graph_layers: int = 3  # L
    attention_dropout: float = 0.3
    feature_dropout: float = 0.2
    edge_dropout: float = 0.2  # For graph regularization
    
    # Correlation thresholds for graph construction
    corr_threshold_ct: float = 0.65
    corr_threshold_pet: float = 0.55
    corr_threshold_clinical: float = 0.45
    corr_threshold_genomic: float = 0.35
    
    # Meta-paths configuration
    meta_paths: List[str] = None  # ['immune', 'proliferation', 'treatment']
    num_meta_paths: int = 3
    
    # Multi-task learning
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
    l2_weight_decay: float = 1e-4
    
    # Uncertainty quantification
    num_mc_dropout_passes: int = 50
    mc_dropout_rate: float = 0.1
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 16
    max_epochs: int = 1000
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.001
    
    # Missing data handling
    modality_dropout_prob: float = 0.3
    
    # Model compression
    pruning_threshold: float = 0.01  # For unstructured pruning
    quantization_bits: int = 8
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ['CT', 'PET', 'Clinical', 'Genomic']
        if self.meta_paths is None:
            self.meta_paths = ['immune', 'proliferation', 'treatment']


@dataclass
class DataConfig:
    """Configuration for data processing"""
    
    # Paths
    tcia_data_path: str = "./data/tcia_nsclc"
    tcga_data_path: str = "./data/tcga_luad"
    synthetic_data_path: str = "./data/synthetic"
    output_path: str = "./outputs"
    checkpoint_path: str = "./checkpoints"
    
    # Imaging parameters
    ct_resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # mm
    pet_resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # mm
    ct_window: Tuple[int, int] = (-1000, 1000)  # Hounsfield units
    pet_normalization: str = "zscore"  # 'zscore' or 'suv'
    
    # Radiomics extraction
    radiomics_params: Dict[str, Any] = None
    
    # Genomic processing
    min_counts_per_million: float = 1.0
    min_sample_proportion: float = 0.5
    variance_stabilizing: bool = True
    
    # Data splits
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    random_seed: int = 42
    n_folds: int = 5
    
    def __post_init__(self):
        if self.radiomics_params is None:
            self.radiomics_params = {
                'first_order': True,
                'shape': True,
                'glcm': True,
                'glrlm': True,
                'glszm': True,
                'ngtdm': True,
                'gldm': True
            }

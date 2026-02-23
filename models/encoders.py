# models/encoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import monai.networks.nets as monai_nets


class CTEncoder(nn.Module):
    """3D ResNet-50 encoder pretrained on MedicalNet for CT imaging"""
    
    def __init__(self, embedding_dim: int = 64, pretrained: bool = True):
        super().__init__()
        
        # Using MedicalNet 3D ResNet-50 backbone
        if pretrained:
            # Note: In practice, load actual MedicalNet weights
            self.backbone = monai_nets.ResNet(
                block="bottleneck",
                layers=[3, 4, 6, 3],
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=3,
                n_input_channels=1
            )
        else:
            self.backbone = monai_nets.ResNet(
                block="bottleneck",
                layers=[3, 4, 6, 3],
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=3,
                n_input_channels=1
            )
        
        # Get feature dimension from backbone
        self.backbone_features = 512 * 4  # ResNet-50 feature dimension
        
        # Projection head
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CT volume of shape (B, 1, D, H, W)
        Returns:
            Embedding of shape (B, embedding_dim)
        """
        features = self.backbone(x)
        embedding = self.projection(features)
        return embedding


class PETEncoder(nn.Module):
    """2D ResNet-50 with 3D context aggregation for PET imaging"""
    
    def __init__(self, embedding_dim: int = 64, pretrained: bool = True):
        super().__init__()
        
        # 2D ResNet-50 for slice-wise processing
        import torchvision.models as models
        if pretrained:
            self.slice_encoder = models.resnet50(weights='IMAGENET1K_V1')
        else:
            self.slice_encoder = models.resnet50(weights=None)
        
        # Modify first conv for single channel
        self.slice_encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Remove classification head
        self.slice_encoder = nn.Sequential(*list(self.slice_encoder.children())[:-2])
        
        # 3D context aggregation
        self.context_aggregation = nn.Sequential(
            nn.Conv3d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(1024),
            nn.Conv3d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(512),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: PET volume of shape (B, 1, D, H, W)
        Returns:
            Embedding of shape (B, embedding_dim)
        """
        B, C, D, H, W = x.shape
        
        # Process each slice
        slice_features = []
        for d in range(D):
            slice_data = x[:, :, d, :, :]  # (B, 1, H, W)
            slice_feat = self.slice_encoder(slice_data)  # (B, 2048, H//32, W//32)
            slice_features.append(slice_feat)
        
        # Stack and aggregate context
        volume_features = torch.stack(slice_features, dim=2)  # (B, 2048, D, H//32, W//32)
        context_features = self.context_aggregation(volume_features)  # (B, 512)
        embedding = self.projection(context_features)  # (B, embedding_dim)
        
        return embedding


class ClinicalEncoder(nn.Module):
    """MLP encoder for clinical variables"""
    
    def __init__(self, input_dim: int, embedding_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class GenomicEncoder(nn.Module):
    """Encoder for genomic pathway scores"""
    
    def __init__(self, input_dim: int = 50, embedding_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ModalityEncoders(nn.Module):
    """Collection of modality-specific encoders"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embedding_dim = config.modality_embedding_dim
        
        # Initialize encoders
        self.encoders = nn.ModuleDict({
            'CT': CTEncoder(self.embedding_dim),
            'PET': PETEncoder(self.embedding_dim),
            'Clinical': ClinicalEncoder(config.clinical_feature_dim, self.embedding_dim),
            'Genomic': GenomicEncoder(config.genomic_pathway_dim, self.embedding_dim)
        })
        
    def forward(self, modality_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            modality_data: Dictionary mapping modality names to input tensors
        Returns:
            Dictionary mapping modality names to embeddings
        """
        embeddings = {}
        for modality, data in modality_data.items():
            if modality in self.encoders and data is not None:
                embeddings[modality] = self.encoders[modality](data)
        return embeddings

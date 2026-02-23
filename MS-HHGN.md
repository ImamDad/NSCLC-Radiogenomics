MS-HHGN
├── Level 1: Modality Encoders
│   ├── CT Encoder (MedicalNet 3D ResNet-50)
│   ├── PET Encoder (2D ResNet-50 + 3D context)
│   ├── Clinical Encoder (MLP)
│   └── Genomic Encoder (MLP)
├── Level 1: Cross-Modal Attention
├── Level 2: Graph Construction
│   ├── Intra-modal edges (correlation-based)
│   ├── Inter-modal edges (meta-paths)
│   │   ├── Immune response
│   │   ├── Proliferation signaling
│   │   └── Treatment response
│   └── Patient-to-feature edges
├── Level 2: Graph Attention Network
│   ├── Multi-head attention (K=8)
│   ├── L=3 layers
│   └── Semantic attention for meta-paths
└── Level 3: Uncertainty-Aware Fusion
    ├── Monte Carlo dropout (T=50)
    ├── Confidence-based weighting
    └── 95% confidence intervals







@article{mshhgn2024,
    title={MS-HHGN: A Multi-Scale Hierarchical Heterogeneous Graph Network 
           for Survival Prediction and Recurrence Classification in 
           Non-Small Cell Lung Cancer},
    author={Author, First and Author, Second and Author, Third and Author, Fourth},
    journal={arXiv preprint arXiv:2401.00000},
    year={2024}
}

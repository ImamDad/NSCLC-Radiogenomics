# MS-HGNN: Multi-Scale Hierarchical Graph Neural Network for NSCLC Prognosis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

MS-HGNN is a novel deep learning framework for simultaneous survival prediction and recurrence classification in Non-Small Cell Lung Cancer (NSCLC). The model integrates multimodal data (CT, PET, clinical, genomic) through a three-level hierarchical fusion architecture with biologically-informed meta-paths and semantic attention.

### Key Features

- **Multi-modal Integration**: Seamless fusion of CT imaging, PET imaging, clinical variables, and genomic data
- **Biologically-Informed Graph Construction**: Domain-specific meta-paths encoding NSCLC biology
- **Semantic Attention**: Learnable pathway importance with biological interpretability
- **Uncertainty Quantification**: Monte Carlo dropout for confidence estimation
- **Multi-Task Learning**: Joint optimization of survival and recurrence prediction
- **Model Compression**: Pruning and quantization for clinical deployment

### Performance Highlights

- **C-index**: 0.85 (95% CI: 0.82-0.88) for survival prediction
- **AUC**: 0.89 (95% CI: 0.86-0.92) for recurrence classification
- **Hazard Ratio**: 4.15 (95% CI: 3.12-5.52) for high vs. low risk
- **External Validation**: C-index 0.81 on TCGA-LUAD

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended, but CPU mode is supported)
- 16GB+ RAM

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/ImamDad/NSCLC-Radiogenomics.git
cd NSCLC-Radiogenomics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .


**Data Directory Structure**
data/
├── raw/
│   ├── tcia_nsclc/
│   │   ├── ct/
│   │   ├── pet/
│   │   ├── clinical/
│   │   └── genomic/
│   └── tcga_luad/
│       ├── ct/
│       ├── pet/
│       ├── clinical/
│       └── genomic/
├── processed/
│   ├── tcia_nsclc_processed.h5
│   ├── tcga_luad_processed.h5
│   └── features/
└── splits/
    ├── train_indices.npy
    ├── val_indices.npy
    └── test_indices.npy

**Preprocessing**
# Download data (requires TCIA API access)
python scripts/download_data.py --dataset tcia_nsclc

# Preprocess data
python scripts/preprocess_data.py --config configs/config.yaml

**Training**
# Train with default configuration
python scripts/train_model.py --config configs/config.yaml

# Train with specific GPU
python scripts/train_model.py --config configs/config.yaml --gpu 0

# Resume training from checkpoint
python scripts/train_model.py --config configs/config.yaml --resume checkpoints/latest.pth


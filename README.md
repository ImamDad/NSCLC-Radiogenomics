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

📈 Evaluation
Run Evaluation
bash
# Evaluate trained model
python scripts/evaluate_model.py --checkpoint checkpoints/best_model.pth

# Run baseline comparison
python experiments/baseline_comparison.py

# Run ablation study
python experiments/ablation_study.py

# Run compression study
python experiments/compression_study.py
Generate Figures
bash
# Generate all paper figures
python experiments/generate_figures.py
📊 Results
The model's performance across different configurations:

Configuration	C-index (95% CI)	AUC (95% CI)
Full Multimodal	0.85 (0.82-0.88)	0.89 (0.86-0.92)
CT Only	0.78 (0.74-0.82)	0.82 (0.79-0.85)
PET Only	0.80 (0.76-0.84)	0.84 (0.81-0.87)
Clinical Only	0.77 (0.73-0.81)	0.80 (0.76-0.84)
Genomic Only	0.81 (0.77-0.85)	0.85 (0.82-0.88)
📁 Project Structure
text
NSCLC-Radiogenomics/
├── configs/                 # Configuration files
├── data/                    # Data directory
├── src/                     # Source code
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model architectures
│   ├── training/           # Training loop and utilities
│   ├── evaluation/         # Evaluation metrics and visualization
│   └── utils/              # Utility functions
├── experiments/            # Experiment scripts
├── notebooks/              # Jupyter notebooks
├── scripts/                # Command-line scripts
├── tests/                  # Unit tests
└── figures/                # Generated figures
🔬 Reproducibility
To reproduce the paper's results:

Set up environment as described above

Download and preprocess data

Run baseline comparison:

bash
python experiments/baseline_comparison.py
Run ablation study:

bash
python experiments/ablation_study.py
Generate all figures:

bash
python experiments/generate_figures.py
All experiments use fixed random seeds (0, 42, 123, 456, 789) for reproducibility.

🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.

📝 Citation
If you use this code in your research, please cite:

bibtex
@article{dad2026mshgnn,
  title={MS-HGNN: Interpretable Multi-Scale Hierarchical Graph Neural Network for Multimodal Survival and Recurrence Prediction in Non-Small Cell Lung Cancer},
  author={Dad, Imam and He, Jianfeng},
  journal={Journal of Medical Imaging},
  year={2026}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
National Natural Science Foundation of China (Grant No. 82160347)

Kunming University of Science and Technology

Chinese Government Scholarship Council (CSC)

📧 Contact
Imam Dad: GitHub

Dr. Jianfeng He: jfenghe@kust.edu.cn

text

---

### 2. requirements.txt

```txt
# Core dependencies
torch>=1.12.0
torch-geometric>=2.2.0
torch-scatter>=2.1.0
torch-sparse>=0.6.16
torch-cluster>=1.6.1
numpy>=1.21.0
scipy>=1.9.0
pandas>=1.4.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
pyyaml>=6.0
h5py>=3.7.0
tqdm>=4.64.0
wandb>=0.13.0
optuna>=3.0.0

# Optional but recommended for full functionality
pytorch-ignite>=0.4.10
lifelines>=0.27.0
statsmodels>=0.13.0
dgl>=1.0.0  # For graph operations
plotly>=5.10.0  # For interactive plots
networkx>=2.8.0  # For graph visualization


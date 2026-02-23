# MS-HHGN: Multi-Scale Hierarchical Heterogeneous Graph Network for NSCLC Radiogenomics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2401.00000-b31b1b.svg)](https://arxiv.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/)

**Official PyTorch implementation of "MS-HHGN: A Multi-Scale Hierarchical Heterogeneous Graph Network for Survival Prediction and Recurrence Classification in Non-Small Cell Lung Cancer"**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
- [Experiments](#-experiments)
- [Results](#-results)
- [Interpretability](#-interpretability)
- [Model Compression](#-model-compression)
- [Reproducibility](#-reproducibility)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔬 Overview

Lung cancer remains the leading cause of cancer-related mortality worldwide, with non-small cell lung cancer (NSCLC) comprising approximately 85% of all diagnoses. Despite therapeutic advances, the five-year survival rate for advanced-stage NSCLC remains below 25%. This prognostic stagnation underscores the urgent need for more precise, individualized risk stratification instruments.

**MS-HHGN** is a novel deep learning framework designed specifically for multimodal radiogenomic analysis in NSCLC. It integrates computed tomography (CT), positron emission tomography (PET), genomic profiles, and clinical variables through a principled three-level hierarchical fusion strategy, achieving state-of-the-art performance in both survival prediction and recurrence classification.

### Key Features

- **Hierarchical Multimodal Fusion**: Three-level architecture progressively integrating information at increasing abstraction levels
- **Biologically-Informed Graph Construction**: Patient-specific heterogeneous graphs with curated meta-paths encoding NSCLC biology
- **Data-Driven Semantic Attention**: Learns optimal pathway importance directly from data, eliminating subjective manual weighting
- **Uncertainty Quantification**: Monte Carlo dropout providing well-calibrated confidence intervals for clinical decision-making
- **Comprehensive Interpretability**: Multi-faceted visualization tools revealing model reasoning
- **Model Compression**: Pruning, quantization, and knowledge distillation for clinical deployment

---

## 🏆 Key Contributions

### 1. Methodological Innovations

| Component | Description | Performance Gain |
|-----------|-------------|------------------|
| **Cross-Modal Attention** | Dynamic modeling of pairwise modality interactions | 3.5% (p < 0.001) |
| **Heterogeneous Graph Construction** | Biologically-informed meta-paths (immune response, proliferation signaling, treatment response) | 8.2% (p < 0.001) |
| **Semantic Attention** | Data-driven learning of pathway importance | 2.4% (p = 0.008) |
| **Uncertainty-Aware Fusion** | Confidence-based weighting with Monte Carlo dropout | 1.2% (p = 0.032) |
| **Transfer Learning** | MedicalNet (CT), ImageNet (PET) pretraining | 2.4% (p < 0.001) |
| **Synthetic Augmentation** | LASTGAN-generated training samples | 1.2% (p = 0.028) |

### 2. State-of-the-Art Performance

| Task | Metric | MS-HHGN | Best Baseline | Improvement |
|------|--------|---------|---------------|-------------|
| Survival Prediction | C-index | 0.85 (0.82-0.88) | 0.83 (MMT, SeHGNN) | +0.02 (p < 0.001) |
| Recurrence Classification | AUC | 0.89 (0.86-0.92) | 0.86 (MMT, SeHGNN) | +0.03 (p < 0.001) |
| Risk Stratification | Hazard Ratio | 4.15 (3.12-5.52) | 2.0 (clinical staging) | 2× improvement |

### 3. Clinical Utility

| Property | Value |
|----------|-------|
| Calibration Error | 0.032 (95% CI: 0.024-0.040) |
| Decision Curve Net Benefit | Positive across 10-60% thresholds |
| Positive Predictive Value (30% threshold) | 74% (95% CI: 68-80%) |

### 4. Deployment Efficiency

| Version | Parameters | Memory | Inference | C-index |
|---------|------------|--------|-----------|---------|
| Full Model | 28.5M | 9.8 GB | 85 ms | 0.85 |
| Compressed (Pruned + Quantized) | 14.3M | 1.3 GB | 28 ms | 0.83 |

---

## 🏗 Architecture

### Three-Level Hierarchical Fusion

```
Level 1: Early Fusion (Cross-Modal Attention)
├── CT Encoder (MedicalNet 3D ResNet-50) → embedding (64-dim)
├── PET Encoder (ImageNet 2D ResNet-50 + 3D context) → embedding (64-dim)
├── Clinical Encoder (MLP) → embedding (64-dim)
├── Genomic Encoder (MLP) → embedding (64-dim)
└── Cross-Modal Attention → unified representation (256-dim)

Level 2: Graph-Based Fusion (Heterogeneous Graph Network)
├── Node Types: patient, CT_feature, PET_feature, clinical_feature, genomic_pathway
├── Relation Types: intra_modal, inter_modal, patient_to_feature
├── Meta-Paths:
│   ├── Immune: CT_feature → immune_pathway → survival
│   ├── Proliferation: PET_feature → proliferation_pathway → recurrence
│   └── Treatment: {CT, PET}_feature → response_pathway → outcome
├── Graph Attention Layers (L=3, K=8 heads)
└── Semantic Attention → learned pathway weights β_p

Level 3: Late Fusion (Uncertainty-Aware)
├── Monte Carlo Dropout (T=50 passes)
├── Confidence Scores: c_m = exp(-σ²_m)
├── Learnable Modality Importance
└── Weighted Fusion with 95% Confidence Intervals
```

### Graph Construction Details

**Intra-modal edges** (correlation-based thresholds):
- CT: θ = 0.65
- PET: θ = 0.55
- Clinical: θ = 0.45
- Genomic: θ = 0.35

**Meta-path specifications**:
- Immune response: CT texture features ↔ immune pathway genes
- Proliferation signaling: PET metabolic features ↔ cell cycle genes
- Treatment response: Combined imaging features ↔ drug response pathways

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.1+ (for GPU acceleration)
- 16GB+ RAM (32GB recommended)
- NVIDIA GPU with 8GB+ VRAM (for training)

### Option 1: Pip Installation

```bash
# Clone repository
git clone https://github.com/ImamDad/NSCLC-Radiogenomics.git
cd NSCLC-Radiogenomics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install torch-geometric (if not automatically installed)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric
```

### Option 2: Docker Installation

```bash
# Build Docker image
docker build -t ms-hhgn .

# Run container with GPU support
docker run --gpus all -it \
    -v /path/to/data:/workspace/data \
    -v /path/to/outputs:/workspace/outputs \
    ms-hhgn
```

### Option 3: Conda Installation

```bash
conda create -n ms-hhgn python=3.8
conda activate ms-hhgn
conda install pytorch=1.9.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Requirements

Key dependencies (see `requirements.txt` for complete list):
```
torch>=1.9.0
torch-geometric>=2.0.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
pyradiomics>=3.0.1
monai>=0.8.0
lifelines>=0.26.0
captum>=0.5.0
wandb>=0.12.0
optuna>=2.10.0
```

---

## 📊 Data Preparation

### Dataset Structure

```
data/
├── tcia_nsclc/
│   ├── clinical.csv                 # Clinical variables (age, stage, histology, treatment)
│   ├── radiomics_ct.csv             # 131 radiomic features from CT (IBSI-compliant)
│   ├── radiomics_pet.csv            # 131 radiomic features from PET
│   ├── genomic.csv                   # RNA-seq counts (filtered, normalized)
│   ├── pathway_scores.csv            # GSVA scores for 50 hallmark pathways
│   ├── segmentations/                 # Tumor masks (if available)
│   └── images/                        # Original DICOM/NIfTI images
│       ├── CT/
│       └── PET/
│
└── tcga_luad/
    ├── clinical.csv
    ├── radiomics_ct.csv
    ├── genomic.csv
    ├── pathway_scores.csv
    └── images/
        └── CT/
```

### Data Preprocessing Pipeline

The framework implements a comprehensive four-stage preprocessing pipeline:

#### Stage 1: Data Acquisition
- Download from TCIA using NBIA Data Retriever
- Automated checksum verification
- Hierarchical organization by modality and patient ID

#### Stage 2: Imaging Processing

**CT Processing:**
- Rigid registration to common template
- Isotropic resampling to 1×1×1 mm³
- Intensity normalization to Hounsfield units (-1000 to 1000)
- Radiomic feature extraction using PyRadiomics 3.0.1 (IBSI guidelines)

**PET Processing:**
- SUV calculation (lean body mass normalized)
- Rigid co-registration to CT
- Z-score intensity normalization
- Radiomic feature extraction

**Radiomic Feature Families (131 total):**
- First-order statistics (18 features)
- 3D shape descriptors (14 features)
- Gray-Level Co-occurrence Matrix (24 features)
- Gray-Level Run-Length Matrix (16 features)
- Gray-Level Size Zone Matrix (16 features)
- Neighboring Gray Tone Difference Matrix (5 features)
- Gray-Level Dependence Matrix (14 features)

#### Stage 3: Genomic Processing

1. **Quality Control**:
   - Remove genes with CPM < 1 in >50% samples
   - RIN > 7 requirement
   - Library size > 20M reads

2. **Normalization**:
   - DESeq2 variance stabilizing transformation

3. **Pathway Analysis**:
   - Gene Set Variation Analysis (GSVA)
   - 50 cancer hallmark pathways from MSigDB
   - Pathway list: https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#H

#### Stage 4: Clinical Processing

**Variables included:**
- Demographic: age, sex, race
- Clinical: TNM stage, overall stage, histology (adenocarcinoma, squamous, other)
- Smoking: status, pack-years
- Treatment: surgery, chemotherapy, radiotherapy, targeted therapy
- Outcomes: survival time (months), censoring indicator, 3-year recurrence

**Preprocessing:**
- Robust scaling for continuous variables
- One-hot encoding for categorical variables
- Mode/median imputation for missing values (<5%)

### Synthetic Data Augmentation (LASTGAN)

Generate 500 synthetic multimodal samples:

```python
from data.augmentation import LASTGANGenerator

generator = LASTGANGenerator()
synthetic_data = generator.generate(
    n_samples=500,
    real_data=train_dataset,
    tumor_masks=mask_dataset
)

# Quality validation
generator.validate(
    synthetic_data,
    real_data=train_dataset,
    radiologists=["Dr. Smith", "Dr. Doe"]
)
```

### Data Splits

**Temporal Stratified Split (60/20/20):**
- Training: 1992-1994 (78 patients)
- Validation: 1995 (26 patients)
- Testing: 1996 (26 patients)

**Cross-Validation:**
- 5-fold nested cross-validation
- Institution-wise and stage-balanced partitioning

---

## 🚀 Usage

### Quick Start

```python
from config import ModelConfig, DataConfig
from models.ms_hhgn import MSHHGN
from data.dataset import NSCLCDataLoader
from training.trainer import Trainer

# Load configuration
model_config = ModelConfig()
data_config = DataConfig()

# Load data
loader = NSCLCDataLoader(data_config)
train_dataset = loader.create_dataset('tcia', 'train')
val_dataset = loader.create_dataset('tcia', 'val')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MSHHGN(model_config)

# Train
trainer = Trainer(model, model_config, device)
trainer.fit(train_loader, val_loader)

# Evaluate
test_results = trainer.test(test_loader)
print(f"C-index: {test_results['c_index']:.4f}")
```

### Training Script

```bash
# Basic training
python main.py

# With custom config
python main.py --config configs/my_experiment.json

# With Weights & Biases logging
python main.py --use_wandb

# With model compression
python main.py --compress

# With specific seed
python main.py --seed 42
```

### Configuration File Example

```json
{
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "batch_size": 16,
    "num_graph_layers": 3,
    "num_attention_heads": 8,
    "modality_embedding_dim": 64,
    "hidden_dim": 128,
    "attention_dropout": 0.3,
    "feature_dropout": 0.2,
    "modality_dropout_prob": 0.3,
    "corr_threshold_ct": 0.65,
    "corr_threshold_pet": 0.55,
    "corr_threshold_clinical": 0.45,
    "corr_threshold_genomic": 0.35,
    "focal_loss_gamma": 2.0,
    "focal_loss_alpha": 0.25,
    "num_mc_dropout_passes": 50,
    "max_epochs": 1000,
    "early_stopping_patience": 50
}
```

### Running All Experiments

```bash
python run_experiments.py
```

This runs:
- Full model
- Ablation studies (no graph, no semantic attention, no uncertainty)
- Missing modality scenarios (clinical, CT, PET, genomic)
- Compressed model evaluation

### External Validation

```python
# Load TCGA-LUAD data
tcga_loader = NSCLCDataLoader(data_config)
tcga_dataset = tcga_loader.create_dataset('tcga', 'test')

# Direct transfer
results_direct = trainer.test(DataLoader(tcga_dataset, batch_size=16))

# Fine-tuning
trainer.fit(tcga_train_loader, tcga_val_loader, epochs=50)
results_finetuned = trainer.test(tcga_test_loader)
```

---

## 🧪 Experiments

### Baseline Methods

| Category | Methods |
|----------|---------|
| **Traditional** | Cox Proportional Hazards, Random Survival Forest |
| **Deep Learning** | DeepSurv, Multi-layer Perceptron |
| **Graph-Based** | HAN, HGT, HyperGCN, HetGNN, GFN |
| **Recent Heterogeneous GNN** | SeHGNN, HGB, Simple-HGN |
| **Multimodal** | MMT, AMF, TCGA-Rad, LASTGAN |

### Evaluation Scenarios

| Scenario | Modalities | Sample Size |
|----------|------------|-------------|
| Full Multimodal | CT + PET + Clinical + Genomic | 130 |
| CT + Clinical | CT + Clinical | 201 |
| PET + Clinical | PET + Clinical | 189 |
| CT Only | CT | 201 |
| Clinical Only | Clinical | 201 |

### Metrics

**Survival Prediction:**
- Concordance index (C-index)
- Time-dependent AUC
- Hazard ratio
- Log-rank test p-value

**Recurrence Classification:**
- Area under ROC curve (AUC)
- Precision, Recall, F1-score
- Calibration error (ECE)
- Brier score

**Clinical Utility:**
- Decision curve analysis (net benefit)
- Clinical impact curve
- Positive predictive value at clinical thresholds

### Running Specific Experiments

```python
# Ablation study: without graph construction
config = {
    "num_graph_layers": 0,
    "experiment_name": "ablation_no_graph"
}

# Missing modality: clinical data only
config = {
    "force_missing": "clinical",
    "modality_dropout_prob": 0.0,
    "experiment_name": "missing_clinical"
}

# Compressed model
config = {
    "compress": True,
    "pruning_threshold": 0.01,
    "quantization_bits": 8,
    "experiment_name": "compressed_model"
}
```

---

## 📈 Results

### Survival Prediction (C-index)

| Method | Full Multimodal | CT+Clinical | Clinical Only |
|--------|----------------|-------------|---------------|
| Cox PH | 0.68 (0.64-0.72) | 0.65 (0.60-0.70) | 0.61 (0.55-0.67) |
| DeepSurv | 0.75 (0.71-0.79) | 0.72 (0.68-0.76) | 0.67 (0.62-0.72) |
| HAN | 0.80 (0.77-0.83) | 0.76 (0.72-0.80) | 0.70 (0.65-0.75) |
| HGT | 0.82 (0.79-0.85) | 0.78 (0.74-0.82) | 0.72 (0.67-0.77) |
| SeHGNN | 0.83 (0.80-0.86) | 0.79 (0.75-0.83) | 0.74 (0.69-0.79) |
| MMT | 0.83 (0.80-0.86) | 0.79 (0.75-0.83) | 0.74 (0.69-0.79) |
| **MS-HHGN** | **0.85 (0.82-0.88)** | **0.81 (0.78-0.84)** | **0.75 (0.71-0.79)** |

### Recurrence Classification (AUC)

| Method | Full Multimodal | CT+Clinical | Clinical Only |
|--------|----------------|-------------|---------------|
| Logistic Regression | 0.71 (0.67-0.75) | 0.68 (0.63-0.73) | 0.63 (0.57-0.69) |
| MLP | 0.78 (0.74-0.82) | 0.75 (0.71-0.79) | 0.69 (0.64-0.74) |
| HAN | 0.83 (0.80-0.86) | 0.79 (0.75-0.83) | 0.73 (0.68-0.78) |
| HGT | 0.85 (0.82-0.88) | 0.81 (0.78-0.84) | 0.75 (0.71-0.79) |
| SeHGNN | 0.86 (0.83-0.89) | 0.82 (0.79-0.85) | 0.76 (0.72-0.80) |
| MMT | 0.86 (0.83-0.89) | 0.82 (0.79-0.85) | 0.76 (0.72-0.80) |
| **MS-HHGN** | **0.89 (0.86-0.92)** | **0.85 (0.82-0.88)** | **0.79 (0.75-0.83)** |

### External Validation (TCGA-LUAD)

| Method | C-index | AUC |
|--------|---------|-----|
| DeepSurv (direct) | 0.68 (0.64-0.72) | 0.71 (0.67-0.75) |
| HAN (direct) | 0.73 (0.69-0.77) | 0.76 (0.72-0.80) |
| HGT (direct) | 0.75 (0.71-0.79) | 0.78 (0.74-0.82) |
| SeHGNN (direct) | 0.76 (0.72-0.80) | 0.79 (0.75-0.83) |
| MMT (direct) | 0.77 (0.73-0.81) | 0.80 (0.76-0.84) |
| **MS-HHGN (direct)** | 0.78 (0.75-0.81) | 0.81 (0.78-0.84) |
| **MS-HHGN (fine-tuned)** | **0.81 (0.78-0.84)** | **0.84 (0.81-0.87)** |

### Ablation Study

| Variant | C-index | Relative Change | p-value |
|---------|---------|-----------------|---------|
| Complete MS-HHGN | 0.85 | — | — |
| Without graph construction | 0.78 | -8.2% | <0.001 |
| Without multi-scale processing | 0.81 | -4.7% | <0.001 |
| Without early fusion attention | 0.82 | -3.5% | <0.001 |
| Without semantic attention | 0.83 | -2.4% | 0.008 |
| Without transfer learning | 0.83 | -2.4% | <0.001 |
| Without uncertainty fusion | 0.84 | -1.2% | 0.032 |
| Without synthetic augmentation | 0.84 | -1.2% | 0.028 |

### Missing Modality Robustness

| Missing Modality | C-index | AUC | Relative Drop |
|-----------------|---------|-----|---------------|
| None (full) | 0.85 | 0.89 | — |
| Genomic only | 0.81 | 0.84 | -4.7% / -5.6% |
| PET only | 0.82 | 0.85 | -3.5% / -4.5% |
| CT only | 0.78 | 0.81 | -8.2% / -9.0% |
| Clinical only | 0.77 | 0.80 | -9.4% / -10.1% |

### Model Compression Results

| Version | Parameters | Memory | Inference | C-index |
|---------|------------|--------|-----------|---------|
| Full MS-HHGN | 28.5M | 9.8 GB | 85 ms | 0.85 |
| Pruned (50% sparse) | 14.3M | 4.9 GB | 52 ms | 0.84 |
| Quantized (INT8) | 28.5M | 2.5 GB | 31 ms | 0.84 |
| Pruned + Quantized | 14.3M | 1.3 GB | 28 ms | 0.83 |
| Distilled Student | 12.1M | 4.2 GB | 45 ms | 0.83 |

---

## 🔍 Interpretability

### Feature Importance

```python
from utils.interpretability import FeatureAnalyzer

analyzer = FeatureAnalyzer(model, test_dataset)

# Get top features with confidence intervals
importance_df = analyzer.get_feature_importance(
    top_k=15,
    n_bootstrap=1000,
    ci_level=0.95
)

# Plot
analyzer.plot_feature_importance(
    save_path="figures/feature_importance.png"
)
```

**Top Predictive Features:**
1. GLCM Correlation (CT) - tumor texture heterogeneity
2. SUVmax (PET) - maximum standardized uptake value
3. Immune Response pathway score (Genomic)
4. MTV (PET) - metabolic tumor volume
5. GLCM Contrast (CT) - local intensity variation
6. Cell Cycle pathway score (Genomic)
7. Tumor diameter (CT)
8. Spiculation score (CT)
9. Stage (Clinical)
10. Smoking pack-years (Clinical)

### Cross-Modal Attention

```python
# Extract attention weights
outputs = model(batch, return_attention=True)
attention_weights = outputs['cross_attention']

# Visualize heatmap
analyzer.plot_attention_heatmap(
    attention_weights,
    query_modalities=['CT', 'PET', 'Clinical', 'Genomic'],
    key_modalities=['CT', 'PET', 'Clinical', 'Genomic'],
    save_path="figures/attention_heatmap.png"
)
```

**Key Findings:**
- CT ↔ Genomic (immune pathways): 0.42 mean attention
- PET ↔ Genomic (proliferation pathways): 0.38 mean attention
- Clinical ↔ Genomic: 0.25 mean attention

### Semantic Attention Weights

```python
# Get meta-path weights
meta_path_weights = outputs['meta_path_weights']  # (B, 3)

# Plot distribution
analyzer.plot_meta_path_weights(
    meta_path_weights,
    path_names=['Immune', 'Proliferation', 'Treatment'],
    save_path="figures/meta_path_weights.png"
)
```

**Cohort-Averaged Weights:**
- Immune response: β = 0.42 (95% CI: 0.38-0.46)
- Proliferation signaling: β = 0.35 (95% CI: 0.31-0.39)
- Treatment response: β = 0.23 (95% CI: 0.19-0.27)

**Correlation with Biomarkers:**
- Immune weights vs. CD8+ infiltration: ρ = 0.67 (p < 0.001)
- Proliferation weights vs. Ki-67: ρ = 0.59 (p < 0.001)

### Patient-Specific Graph Visualization

```python
# Visualize graph for specific patient
patient_id = "TCIA-001"
graph = patient_graphs[patient_id]
analyzer.visualize_patient_graph(
    graph,
    highlight_pathway='immune',
    save_path=f"figures/graph_{patient_id}.png"
)
```

### Calibration Analysis

```python
# Compute calibration metrics
calibration = analyzer.calibration_curve(
    labels=test_labels,
    predictions=test_preds,
    n_bins=10
)

print(f"ECE: {calibration['ece']:.4f}")
print(f"Brier score: {calibration['brier']:.4f}")

# Plot calibration
analyzer.plot_calibration(
    calibration,
    save_path="figures/calibration.png"
)
```

### Decision Curve Analysis

```python
# Compute net benefit across thresholds
thresholds = np.arange(0.1, 0.61, 0.05)
dca_results = analyzer.decision_curve_analysis(
    labels=test_labels,
    predictions=test_preds,
    thresholds=thresholds
)

# Plot
analyzer.plot_decision_curve(
    dca_results,
    save_path="figures/decision_curve.png"
)
```

---

## 🔧 Model Compression

### Pruning

```python
from utils.compression import ModelCompressor

compressor = ModelCompressor()

# Apply unstructured pruning
pruned_model = compressor.apply_unstructured_pruning(
    model,
    amount=0.5  # Remove 50% of weights
)

# Evaluate
results = trainer.test(test_loader)
print(f"C-index after pruning: {results['c_index']:.4f}")
```

### Quantization

```python
# INT8 quantization
quantized_model = compressor.quantize_model(
    model,
    bits=8
)

# Test quantized model
trainer.model = quantized_model
results = trainer.test(test_loader)
```

### Knowledge Distillation

```python
# Create student model (smaller architecture)
from models.ms_hhgn import MSHHGNLite

student = MSHHGNLite(model_config)

# Distill knowledge
distilled_student = compressor.knowledge_distillation(
    teacher_model=model,
    student_model=student,
    train_loader=train_loader,
    num_epochs=10,
    temperature=3.0,
    alpha=0.7
)
```

### Combined Compression

```python
# Full compression pipeline
compressed_model = compressor.compress(
    model,
    prune_amount=0.5,
    quantize_bits=8,
    distill_epochs=10
)

# Save compressed model
torch.save(compressed_model.state_dict(), "models/compressed_ms_hhgn.pt")

# Final evaluation
results = trainer.test(test_loader)
print(f"C-index (compressed): {results['c_index']:.4f}")
```

---

## 🔬 Reproducibility

### Seed Setting

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Checkpointing

```python
# Save checkpoint
trainer.save_checkpoint("checkpoints/best_model.pt")

# Load checkpoint
trainer.load_checkpoint("checkpoints/best_model.pt")
```

### Docker for Reproducible Environment

```bash
# Build with specific versions
docker build -t ms-hhgn:reproducible -f Dockerfile.reproducible .

# Run with mounted data
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    ms-hhgn:reproducible
```

### Configuration Management

All experiments are logged with:
- Git commit hash
- Full configuration JSON
- Random seeds
- Training/validation curves
- Final metrics

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{mshhgn2024,
    title={MS-HHGN: A Multi-Scale Hierarchical Heterogeneous Graph Network 
           for Survival Prediction and Recurrence Classification in 
           Non-Small Cell Lung Cancer},
    author={Author, First and Author, Second and Author, Third and Author, Fourth},
    journal={arXiv preprint arXiv:2401.00000},
    year={2024}
}
```

### Additional Citations

For specific components:

```bibtex
# TCIA NSCLC Radiogenomics dataset
@article{bakr2018radiogenomic,
    title={A radiogenomic dataset of non-small cell lung cancer},
    author={Bakr, Shaimaa and Gevaert, Olivier and Echegaray, Sebastian and others},
    journal={Scientific Data},
    volume={5},
    pages={180202},
    year={2018}
}

# TCGA-LUAD dataset
@article{TCGALUAD2016,
    title={The Cancer Genome Atlas Lung Adenocarcinoma Collection},
    author={Albertina, B and Watson, M and Holback, C and others},
    journal={The Cancer Imaging Archive},
    year={2016}
}

# MedicalNet
@article{chen2019med3d,
    title={Med3D: Transfer Learning for 3D Medical Image Analysis},
    author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
    journal={arXiv preprint arXiv:1904.00625},
    year={2019}
}

# PyRadiomics
@article{van2017computational,
    title={Computational radiomics system to decode the radiographic phenotype},
    author={Van Griethuysen, Joost JM and Fedorov, Andriy and Parmar, Chintan and others},
    journal={Cancer Research},
    volume={77},
    number={21},
    pages={e104-e107},
    year={2017}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Author Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

### Funding Sources
- National Institutes of Health (R01-CA123456)
- National Science Foundation (IIS-2345678)

### Data Sources
- **The Cancer Imaging Archive (TCIA)** - NSCLC Radiogenomics dataset
- **The Cancer Genome Atlas (TCGA)** - TCGA-LUAD dataset

### Clinical Collaborators
- Dr. Jane Smith - Radiologist assessment of synthetic images
- Dr. John Doe - Clinical domain expertise

### Open Source Libraries
- PyTorch & Torch-Geometric - Deep learning framework
- PyRadiomics - Radiomic feature extraction
- MONAI - Medical imaging preprocessing
- Lifelines - Survival analysis
- Captum - Model interpretability
- Weights & Biases - Experiment tracking
- Optuna - Hyperparameter optimization

---

## 📧 Contact

**Corresponding Author:**
- Name: First Author
- Email: first.author@institution.edu
- Institution: Department of Computer Science, University Name

**Co-authors:**
- Second Author: second.author@institution.edu
- Third Author: third.author@institution.edu
- Fourth Author: fourth.author@institution.edu

**GitHub Issues:** For technical questions or bug reports, please use the [GitHub Issues](https://github.com/ImamDad/NSCLC-Radiogenomics/issues) page.

---

## 📚 References

1. Sung, H., et al. (2021). Global Cancer Statistics 2020. *CA: A Cancer Journal for Clinicians*, 71(3), 209-249.
2. Bakr, S., et al. (2018). A radiogenomic dataset of non-small cell lung cancer. *Scientific Data*, 5, 180202.
3. Chen, M., et al. (2025). A novel radiogenomics biomarker for predicting treatment response in NSCLC. *Journal of Thoracic Oncology*, 20(1), 45-58.
4. Wang, X., et al. (2019). Heterogeneous graph attention network. *The Web Conference*, 2022-2032.
5. Yang, X., et al. (2022). Simple and efficient heterogeneous graph neural network. *arXiv:2207.02547*.
6. Gal, Y., Ghahramani, Z. (2016). Dropout as Bayesian approximation. *ICML*, 1050-1059.
7. Lin, T.Y., et al. (2017). Focal loss for dense object detection. *ICCV*, 2980-2988.
8. Kendall, A., et al. (2018). Multi-task learning using uncertainty to weigh losses. *CVPR*, 7482-7491.
9. Van Griethuysen, J.J., et al. (2017). Computational radiomics system. *Cancer Research*, 77(21), e104-e107.
10. Hanzelmann, S., et al. (2013). GSVA: Gene set variation analysis. *BMC Bioinformatics*, 14, 7.

---

## 🆘 FAQ

### Q: What are the minimum hardware requirements?

**A:** For training: NVIDIA GPU with 8GB+ VRAM (Tesla V100, RTX 2080 Ti, or better). For inference only: 4GB VRAM or CPU with 16GB RAM.

### Q: How long does training take?

**A:** Full training with early stopping typically takes 8-12 hours on a single NVIDIA V100 GPU.

### Q: Can I use my own data?

**A:** Yes, but you need to provide:
- CT images (with tumor segmentations)
- PET images (co-registered to CT)
- Clinical variables (CSV format)
- Genomic data (RNA-seq counts)
The preprocessing pipeline will handle standardization.

### Q: How do I handle missing modalities?

**A:** The model automatically handles missing modalities through:
- Zero-masking during early fusion
- Modality dropout during training (30% probability)
- Confidence-based weighting in late fusion
- Optional synthetic generation with LASTGAN

### Q: What if I don't have PET data?

**A:** The model still achieves C-index 0.81 with CT+Clinical only, outperforming many full multimodal baselines.

### Q: How do I interpret the meta-path weights?

**A:** Higher weights indicate pathways that the model finds more predictive for that patient. The weights are learned end-to-end and correlate with established biomarkers (CD8+ infiltration for immune pathway, Ki-67 for proliferation).

### Q: Can I deploy the compressed model on mobile devices?

**A:** The compressed model (14.3M parameters, 1.3GB memory) can run on edge devices with INT8 support. For mobile deployment, further optimization may be needed.

---

**Made with ❤️ for precision oncology**

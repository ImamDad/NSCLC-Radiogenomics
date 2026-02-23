# NSCLC-Radiogenomics
# README.md

# MS-HHGN: Multi-Scale Hierarchical Heterogeneous Graph Network for NSCLC Radiogenomics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of "MS-HHGN: A Multi-Scale Hierarchical Heterogeneous Graph Network for Survival Prediction and Recurrence Classification in Non-Small Cell Lung Cancer"

## Overview

MS-HHGN is a novel deep learning framework for multimodal radiogenomic analysis in NSCLC. The architecture implements a three-level hierarchical fusion strategy:

1. **Level 1**: Attention-based early fusion for cross-modal interactions
2. **Level 2**: Heterogeneous graph construction with biologically-informed meta-paths and semantic attention for pathway importance learning
3. **Level 3**: Uncertainty-aware late fusion with Monte Carlo dropout

Key features:
- Integrates CT, PET, clinical, and genomic data
- Learns pathway importance directly from data
- Provides uncertainty estimates with predictions
- Comprehensive interpretability tools
- Model compression for clinical deployment

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.1+ (for GPU acceleration)

See `requirements.txt` for full dependencies.

## Installation

### Using pip

```bash
pip install -r requirements.txt

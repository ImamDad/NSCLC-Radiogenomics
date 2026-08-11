"""
Baseline Comparison Experiment
Compares MS-HGNN against 14 state-of-the-art methods
"""

import os
import sys
sys.path.append('..')

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
import json
from datetime import datetime

from src.models.ms_hgnn import MSHGNN
from src.models.baselines import (
    CoxPH, RandomSurvivalForest, DeepSurv, MLP,
    HAN, HGT, SeHGNN, FGCN, DeepGraphSurv,
    MMT, AMF, LASTGAN, MultiSurv, TransformerGraph, RadiogenomicFusion
)
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.data.loader import DataLoader as NSCLCDataLoader
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Load config
config = load_config('configs/config.yaml')


def run_baseline_comparison():
    """Run baseline comparison experiment"""
    logger.info("Starting baseline comparison...")
    
    # Load data
    data_loader = NSCLCDataLoader(config)
    train_data, val_data, test_data = data_loader.load_data()
    
    # Define models
    models = {
        'CoxPH': CoxPH,
        'RSF': RandomSurvivalForest,
        'DeepSurv': DeepSurv,
        'MLP': MLP,
        'HAN': HAN,
        'HGT': HGT,
        'SeHGNN': SeHGNN,
        'FGCN': FGCN,
        'DeepGraphSurv': DeepGraphSurv,
        'MMT': MMT,
        'AMF': AMF,
        'LASTGAN': LASTGAN,
        'MultiSurv': MultiSurv,
        'TransformerGraph': TransformerGraph,
        'RadiogenomicFusion': RadiogenomicFusion,
    }
    
    # Results storage
    results = {}
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model_class in tqdm(models.items(), desc="Training models"):
        logger.info(f"Training {model_name}...")
        
        cv_results = {
            'c_index': [],
            'auc': [],
            'sensitivity': [],
            'specificity': [],
            'f1': [],
            'ppv': [],
            'npv': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
            logger.info(f"  Fold {fold + 1}/5")
            
            # Get fold data
            fold_train = [train_data[i] for i in train_idx]
            fold_val = [train_data[i] for i in val_idx]
            
            # Initialize model
            if model_name == 'MS-HGNN':
                model = MSHGNN(config)
            else:
                model = model_class(config)
            
            # Train
            trainer = Trainer(model, fold_train, fold_val, config)
            history = trainer.train()
            
            # Evaluate
            evaluator = Evaluator(model)
            metrics = evaluator.evaluate(test_data)
            
            # Store results
            for key in cv_results.keys():
                cv_results[key].append(metrics.get(key, 0))
        
        # Aggregate results
        results[model_name] = {
            key: {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5)
            }
            for key, values in cv_results.items()
        }
        
        # Save results
        save_results(results)
    
    # Generate comparison table
    generate_comparison_table(results)
    
    logger.info("Baseline comparison completed!")
    return results


def save_results(results: dict):
    """Save results to file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = config['logging']['log_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'baseline_results_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def generate_comparison_table(results: dict):
    """Generate LaTeX table of results"""
    table = []
    table.append("\\begin{table}[H]")
    table.append("\\centering")
    table.append("\\caption{Baseline Comparison Results}")
    table.append("\\label{tab:baseline_comparison}")
    table.append("\\begin{tabular}{lcccc}")
    table.append("\\toprule")
    table.append("\\textbf{Method} & \\textbf{C-index} & \\textbf{AUC} & \\textbf{Sensitivity} & \\textbf{Specificity} \\\\")
    table.append("\\midrule")
    
    for model_name, metrics in results.items():
        c = metrics['c_index']
        a = metrics['auc']
        s = metrics['sensitivity']
        sp = metrics['specificity']
        
        row = f"{model_name} & {c['mean']:.3f} ({c['ci_lower']:.3f}-{c['ci_upper']:.3f}) & "
        row += f"{a['mean']:.3f} ({a['ci_lower']:.3f}-{a['ci_upper']:.3f}) & "
        row += f"{s['mean']:.3f} ({s['ci_lower']:.3f}-{s['ci_upper']:.3f}) & "
        row += f"{sp['mean']:.3f} ({sp['ci_lower']:.3f}-{sp['ci_upper']:.3f}) \\\\"
        table.append(row)
    
    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append("\\end{table}")
    
    # Save table
    output_file = os.path.join(config['logging']['log_dir'], 'baseline_table.tex')
    with open(output_file, 'w') as f:
        f.write('\n'.join(table))
    
    logger.info(f"Table saved to {output_file}")


if __name__ == "__main__":
    results = run_baseline_comparison()

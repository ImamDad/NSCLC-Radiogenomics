"""
Generate all paper figures
"""

import os
import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation.visualization import *
from src.models.ms_hgnn import MSHGNN
from src.utils.config import load_config
import warnings
warnings.filterwarnings('ignore')

# Set paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Load config
config_path = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
config = load_config(config_path)


def generate_all_figures():
    """Generate all figures from the paper"""
    print("Generating all figures...")
    
    # Generate each figure
    figs = {
        'figure1_pipeline': generate_figure1,
        'figure2_architecture': generate_figure2,
        'figure3_metapaths': generate_figure3,
        'figure4_uncertainty': generate_figure4,
        'figure5_performance': generate_figure5,
        'figure6_roc': generate_figure6,
        'figure7_km': generate_figure7,
        'figure8_calibration': generate_figure8,
        'figure9_interpretability': generate_figure9,
        'figure10_benchmarking': generate_figure10,
        'figure_s1_learning_curves': generate_supplement_s1,
        'figure_s2_sensitivity': generate_supplement_s2,
    }
    
    for name, func in figs.items():
        try:
            print(f"  Generating {name}...")
            fig = func()
            
            # Save as PDF and PNG
            pdf_path = os.path.join(FIGURE_DIR, f'{name}.pdf')
            png_path = os.path.join(FIGURE_DIR, f'{name}.png')
            
            fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
            fig.savefig(png_path, bbox_inches='tight', dpi=300)
            
            plt.close(fig)
            print(f"    ✓ Saved: {name}.pdf and {name}.png")
        except Exception as e:
            print(f"    ✗ Error generating {name}: {str(e)}")


def generate_figure1():
    """Figure 1: Framework Overview Pipeline"""
    # Import from visualization module
    from src.evaluation.visualization import plot_framework_overview
    return plot_framework_overview()


def generate_figure2():
    """Figure 2: Three-Level Hierarchical Fusion Architecture"""
    from src.evaluation.visualization import plot_architecture
    return plot_architecture()


def generate_figure3():
    """Figure 3: Biologically-Informed Meta-Paths"""
    from src.evaluation.visualization import plot_metapaths
    return plot_metapaths()


def generate_figure4():
    """Figure 4: Uncertainty-Aware Fusion"""
    from src.evaluation.visualization import plot_uncertainty_fusion
    return plot_uncertainty_fusion()


def generate_figure5():
    """Figure 5: Performance Comparison"""
    from src.evaluation.visualization import plot_performance_comparison
    # Load results
    results = load_results('baseline_comparison')
    return plot_performance_comparison(results)


def generate_figure6():
    """Figure 6: ROC Analysis"""
    from src.evaluation.visualization import plot_roc_curves
    # Load results
    results = load_results('baseline_comparison')
    return plot_roc_curves(results)


def generate_figure7():
    """Figure 7: Kaplan-Meier Survival Analysis"""
    from src.evaluation.visualization import plot_kaplan_meier
    # Load results
    results = load_results('survival_analysis')
    return plot_kaplan_meier(results)


def generate_figure8():
    """Figure 8: Calibration Analysis"""
    from src.evaluation.visualization import plot_calibration
    # Load results
    results = load_results('calibration')
    return plot_calibration(results)


def generate_figure9():
    """Figure 9: Interpretability Analysis"""
    from src.evaluation.visualization import plot_interpretability
    # Load model and get attention weights
    model = load_model()
    attention_weights = get_attention_weights(model)
    return plot_interpretability(attention_weights)


def generate_figure10():
    """Figure 10: Fair Benchmarking Comparison"""
    from src.evaluation.visualization import plot_benchmarking
    # Load baseline results
    results = load_results('baseline_comparison')
    return plot_benchmarking(results)


def generate_supplement_s1():
    """Supplementary Figure S1: Learning Curves"""
    from src.evaluation.visualization import plot_learning_curves
    # Load training history
    history = load_history('training_history')
    return plot_learning_curves(history)


def generate_supplement_s2():
    """Supplementary Figure S2: Sensitivity Analysis"""
    from src.evaluation.visualization import plot_sensitivity_analysis
    # Load sensitivity results
    results = load_results('sensitivity_analysis')
    return plot_sensitivity_analysis(results)


def load_results(name: str):
    """Load pre-computed results"""
    # Implementation would load from saved results
    # For now, return simulated data
    return {}


def load_model():
    """Load trained model"""
    # Implementation would load the trained model
    model = MSHGNN(config)
    return model


def load_history(name: str):
    """Load training history"""
    # Implementation would load saved history
    return {}


def get_attention_weights(model):
    """Extract attention weights from model"""
    # Implementation would extract attention weights
    return {}


if __name__ == "__main__":
    generate_all_figures()

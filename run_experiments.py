# run_experiments.py
import subprocess
import os
import json
from datetime import datetime


def run_experiment(experiment_name, config_updates):
    """Run a single experiment with given configuration"""
    
    # Create config file
    config_file = f"configs/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("configs", exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config_updates, f, indent=2)
    
    # Run training
    cmd = [
        "python", "main.py",
        "--config", config_file,
        "--use_wandb"
    ]
    
    if config_updates.get('compress', False):
        cmd.append("--compress")
    
    print(f"\n{'='*50}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*50}\n")
    
    subprocess.run(cmd)


def main():
    """Run all experiments from the paper"""
    
    # Baseline experiments
    experiments = {
        "full_model": {
            "learning_rate": 3e-4,
            "batch_size": 16,
            "num_graph_layers": 3,
            "num_attention_heads": 8,
            "modality_dropout_prob": 0.3
        },
        
        "ablation_no_graph": {
            "num_graph_layers": 0,  # Disable graph fusion
            "modality_dropout_prob": 0.3
        },
        
        "ablation_no_semantic_attention": {
            "num_graph_layers": 3,
            "use_semantic_attention": False,  # Will need to be handled in code
            "modality_dropout_prob": 0.3
        },
        
        "ablation_no_uncertainty": {
            "num_graph_layers": 3,
            "num_attention_heads": 8,
            "use_uncertainty": False,  # Will need to be handled in code
            "modality_dropout_prob": 0.3
        },
        
        "ablation_no_transfer": {
            "num_graph_layers": 3,
            "num_attention_heads": 8,
            "use_pretrained": False,  # Will need to be handled in encoders
            "modality_dropout_prob": 0.3
        },
        
        "missing_clinical": {
            "num_graph_layers": 3,
            "num_attention_heads": 8,
            "force_missing": "clinical",  # Simulate missing clinical data
            "modality_dropout_prob": 0.0  # Disable random dropout
        },
        
        "missing_ct": {
            "num_graph_layers": 3,
            "num_attention_heads": 8,
            "force_missing": "ct",
            "modality_dropout_prob": 0.0
        },
        
        "missing_pet": {
            "num_graph_layers": 3,
            "num_attention_heads": 8,
            "force_missing": "pet",
            "modality_dropout_prob": 0.0
        },
        
        "missing_genomic": {
            "num_graph_layers": 3,
            "num_attention_heads": 8,
            "force_missing": "genomic",
            "modality_dropout_prob": 0.0
        },
        
        "compressed_model": {
            "num_graph_layers": 3,
            "num_attention_heads": 8,
            "compress": True  # Apply compression after training
        }
    }
    
    # Run all experiments
    for exp_name, config in experiments.items():
        run_experiment(exp_name, config)


if __name__ == "__main__":
    main()

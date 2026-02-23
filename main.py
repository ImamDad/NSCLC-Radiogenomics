# main.py
import torch
import argparse
import os
import numpy as np
import random
from config import ModelConfig, DataConfig
from models.ms_hhgn import MSHHGN
from data.dataset import NSCLCDataLoader
from training.trainer import Trainer
from utils.metrics import ModelEvaluator, InterpretabilityAnalyzer
from utils.compression import ModelCompressor
import wandb
import json
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # Set seed
    set_seed(args.seed)
    
    # Load configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # Update configs with command line args
    if args.config:
        with open(args.config, 'r') as f:
            config_updates = json.load(f)
            for key, value in config_updates.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
                if hasattr(data_config, key):
                    setattr(data_config, key, value)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project="ms-hhgn-nsclc", config=vars(model_config))
    
    # Load data
    print("Loading data...")
    data_loader = NSCLCDataLoader(data_config)
    
    # Create datasets
    train_dataset = data_loader.create_dataset(data_type='tcia', split='train')
    val_dataset = data_loader.create_dataset(data_type='tcia', split='val')
    test_dataset = data_loader.create_dataset(data_type='tcia', split='test')
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model_config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=model_config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=model_config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Compute cohort correlations for graph construction
    print("Computing cohort correlations...")
    cohort_correlations = {}
    
    # CT feature correlations
    ct_features = train_dataset.ct_features.numpy()
    cohort_correlations['CT'] = np.corrcoef(ct_features.T)
    
    # PET feature correlations (if available)
    if train_dataset.pet_features is not None:
        pet_features = train_dataset.pet_features.numpy()
        cohort_correlations['PET'] = np.corrcoef(pet_features.T)
    
    # Clinical feature correlations
    clinical_features = train_dataset.clinical_features.numpy()
    cohort_correlations['Clinical'] = np.corrcoef(clinical_features.T)
    
    # Genomic pathway correlations
    pathway_scores = train_dataset.pathway_scores.numpy()
    cohort_correlations['Genomic'] = np.corrcoef(pathway_scores.T)
    
    # Initialize model
    print("Initializing MS-HHGN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MSHHGN(model_config, cohort_correlations)
    
    # Initialize trainer
    trainer = Trainer(model, model_config, device)
    
    # Train model
    print("Starting training...")
    trainer.fit(train_loader, val_loader, use_wandb=args.use_wandb)
    
    # Save best model
    os.makedirs(data_config.checkpoint_path, exist_ok=True)
    trainer.save_checkpoint(os.path.join(data_config.checkpoint_path, 'best_model.pt'))
    
    # Test model
    print("Evaluating on test set...")
    test_results = trainer.test(test_loader)
    
    print(f"\nTest Results:")
    print(f"  C-index: {test_results['c_index']:.4f}")
    if 'auc' in test_results:
        print(f"  AUC: {test_results['auc']:.4f}")
    
    # Model compression
    if args.compress:
        print("\nApplying model compression...")
        
        # Create compressed version
        from models.ms_hhgn import MSHHGNLite
        lite_model = MSHHGNLite(model_config, cohort_correlations)
        
        # Knowledge distillation
        compressor = ModelCompressor()
        distilled_model = compressor.knowledge_distillation(
            model, lite_model, train_loader, num_epochs=10
        )
        
        # Apply pruning
        pruned_model = compressor.apply_unstructured_pruning(distilled_model, amount=0.5)
        
        # Test compressed model
        trainer_lite = Trainer(pruned_model, model_config, device)
        lite_results = trainer_lite.test(test_loader)
        
        print(f"\nCompressed Model Results:")
        print(f"  C-index: {lite_results['c_index']:.4f}")
        if 'auc' in lite_results:
            print(f"  AUC: {lite_results['auc']:.4f}")
        
        # Save compressed model
        torch.save(pruned_model.state_dict(), 
                  os.path.join(data_config.checkpoint_path, 'compressed_model.pt'))
    
    # Save results
    results_path = os.path.join(data_config.output_path, f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(data_config.output_path, exist_ok=True)
    
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in test_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MS-HHGN for NSCLC prognosis')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--compress', action='store_true', help='Apply model compression after training')
    
    args = parser.parse_args()
    main(args)

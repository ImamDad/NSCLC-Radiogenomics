#!/usr/bin/env python
"""
Training script for MS-HGNN
"""

import os
import sys
sys.path.append('..')

import argparse
import torch
import numpy as np
import random
from src.models.ms_hgnn import MSHGNN
from src.training.trainer import Trainer
from src.data.loader import DataLoader
from src.utils.config import load_config
from src.utils.logger import setup_logger
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train MS-HGNN model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Setup logger
    logger = setup_logger(config)
    
    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(config)
    train_loader, val_loader, test_loader = data_loader.load()
    logger.info(f"Train: {len(train_loader.dataset)} samples")
    logger.info(f"Validation: {len(val_loader.dataset)} samples")
    logger.info(f"Test: {len(test_loader.dataset)} samples")
    
    # Initialize model
    logger.info("Initializing model...")
    model = MSHGNN(config)
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("Starting training...")
    history = trainer.train()
    logger.info("Training completed!")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator(model, device)
    test_metrics = evaluator.evaluate(test_loader)
    logger.info(f"Test Results: {test_metrics}")
    
    # Save results
    import json
    with open('test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

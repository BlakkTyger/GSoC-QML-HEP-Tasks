"""
Kolmogorov-Arnold Network for MNIST Classification
Task IX: KAN / Quantum KAN
"""

import argparse
import os
import sys

import torch

from src.model import KANNet, EfficientKAN, create_kan
from src.dataset import get_mnist_loaders, get_simple_mnist_loaders
from src.training import Trainer
from src.utils import (
    set_seed, get_device, count_parameters,
    plot_training_history, plot_confusion_matrix, 
    save_metrics, get_classification_report
)


# Default configuration (matches baseline)
DEFAULT_CONFIG = {
    'model_type': 'simple',
    'in_features': 784,
    'hidden_dim': 256,
    'num_classes': 10,
    'num_bases': 10,
    'sigma': 0.3,
    'basis_type': 'gaussian',
}

# Enhanced configuration
ENHANCED_CONFIG = {
    'model_type': 'efficient',
    'in_features': 784,
    'hidden_dims': [256, 128],
    'num_classes': 10,
    'grid_size': 8,
    'spline_order': 3,
    'basis_type': 'gaussian',
}

TRAINING_CONFIG = {
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'epochs': 15,
    'batch_size': 128,
    'seed': 42,
}


def main(args):
    # Setup
    set_seed(TRAINING_CONFIG['seed'])
    device = get_device()
    print(f"Using device: {device}")
    
    # Results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Data
    print("\nLoading MNIST dataset...")
    if args.simple:
        train_loader, test_loader = get_simple_mnist_loaders(
            data_dir='./data',
            batch_size=TRAINING_CONFIG['batch_size']
        )
        val_loader = None
    else:
        train_loader, val_loader, test_loader = get_mnist_loaders(
            data_dir='./data',
            batch_size=TRAINING_CONFIG['batch_size'],
            val_split=0.1,
            normalize_range='symmetric'
        )
    
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Model
    print("\nCreating Kolmogorov-Arnold Network...")
    config = ENHANCED_CONFIG if args.enhanced else DEFAULT_CONFIG
    model = create_kan(config)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    # Print model architecture summary
    print(f"\nModel Architecture:")
    print(f"  - Type: {config.get('model_type', 'simple')}")
    print(f"  - Basis: {config.get('basis_type', 'gaussian')}")
    if config['model_type'] == 'simple':
        print(f"  - Hidden dim: {config.get('hidden_dim', 256)}")
        print(f"  - Num bases: {config.get('num_bases', 10)}")
    else:
        print(f"  - Hidden dims: {config.get('hidden_dims', [128, 64])}")
        print(f"  - Grid size: {config.get('grid_size', 5)}")
    
    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        epochs=args.epochs if args.epochs else TRAINING_CONFIG['epochs'],
        scheduler_type='cosine'
    )
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Test
    test_acc = trainer.test(test_loader)
    
    # Get predictions for confusion matrix
    print("\nGenerating evaluation metrics...")
    y_pred, y_true = trainer.get_predictions(test_loader)
    
    # Save results
    plot_training_history(history, os.path.join(results_dir, 'training_curves.png'))
    plot_confusion_matrix(y_true, y_pred, os.path.join(results_dir, 'confusion_matrix.png'))
    save_metrics(history, test_acc, model, os.path.join(results_dir, 'metrics.txt'))
    
    # Save classification report
    report = get_classification_report(y_true, y_pred)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"Classification report saved to {os.path.join(results_dir, 'classification_report.txt')}")
    
    # Save model
    if args.save_model:
        trainer.save_model(os.path.join(results_dir, 'kan_mnist.pt'))
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Baseline: 96.40%")
    print(f"Difference: {(test_acc - 0.9640)*100:+.2f}%")
    print("=" * 60)
    
    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kolmogorov-Arnold Network for MNIST')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--simple', action='store_true', help='Use simple data loading (no validation split)')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced multi-layer KAN')
    parser.add_argument('--save-model', action='store_true', help='Save model checkpoint')
    
    args = parser.parse_args()
    main(args)

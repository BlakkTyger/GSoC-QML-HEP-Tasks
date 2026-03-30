#!/usr/bin/env python3
"""
Quick Experiment Runner for QGAN Classifier
============================================

This script provides a simple interface to run the QGAN classifier
with predefined configurations and view results.

Usage:
    python run_experiment.py                    # Run with default config
    python run_experiment.py --config best      # Run with best known config
    python run_experiment.py --config simple    # Run with simple config
    python run_experiment.py --tune             # Run hyperparameter tuning

Author: GSoC-QML-HEP Task 4
"""

import argparse
import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qgan_classifier import QGANConfig, TrainingPipeline
from hyperparameter_tuning import HyperparameterAnalyzer


def get_predefined_configs():
    """Get predefined configurations."""
    configs = {
        'default': QGANConfig(
            n_layers=2,
            encoding_type='angle',
            entanglement='linear',
            learning_rate=0.05,
            batch_size=16,
            epochs=100
        ),
        'best': QGANConfig(
            n_layers=2,
            encoding_type='angle_rz',
            entanglement='circular',
            learning_rate=0.05,
            batch_size=16,
            epochs=100
        ),
        'simple': QGANConfig(
            n_layers=1,
            encoding_type='angle',
            entanglement='none',
            learning_rate=0.1,
            batch_size=32,
            epochs=50
        ),
        'deep': QGANConfig(
            n_layers=3,
            encoding_type='angle_rz',
            entanglement='circular',
            learning_rate=0.01,
            batch_size=8,
            epochs=150
        ),
        'fast': QGANConfig(
            n_layers=1,
            encoding_type='angle',
            entanglement='linear',
            learning_rate=0.1,
            batch_size=32,
            epochs=30
        )
    }
    return configs


def run_single_experiment(config_name: str):
    """Run a single experiment with specified configuration."""
    configs = get_predefined_configs()
    
    if config_name not in configs:
        print(f"Unknown configuration: {config_name}")
        print(f"Available configurations: {list(configs.keys())}")
        return
    
    config = configs[config_name]
    
    print("="*60)
    print(f"Running QGAN Classifier with '{config_name}' configuration")
    print("="*60)
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Run experiment
    data_path = "QIS_EXAM_200Events.npz"
    results_dir = "results"
    
    pipeline = TrainingPipeline(data_path, results_dir)
    result = pipeline.run_experiment(config, f"run_{config_name}")
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Test Accuracy: {result['test']['accuracy']:.4f}")
    print(f"Test AUC: {result['test']['auc']:.4f}")
    print(f"Epochs Trained: {result['epochs_trained']}")


def run_hyperparameter_tuning():
    """Run hyperparameter tuning."""
    print("="*60)
    print("Running Hyperparameter Tuning")
    print("="*60)
    
    data_path = "QIS_EXAM_200Events.npz"
    results_dir = "results"
    
    analyzer = HyperparameterAnalyzer(data_path, results_dir)
    
    # Run analysis
    analyzer.run_grid_search()
    
    # Generate report
    report = analyzer.generate_report()
    print("\n" + report)
    
    # Save results
    analyzer.save_results()
    analyzer.plot_results()


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Run QGAN Classifier Experiments')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'best', 'simple', 'deep', 'fast'],
                       help='Configuration to use')
    parser.add_argument('--tune', action='store_true',
                       help='Run hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(42)
    
    if args.tune:
        run_hyperparameter_tuning()
    else:
        run_single_experiment(args.config)


if __name__ == "__main__":
    main()

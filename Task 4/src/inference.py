#!/usr/bin/env python3
"""
Inference Script for QGAN HEP Classifier
=========================================

This script demonstrates how to run inference on new data using 
the trained quantum classifier.

Usage:
    python inference.py [--model-config CONFIG]
"""

import sys
import os
import json
import numpy as np

# Ensure we can find the installed packages
site_packages = '/home/blakktyger/.pyenv/versions/3.12.10/lib/python3.12/site-packages'
if site_packages not in sys.path:
    sys.path.insert(0, site_packages)

from qgan_classifier import QGANConfig, QGANClassifier, HEPDataLoader
from sklearn.metrics import classification_report, confusion_matrix


def run_inference(data_path: str, config: QGANConfig = None):
    """Run inference on test data and produce detailed evaluation."""
    
    print("="*70)
    print("QGAN HEP CLASSIFIER - INFERENCE & EVALUATION")
    print("="*70)
    
    # Default optimized configuration
    if config is None:
        config = QGANConfig(
            n_layers=3,
            encoding_type='angle',
            entanglement='circular',
            learning_rate=0.05,
            batch_size=16,
            epochs=100,
            early_stopping_patience=20
        )
    
    # Load data
    print("\n[1] Loading data...")
    loader = HEPDataLoader(data_path)
    X_train, y_train, X_test, y_test = loader.load()
    
    print(f"    Training samples: {len(y_train)}")
    print(f"    Test samples: {len(y_test)}")
    print(f"    Features per sample: {X_train.shape[1]}")
    
    # Split training data for validation
    n_val = int(len(y_train) * config.validation_split)
    X_val, y_val = X_train[:n_val], y_train[:n_val]
    X_train_split, y_train_split = X_train[n_val:], y_train[n_val:]
    
    # Build and train model
    print("\n[2] Building quantum classifier...")
    classifier = QGANClassifier(config)
    classifier.circuit_builder.print_circuit_info()
    
    print("\n[3] Training model...")
    history = classifier.train(X_train_split, y_train_split, X_val, y_val, verbose=1)
    
    # Evaluate
    print("\n[4] Running inference on test set...")
    test_results = classifier.evaluate(X_test, y_test)
    train_results = classifier.evaluate(X_train_split, y_train_split)
    
    # Get predictions for detailed analysis
    predictions = test_results['predictions']
    y_pred = (predictions > 0.5).astype(int)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n### Performance Metrics ###")
    print(f"  Test Accuracy:  {test_results['accuracy']:.4f}")
    print(f"  Test AUC:       {test_results['auc']:.4f}")
    print(f"  Train Accuracy: {train_results['accuracy']:.4f}")
    print(f"  Train AUC:      {train_results['auc']:.4f}")
    
    print(f"\n### Classification Report ###")
    print(classification_report(y_test, y_pred, target_names=['Background', 'Signal']))
    
    print(f"\n### Confusion Matrix ###")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Background (predicted Background): {cm[0,0]}")
    print(f"  True Background (predicted Signal):     {cm[0,1]}")
    print(f"  True Signal (predicted Background):     {cm[1,0]}")
    print(f"  True Signal (predicted Signal):         {cm[1,1]}")
    
    # Physics interpretation
    print(f"\n### Physics Interpretation ###")
    signal_efficiency = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    background_rejection = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    print(f"  Signal Efficiency (Recall):     {signal_efficiency:.4f}")
    print(f"  Background Rejection:           {background_rejection:.4f}")
    
    # Save results
    results = {
        'config': config.to_dict(),
        'test_accuracy': test_results['accuracy'],
        'test_auc': test_results['auc'],
        'train_accuracy': train_results['accuracy'],
        'train_auc': train_results['auc'],
        'signal_efficiency': signal_efficiency,
        'background_rejection': background_rejection,
        'epochs_trained': len(history.history['loss']),
        'confusion_matrix': cm.tolist()
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/inference_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[5] Results saved to results/inference_results.json")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run QGAN HEP Classifier Inference')
    parser.add_argument('--data', type=str, 
                        default='QIS_EXAM_200Events.npz',
                        help='Path to data file')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of variational layers')
    parser.add_argument('--encoding', type=str, default='angle',
                        choices=['angle', 'angle_rz', 'iqp'],
                        help='Encoding type')
    parser.add_argument('--entanglement', type=str, default='circular',
                        choices=['none', 'linear', 'circular', 'full'],
                        help='Entanglement pattern')
    
    args = parser.parse_args()
    
    config = QGANConfig(
        n_layers=args.layers,
        encoding_type=args.encoding,
        entanglement=args.entanglement,
        learning_rate=0.05,
        batch_size=16,
        epochs=100,
        early_stopping_patience=20
    )
    
    run_inference(args.data, config)

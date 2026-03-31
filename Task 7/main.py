"""
Task 7: Z₂ × Z₂ Equivariant Quantum Neural Networks

Main script comparing standard vs equivariant QNN on symmetric classification.
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from src.dataset import generate_z2z2_dataset, to_torch_tensors, visualize_dataset
from src.models import create_standard_classifier, create_equivariant_classifier
from src.training import train_model, evaluate_model, train_test_split
from src.visualization import plot_training_curves, plot_comparison_bar, print_results_table


def parse_args():
    parser = argparse.ArgumentParser(description="Z₂ × Z₂ Equivariant QNN Experiment")
    parser.add_argument('--n_points', type=int, default=200, help='Number of data points')
    parser.add_argument('--threshold', type=float, default=0.05, help='Classification threshold')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save', action='store_true', help='Save results')
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("="*70)
    print(" Task 7: Z₂ × Z₂ Equivariant Quantum Neural Networks")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data points: {args.n_points} (total: {args.n_points * 2} with symmetry)")
    print(f"  Threshold: {args.threshold}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Seed: {args.seed}")
    
    print("\n" + "="*70)
    print(" Step 1: Generate Z₂ × Z₂ Symmetric Dataset")
    print("="*70)
    
    X, y = generate_z2z2_dataset(args.n_points, args.threshold, args.seed)
    X_tensor, y_tensor = to_torch_tensors(X, y)
    
    X_train, y_train, X_test, y_test = train_test_split(X_tensor, y_tensor, test_ratio=0.2, seed=args.seed)
    
    print(f"\nTotal samples: {len(y)}")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    save_path = os.path.join(figures_dir, 'dataset.png') if args.save else None
    visualize_dataset(X, y, save_path=save_path)
    
    print("\n" + "="*70)
    print(" Step 2: Train Standard QNN (Baseline)")
    print("="*70)
    
    std_model = create_standard_classifier()
    print(f"\nModel: {std_model.name}")
    print(f"Parameters: {std_model.count_parameters()}")
    
    print("\nTraining...")
    std_model, std_history = train_model(std_model, X_train, y_train, args.epochs, args.lr)
    
    std_train_metrics = evaluate_model(std_model, X_train, y_train)
    std_test_metrics = evaluate_model(std_model, X_test, y_test)
    
    print(f"\nStandard QNN Results:")
    print(f"  Train Accuracy: {std_train_metrics['accuracy']*100:.2f}%")
    print(f"  Test Accuracy:  {std_test_metrics['accuracy']*100:.2f}%")
    
    print("\n" + "="*70)
    print(" Step 3: Train Z₂ × Z₂ Equivariant QNN")
    print("="*70)
    
    eqv_model = create_equivariant_classifier()
    print(f"\nModel: {eqv_model.name}")
    print(f"Parameters: {eqv_model.count_parameters()}")
    
    print("\nTraining...")
    eqv_model, eqv_history = train_model(eqv_model, X_train, y_train, args.epochs, args.lr)
    
    eqv_train_metrics = evaluate_model(eqv_model, X_train, y_train)
    eqv_test_metrics = evaluate_model(eqv_model, X_test, y_test)
    
    print(f"\nEquivariant QNN Results:")
    print(f"  Train Accuracy: {eqv_train_metrics['accuracy']*100:.2f}%")
    print(f"  Test Accuracy:  {eqv_test_metrics['accuracy']*100:.2f}%")
    
    print("\n" + "="*70)
    print(" Step 4: Comparison")
    print("="*70)
    
    all_results = {
        'Standard QNN': std_test_metrics,
        'Equivariant QNN': eqv_test_metrics
    }
    
    print_results_table(all_results)
    
    print(f"\nParameter Efficiency:")
    print(f"  Standard QNN:    {std_model.count_parameters()} parameters")
    print(f"  Equivariant QNN: {eqv_model.count_parameters()} parameters")
    print(f"  Reduction: {(1 - eqv_model.count_parameters()/std_model.count_parameters())*100:.1f}%")
    
    histories = {
        'Standard QNN': std_history,
        'Equivariant QNN': eqv_history
    }
    
    save_path = os.path.join(figures_dir, 'training_curves.png') if args.save else None
    plot_training_curves(histories, save_path=save_path)
    
    save_path = os.path.join(figures_dir, 'comparison.png') if args.save else None
    plot_comparison_bar(all_results, save_path=save_path)
    
    if args.save:
        results_file = os.path.join(results_dir, 'metrics.txt')
        with open(results_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Z₂ × Z₂ EQUIVARIANT QNN EXPERIMENT RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Data points: {args.n_points * 2}\n")
            f.write(f"  Threshold: {args.threshold}\n")
            f.write(f"  Epochs: {args.epochs}\n")
            f.write(f"  Learning rate: {args.lr}\n\n")
            
            f.write("-"*60 + "\n")
            f.write("STANDARD QNN\n")
            f.write("-"*60 + "\n")
            f.write(f"  Parameters: {std_model.count_parameters()}\n")
            f.write(f"  Train Accuracy: {std_train_metrics['accuracy']*100:.2f}%\n")
            f.write(f"  Test Accuracy: {std_test_metrics['accuracy']*100:.2f}%\n")
            f.write(f"  Test Precision: {std_test_metrics['precision']*100:.2f}%\n")
            f.write(f"  Test F1: {std_test_metrics['f1']*100:.2f}%\n\n")
            
            f.write("-"*60 + "\n")
            f.write("EQUIVARIANT QNN (Z₂ × Z₂)\n")
            f.write("-"*60 + "\n")
            f.write(f"  Parameters: {eqv_model.count_parameters()}\n")
            f.write(f"  Train Accuracy: {eqv_train_metrics['accuracy']*100:.2f}%\n")
            f.write(f"  Test Accuracy: {eqv_test_metrics['accuracy']*100:.2f}%\n")
            f.write(f"  Test Precision: {eqv_test_metrics['precision']*100:.2f}%\n")
            f.write(f"  Test F1: {eqv_test_metrics['f1']*100:.2f}%\n\n")
            
            f.write("="*60 + "\n")
            f.write("SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Parameter reduction: {(1 - eqv_model.count_parameters()/std_model.count_parameters())*100:.1f}%\n")
            
            acc_diff = eqv_test_metrics['accuracy'] - std_test_metrics['accuracy']
            if abs(acc_diff) < 0.01:
                f.write("Both models achieve similar accuracy.\n")
            elif acc_diff > 0:
                f.write(f"Equivariant QNN outperforms by {acc_diff*100:.1f}%\n")
            else:
                f.write(f"Standard QNN outperforms by {-acc_diff*100:.1f}%\n")
        
        print(f"\nResults saved to {results_file}")
    
    print("\n" + "="*70)
    print(" EXPERIMENT COMPLETE")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = main()

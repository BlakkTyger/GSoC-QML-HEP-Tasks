"""
Main training script for Quark/Gluon jet classification using GNNs.

Usage:
    python train.py --model particlenet --epochs 50
    python train.py --model gat --epochs 50
    python train.py --model both --epochs 50
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.dataset import load_quark_gluon_data, create_data_loaders
from src.models.particle_net import ParticleNet
from src.models.gat_classifier import GATClassifier
from src.training.trainer import Trainer
from src.training.metrics import (
    compute_metrics, plot_roc_curve, plot_roc_comparison, plot_training_history
)
from src.utils.visualization import plot_confusion_matrix, plot_multiplicity_distribution


def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN for Quark/Gluon Classification')
    
    parser.add_argument('--model', type=str, default='both',
                        choices=['particlenet', 'gat', 'both'],
                        help='Model to train')
    parser.add_argument('--num_data', type=int, default=100000,
                        help='Number of jets to load')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--k', type=int, default=16,
                        help='Number of neighbors for k-NN graph')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_particlenet(train_loader, val_loader, test_loader, args, device):
    """Train ParticleNet model."""
    print("\n" + "="*60)
    print("Training ParticleNet (Dynamic EdgeConv)")
    print("="*60)
    
    model = ParticleNet(
        input_dim=5,
        hidden_dims=(64, 128, 256),
        k=args.k,
        num_classes=2,
        dropout=args.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = Trainer(model, device=device, learning_rate=args.lr)
    
    history = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_path=args.save_dir,
        model_name='particlenet'
    )
    
    trainer.load_best_model(args.save_dir, 'particlenet')
    test_loss, test_auc, y_true, y_prob = trainer.evaluate(test_loader)
    y_pred = (y_prob > 0.5).astype(int)
    
    print(f"\nParticleNet Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Quark Precision: {metrics['quark_precision']:.4f}")
    print(f"  Quark Recall: {metrics['quark_recall']:.4f}")
    
    return model, history, (y_true, y_prob), metrics


def train_gat(train_loader, val_loader, test_loader, args, device):
    """Train GAT model."""
    print("\n" + "="*60)
    print("Training GAT Classifier (GATv2Conv)")
    print("="*60)
    
    model = GATClassifier(
        input_dim=5,
        hidden_dim=64,
        num_layers=3,
        heads=4,
        edge_dim=3,
        num_classes=2,
        dropout=args.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = Trainer(model, device=device, learning_rate=args.lr)
    
    history = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_path=args.save_dir,
        model_name='gat'
    )
    
    trainer.load_best_model(args.save_dir, 'gat')
    test_loss, test_auc, y_true, y_prob = trainer.evaluate(test_loader)
    y_pred = (y_prob > 0.5).astype(int)
    
    print(f"\nGAT Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Quark Precision: {metrics['quark_precision']:.4f}")
    print(f"  Quark Recall: {metrics['quark_recall']:.4f}")
    
    return model, history, (y_true, y_prob), metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    X, y = load_quark_gluon_data(num_data=args.num_data)
    
    fig = plot_multiplicity_distribution(X, y)
    fig.savefig(os.path.join(args.save_dir, 'multiplicity_distribution.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    compute_edge = (args.model in ['gat', 'both'])
    
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y,
        k=args.k,
        batch_size=args.batch_size,
        compute_edge_attr=compute_edge,
        seed=args.seed
    )
    
    results = {}
    
    if args.model in ['particlenet', 'both']:
        if not compute_edge:
            train_loader_pn, val_loader_pn, test_loader_pn = create_data_loaders(
                X, y, k=args.k, batch_size=args.batch_size,
                compute_edge_attr=False, seed=args.seed
            )
        else:
            train_loader_pn, val_loader_pn, test_loader_pn = train_loader, val_loader, test_loader
        
        pn_model, pn_history, pn_results, pn_metrics = train_particlenet(
            train_loader_pn, val_loader_pn, test_loader_pn, args, device
        )
        results['ParticleNet'] = pn_results
        
        plot_training_history(pn_history, 'ParticleNet',
                              os.path.join(args.save_dir, 'particlenet_history.png'))
        plt.close()
        
        plot_confusion_matrix(pn_results[0], (pn_results[1] > 0.5).astype(int),
                              'ParticleNet',
                              os.path.join(args.save_dir, 'particlenet_confusion.png'))
        plt.close()
    
    if args.model in ['gat', 'both']:
        gat_model, gat_history, gat_results, gat_metrics = train_gat(
            train_loader, val_loader, test_loader, args, device
        )
        results['GAT'] = gat_results
        
        plot_training_history(gat_history, 'GAT',
                              os.path.join(args.save_dir, 'gat_history.png'))
        plt.close()
        
        plot_confusion_matrix(gat_results[0], (gat_results[1] > 0.5).astype(int),
                              'GAT',
                              os.path.join(args.save_dir, 'gat_confusion.png'))
        plt.close()
    
    if len(results) > 0:
        fig, ax = plot_roc_comparison(results)
        fig.savefig(os.path.join(args.save_dir, 'roc_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Results saved to: {args.save_dir}/")
    
    if len(results) == 2:
        print("\n--- Final Comparison ---")
        for name, (y_true, y_prob) in results.items():
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_prob)
            acc = ((y_prob > 0.5).astype(int) == y_true).mean()
            print(f"{name}: AUC = {auc:.4f}, Accuracy = {acc:.4f}")


if __name__ == '__main__':
    main()

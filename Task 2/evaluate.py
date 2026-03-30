"""
Evaluation script for trained GNN models.

Usage:
    python evaluate.py --model particlenet --checkpoint results/particlenet_best.pt
    python evaluate.py --model gat --checkpoint results/gat_best.pt
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.dataset import load_quark_gluon_data, create_data_loaders
from src.models.particle_net import ParticleNet
from src.models.gat_classifier import GATClassifier
from src.training.metrics import compute_metrics, plot_roc_curve
from src.utils.visualization import (
    plot_confusion_matrix, plot_jet_with_graph, plot_attention_weights
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained GNN model')
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['particlenet', 'gat'],
                        help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_data', type=int, default=100000,
                        help='Number of jets to load')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--k', type=int, default=16,
                        help='Number of neighbors for k-NN graph')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def load_model(model_type: str, checkpoint_path: str, device: str):
    """Load a trained model."""
    if model_type == 'particlenet':
        model = ParticleNet(
            input_dim=5,
            hidden_dims=(64, 128, 256),
            k=16,
            num_classes=2,
            dropout=0.3
        )
    else:
        model = GATClassifier(
            input_dim=5,
            hidden_dim=64,
            num_layers=3,
            heads=4,
            edge_dim=3,
            num_classes=2,
            dropout=0.3
        )
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} from {checkpoint_path}")
    return model


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    all_labels = []
    all_probs = []
    
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        if isinstance(out, tuple):
            out = out[0]
        
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        labels = data.y.view(-1).cpu().numpy()
        
        all_probs.extend(probs)
        all_labels.extend(labels)
    
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)
    
    return y_true, y_pred, y_prob


def visualize_jets(model, test_loader, model_type, save_dir, device, num_jets=4):
    """Visualize sample jets with predictions."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    data_iter = iter(test_loader)
    batch = next(data_iter).to(device)
    
    with torch.no_grad():
        if model_type == 'gat':
            logits, attentions = model(batch, return_attention=True)
        else:
            logits = model(batch)
    
    probs = torch.softmax(logits, dim=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    ptr = 0
    for i in range(min(num_jets, batch.num_graphs)):
        mask = batch.batch == i
        coords = batch.coords[mask].cpu().numpy()
        x = batch.x[mask].cpu().numpy()
        pt = np.exp(x[:, 2])
        
        edge_mask = (batch.edge_index[0] >= ptr) & (batch.edge_index[0] < ptr + mask.sum())
        local_edges = batch.edge_index[:, edge_mask].cpu().numpy()
        local_edges = local_edges - ptr
        
        true_label = batch.y[i].item()
        pred_prob = probs[i, 1].item()
        pred_label = 1 if pred_prob > 0.5 else 0
        
        true_str = "Quark" if true_label == 1 else "Gluon"
        pred_str = "Quark" if pred_label == 1 else "Gluon"
        
        ax = axes[i]
        
        for j in range(local_edges.shape[1]):
            src, dst = local_edges[0, j], local_edges[1, j]
            if src < len(coords) and dst < len(coords):
                ax.plot(
                    [coords[src, 0], coords[dst, 0]],
                    [coords[src, 1], coords[dst, 1]],
                    'gray', alpha=0.2, linewidth=0.5
                )
        
        sizes = 80 * (pt / pt.max()) ** 0.5
        sizes = np.clip(sizes, 10, 200)
        
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            s=sizes, c=np.log(pt + 1e-8),
            cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5, zorder=10
        )
        
        correct = "✓" if true_label == pred_label else "✗"
        ax.set_title(f"True: {true_str} | Pred: {pred_str} ({pred_prob:.2f}) {correct}", fontsize=12)
        ax.set_xlabel('Δη')
        ax.set_ylabel('Δφ')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        ptr += mask.sum().item()
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{model_type}_sample_jets.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sample jets visualization to {save_dir}/{model_type}_sample_jets.png")


def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = load_model(args.model, args.checkpoint, device)
    
    print("Loading test data...")
    X, y = load_quark_gluon_data(num_data=args.num_data)
    
    compute_edge = (args.model == 'gat')
    _, _, test_loader = create_data_loaders(
        X, y,
        k=args.k,
        batch_size=args.batch_size,
        compute_edge_attr=compute_edge,
        seed=args.seed
    )
    
    print("Evaluating model...")
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    print("\n" + "="*50)
    print(f"Evaluation Results: {args.model.upper()}")
    print("="*50)
    print(f"AUC Score:        {metrics['auc']:.4f}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Quark Precision:  {metrics['quark_precision']:.4f}")
    print(f"Quark Recall:     {metrics['quark_recall']:.4f}")
    print(f"Gluon Precision:  {metrics['gluon_precision']:.4f}")
    print(f"Gluon Recall:     {metrics['gluon_recall']:.4f}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    plot_roc_curve(y_true, y_prob, args.model.upper(),
                   save_path=os.path.join(args.save_dir, f'{args.model}_roc.png'))
    plt.close()
    
    plot_confusion_matrix(y_true, y_pred, args.model.upper(),
                          save_path=os.path.join(args.save_dir, f'{args.model}_confusion.png'))
    plt.close()
    
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_jets(model, test_loader, args.model, args.save_dir, device)
    
    print(f"\nResults saved to {args.save_dir}/")


if __name__ == '__main__':
    main()

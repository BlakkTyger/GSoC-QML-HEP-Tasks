"""
Visualization utilities for QNN experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None
) -> None:
    """Plot training curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = ['#2ecc71', '#e74c3c']
    
    for i, (name, history) in enumerate(histories.items()):
        epochs = range(len(history['loss']))
        axes[0].plot(epochs, history['loss'], label=name, color=colors[i], linewidth=2)
        axes[1].plot(epochs, [a * 100 for a in history['accuracy']], label=name, color=colors[i], linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_comparison_bar(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """Create bar chart comparing model metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(metrics))
    width = 0.35
    colors = ['#2ecc71', '#e74c3c']
    
    for i, (name, result) in enumerate(results.items()):
        values = [result.get(m, 0) * 100 for m in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=colors[i])
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print results as formatted table."""
    print("\n" + "="*60)
    print(" MODEL COMPARISON RESULTS")
    print("="*60)
    
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'F1':>10}"
    print(header)
    print("-"*60)
    
    for name, metrics in results.items():
        row = f"{name:<25} {metrics['accuracy']*100:>9.2f}% {metrics['precision']*100:>9.2f}% {metrics['f1']*100:>9.2f}%"
        print(row)
    
    print("="*60)

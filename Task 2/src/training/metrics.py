"""
Metrics computation and plotting utilities.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple, List, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities for positive class
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob)
    }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['quark_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['quark_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['gluon_precision'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics['gluon_recall'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        model_name: Name for legend
        ax: Matplotlib axes (creates new if None)
        save_path: Path to save figure
        
    Returns:
        Figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    
    ax.set_xlabel('False Positive Rate (Gluon Misidentification)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Quark Efficiency)', fontsize=12)
    ax.set_title('ROC Curve for Quark/Gluon Classification', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_roc_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ROC curves for multiple models.
    
    Args:
        results: Dict mapping model names to (y_true, y_prob) tuples
        save_path: Path to save figure
        
    Returns:
        Figure and axes
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.tab10.colors
    
    for i, (name, (y_true, y_prob)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', 
                linewidth=2, color=colors[i % len(colors)])
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Comparison: Quark/Gluon Classification', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot training history (loss and metrics over epochs).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_auc', 'val_auc'
        model_name: Model name for title
        save_path: Path to save figure
        
    Returns:
        Figure and axes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name}: Training Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    if 'train_auc' in history:
        axes[1].plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
        axes[1].plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('AUC', fontsize=12)
        axes[1].set_title(f'{model_name}: AUC Score', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes

"""
Visualization utilities for jet data and model analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from typing import Optional, List
import torch


def plot_jet(
    coords: np.ndarray,
    pt: np.ndarray,
    label: int,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a jet in (eta, phi) space.
    
    Args:
        coords: (N, 2) array of (delta_eta, delta_phi)
        pt: (N,) array of pT values
        label: Jet label (0=gluon, 1=quark)
        ax: Matplotlib axes
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    sizes = 50 * (pt / pt.max()) ** 0.5
    sizes = np.clip(sizes, 5, 200)
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        s=sizes, c=np.log(pt + 1e-8),
        cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log(pT)', fontsize=10)
    
    label_str = "Quark" if label == 1 else "Gluon"
    if title is None:
        title = f"{label_str} Jet (N={len(pt)} particles)"
    
    ax.set_xlabel('Δη', fontsize=12)
    ax.set_ylabel('Δφ', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_jet_with_graph(
    coords: np.ndarray,
    pt: np.ndarray,
    edge_index: np.ndarray,
    label: int,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a jet with k-NN graph edges.
    
    Args:
        coords: (N, 2) array of coordinates
        pt: (N,) array of pT values
        edge_index: (2, E) array of edge indices
        label: Jet label
        ax: Matplotlib axes
        title: Plot title
        save_path: Path to save figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        ax.plot(
            [coords[src, 0], coords[dst, 0]],
            [coords[src, 1], coords[dst, 1]],
            'gray', alpha=0.2, linewidth=0.5
        )
    
    sizes = 100 * (pt / pt.max()) ** 0.5
    sizes = np.clip(sizes, 10, 300)
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        s=sizes, c=np.log(pt + 1e-8),
        cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5, zorder=10
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log(pT)', fontsize=10)
    
    label_str = "Quark" if label == 1 else "Gluon"
    if title is None:
        title = f"{label_str} Jet Graph (N={len(pt)}, E={edge_index.shape[1]})"
    
    ax.set_xlabel('Δη', fontsize=12)
    ax.set_ylabel('Δφ', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attention_weights(
    coords: np.ndarray,
    pt: np.ndarray,
    edge_index: np.ndarray,
    attention_weights: np.ndarray,
    label: int,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot jet with attention-weighted edges.
    
    Args:
        coords: (N, 2) array of coordinates
        pt: (N,) array of pT
        edge_index: (2, E) edge indices
        attention_weights: (E,) attention weights
        label: Jet label
        ax: Matplotlib axes
        title: Plot title
        save_path: Path to save figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    attn_norm = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        alpha = 0.1 + 0.8 * attn_norm[i]
        width = 0.5 + 3 * attn_norm[i]
        ax.plot(
            [coords[src, 0], coords[dst, 0]],
            [coords[src, 1], coords[dst, 1]],
            color=plt.cm.Reds(attn_norm[i]),
            alpha=alpha, linewidth=width
        )
    
    sizes = 100 * (pt / pt.max()) ** 0.5
    sizes = np.clip(sizes, 10, 300)
    
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=sizes, c='steelblue',
        alpha=0.8, edgecolors='black', linewidth=0.5, zorder=10
    )
    
    label_str = "Quark" if label == 1 else "Gluon"
    if title is None:
        title = f"{label_str} Jet - Attention Weights"
    
    ax.set_xlabel('Δη', fontsize=12)
    ax.set_ylabel('Δφ', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Model name for title
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Gluon', 'Quark'],
        yticklabels=['Gluon', 'Quark'],
        ax=ax
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'{model_name}: Confusion Matrix', fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_multiplicity_distribution(
    X: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot particle multiplicity distribution for quarks vs gluons.
    
    Args:
        X: (N, M, 4) jet data
        y: (N,) labels
        save_path: Path to save figure
    """
    multiplicities = (X[:, :, 0] > 0).sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    quark_mult = multiplicities[y == 1]
    gluon_mult = multiplicities[y == 0]
    
    bins = np.arange(0, multiplicities.max() + 5, 5)
    
    ax.hist(quark_mult, bins=bins, alpha=0.6, label=f'Quark (mean={quark_mult.mean():.1f})', 
            density=True, color='blue')
    ax.hist(gluon_mult, bins=bins, alpha=0.6, label=f'Gluon (mean={gluon_mult.mean():.1f})', 
            density=True, color='red')
    
    ax.set_xlabel('Particle Multiplicity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Particle Multiplicity: Quark vs Gluon Jets', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

"""
Dataset generation for Z₂ × Z₂ symmetric binary classification.

Classification rule: |x₁ - x₂| >= threshold
This naturally respects Z₂ × Z₂ symmetry (swap invariance).
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, Optional


def generate_z2z2_dataset(
    n_points: int = 200,
    threshold: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Z₂ × Z₂ symmetric dataset.
    
    Classification rule:
    - Class 0: |x₁ - x₂| < threshold (near diagonal)
    - Class 1: |x₁ - x₂| >= threshold (away from diagonal)
    
    Symmetry is enforced by adding swapped points (x₂, x₁).
    
    Args:
        n_points: Number of base points (total will be 2x due to symmetry)
        threshold: Distance threshold from diagonal
        seed: Random seed
        
    Returns:
        X: Features array of shape (2*n_points, 2)
        y: Labels array of shape (2*n_points,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.rand(n_points, 2)
    y = (np.abs(X[:, 0] - X[:, 1]) >= threshold).astype(int)
    
    X_sym = np.column_stack((X[:, 1], X[:, 0]))
    X_full = np.concatenate([X, X_sym], axis=0)
    y_full = np.concatenate([y, y])
    
    return X_full, y_full


def to_torch_tensors(
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert numpy arrays to PyTorch tensors."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor


def visualize_dataset(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Z₂ × Z₂ Symmetric Dataset",
    save_path: Optional[str] = None
) -> None:
    """Visualize the dataset."""
    plt.figure(figsize=(8, 6))
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='white', linewidth=0.5)
    plt.colorbar(scatter, label='Class')
    
    x_line = np.linspace(0, 1, 100)
    plt.plot(x_line, x_line, 'r--', alpha=0.5, label='y = x (symmetry axis)')
    
    plt.xlabel("x₁", fontsize=12)
    plt.ylabel("x₂", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def verify_symmetry(X: np.ndarray, y: np.ndarray) -> bool:
    """Verify dataset respects Z₂ × Z₂ symmetry."""
    n = len(y) // 2
    
    X_swapped = np.column_stack((X[:n, 1], X[:n, 0]))
    labels_consistent = np.allclose(
        (np.abs(X[:n, 0] - X[:n, 1]) >= 0.05).astype(int),
        (np.abs(X_swapped[:, 0] - X_swapped[:, 1]) >= 0.05).astype(int)
    )
    
    return labels_consistent


if __name__ == "__main__":
    X, y = generate_z2z2_dataset(200, threshold=0.05, seed=42)
    print(f"Dataset: {X.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Symmetry verified: {verify_symmetry(X, y)}")
    visualize_dataset(X, y)

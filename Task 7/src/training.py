"""
Training utilities for QNN models using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import time


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a QNN classifier using Adam optimizer and NLL loss.
    
    Args:
        model: QNNClassifier model
        X: Input features tensor
        y: Labels tensor (long type)
        n_epochs: Number of training epochs
        lr: Learning rate
        verbose: Print progress
        
    Returns:
        Trained model and history dict
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    history = {
        'loss': [],
        'accuracy': [],
        'epoch_time': []
    }
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        model.train()
        optimizer.zero_grad()
        
        output = model(X)
        loss = criterion(output, y)
        
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        
        epoch_time = time.time() - start_time
        
        history['loss'].append(loss.item())
        history['accuracy'].append(acc)
        history['epoch_time'].append(epoch_time)
        
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}, Accuracy = {acc*100:.2f}%")
    
    return model, history


def evaluate_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained QNNClassifier
        X: Input features
        y: True labels
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    with torch.no_grad():
        output = model(X)
        loss = nn.NLLLoss()(output, y)
        pred = output.argmax(dim=1)
        
        accuracy = (pred == y).float().mean().item()
        
        tp = ((pred == 1) & (y == 1)).sum().item()
        tn = ((pred == 0) & (y == 0)).sum().item()
        fp = ((pred == 1) & (y == 0)).sum().item()
        fn = ((pred == 0) & (y == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_test_split(
    X: torch.Tensor,
    y: torch.Tensor,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split data into train and test sets."""
    torch.manual_seed(seed)
    n = len(y)
    indices = torch.randperm(n)
    
    n_test = int(n * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

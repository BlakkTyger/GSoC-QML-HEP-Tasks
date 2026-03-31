"""Data generation utilities for PQC embedding task."""

import torch
from torch.utils.data import TensorDataset, DataLoader


def create_target(x: torch.Tensor) -> torch.Tensor:
    """
    Create target values for input x.
    Target = [x, sin(x), cos(x), x²]
    
    Args:
        x: Input tensor of shape (N, 1)
    
    Returns:
        Target tensor of shape (N, 4)
    """
    t1 = x
    t2 = torch.sin(x)
    t3 = torch.cos(x)
    t4 = x ** 2
    return torch.cat([t1, t2, t3, t4], dim=1)


def generate_data(
    num_samples: int = 2048,
    seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate normally distributed input data and corresponding targets.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X_data, Y_data) tensors
    """
    torch.manual_seed(seed)
    X_data = torch.randn(num_samples, 1)
    Y_data = create_target(X_data)
    return X_data, Y_data


def create_dataloaders(
    X_data: torch.Tensor,
    Y_data: torch.Tensor,
    batch_size: int = 32,
    train_split: float = 0.8,
    seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        X_data: Input tensor
        Y_data: Target tensor
        batch_size: Batch size for training
        train_split: Fraction of data for training
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    torch.manual_seed(seed)
    
    num_samples = X_data.shape[0]
    num_train = int(num_samples * train_split)
    
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    X_train, Y_train = X_data[train_indices], Y_data[train_indices]
    X_test, Y_test = X_data[test_indices], Y_data[test_indices]
    
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

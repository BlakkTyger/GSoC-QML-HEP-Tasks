"""MNIST Dataset Loading and Preprocessing."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_loaders(
    data_dir='./data',
    batch_size=128,
    val_split=0.1,
    num_workers=2,
    normalize_range='symmetric'
):
    """
    Create MNIST data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation
        num_workers: Number of workers for data loading
        normalize_range: 'symmetric' for [-1,1], 'standard' for mean/std normalization
        
    Returns:
        train_loader, val_loader, test_loader
    """
    if normalize_range == 'symmetric':
        # Rescale to [-1, 1] as in baseline
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0)
        ])
    else:
        # Standard MNIST normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Split training data into train/val
    n_train = len(train_dataset)
    n_val = int(n_train * val_split)
    n_train = n_train - n_val
    
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        train_dataset, 
        [n_train, n_val],
        generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_simple_mnist_loaders(data_dir='./data', batch_size=128):
    """
    Simplified MNIST loaders (train and test only, no validation split).
    Matches the baseline implementation.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2.0 * x - 1.0)  # Rescale to [-1, 1]
    ])
    
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

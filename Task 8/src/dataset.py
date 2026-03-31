"""MNIST Dataset Loading and Preprocessing."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_loaders(
    data_dir='./data',
    batch_size=128,
    val_split=0.1,
    num_workers=2,
    augment=True
):
    """
    Create MNIST data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation
        num_workers: Number of workers for data loading
        augment: Whether to apply data augmentation to training data
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # MNIST statistics
    mean = (0.1307,)
    std = (0.3081,)
    
    # Test/validation transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Training transform (with optional augmentation)
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = test_transform
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    # For validation, we need the same data but with test transform
    val_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=test_transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Split training data into train/val
    n_train = len(train_dataset)
    n_val = int(n_train * val_split)
    n_train = n_train - n_val
    
    # Use same random split for both datasets
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(train_dataset)), 
        [n_train, n_val],
        generator=generator
    )
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
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
        transforms.Normalize((0.1307,), (0.3081,))
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

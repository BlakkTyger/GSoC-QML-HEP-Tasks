"""
PyTorch Geometric Dataset for Quark/Gluon jet classification.
"""

import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from typing import Tuple, List, Optional
import energyflow as ef

from .preprocessing import preprocess_jet, build_knn_graph, compute_edge_features


class JetGraphDataset(Dataset):
    """
    Dataset that converts jets to graphs for GNN classification.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 16,
        compute_edge_attr: bool = True,
        transform=None,
        pre_transform=None
    ):
        """
        Args:
            X: (N, M, 4) array of jet constituents
            y: (N,) array of labels (0=gluon, 1=quark)
            k: Number of neighbors for k-NN graph
            compute_edge_attr: Whether to compute edge features
        """
        super().__init__(None, transform, pre_transform)
        self.X = X
        self.y = y
        self.k = k
        self.compute_edge_attr = compute_edge_attr
    
    def len(self) -> int:
        return len(self.y)
    
    def get(self, idx: int) -> Data:
        """Get a single graph from the dataset."""
        particles = self.X[idx]
        label = self.y[idx]
        
        coords, features = preprocess_jet(particles)
        
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        edge_index = build_knn_graph(coords_tensor, k=self.k)
        
        edge_attr = None
        if self.compute_edge_attr and edge_index.size(1) > 0:
            edge_attr = compute_edge_features(coords_tensor, features_tensor, edge_index)
        
        data = Data(
            x=features_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long),
            coords=coords_tensor
        )
        
        return data


def load_quark_gluon_data(
    num_data: int = 100000,
    cache_dir: str = '~/.energyflow',
    with_bc: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load quark/gluon jet data using EnergyFlow package.
    
    Args:
        num_data: Number of jets to load
        cache_dir: Directory to cache downloaded data
        with_bc: Whether to include b/c jets
        
    Returns:
        X: (N, M, 4) array of jet constituents
        y: (N,) array of labels
    """
    print(f"Loading {num_data} jets from EnergyFlow...")
    
    X, y = ef.qg_jets.load(num_data=num_data, cache_dir=cache_dir, with_bc=with_bc)
    
    print(f"Loaded {len(y)} jets")
    print(f"  - Quarks: {(y == 1).sum()}")
    print(f"  - Gluons: {(y == 0).sum()}")
    print(f"  - Max multiplicity: {X.shape[1]}")
    
    return X, y


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 16,
    batch_size: int = 128,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    compute_edge_attr: bool = True,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders.
    
    Args:
        X: Jet constituents array
        y: Labels array
        k: Number of neighbors for k-NN
        batch_size: Batch size for data loaders
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        compute_edge_attr: Whether to compute edge features
        num_workers: Number of workers for data loading
        seed: Random seed for splitting
        
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset = JetGraphDataset(X, y, k=k, compute_edge_attr=compute_edge_attr)
    
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    
    print(f"Dataset splits: train={n_train}, val={n_val}, test={n_test}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

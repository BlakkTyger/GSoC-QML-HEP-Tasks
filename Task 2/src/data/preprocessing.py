"""
Preprocessing utilities for jet data.
Handles feature engineering, graph construction, and data transformation.
"""

import numpy as np
import torch
from typing import Tuple, Optional


def preprocess_jet(
    particles: np.ndarray,
    center_jet: bool = True,
    log_pt: bool = True,
    compute_pt_frac: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a single jet's particle features.
    
    Args:
        particles: (M, 4) array with columns [pT, rapidity, phi, pdgid]
        center_jet: Whether to center the jet at (eta=0, phi=0)
        log_pt: Whether to use log(pT) instead of pT
        compute_pt_frac: Whether to compute pT fraction
        
    Returns:
        coords: (N, 2) array of (delta_eta, delta_phi) for graph construction
        features: (N, D) array of node features
    """
    mask = particles[:, 0] > 0
    particles = particles[mask]
    
    if len(particles) == 0:
        return np.zeros((1, 2)), np.zeros((1, 5))
    
    pt = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]
    
    if center_jet:
        pt_sum = pt.sum()
        eta_center = (pt * eta).sum() / pt_sum
        phi_center = (pt * phi).sum() / pt_sum
        
        delta_eta = eta - eta_center
        delta_phi = phi - phi_center
        delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
    else:
        delta_eta = eta
        delta_phi = phi
    
    coords = np.stack([delta_eta, delta_phi], axis=1)
    
    energy = pt * np.cosh(eta)
    
    feature_list = [delta_eta, delta_phi]
    
    if log_pt:
        feature_list.append(np.log(pt + 1e-8))
        feature_list.append(np.log(energy + 1e-8))
    else:
        feature_list.append(pt)
        feature_list.append(energy)
    
    if compute_pt_frac:
        pt_frac = pt / pt.sum()
        feature_list.append(pt_frac)
    
    features = np.stack(feature_list, axis=1)
    
    return coords.astype(np.float32), features.astype(np.float32)


def build_knn_graph(
    coords: torch.Tensor,
    k: int = 16,
    loop: bool = False
) -> torch.Tensor:
    """
    Build k-nearest neighbor graph from coordinates (no torch-cluster required).
    
    Args:
        coords: (N, 2) tensor of (eta, phi) coordinates
        k: Number of nearest neighbors
        loop: Whether to include self-loops
        
    Returns:
        edge_index: (2, E) tensor of edge indices
    """
    n = coords.size(0)
    
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    
    k_actual = min(k, n - 1) if not loop else min(k, n)
    
    if k_actual < 1:
        return torch.zeros((2, 0), dtype=torch.long)
    
    dist = torch.cdist(coords, coords)
    
    if not loop:
        dist.fill_diagonal_(float('inf'))
    
    _, knn_idx = dist.topk(k_actual, dim=1, largest=False)
    
    src = torch.arange(n).unsqueeze(1).expand(-1, k_actual).reshape(-1)
    dst = knn_idx.reshape(-1)
    
    edge_index = torch.stack([src, dst])
    
    return edge_index


def compute_edge_features(
    coords: torch.Tensor,
    features: torch.Tensor,
    edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Compute edge features for GAT model.
    
    Args:
        coords: (N, 2) tensor of coordinates
        features: (N, D) tensor of node features
        edge_index: (2, E) tensor of edge indices
        
    Returns:
        edge_attr: (E, 3) tensor with (delta_R, delta_eta, delta_phi)
    """
    src, dst = edge_index[0], edge_index[1]
    
    delta_eta = coords[dst, 0] - coords[src, 0]
    delta_phi = coords[dst, 1] - coords[src, 1]
    
    delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)
    
    edge_attr = torch.stack([delta_R, delta_eta, delta_phi], dim=1)
    
    return edge_attr


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        features: (N, D) array of features
        mean: Pre-computed mean (for test set)
        std: Pre-computed std (for test set)
        
    Returns:
        normalized: Normalized features
        mean: Feature means
        std: Feature stds
    """
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0)
        std[std < 1e-8] = 1.0
    
    normalized = (features - mean) / std
    
    return normalized, mean, std

"""
Preprocessing utilities for QGNN.
Handles particle selection, feature normalization, and graph construction.
"""

import numpy as np
from typing import Tuple, List, Optional


def select_top_particles(
    jet: np.ndarray,
    n_particles: int = 8,
    min_pt: float = 0.0
) -> np.ndarray:
    """
    Select top-n particles by pT from a jet.
    
    Args:
        jet: (M, 4) array of particles [pT, rapidity, phi, pdgid]
        n_particles: Number of particles to select
        min_pt: Minimum pT threshold
        
    Returns:
        selected: (n_particles, 4) array of selected particles
    """
    # Remove zero-padded particles
    mask = jet[:, 0] > min_pt
    valid_particles = jet[mask]
    
    if len(valid_particles) == 0:
        return np.zeros((n_particles, 4))
    
    # Sort by pT (descending)
    sorted_idx = np.argsort(-valid_particles[:, 0])
    sorted_particles = valid_particles[sorted_idx]
    
    # Take top-n, pad if necessary
    if len(sorted_particles) >= n_particles:
        return sorted_particles[:n_particles]
    else:
        # Pad with zeros
        padded = np.zeros((n_particles, 4))
        padded[:len(sorted_particles)] = sorted_particles
        return padded


def compute_jet_center(particles: np.ndarray) -> Tuple[float, float]:
    """
    Compute pT-weighted centroid of jet in (eta, phi) space.
    
    Args:
        particles: (N, 4) array [pT, eta, phi, pdgid]
        
    Returns:
        (eta_center, phi_center)
    """
    pt = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]
    
    total_pt = np.sum(pt) + 1e-10
    eta_center = np.sum(pt * eta) / total_pt
    phi_center = np.sum(pt * phi) / total_pt
    
    return eta_center, phi_center


def normalize_features(
    particles: np.ndarray,
    center: bool = True
) -> np.ndarray:
    """
    Normalize particle features for quantum encoding.
    
    Features after normalization:
    - Δη: relative rapidity, scaled to [-π, π]
    - Δφ: relative azimuthal angle, scaled to [-π, π]
    - log_pT: log(pT), scaled to [-π, π]
    
    Args:
        particles: (N, 4) array [pT, eta, phi, pdgid]
        center: Whether to center the jet
        
    Returns:
        features: (N, 3) array [Δη_scaled, Δφ_scaled, log_pT_scaled]
    """
    pt = particles[:, 0].copy()
    eta = particles[:, 1].copy()
    phi = particles[:, 2].copy()
    
    # Center the jet
    if center:
        eta_c, phi_c = compute_jet_center(particles)
        eta = eta - eta_c
        phi = phi - phi_c
        # Wrap phi to [-π, π]
        phi = np.arctan2(np.sin(phi), np.cos(phi))
    
    # Scale eta to [-π, π] (typical range is ~[-0.5, 0.5])
    eta_scaled = np.clip(eta * np.pi / 0.5, -np.pi, np.pi)
    
    # Phi is already in [-π, π]
    phi_scaled = phi
    
    # Log pT scaled to [-π, π]
    # pT range typically 1-500 GeV, log range ~0-6
    log_pt = np.log(pt + 1e-10)
    log_pt_scaled = np.clip((log_pt - 3) * np.pi / 3, -np.pi, np.pi)
    
    features = np.stack([eta_scaled, phi_scaled, log_pt_scaled], axis=1)
    
    return features


def build_knn_graph(
    particles: np.ndarray,
    k: int = 4
) -> np.ndarray:
    """
    Build k-nearest neighbor graph from particle coordinates.
    
    Args:
        particles: (N, 4) array [pT, eta, phi, pdgid]
        k: Number of nearest neighbors
        
    Returns:
        edge_index: (2, E) array of edge indices
    """
    n = len(particles)
    eta = particles[:, 1]
    phi = particles[:, 2]
    
    # Compute pairwise distances in (eta, phi) space
    coords = np.stack([eta, phi], axis=1)
    
    # Distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # Handle phi wraparound
    diff[:, :, 1] = np.arctan2(np.sin(diff[:, :, 1]), np.cos(diff[:, :, 1]))
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set self-distance to infinity
    np.fill_diagonal(dist, np.inf)
    
    # Find k nearest neighbors for each node
    edge_list = []
    for i in range(n):
        # Skip if particle has zero pT (padding)
        if particles[i, 0] < 1e-10:
            continue
            
        # Get indices of k nearest valid neighbors
        valid_mask = particles[:, 0] > 1e-10
        dists_i = dist[i].copy()
        dists_i[~valid_mask] = np.inf
        
        neighbors = np.argsort(dists_i)[:k]
        
        for j in neighbors:
            if dists_i[j] < np.inf:
                edge_list.append([i, j])
    
    if len(edge_list) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    
    edge_index = np.array(edge_list, dtype=np.int64).T
    
    return edge_index


def preprocess_jet(
    jet: np.ndarray,
    n_particles: int = 8,
    k_neighbors: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a single jet.
    
    Args:
        jet: (M, 4) array of particles [pT, rapidity, phi, pdgid]
        n_particles: Number of particles to select
        k_neighbors: Number of neighbors for k-NN graph
        
    Returns:
        features: (n_particles, 3) normalized features for quantum encoding
        edge_index: (2, E) edge indices for graph structure
    """
    # Select top particles
    selected = select_top_particles(jet, n_particles)
    
    # Normalize features
    features = normalize_features(selected)
    
    # Build graph
    edge_index = build_knn_graph(selected, k=k_neighbors)
    
    return features, edge_index


def preprocess_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_particles: int = 8,
    k_neighbors: int = 4,
    max_samples: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Preprocess entire dataset.
    
    Args:
        X: (N_jets, M, 4) jet data
        y: (N_jets,) labels
        n_particles: Number of particles per jet
        k_neighbors: k-NN parameter
        max_samples: Maximum samples to process
        
    Returns:
        features_list: List of (n_particles, 3) feature arrays
        edge_list: List of (2, E) edge index arrays
        labels: (N,) label array
    """
    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]
    
    features_list = []
    edge_list = []
    
    for jet in X:
        features, edge_index = preprocess_jet(jet, n_particles, k_neighbors)
        features_list.append(features)
        edge_list.append(edge_index)
    
    return features_list, edge_list, y


def get_unique_edges(edge_index: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get unique undirected edges from edge index.
    
    Args:
        edge_index: (2, E) edge indices
        
    Returns:
        List of (i, j) tuples where i < j
    """
    edges = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        edge = (min(src, dst), max(src, dst))
        edges.add(edge)
    return sorted(list(edges))

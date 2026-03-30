"""
ParticleNet implementation using Dynamic EdgeConv.
Based on: "ParticleNet: Jet Tagging via Particle Clouds" (arXiv:1902.08570)

This implementation uses a custom EdgeConv that doesn't require torch-cluster.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Tuple


def knn_graph_batch(x: torch.Tensor, k: int, batch: torch.Tensor) -> torch.Tensor:
    """
    Compute k-NN graph for batched point clouds without torch-cluster.
    
    Args:
        x: (N, D) node features
        k: number of neighbors
        batch: (N,) batch assignment
        
    Returns:
        edge_index: (2, N*k) edge indices
    """
    device = x.device
    batch_size = batch.max().item() + 1
    
    edge_sources = []
    edge_targets = []
    
    for b in range(batch_size):
        mask = (batch == b)
        indices = torch.where(mask)[0]
        x_b = x[mask]
        n = x_b.size(0)
        
        if n <= 1:
            continue
        
        k_actual = min(k, n - 1)
        
        dist = torch.cdist(x_b, x_b)
        dist.fill_diagonal_(float('inf'))
        
        _, knn_idx = dist.topk(k_actual, dim=1, largest=False)
        
        src = indices.unsqueeze(1).expand(-1, k_actual).reshape(-1)
        dst = indices[knn_idx.reshape(-1)]
        
        edge_sources.append(src)
        edge_targets.append(dst)
    
    if len(edge_sources) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    edge_index = torch.stack([
        torch.cat(edge_sources),
        torch.cat(edge_targets)
    ])
    
    return edge_index


class EdgeConvBlock(nn.Module):
    """
    EdgeConv block with MLP and batch normalization.
    Uses custom k-NN implementation (no torch-cluster dependency).
    """
    
    def __init__(self, in_channels: int, out_channels: int, k: int = 16):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EdgeConv block.
        
        Args:
            x: (N, C) node features
            batch: (N,) batch indices
            
        Returns:
            x: (N, C') updated node features
        """
        edge_index = knn_graph_batch(x, self.k, batch)
        
        if edge_index.size(1) == 0:
            return self.shortcut(x)
        
        src, dst = edge_index[0], edge_index[1]
        
        x_src = x[src]
        x_dst = x[dst]
        edge_features = torch.cat([x_src, x_dst - x_src], dim=1)
        
        edge_out = self.mlp(edge_features)
        
        x_out = torch.zeros(x.size(0), edge_out.size(1), device=x.device)
        
        x_out = x_out.scatter_reduce(
            0, 
            src.unsqueeze(1).expand(-1, edge_out.size(1)), 
            edge_out, 
            reduce='amax',
            include_self=False
        )
        
        x_shortcut = self.shortcut(x)
        
        return x_out + x_shortcut


class ParticleNet(nn.Module):
    """
    ParticleNet architecture for jet classification.
    
    Uses dynamic k-NN graph construction in learned feature space
    with EdgeConv message passing.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: Tuple[int, ...] = (64, 128, 256),
        k: int = 16,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Number of input node features
            hidden_dims: Hidden dimensions for each EdgeConv block
            k: Number of neighbors for k-NN
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.k = k
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        self.edge_convs = nn.ModuleList()
        in_channels = input_dim
        
        for out_channels in hidden_dims:
            self.edge_convs.append(
                EdgeConvBlock(in_channels, out_channels, k=k)
            )
            in_channels = out_channels
        
        pool_dim = hidden_dims[-1] * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass through ParticleNet.
        
        Args:
            data: PyG Data object with x, batch attributes
            
        Returns:
            logits: (B, num_classes) classification logits
        """
        x, batch = data.x, data.batch
        
        x = self.input_bn(x)
        
        for edge_conv in self.edge_convs:
            x = edge_conv(x, batch)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        logits = self.classifier(x)
        
        return logits
    
    def get_embeddings(self, data) -> torch.Tensor:
        """Get node embeddings before global pooling."""
        x, batch = data.x, data.batch
        
        x = self.input_bn(x)
        
        for edge_conv in self.edge_convs:
            x = edge_conv(x, batch)
        
        return x

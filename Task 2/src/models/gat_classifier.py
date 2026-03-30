"""
Graph Attention Network (GAT) for jet classification.
Uses GATv2Conv for improved attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from typing import Tuple, Optional


class GATBlock(nn.Module):
    """
    GAT block with multi-head attention and residual connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.concat = concat
        actual_out = out_channels * heads if concat else out_channels
        
        self.gat = GATv2Conv(
            in_channels,
            out_channels,
            heads=heads,
            concat=concat,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=True
        )
        
        self.bn = nn.BatchNorm1d(actual_out)
        
        if in_channels != actual_out:
            self.shortcut = nn.Linear(in_channels, actual_out)
        else:
            self.shortcut = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through GAT block.
        
        Args:
            x: (N, C) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, D) edge features (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            x: (N, C') updated node features
            attention: (E, H) attention weights if return_attention=True
        """
        if return_attention:
            x_out, attention = self.gat(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
        else:
            x_out = self.gat(x, edge_index, edge_attr=edge_attr)
            attention = None
        
        x_out = self.bn(x_out)
        x_out = F.relu(x_out)
        
        x_shortcut = self.shortcut(x)
        x_out = x_out + x_shortcut
        
        return x_out, attention


class GATClassifier(nn.Module):
    """
    GAT-based classifier for quark/gluon jet classification.
    
    Uses static k-NN graph with attention mechanism for
    adaptive neighbor aggregation.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        edge_dim: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Number of input node features
            hidden_dim: Hidden dimension per attention head
            num_layers: Number of GAT layers
            heads: Number of attention heads
            edge_dim: Number of edge features (set to None to disable)
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        self.gat_blocks = nn.ModuleList()
        
        in_channels = input_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            concat = not is_last
            
            self.gat_blocks.append(
                GATBlock(
                    in_channels,
                    hidden_dim,
                    heads=heads,
                    concat=concat,
                    edge_dim=edge_dim,
                    dropout=dropout if not is_last else 0.0
                )
            )
            
            in_channels = hidden_dim * heads if concat else hidden_dim
        
        pool_dim = hidden_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(
        self,
        data,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through GAT classifier.
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (B, num_classes) classification logits
            attentions: List of attention weights per layer (if requested)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch
        
        x = self.input_bn(x)
        
        attentions = []
        for gat_block in self.gat_blocks:
            x, attn = gat_block(x, edge_index, edge_attr, return_attention)
            if return_attention and attn is not None:
                attentions.append(attn)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        logits = self.classifier(x)
        
        if return_attention:
            return logits, attentions
        return logits
    
    def get_attention_weights(self, data) -> list:
        """Get attention weights for visualization."""
        _, attentions = self.forward(data, return_attention=True)
        return attentions

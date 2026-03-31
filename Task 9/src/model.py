"""Kolmogorov-Arnold Network Models for MNIST Classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLayer(nn.Module):
    """
    KAN Layer with Gaussian basis functions.
    
    Each input feature is expanded using Gaussian basis functions centered
    at fixed points, then linearly combined with learnable weights.
    """
    
    def __init__(self, in_features, out_features, num_bases=10, sigma=0.3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bases = num_bases
        self.sigma = sigma
        
        # Fixed centers uniformly distributed in [-1, 1]
        self.register_buffer("centers", torch.linspace(-1, 1, num_bases))
        
        # Learnable coefficients for basis functions
        self.coefficients = nn.Parameter(torch.randn(in_features, num_bases, out_features) * 0.1)
        
        # Optional base linear transformation (residual-like)
        self.base_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        
    def forward(self, x):
        # x: (batch, in_features)
        batch_size = x.shape[0]
        
        # Expand input for basis computation: (batch, in_features, 1)
        x_expanded = x.unsqueeze(2)
        
        # Centers: (1, 1, num_bases)
        centers = self.centers.view(1, 1, self.num_bases)
        
        # Gaussian basis values: (batch, in_features, num_bases)
        basis = torch.exp(-((x_expanded - centers) ** 2) / (2 * self.sigma ** 2))
        
        # Apply learnable coefficients
        # basis: (batch, in_features, num_bases)
        # coefficients: (in_features, num_bases, out_features)
        # Result: (batch, in_features, out_features)
        spline_out = torch.einsum('bin,ino->bio', basis, self.coefficients)
        
        # Sum over input features: (batch, out_features)
        spline_out = spline_out.sum(dim=1)
        
        # Add base transformation
        base_out = torch.matmul(x, self.base_weight)
        
        return spline_out + base_out


class BSplineKANLayer(nn.Module):
    """
    KAN Layer with B-spline basis functions.
    
    Implements cubic B-splines (order 3) for smooth function approximation.
    """
    
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Number of basis functions = grid_size + spline_order
        self.num_bases = grid_size + spline_order
        
        # Create grid points (extended for B-spline boundary conditions)
        h = 2.0 / grid_size  # Grid spacing for [-1, 1]
        grid = torch.linspace(-1 - h * spline_order, 1 + h * spline_order, 
                              grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)
        
        # Learnable spline coefficients
        self.coefficients = nn.Parameter(
            torch.randn(in_features, self.num_bases, out_features) * 0.1
        )
        
        # Base weight for residual connection
        self.base_weight = nn.Parameter(
            torch.randn(in_features, out_features) * 0.1
        )
        
    def _bspline_basis(self, x):
        """
        Compute B-spline basis values using Cox-de Boor recursion.
        
        Args:
            x: (batch, in_features) input values
            
        Returns:
            basis: (batch, in_features, num_bases) B-spline basis values
        """
        batch_size, in_features = x.shape
        grid = self.grid
        k = self.spline_order
        
        # Initialize order-0 basis (piecewise constant)
        # Shape: (batch, in_features, num_bases)
        x_expanded = x.unsqueeze(2)  # (batch, in_features, 1)
        
        # B_{i,0}(x) = 1 if grid[i] <= x < grid[i+1], else 0
        bases = ((x_expanded >= grid[:-1].view(1, 1, -1)) & 
                 (x_expanded < grid[1:].view(1, 1, -1))).float()
        
        # Cox-de Boor recursion for higher orders
        for p in range(1, k + 1):
            # Number of basis functions at order p
            n_basis = len(grid) - p - 1
            
            # Left term: (x - t_i) / (t_{i+p} - t_i) * B_{i,p-1}(x)
            left_num = x_expanded - grid[:-p-1].view(1, 1, -1)
            left_den = (grid[p:-1] - grid[:-p-1]).view(1, 1, -1)
            left_den = torch.clamp(left_den, min=1e-8)  # Avoid division by zero
            left = (left_num / left_den) * bases[:, :, :-1]
            
            # Right term: (t_{i+p+1} - x) / (t_{i+p+1} - t_{i+1}) * B_{i+1,p-1}(x)
            right_num = grid[p+1:].view(1, 1, -1) - x_expanded
            right_den = (grid[p+1:] - grid[1:-p]).view(1, 1, -1)
            right_den = torch.clamp(right_den, min=1e-8)
            right = (right_num / right_den) * bases[:, :, 1:]
            
            bases = left + right
        
        return bases[:, :, :self.num_bases]
    
    def forward(self, x):
        # x: (batch, in_features)
        
        # Compute B-spline basis: (batch, in_features, num_bases)
        basis = self._bspline_basis(x)
        
        # Apply coefficients: (batch, in_features, out_features)
        spline_out = torch.einsum('bin,ino->bio', basis, self.coefficients)
        
        # Sum over inputs: (batch, out_features)
        spline_out = spline_out.sum(dim=1)
        
        # Base transformation
        base_out = torch.matmul(x, self.base_weight)
        
        return spline_out + base_out


class KANNet(nn.Module):
    """
    Simple KAN Network with single KAN layer (matches baseline).
    """
    
    def __init__(self, in_features=784, hidden_dim=256, num_classes=10,
                 num_bases=10, sigma=0.3, basis_type='gaussian'):
        super().__init__()
        
        if basis_type == 'gaussian':
            self.kan = KANLayer(in_features, hidden_dim, num_bases, sigma)
        else:
            self.kan = BSplineKANLayer(in_features, hidden_dim, 
                                        grid_size=num_bases, spline_order=3)
        
        self.activation = nn.SiLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Flatten input: (B, 1, 28, 28) -> (B, 784)
        x = x.view(x.size(0), -1)
        x = self.kan(x)
        x = self.activation(x)
        x = self.classifier(x)
        return x


class EfficientKAN(nn.Module):
    """
    Multi-layer KAN with efficient implementation.
    """
    
    def __init__(self, in_features=784, hidden_dims=[128, 64], num_classes=10,
                 grid_size=5, spline_order=3, basis_type='gaussian'):
        super().__init__()
        
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            if basis_type == 'gaussian':
                layers.append(KANLayer(prev_dim, hidden_dim, num_bases=grid_size + spline_order))
            else:
                layers.append(BSplineKANLayer(prev_dim, hidden_dim, grid_size, spline_order))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim
        
        self.kan_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.kan_layers(x)
        x = self.classifier(x)
        return x


def create_kan(config):
    """Factory function to create KAN from config dict."""
    model_type = config.get('model_type', 'simple')
    
    if model_type == 'simple':
        return KANNet(
            in_features=config.get('in_features', 784),
            hidden_dim=config.get('hidden_dim', 256),
            num_classes=config.get('num_classes', 10),
            num_bases=config.get('num_bases', 10),
            sigma=config.get('sigma', 0.3),
            basis_type=config.get('basis_type', 'gaussian')
        )
    else:
        return EfficientKAN(
            in_features=config.get('in_features', 784),
            hidden_dims=config.get('hidden_dims', [128, 64]),
            num_classes=config.get('num_classes', 10),
            grid_size=config.get('grid_size', 5),
            spline_order=config.get('spline_order', 3),
            basis_type=config.get('basis_type', 'gaussian')
        )

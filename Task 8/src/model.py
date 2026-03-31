"""Vision Transformer Model for MNIST Classification."""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings using Conv2d."""
    
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial dimensions: (B, embed_dim, n_patches)
        x = x.flatten(2)
        # Transpose: (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block with Multi-Head Attention and MLP."""
    
    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # Pre-norm architecture
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image classification."""
    
    def __init__(
        self,
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        depth=6,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers and layer norms
        self.apply(self._init_module_weights)
        
    def _init_module_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding: (B, n_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Prepend class token: (B, n_patches+1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Classification using class token
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits


def create_vit(config):
    """Factory function to create ViT from config dict."""
    return VisionTransformer(
        img_size=config.get('img_size', 28),
        patch_size=config.get('patch_size', 7),
        in_channels=config.get('in_channels', 1),
        num_classes=config.get('num_classes', 10),
        embed_dim=config.get('embed_dim', 64),
        depth=config.get('depth', 6),
        num_heads=config.get('num_heads', 4),
        mlp_ratio=config.get('mlp_ratio', 2.0),
        dropout=config.get('dropout', 0.1)
    )

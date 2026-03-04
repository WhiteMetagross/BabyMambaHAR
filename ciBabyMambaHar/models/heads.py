"""
Classification Head for BabyMamba

Provides the output layer that maps from model features to class predictions.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification Head with Global Average Pooling.
    
    Architecture:
    1. Global Average Pooling over time dimension
    2. Linear projection to num_classes
    
    Args:
        inFeatures: Input feature dimension (d_model)
        numClasses: Number of output classes
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(
        self,
        inFeatures: int,
        numClasses: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(inFeatures, numClasses)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, D, T] (batch, features, time)
        Returns:
            Logits [B, numClasses]
        """
        # Global Average Pooling
        x = self.pooling(x).squeeze(-1)  # [B, D]
        
        # Dropout + Linear
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class MultiHeadClassificationHead(nn.Module):
    """
    Multi-Head Classification for ensemble predictions.
    
    Creates multiple classification heads and averages predictions.
    Useful for uncertainty estimation.
    """
    
    def __init__(
        self,
        inFeatures: int,
        numClasses: int,
        numHeads: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.heads = nn.ModuleList([
            ClassificationHead(inFeatures, numClasses, dropout)
            for _ in range(numHeads)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns averaged predictions from all heads."""
        outputs = [head(x) for head in self.heads]
        return torch.stack(outputs).mean(dim=0)

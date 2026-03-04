"""
Checkpoint Utilities

Save and load model checkpoints with training state.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
import json


def saveCheckpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    filepath: str,
    scheduler: Optional[Any] = None,
    isBest: bool = False
):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Evaluation metrics
        config: Training configuration
        filepath: Path to save checkpoint
        scheduler: Optional LR scheduler
        isBest: If True, also save as best.pt
    """
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)
    
    # Save as best if specified
    if isBest:
        bestPath = Path(filepath).parent / 'best.pt'
        torch.save(checkpoint, bestPath)
    
    # Also save metrics as JSON for easy reading
    metricsPath = Path(filepath).with_suffix('.json')
    with open(metricsPath, 'w') as f:
        json.dump({
            'epoch': epoch,
            'metrics': metrics,
        }, f, indent=2)


def loadCheckpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        device: Device to load to
    
    Returns:
        Checkpoint dictionary with metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {}),
    }


def saveModelOnly(model: nn.Module, filepath: str):
    """
    Save only the model weights (for inference/deployment).
    
    Args:
        model: PyTorch model
        filepath: Path to save
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)


def loadModelOnly(model: nn.Module, filepath: str, device: str = 'cuda'):
    """
    Load only model weights.
    
    Args:
        model: PyTorch model
        filepath: Path to checkpoint
        device: Device to load to
    """
    state_dict = torch.load(filepath, map_location=device)
    
    # Handle case where checkpoint contains full training state
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)

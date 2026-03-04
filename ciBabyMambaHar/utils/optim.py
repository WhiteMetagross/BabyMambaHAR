"""
Optimizer and Scheduler Utilities
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Dict, Any


def getOptimizer(
    model: nn.Module,
    name: str = 'adamw',
    lr: float = 1e-3,
    weightDecay: float = 0.01,
    momentum: float = 0.9,
    **kwargs
) -> Optimizer:
    """
    Get optimizer for model.
    
    Args:
        model: PyTorch model
        name: Optimizer name ('adamw', 'adam', 'sgd')
        lr: Learning rate
        weightDecay: Weight decay
        momentum: Momentum (for SGD)
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer instance
    """
    # Separate parameters that should not have weight decay
    decayParams = []
    noDecayParams = []
    
    for name_, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name_ or 'norm' in name_ or 'bn' in name_:
            noDecayParams.append(param)
        else:
            decayParams.append(param)
    
    paramGroups = [
        {'params': decayParams, 'weight_decay': weightDecay},
        {'params': noDecayParams, 'weight_decay': 0.0}
    ]
    
    name = name.lower()
    
    if name == 'adamw':
        return torch.optim.AdamW(paramGroups, lr=lr, **kwargs)
    elif name == 'adam':
        return torch.optim.Adam(paramGroups, lr=lr, **kwargs)
    elif name == 'sgd':
        return torch.optim.SGD(paramGroups, lr=lr, momentum=momentum, **kwargs)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(paramGroups, lr=lr, momentum=momentum, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def getScheduler(
    optimizer: Optimizer,
    name: str = 'cosine',
    epochs: int = 100,
    stepsPerEpoch: int = 100,
    warmupEpochs: int = 5,
    minLr: float = 1e-6,
    **kwargs
) -> Optional[_LRScheduler]:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        name: Scheduler name ('cosine', 'step', 'multistep', 'linear', 'none')
        epochs: Total training epochs
        stepsPerEpoch: Steps per epoch (for warmup)
        warmupEpochs: Number of warmup epochs
        minLr: Minimum learning rate
        **kwargs: Additional scheduler arguments
    
    Returns:
        Scheduler instance or None
    """
    totalSteps = epochs * stepsPerEpoch
    warmupSteps = warmupEpochs * stepsPerEpoch
    
    name = name.lower()
    
    if name == 'none':
        return None
    
    elif name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=minLr
        )
        
        # Add warmup if specified
        if warmupEpochs > 0:
            scheduler = WarmupScheduler(
                optimizer,
                scheduler,
                warmupEpochs=warmupEpochs
            )
        
        return scheduler
    
    elif name == 'step':
        stepSize = kwargs.get('stepSize', epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=stepSize,
            gamma=gamma
        )
    
    elif name == 'multistep':
        milestones = kwargs.get('milestones', [epochs // 3, 2 * epochs // 3])
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
    
    elif name == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=minLr / optimizer.param_groups[0]['lr'],
            total_iters=epochs
        )
    
    elif name == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=totalSteps,
            pct_start=warmupSteps / totalSteps if totalSteps > 0 else 0.1,
            anneal_strategy='cos',
            final_div_factor=1000
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {name}")


class WarmupScheduler(_LRScheduler):
    """
    Linear warmup wrapper for any scheduler.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        mainScheduler: _LRScheduler,
        warmupEpochs: int = 5,
        lastEpoch: int = -1
    ):
        self.mainScheduler = mainScheduler
        self.warmupEpochs = warmupEpochs
        self.finished = False
        super().__init__(optimizer, lastEpoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmupEpochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / self.warmupEpochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            return self.mainScheduler.get_last_lr()
    
    def step(self, epoch=None):
        if self.last_epoch >= self.warmupEpochs - 1:
            if epoch is None:
                self.mainScheduler.step()
            else:
                self.mainScheduler.step(epoch - self.warmupEpochs)
        super().step(epoch)

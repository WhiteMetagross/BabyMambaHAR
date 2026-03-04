"""
CiBabyMambaHar Ablation Model Variants

These models are used for ablation studies to demonstrate the importance
of each architectural component in CiBabyMambaHar (BabyMamba-Crossover-BiDir).

Ablation Variants:
1. CiBabyMambaHarFull: Complete model (baseline)
2. CiBabyMambaHarUnidirectional: Without bidirectional processing
3. CiBabyMambaHar2Layer: Only 2 layers instead of 4
4. CiBabyMambaHarNoPatching: Without discrete patching (direct SSM on stem output)
5. CiBabyMambaHarCnnOnly: Replace SSM with CNN (no Mamba)

Run: python scripts/runCiBabyMambaHarAblations.py --dataset ucihar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from ciBabyMambaHar.models.ciBabyMambaBlock import (
    WeightTiedBiDirMambaBlock,
    PureSelectiveScan,
)


# ============================================================================
# FROZEN ARCHITECTURE CONFIG (same as main model)
# ============================================================================
CI_BABYMAMBA_HAR_CONFIG = {
    'dModel': 26,
    'dState': 8,
    'nLayers': 4,
    'expand': 2,
    'dtRank': 2,
    'dConv': 4,
    'stemKernel': 5,
    'patchKernel': 16,
    'patchStride': 4,
}


# ============================================================================
# ABLATION A: FULL MODEL (BASELINE)
# ============================================================================
class CiBabyMambaHarFull(nn.Module):
    """
    Ablation A: Full CiBabyMambaHar Model (Baseline).
    
    Complete architecture with all components:
    - Stem: Conv1D + BatchNorm + SiLU
    - Discrete Patching: Depthwise + Pointwise + BatchNorm + Pos Embed
    - 4 × WeightTiedBiDirMambaBlock
    - Classification Head: GlobalPool + LayerNorm + Linear
    
    This is the baseline for ablation comparisons.
    """
    
    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dModel = CI_BABYMAMBA_HAR_CONFIG['dModel']
        self.dState = CI_BABYMAMBA_HAR_CONFIG['dState']
        self.nLayers = CI_BABYMAMBA_HAR_CONFIG['nLayers']
        self.expand = CI_BABYMAMBA_HAR_CONFIG['expand']
        self.dtRank = CI_BABYMAMBA_HAR_CONFIG['dtRank']
        self.dConv = CI_BABYMAMBA_HAR_CONFIG['dConv']
        self.stemKernel = CI_BABYMAMBA_HAR_CONFIG['stemKernel']
        self.patchKernel = CI_BABYMAMBA_HAR_CONFIG['patchKernel']
        self.patchStride = CI_BABYMAMBA_HAR_CONFIG['patchStride']
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        self.numPatches = (seqLen - self.patchKernel) // self.patchStride + 1
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(inChannels, self.dModel, kernel_size=self.stemKernel,
                      padding=self.stemKernel // 2, stride=1, bias=False),
            nn.BatchNorm1d(self.dModel),
            nn.SiLU(inplace=True),
        )
        
        # Patch Embed
        self.patchDepthwise = nn.Conv1d(self.dModel, self.dModel,
                                        kernel_size=self.patchKernel,
                                        stride=self.patchStride,
                                        padding=self.patchKernel // 4,
                                        groups=self.dModel, bias=False)
        self.patchPointwise = nn.Conv1d(self.dModel, self.dModel, kernel_size=1, bias=False)
        self.patchNorm = nn.BatchNorm1d(self.dModel)
        self.posEmbed = nn.Parameter(torch.zeros(1, self.numPatches, self.dModel))
        nn.init.trunc_normal_(self.posEmbed, std=0.02)
        
        # Mamba Layers
        self.mambaLayers = nn.ModuleList([
            WeightTiedBiDirMambaBlock(
                dModel=self.dModel, dState=self.dState, dConv=self.dConv,
                expand=self.expand, dtRank=self.dtRank
            )
            for _ in range(self.nLayers)
        ])
        
        # Head
        self.headNorm = nn.LayerNorm(self.dModel)
        self.headDropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.headLinear = nn.Linear(self.dModel, numClasses)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] == self.inChannels:
            x = x.transpose(1, 2)
        
        x = self.stem(x)
        x = self.patchDepthwise(x)
        x = self.patchPointwise(x)
        x = self.patchNorm(x)
        x = F.silu(x)
        x = x.transpose(1, 2)
        
        if x.size(1) != self.posEmbed.size(1):
            posEmbed = F.interpolate(self.posEmbed.transpose(1, 2),
                                     size=x.size(1), mode='linear',
                                     align_corners=False).transpose(1, 2)
        else:
            posEmbed = self.posEmbed
        x = x + posEmbed
        
        for layer in self.mambaLayers:
            x = layer(x)
        
        x = x.mean(dim=1)
        x = self.headNorm(x)
        x = self.headDropout(x)
        x = self.headLinear(x)
        
        return x


# ============================================================================
# ABLATION B: UNIDIRECTIONAL (NO BIDIRECTIONAL)
# ============================================================================
class UnidirectionalMambaBlock(nn.Module):
    """Single-direction Mamba block (forward only)."""
    
    def __init__(self, dModel: int, dState: int, dConv: int, expand: int, dtRank: int):
        super().__init__()
        self.preNorm = nn.LayerNorm(dModel)
        self.ssm = PureSelectiveScan(dModel=dModel, dState=dState, dConv=dConv,
                                      expand=expand, dtRank=dtRank)
        self.postNorm = nn.LayerNorm(dModel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.preNorm(x)
        x = self.ssm(x)  # Forward only
        out = residual + x
        out = self.postNorm(out)
        return out


class CiBabyMambaHarUnidirectional(nn.Module):
    """
    Ablation B: Without Bidirectional Processing.
    
    Uses unidirectional SSM (forward only) instead of weight-tied bidirectional.
    Expected: Lower accuracy due to missing backward context.
    """
    
    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dModel = CI_BABYMAMBA_HAR_CONFIG['dModel']
        self.dState = CI_BABYMAMBA_HAR_CONFIG['dState']
        self.nLayers = CI_BABYMAMBA_HAR_CONFIG['nLayers']
        self.expand = CI_BABYMAMBA_HAR_CONFIG['expand']
        self.dtRank = CI_BABYMAMBA_HAR_CONFIG['dtRank']
        self.dConv = CI_BABYMAMBA_HAR_CONFIG['dConv']
        self.stemKernel = CI_BABYMAMBA_HAR_CONFIG['stemKernel']
        self.patchKernel = CI_BABYMAMBA_HAR_CONFIG['patchKernel']
        self.patchStride = CI_BABYMAMBA_HAR_CONFIG['patchStride']
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        self.numPatches = (seqLen - self.patchKernel) // self.patchStride + 1
        
        # Same stem and patch embed
        self.stem = nn.Sequential(
            nn.Conv1d(inChannels, self.dModel, kernel_size=self.stemKernel,
                      padding=self.stemKernel // 2, stride=1, bias=False),
            nn.BatchNorm1d(self.dModel),
            nn.SiLU(inplace=True),
        )
        
        self.patchDepthwise = nn.Conv1d(self.dModel, self.dModel,
                                        kernel_size=self.patchKernel,
                                        stride=self.patchStride,
                                        padding=self.patchKernel // 4,
                                        groups=self.dModel, bias=False)
        self.patchPointwise = nn.Conv1d(self.dModel, self.dModel, kernel_size=1, bias=False)
        self.patchNorm = nn.BatchNorm1d(self.dModel)
        self.posEmbed = nn.Parameter(torch.zeros(1, self.numPatches, self.dModel))
        nn.init.trunc_normal_(self.posEmbed, std=0.02)
        
        # Unidirectional Mamba Layers
        self.mambaLayers = nn.ModuleList([
            UnidirectionalMambaBlock(
                dModel=self.dModel, dState=self.dState, dConv=self.dConv,
                expand=self.expand, dtRank=self.dtRank
            )
            for _ in range(self.nLayers)
        ])
        
        # Head
        self.headNorm = nn.LayerNorm(self.dModel)
        self.headDropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.headLinear = nn.Linear(self.dModel, numClasses)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] == self.inChannels:
            x = x.transpose(1, 2)
        
        x = self.stem(x)
        x = self.patchDepthwise(x)
        x = self.patchPointwise(x)
        x = self.patchNorm(x)
        x = F.silu(x)
        x = x.transpose(1, 2)
        
        if x.size(1) != self.posEmbed.size(1):
            posEmbed = F.interpolate(self.posEmbed.transpose(1, 2),
                                     size=x.size(1), mode='linear',
                                     align_corners=False).transpose(1, 2)
        else:
            posEmbed = self.posEmbed
        x = x + posEmbed
        
        for layer in self.mambaLayers:
            x = layer(x)
        
        x = x.mean(dim=1)
        x = self.headNorm(x)
        x = self.headDropout(x)
        x = self.headLinear(x)
        
        return x


# ============================================================================
# ABLATION C: 2-LAYER (INSTEAD OF 4)
# ============================================================================
class CiBabyMambaHar2Layer(nn.Module):
    """
    Ablation C: Only 2 Layers (instead of 4).
    
    Uses half the depth to demonstrate importance of 4-layer design.
    Expected: Lower accuracy, especially for complex patterns.
    """
    
    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dModel = CI_BABYMAMBA_HAR_CONFIG['dModel']
        self.dState = CI_BABYMAMBA_HAR_CONFIG['dState']
        self.nLayers = 2  # ABLATION: Only 2 layers
        self.expand = CI_BABYMAMBA_HAR_CONFIG['expand']
        self.dtRank = CI_BABYMAMBA_HAR_CONFIG['dtRank']
        self.dConv = CI_BABYMAMBA_HAR_CONFIG['dConv']
        self.stemKernel = CI_BABYMAMBA_HAR_CONFIG['stemKernel']
        self.patchKernel = CI_BABYMAMBA_HAR_CONFIG['patchKernel']
        self.patchStride = CI_BABYMAMBA_HAR_CONFIG['patchStride']
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        self.numPatches = (seqLen - self.patchKernel) // self.patchStride + 1
        
        # Same stem and patch embed
        self.stem = nn.Sequential(
            nn.Conv1d(inChannels, self.dModel, kernel_size=self.stemKernel,
                      padding=self.stemKernel // 2, stride=1, bias=False),
            nn.BatchNorm1d(self.dModel),
            nn.SiLU(inplace=True),
        )
        
        self.patchDepthwise = nn.Conv1d(self.dModel, self.dModel,
                                        kernel_size=self.patchKernel,
                                        stride=self.patchStride,
                                        padding=self.patchKernel // 4,
                                        groups=self.dModel, bias=False)
        self.patchPointwise = nn.Conv1d(self.dModel, self.dModel, kernel_size=1, bias=False)
        self.patchNorm = nn.BatchNorm1d(self.dModel)
        self.posEmbed = nn.Parameter(torch.zeros(1, self.numPatches, self.dModel))
        nn.init.trunc_normal_(self.posEmbed, std=0.02)
        
        # Only 2 Mamba Layers
        self.mambaLayers = nn.ModuleList([
            WeightTiedBiDirMambaBlock(
                dModel=self.dModel, dState=self.dState, dConv=self.dConv,
                expand=self.expand, dtRank=self.dtRank
            )
            for _ in range(self.nLayers)
        ])
        
        # Head
        self.headNorm = nn.LayerNorm(self.dModel)
        self.headDropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.headLinear = nn.Linear(self.dModel, numClasses)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] == self.inChannels:
            x = x.transpose(1, 2)
        
        x = self.stem(x)
        x = self.patchDepthwise(x)
        x = self.patchPointwise(x)
        x = self.patchNorm(x)
        x = F.silu(x)
        x = x.transpose(1, 2)
        
        if x.size(1) != self.posEmbed.size(1):
            posEmbed = F.interpolate(self.posEmbed.transpose(1, 2),
                                     size=x.size(1), mode='linear',
                                     align_corners=False).transpose(1, 2)
        else:
            posEmbed = self.posEmbed
        x = x + posEmbed
        
        for layer in self.mambaLayers:
            x = layer(x)
        
        x = x.mean(dim=1)
        x = self.headNorm(x)
        x = self.headDropout(x)
        x = self.headLinear(x)
        
        return x


# ============================================================================
# ABLATION D: NO PATCHING
# ============================================================================
class CiBabyMambaHarNoPatching(nn.Module):
    """
    Ablation D: Without Discrete Patching.
    
    Applies SSM directly on stem output without patching.
    Expected: Lower efficiency, potentially lower accuracy due to longer sequences.
    """
    
    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dModel = CI_BABYMAMBA_HAR_CONFIG['dModel']
        self.dState = CI_BABYMAMBA_HAR_CONFIG['dState']
        self.nLayers = CI_BABYMAMBA_HAR_CONFIG['nLayers']
        self.expand = CI_BABYMAMBA_HAR_CONFIG['expand']
        self.dtRank = CI_BABYMAMBA_HAR_CONFIG['dtRank']
        self.dConv = CI_BABYMAMBA_HAR_CONFIG['dConv']
        self.stemKernel = CI_BABYMAMBA_HAR_CONFIG['stemKernel']
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        
        # Stem only, no patching
        self.stem = nn.Sequential(
            nn.Conv1d(inChannels, self.dModel, kernel_size=self.stemKernel,
                      padding=self.stemKernel // 2, stride=1, bias=False),
            nn.BatchNorm1d(self.dModel),
            nn.SiLU(inplace=True),
        )
        
        # Positional embedding for full sequence
        self.posEmbed = nn.Parameter(torch.zeros(1, seqLen, self.dModel))
        nn.init.trunc_normal_(self.posEmbed, std=0.02)
        
        # Mamba Layers
        self.mambaLayers = nn.ModuleList([
            WeightTiedBiDirMambaBlock(
                dModel=self.dModel, dState=self.dState, dConv=self.dConv,
                expand=self.expand, dtRank=self.dtRank
            )
            for _ in range(self.nLayers)
        ])
        
        # Head
        self.headNorm = nn.LayerNorm(self.dModel)
        self.headDropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.headLinear = nn.Linear(self.dModel, numClasses)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] == self.inChannels:
            x = x.transpose(1, 2)
        
        x = self.stem(x)  # [B, dModel, seqLen]
        x = x.transpose(1, 2)  # [B, seqLen, dModel]
        
        if x.size(1) != self.posEmbed.size(1):
            posEmbed = F.interpolate(self.posEmbed.transpose(1, 2),
                                     size=x.size(1), mode='linear',
                                     align_corners=False).transpose(1, 2)
        else:
            posEmbed = self.posEmbed
        x = x + posEmbed
        
        for layer in self.mambaLayers:
            x = layer(x)
        
        x = x.mean(dim=1)
        x = self.headNorm(x)
        x = self.headDropout(x)
        x = self.headLinear(x)
        
        return x


# ============================================================================
# ABLATION E: CNN ONLY (NO MAMBA)
# ============================================================================
class CNNBlock(nn.Module):
    """CNN block to replace Mamba."""
    
    def __init__(self, dModel: int, expand: int = 2, kernelSize: int = 5):
        super().__init__()
        dInner = dModel * expand
        self.conv = nn.Sequential(
            nn.Conv1d(dModel, dInner, kernelSize, padding=kernelSize // 2, groups=1),
            nn.BatchNorm1d(dInner),
            nn.SiLU(inplace=True),
            nn.Conv1d(dInner, dModel, 1),
            nn.BatchNorm1d(dModel),
        )
        self.norm = nn.LayerNorm(dModel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        residual = x
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, L, D]
        x = residual + x
        x = self.norm(x)
        return x


class CiBabyMambaHarCnnOnly(nn.Module):
    """
    Ablation E: CNN Only (No Mamba/SSM).
    
    Replaces SSM blocks with CNN blocks.
    Expected: Lower accuracy on longer sequences, but may work for short sequences.
    """
    
    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dModel = CI_BABYMAMBA_HAR_CONFIG['dModel']
        self.nLayers = CI_BABYMAMBA_HAR_CONFIG['nLayers']
        self.expand = CI_BABYMAMBA_HAR_CONFIG['expand']
        self.stemKernel = CI_BABYMAMBA_HAR_CONFIG['stemKernel']
        self.patchKernel = CI_BABYMAMBA_HAR_CONFIG['patchKernel']
        self.patchStride = CI_BABYMAMBA_HAR_CONFIG['patchStride']
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        self.numPatches = (seqLen - self.patchKernel) // self.patchStride + 1
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(inChannels, self.dModel, kernel_size=self.stemKernel,
                      padding=self.stemKernel // 2, stride=1, bias=False),
            nn.BatchNorm1d(self.dModel),
            nn.SiLU(inplace=True),
        )
        
        # Patch Embed
        self.patchDepthwise = nn.Conv1d(self.dModel, self.dModel,
                                        kernel_size=self.patchKernel,
                                        stride=self.patchStride,
                                        padding=self.patchKernel // 4,
                                        groups=self.dModel, bias=False)
        self.patchPointwise = nn.Conv1d(self.dModel, self.dModel, kernel_size=1, bias=False)
        self.patchNorm = nn.BatchNorm1d(self.dModel)
        self.posEmbed = nn.Parameter(torch.zeros(1, self.numPatches, self.dModel))
        nn.init.trunc_normal_(self.posEmbed, std=0.02)
        
        # CNN Layers (instead of Mamba)
        self.cnnLayers = nn.ModuleList([
            CNNBlock(dModel=self.dModel, expand=self.expand)
            for _ in range(self.nLayers)
        ])
        
        # Head
        self.headNorm = nn.LayerNorm(self.dModel)
        self.headDropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.headLinear = nn.Linear(self.dModel, numClasses)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] == self.inChannels:
            x = x.transpose(1, 2)
        
        x = self.stem(x)
        x = self.patchDepthwise(x)
        x = self.patchPointwise(x)
        x = self.patchNorm(x)
        x = F.silu(x)
        x = x.transpose(1, 2)
        
        if x.size(1) != self.posEmbed.size(1):
            posEmbed = F.interpolate(self.posEmbed.transpose(1, 2),
                                     size=x.size(1), mode='linear',
                                     align_corners=False).transpose(1, 2)
        else:
            posEmbed = self.posEmbed
        x = x + posEmbed
        
        for layer in self.cnnLayers:
            x = layer(x)
        
        x = x.mean(dim=1)
        x = self.headNorm(x)
        x = self.headDropout(x)
        x = self.headLinear(x)
        
        return x


# ============================================================================
# FACTORY FUNCTION
# ============================================================================
ABLATION_MODELS = {
    'full': CiBabyMambaHarFull,
    'unidirectional': CiBabyMambaHarUnidirectional,
    '2layer': CiBabyMambaHar2Layer,
    'nopatching': CiBabyMambaHarNoPatching,
    'cnnonly': CiBabyMambaHarCnnOnly,
}


def getAblationModel(variant: str, **kwargs) -> nn.Module:
    """
    Factory function to get CiBabyMambaHar ablation model.
    
    Args:
        variant: One of 'full', 'unidirectional', '2layer', 'nopatching', 'cnnonly'
        **kwargs: Model arguments (numClasses, inChannels, etc.)
    
    Returns:
        Ablation model instance
    """
    if variant.lower() not in ABLATION_MODELS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(ABLATION_MODELS.keys())}")
    
    return ABLATION_MODELS[variant.lower()](**kwargs)


if __name__ == "__main__":
    # Quick test
    print("CiBabyMambaHar Ablation Models")
    print("=" * 60)
    
    for name, ModelClass in ABLATION_MODELS.items():
        model = ModelClass(numClasses=6, inChannels=9)
        params = sum(p.numel() for p in model.parameters())
        
        x = torch.randn(2, 128, 9)
        y = model(x)
        
        print(f"{name:15s}: {params:,} params, output shape: {y.shape}")

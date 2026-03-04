"""
BabyMamba-Crossover-BiDir Ablation Model Variants

These models are used for ablation studies to demonstrate the importance
of each architectural component in BabyMamba-Crossover-BiDir.

Ablation Variants (matching source ciBabyMambaHar/models/ciBabyMambaAblations.py):
    1. CrossoverBiDirBabyMambaHarFull: Complete model (baseline)
    2. CrossoverBiDirBabyMambaHarUnidirectional: Without bidirectional processing
    3. CrossoverBiDirBabyMambaHar2Layer: Only 2 layers instead of 4
    4. CrossoverBiDirBabyMambaHarNoPatching: Without discrete patching
    5. CrossoverBiDirBabyMambaHarCnnOnly: Replace SSM with CNN (no Mamba)

All ablations use the FROZEN hyperparameters:
    d_model = 26, d_state = 8, n_layers = 4 (unless ablated),
    expand = 2, dt_rank = 2, d_conv = 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .crossoverBiDirBlock import WeightTiedBiDirMambaBlock, PureSelectiveScan


# ============================================================================
# FROZEN ARCHITECTURE CONFIG (same as main model)
# ============================================================================
BABYMAMBA_CONFIG = {
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
class CrossoverBiDirBabyMambaHarFull(nn.Module):
    """
    Ablation A: Full BabyMamba-Crossover-BiDir Model (Baseline).
    
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
        
        self.dModel = BABYMAMBA_CONFIG['dModel']
        self.dState = BABYMAMBA_CONFIG['dState']
        self.nLayers = BABYMAMBA_CONFIG['nLayers']
        self.expand = BABYMAMBA_CONFIG['expand']
        self.dtRank = BABYMAMBA_CONFIG['dtRank']
        self.dConv = BABYMAMBA_CONFIG['dConv']
        self.stemKernel = BABYMAMBA_CONFIG['stemKernel']
        self.patchKernel = BABYMAMBA_CONFIG['patchKernel']
        self.patchStride = BABYMAMBA_CONFIG['patchStride']
        
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


class CrossoverBiDirBabyMambaHarUnidirectional(nn.Module):
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
        
        self.dModel = BABYMAMBA_CONFIG['dModel']
        self.dState = BABYMAMBA_CONFIG['dState']
        self.nLayers = BABYMAMBA_CONFIG['nLayers']
        self.expand = BABYMAMBA_CONFIG['expand']
        self.dtRank = BABYMAMBA_CONFIG['dtRank']
        self.dConv = BABYMAMBA_CONFIG['dConv']
        self.stemKernel = BABYMAMBA_CONFIG['stemKernel']
        self.patchKernel = BABYMAMBA_CONFIG['patchKernel']
        self.patchStride = BABYMAMBA_CONFIG['patchStride']
        
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
class CrossoverBiDirBabyMambaHar2Layer(nn.Module):
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
        
        self.dModel = BABYMAMBA_CONFIG['dModel']
        self.dState = BABYMAMBA_CONFIG['dState']
        self.nLayers = 2  # ABLATION: Only 2 layers
        self.expand = BABYMAMBA_CONFIG['expand']
        self.dtRank = BABYMAMBA_CONFIG['dtRank']
        self.dConv = BABYMAMBA_CONFIG['dConv']
        self.stemKernel = BABYMAMBA_CONFIG['stemKernel']
        self.patchKernel = BABYMAMBA_CONFIG['patchKernel']
        self.patchStride = BABYMAMBA_CONFIG['patchStride']
        
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
class CrossoverBiDirBabyMambaHarNoPatching(nn.Module):
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
        
        self.dModel = BABYMAMBA_CONFIG['dModel']
        self.dState = BABYMAMBA_CONFIG['dState']
        self.nLayers = BABYMAMBA_CONFIG['nLayers']
        self.expand = BABYMAMBA_CONFIG['expand']
        self.dtRank = BABYMAMBA_CONFIG['dtRank']
        self.dConv = BABYMAMBA_CONFIG['dConv']
        self.stemKernel = BABYMAMBA_CONFIG['stemKernel']
        
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


class CrossoverBiDirBabyMambaHarCnnOnly(nn.Module):
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
        
        self.dModel = BABYMAMBA_CONFIG['dModel']
        self.nLayers = BABYMAMBA_CONFIG['nLayers']
        self.expand = BABYMAMBA_CONFIG['expand']
        self.stemKernel = BABYMAMBA_CONFIG['stemKernel']
        self.patchKernel = BABYMAMBA_CONFIG['patchKernel']
        self.patchStride = BABYMAMBA_CONFIG['patchStride']
        
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
        
        # CNN Layers instead of Mamba
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
# FACTORY FUNCTIONS
# ============================================================================
ABLATION_MODELS = {
    'full': CrossoverBiDirBabyMambaHarFull,
    'unidirectional': CrossoverBiDirBabyMambaHarUnidirectional,
    '2layer': CrossoverBiDirBabyMambaHar2Layer,
    'no_patching': CrossoverBiDirBabyMambaHarNoPatching,
    'cnn_only': CrossoverBiDirBabyMambaHarCnnOnly,
}


def createAblationModel(
    ablationType: str,
    numClasses: int = 6,
    inChannels: int = 9,
    seqLen: int = 128,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Factory function for creating ablation models.
    
    Args:
        ablationType: 'full', 'unidirectional', '2layer', 'no_patching', 'cnn_only'
        numClasses: Number of classes
        inChannels: Number of input channels
        seqLen: Sequence length
        dropout: Dropout rate
    
    Returns:
        Ablation model
    """
    if ablationType not in ABLATION_MODELS:
        raise ValueError(f"Unknown ablation: {ablationType}. "
                        f"Choose from: {list(ABLATION_MODELS.keys())}")
    
    return ABLATION_MODELS[ablationType](
        numClasses=numClasses,
        inChannels=inChannels,
        seqLen=seqLen,
        dropout=dropout
    )


if __name__ == "__main__":
    print("=" * 60)
    print("BabyMamba-Crossover-BiDir Ablation Models")
    print("=" * 60)
    
    # Test all ablations
    x = torch.randn(2, 9, 128)  # UCI-HAR format
    
    for name, Model in ABLATION_MODELS.items():
        model = Model(numClasses=6, inChannels=9, seqLen=128)
        params = sum(p.numel() for p in model.parameters())
        out = model(x)
        print(f"{name:15s}: {params:6,} params, output: {out.shape}")

"""
CI-BabyMamba-HAR: Channel-Independent State Space Model with Context-Gated Temporal Attention

Evolution from BabyMamba-HAR / CiBabyMambaHar to address performance failures on 
PAMAP2, Daphnet, and SKODA datasets by isolating sensor noise and spotlighting 
transient temporal events.

============== FROZEN ARCHITECTURE SPECIFICATION ==============

Target: 27K-29K parameters (FROZEN across all datasets)
Target Accuracy: Improved performance on SKODA, PAMAP2, Daphnet

============== HIGH-LEVEL DESIGN ==============

Name: CI-BabyMamba-HAR
Architecture Type: Channel-Independent SSM with Context-Gated Attention
Target Inference Cost: O(N) (Linear)
Total Parameters: ~27,700-28,700 (FROZEN)

============== KEY INNOVATIONS ==============

1. Channel-Independent (CI) Stem:
   - Treats each sensor channel as an independent sample
   - Prevents noise in one sensor from "poisoning" others
   - Shared 1D convolution kernel across all channels

2. Expanded State Memory (N=16):
   - Increased from 8 to 16 for nuanced activity recognition
   - Critical for PAMAP2's 12-class problem

3. Context-Gated Temporal Attention:
   - Replaces simple global mean pooling
   - Prevents dilution of transient events (Freeze in Daphnet)
   - Tanh gating + Softmax attention over patches

4. Weight-Tied Bidirectional SSM:
   - Same weights for forward and backward passes
   - Full temporal context without parameter doubling

============== THE FROZEN CONFIG ==============

| Hyperparameter    | Value | Reasoning                                    |
|-------------------|-------|----------------------------------------------|
| d_model           | 24    | Optimal for param budget with CI overhead    |
| d_state           | 16    | INCREASED: Better for 12-class problems      |
| n_layers          | 4     | Deep hierarchical features                   |
| expand            | 2     | Inner dimension = 48. Standard Mamba.        |
| dt_rank           | 2     | Minimal rank for delta discretization        |
| kernel_size       | 4     | Local 1D conv inside SSM                     |
| Bi-Directionality | True  | Weight-Tied. Scans t->T and T->t.            |
| Gated Attention   | True  | Spotlights transient events                  |

============== ARCHITECTURE STAGES ==============

STAGE I: Channel-Independent Stem
    - Reshape: (B, T, C) -> (B*C, 1, T)
    - Conv1D: 1 -> 24, k=5, shared across channels
    - BatchNorm + SiLU

STAGE II: Patch Embedding  
    - DepthwiseSep Conv: k=16, stride=4
    - Positional Embedding

STAGE III: Weight-Tied Bi-Directional SSM Backbone
    - 4x WeightTiedBiDirMambaBlock (d_state=16)

STAGE IV: Context-Gated Temporal Attention
    - Tanh(Linear) gating
    - Softmax attention weights
    - Weighted sum over patches

STAGE V: Channel Fusion & Head
    - Unfold: (B*C, 24) -> (B, C, 24)
    - Mean pool across channels
    - LayerNorm + Dropout + Linear

============== PARAMETER BUDGET ==============

| Component       | Parameters | Notes                              |
|-----------------|------------|------------------------------------|
| CI-Stem         | ~168       | Shared across all channels         |
| Patch Embed     | ~1,704     | 4x sequence compression            |
| Mamba Backbone  | ~25,344    | 4 layers, d_state=16               |
| Gated Attention | ~600       | Spotlight sparse events            |
| Head            | ~200-350   | Dataset-dependent classes          |
| TOTAL           | ~27,800-28,700                              |
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from ciBabyMambaHar.models.ciBabyMambaBlock import (
    WeightTiedBiDirMambaBlock,
    PureSelectiveScan,
    MAMBA_AVAILABLE,
)


class _UnidirectionalMambaBlock(nn.Module):
    """Unidirectional (forward-only) Mamba block used for ablations."""

    def __init__(
        self,
        dModel: int,
        dState: int,
        dConv: int,
        expand: int,
        dtRank: int,
        dropPath: float = 0.0,
    ):
        super().__init__()
        self.preNorm = nn.LayerNorm(dModel)

        if MAMBA_AVAILABLE:
            # Mamba symbol is provided by BabyMambaBlock when available
            from ciBabyMambaHar.models import BabyMambaBlock as _block
            self.ssm = _block.Mamba(
                d_model=dModel,
                d_state=dState,
                d_conv=dConv,
                expand=expand,
            )
        else:
            self.ssm = PureSelectiveScan(
                dModel=dModel,
                dState=dState,
                dConv=dConv,
                expand=expand,
                dtRank=dtRank,
            )

        self.postNorm = nn.LayerNorm(dModel)

        if dropPath and dropPath > 0:
            from ciBabyMambaHar.models.ciBabyMambaBlock import DropPath
            self.dropPath = DropPath(dropPath)
        else:
            self.dropPath = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.preNorm(x)
        h = self.ssm(x)
        h = self.dropPath(h)
        out = residual + h
        out = self.postNorm(out)
        return out


class _FusedStem(nn.Module):
    """Early-fusion stem: mixes channels immediately (Conv1D: C -> dModel)."""

    def __init__(self, inChannels: int, dModel: int, kernelSize: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=inChannels,
            out_channels=dModel,
            kernel_size=kernelSize,
            padding=kernelSize // 2,
            stride=1,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(dModel)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# ============================================================================
# FROZEN ARCHITECTURE - DO NOT MODIFY
# ============================================================================
# These parameters are LOCKED for the research paper.
# Only training hyperparameters (lr, weight_decay, dropout) are tuned via HPO.
#
CI_BABYMAMBA_HAR_CONFIG = {
    'dModel': 24,        # FROZEN - model dimension
    'dState': 16,        # FROZEN - INCREASED from 8 for nuanced activities
    'nLayers': 4,        # FROZEN - deep hierarchical features
    'expand': 2,         # FROZEN - inner dimension = 48
    'dtRank': 2,         # FROZEN - minimal delta rank
    'dConv': 4,          # FROZEN - local conv inside SSM
    'stemKernel': 5,     # FROZEN - CI stem convolution
    'patchKernel': 16,   # FROZEN - discrete patching
    'patchStride': 4,    # FROZEN - 4x sequence compression
    'useGatedAttention': True,  # FROZEN - spotlight transient events
}


class GatedTemporalAttention(nn.Module):
    """
    Context-Gated Temporal Attention.
    
    Replaces simple global mean pooling to prevent "dilution" of short events
    like gait freezing (Daphnet) or impact dynamics (PAMAP2).
    
    Mathematical Formulation:
        u = Tanh(W @ H)           # Gating transformation
        e = u @ v                 # Importance score per patch
        alpha = Softmax(e)        # Normalized attention weights
        c = sum(alpha * H)        # Context vector (attended output)
    
    Args:
        dModel: Feature dimension
    """
    
    def __init__(self, dModel: int):
        super().__init__()
        self.dModel = dModel
        
        # Linear projection for gating
        self.projection = nn.Linear(dModel, dModel, bias=True)
        
        # Learnable context vector
        self.context = nn.Parameter(torch.randn(dModel))
        nn.init.normal_(self.context, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D] (batch, sequence, features)
        Returns:
            Attended features [B, D]
        """
        # Gating with Tanh activation
        u = torch.tanh(self.projection(x))  # [B, L, D]
        
        # Score each position using context vector
        scores = torch.einsum('bld,d->bl', u, self.context)  # [B, L]
        
        # Softmax attention weights over sequence dimension
        alpha = F.softmax(scores, dim=-1)  # [B, L]
        
        # Weighted sum to get attended features
        c = torch.einsum('bl,bld->bd', alpha, x)  # [B, D]
        
        return c


class ChannelIndependentStem(nn.Module):
    """
    Channel-Independent (CI) Shared Stem.
    
    Instead of mixing sensor channels early, this stage treats each channel 
    as an independent sample to ensure noise in one sensor does not "poison" 
    others. A single-channel input kernel is shared across all channels.
    
    SKODA: Forces model to ignore uncorrelated high-frequency machine noise
    PAMAP2: Prevents heart rate noise from affecting accelerometer features
    Daphnet: Isolates ankle/thigh/trunk sensor artifacts
    
    Args:
        dModel: Output feature dimension (default: 24)
        kernelSize: Convolution kernel size (default: 5)
    """
    
    def __init__(self, dModel: int = 24, kernelSize: int = 5):
        super().__init__()
        self.dModel = dModel
        self.kernelSize = kernelSize
        
        # Single-channel convolution shared across all sensor channels
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=dModel,
            kernel_size=kernelSize,
            padding=kernelSize // 2,
            stride=1,
            bias=False
        )
        self.norm = nn.BatchNorm1d(dModel)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B*C, 1, T] - each channel treated independently
        Returns:
            Features [B*C, dModel, T]
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class CiBabyMambaHar(nn.Module):
    """
    CI-BabyMamba-HAR: Channel-Independent SSM with Context-Gated Temporal Attention.
    
    Evolution from BabyMamba-HAR to address PAMAP2, Daphnet, and SKODA failures.
    
    Target: ~27,000-28,000 parameters with O(N) inference complexity.
    
    Architecture:
        Input[B,T,C] -> CI-Stem -> PatchEmbed -> 4xBiDirMambaBlock -> 
        GatedAttention -> ChannelFusion -> Head -> [B,num_classes]
    
    FROZEN Configuration (DO NOT CHANGE):
        - d_model = 24 (model width)
        - d_state = 16 (SSM state dimension - INCREASED from 8)
        - n_layers = 4 (backbone depth)
        - expand = 2 (SSM expansion)
        - dt_rank = 2 (delta discretization rank)
        - d_conv = 4 (local conv kernel)
        - bidirectional = True (weight-tied)
        - gated_attention = True (spotlight transient events)
    
    HPO-Tunable (Training Hyperparameters ONLY):
        - dropout: 0.0 - 0.3
        - dropPath: 0.0 - 0.2 (stochastic depth)
        - learning_rate: 0.0003 - 0.003 (log scale)
        - weight_decay: 0.005 - 0.05 (log scale)
        - label_smoothing: 0.0 - 0.2
    
    Args:
        numClasses: Number of activity classes
        inChannels: Number of input sensor channels
        seqLen: Sequence length (default: 128)
        dropout: Dropout rate in classifier (default: 0.1)
        dropPath: Stochastic depth rate (default: 0.0)
        dModel: Override model dimension (ablation only)
        dState: Override state dimension (ablation only)
        nLayers: Override layer count (ablation only)
    """
    
    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        dropout: float = 0.1,
        dropPath: float = 0.0,
        # Architecture overrides for ablation studies only
        dModel: Optional[int] = None,
        dState: Optional[int] = None,
        nLayers: Optional[int] = None,
        expand: Optional[int] = None,
        dtRank: Optional[int] = None,
        dConv: Optional[int] = None,
        useGatedAttention: Optional[bool] = None,
        channelIndependent: bool = True,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        # Use frozen config, allow ablation overrides
        self.dModel = dModel or CI_BABYMAMBA_HAR_CONFIG['dModel']
        self.dState = dState or CI_BABYMAMBA_HAR_CONFIG['dState']
        self.nLayers = nLayers or CI_BABYMAMBA_HAR_CONFIG['nLayers']
        self.expand = expand or CI_BABYMAMBA_HAR_CONFIG['expand']
        self.dtRank = dtRank or CI_BABYMAMBA_HAR_CONFIG['dtRank']
        self.dConv = dConv or CI_BABYMAMBA_HAR_CONFIG['dConv']
        self.stemKernel = CI_BABYMAMBA_HAR_CONFIG['stemKernel']
        self.patchKernel = CI_BABYMAMBA_HAR_CONFIG['patchKernel']
        self.patchStride = CI_BABYMAMBA_HAR_CONFIG['patchStride']
        self.useGatedAttention = CI_BABYMAMBA_HAR_CONFIG['useGatedAttention'] if useGatedAttention is None else bool(useGatedAttention)
        self.channelIndependent = bool(channelIndependent)
        self.bidirectional = bool(bidirectional)
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        self.dropout = dropout
        self.dropPathRate = dropPath
        
        # Calculate number of patches after patching
        self.numPatches = (seqLen - self.patchKernel) // self.patchStride + 1
        
        # ============== STAGE I: Stem ==============
        # CI-stem: treat each channel as independent sample
        # Fused stem: mix channels early (C -> dModel)
        if self.channelIndependent:
            self.ciStem = ChannelIndependentStem(
                dModel=self.dModel,
                kernelSize=self.stemKernel,
            )
            self.fusedStem = None
        else:
            self.fusedStem = _FusedStem(
                inChannels=inChannels,
                dModel=self.dModel,
                kernelSize=self.stemKernel,
            )
            self.ciStem = None
        
        # ============== STAGE II: Patch Embedding ==============
        # Depthwise: spatial mixing per channel
        self.patchDepthwise = nn.Conv1d(
            self.dModel, self.dModel,
            kernel_size=self.patchKernel,
            stride=self.patchStride,
            padding=self.patchKernel // 4,
            groups=self.dModel,  # Depthwise
            bias=False
        )
        # Pointwise: channel mixing
        self.patchPointwise = nn.Conv1d(
            self.dModel, self.dModel,
            kernel_size=1,
            bias=False
        )
        self.patchNorm = nn.BatchNorm1d(self.dModel)
        
        # Learnable positional embedding
        self.posEmbed = nn.Parameter(torch.zeros(1, self.numPatches, self.dModel))
        nn.init.trunc_normal_(self.posEmbed, std=0.02)
        
        # ============== STAGE III: Weight-Tied Bi-Directional SSM Backbone ==============
        # 4 layers with d_state=16 for expanded memory
        dpRates = [x.item() for x in torch.linspace(0, dropPath, self.nLayers)]
        Block = WeightTiedBiDirMambaBlock if self.bidirectional else _UnidirectionalMambaBlock
        self.mambaLayers = nn.ModuleList([
            Block(
                dModel=self.dModel,
                dState=self.dState,
                dConv=self.dConv,
                expand=self.expand,
                dtRank=self.dtRank,
                dropPath=dpRates[i],
            )
            for i in range(self.nLayers)
        ])
        
        # ============== STAGE IV: Context-Gated Temporal Attention ==============
        if self.useGatedAttention:
            self.gatedAttention = GatedTemporalAttention(self.dModel)
        else:
            self.gatedAttention = None
        
        # ============== STAGE V: Channel Fusion & Head ==============
        self.headNorm = nn.LayerNorm(self.dModel)
        self.headDropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.headLinear = nn.Linear(self.dModel, numClasses)
        
        # Initialize weights
        self._initWeights()
    
    def _initWeights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Channel-Independent processing.
        
        Args:
            x: Input tensor [B, T, C] or [B, C, T]
        Returns:
            Logits [B, numClasses]
        """
        # Ensure input is [B, T, C] format
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # Handle [B, C, T] -> [B, T, C]
        if x.shape[-1] != self.inChannels and x.shape[1] == self.inChannels:
            x = x.transpose(1, 2)
        
        B, T, C = x.shape
        
        # ============== STAGE I: Stem ==============
        if self.channelIndependent:
            # Reshape to treat each channel independently
            x = x.permute(0, 2, 1)              # [B, C, T]
            x = x.reshape(B * C, 1, T)          # [B*C, 1, T]
            x = self.ciStem(x)                  # [B*C, dModel, T]
        else:
            # Early fusion on full multivariate signal
            x = x.permute(0, 2, 1)              # [B, C, T]
            x = self.fusedStem(x)               # [B, dModel, T]
        
        # ============== STAGE II: Patch Embedding ==============
        x = self.patchDepthwise(x)          # [B*C, dModel, numPatches]
        x = self.patchPointwise(x)          # [B*C, dModel, numPatches]
        x = self.patchNorm(x)
        x = F.silu(x)
        
        # Transpose for Mamba: [B*C, dModel, P] -> [B*C, P, dModel]
        x = x.transpose(1, 2)
        
        # Add positional embedding (handle dynamic sequence length)
        if x.size(1) != self.posEmbed.size(1):
            posEmbed = F.interpolate(
                self.posEmbed.transpose(1, 2),
                size=x.size(1),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            posEmbed = self.posEmbed
        x = x + posEmbed
        
        # ============== STAGE III: SSM Backbone ==============
        for layer in self.mambaLayers:
            x = layer(x)                    # [B*C, P, dModel]
        
        # ============== STAGE IV: Context-Gated Temporal Attention ==============
        if self.gatedAttention is not None:
            x = self.gatedAttention(x)      # [B*C, dModel]
        else:
            # Fallback to global mean pooling
            x = x.mean(dim=1)               # [B*C, dModel]
        
        # ============== STAGE V: Channel Fusion & Head ==============
        if self.channelIndependent:
            # Unfold back to batch dimension
            x = x.view(B, C, self.dModel)       # [B, C, dModel]
            # Mean pool across channels (filters uncorrelated noise)
            x = x.mean(dim=1)                   # [B, dModel]
        else:
            # Already fused; x is [B, dModel]
            pass
        
        x = self.headNorm(x)
        x = self.headDropout(x)
        x = self.headLinear(x)              # [B, numClasses]
        
        return x
    
    def countParameters(self) -> Dict[str, int]:
        """Count model parameters by component."""
        if self.channelIndependent:
            stemParams = sum(p.numel() for p in self.ciStem.parameters()) if self.ciStem is not None else 0
        else:
            stemParams = sum(p.numel() for p in self.fusedStem.parameters()) if self.fusedStem is not None else 0
        
        patchParams = (
            sum(p.numel() for p in self.patchDepthwise.parameters()) +
            sum(p.numel() for p in self.patchPointwise.parameters()) +
            sum(p.numel() for p in self.patchNorm.parameters()) +
            self.posEmbed.numel()
        )
        
        mambaParams = sum(p.numel() for p in self.mambaLayers.parameters())
        
        attentionParams = 0
        if self.gatedAttention is not None:
            attentionParams = sum(p.numel() for p in self.gatedAttention.parameters())
        
        headParams = (
            sum(p.numel() for p in self.headNorm.parameters()) +
            sum(p.numel() for p in self.headLinear.parameters())
        )
        
        totalParams = sum(p.numel() for p in self.parameters())
        
        return {
            'stem': stemParams,
            'patch': patchParams,
            'mamba': mambaParams,
            'attention': attentionParams,
            'head': headParams,
            'total': totalParams,
        }
    
    def getConfigDict(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            'architecture': 'CI-BabyMamba-HAR',
            'numClasses': self.numClasses,
            'inChannels': self.inChannels,
            'seqLen': self.seqLen,
            'dModel': self.dModel,
            'dState': self.dState,
            'nLayers': self.nLayers,
            'expand': self.expand,
            'dtRank': self.dtRank,
            'dConv': self.dConv,
            'stemKernel': self.stemKernel,
            'patchKernel': self.patchKernel,
            'patchStride': self.patchStride,
            'numPatches': self.numPatches,
            'dropout': self.dropout,
            'dropPath': self.dropPathRate,
            'bidirectional': self.bidirectional,
            'weightTied': self.bidirectional,
            'useGatedAttention': self.useGatedAttention,
            'channelIndependent': self.channelIndependent,
        }


# Legacy alias for backward compatibility
BabyMamba = CiBabyMambaHar


def createCiBabyMambaHar(
    dataset: str = 'ucihar',
    dropout: float = 0.1,
    dState: Optional[int] = None,  # For ablation: test d_state=8/16/24
    dModel: Optional[int] = None,
    nLayers: Optional[int] = None,
    expand: Optional[int] = None,
    useGatedAttention: Optional[bool] = None,
    channelIndependent: bool = True,
    bidirectional: bool = True,
) -> CiBabyMambaHar:
    """
    Factory function to create CiBabyMambaHar for specific datasets.
    
    Args:
        dataset: Dataset name ('ucihar', 'motionsense', 'wisdm', 'pamap2', 'skoda', 'daphnet')
        dropout: Dropout rate in classifier head
        dState: Optional override for d_state (ablation studies only)
    
    Returns:
        Configured CiBabyMambaHar model (~27k-28k params)
    """
    datasetConfigs = {
        'ucihar': {'numClasses': 6, 'inChannels': 9, 'seqLen': 128},
        # MotionSense in this repo is 6 channels (acc + gyro)
        'motionsense': {'numClasses': 6, 'inChannels': 6, 'seqLen': 128},
        'wisdm': {'numClasses': 6, 'inChannels': 3, 'seqLen': 128},
        'pamap2': {'numClasses': 12, 'inChannels': 19, 'seqLen': 128},
        'skoda': {'numClasses': 11, 'inChannels': 30, 'seqLen': 98},
        'daphnet': {'numClasses': 2, 'inChannels': 9, 'seqLen': 64},
        # Keep these for convenience, but core training uses dataset specs elsewhere
        'opportunity': {'numClasses': 5, 'inChannels': 79, 'seqLen': 128},
        'unimib': {'numClasses': 9, 'inChannels': 3, 'seqLen': 128},
    }
    
    if dataset.lower() not in datasetConfigs:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(datasetConfigs.keys())}")
    
    config = datasetConfigs[dataset.lower()]
    return CiBabyMambaHar(
        dropout=dropout,
        dState=dState,
        dModel=dModel,
        nLayers=nLayers,
        expand=expand,
        useGatedAttention=useGatedAttention,
        channelIndependent=channelIndependent,
        bidirectional=bidirectional,
        **config
    )


# Legacy alias
createBabyMamba = createCiBabyMambaHar


if __name__ == "__main__":
    print("=" * 70)
    print("CI-BabyMamba-HAR: Channel-Independent SSM with Gated Attention")
    print("=" * 70)
    
    # Test UCI-HAR configuration (default)
    model = CiBabyMambaHar(numClasses=6, inChannels=9)
    params = model.countParameters()
    config = model.getConfigDict()
    
    print(f"\n=== FROZEN CONFIGURATION ===")
    print(f"Architecture: {config['architecture']}")
    print(f"Channel-Independent: {config['channelIndependent']}")
    print(f"Gated Attention: {config['useGatedAttention']}")
    print(f"Bidirectional: {config['bidirectional']} (Weight-Tied: {config['weightTied']})")
    print(f"d_model: {config['dModel']}")
    print(f"d_state: {config['dState']} (INCREASED from 8)")
    print(f"n_layers: {config['nLayers']}")
    print(f"expand: {config['expand']}")
    
    print(f"\n=== PARAMETER BUDGET (UCI-HAR) ===")
    print(f"  CI-Stem:    {params['stem']:,} params")
    print(f"  Patch:      {params['patch']:,} params")
    print(f"  Mamba:      {params['mamba']:,} params")
    print(f"  Attention:  {params['attention']:,} params")
    print(f"  Head:       {params['head']:,} params")
    print(f"  TOTAL:      {params['total']:,} params")
    
    print(f"\n=== TARGET vs ACTUAL ===")
    print(f"  Target: 24,000-28,000 params")
    print(f"  Actual: {params['total']:,} params")
    inBudget = 24000 <= params['total'] <= 30000
    print(f"  Within budget: {'YES' if inBudget else 'NO'}")
    
    # Test forward pass
    x = torch.randn(2, 128, 9)  # [B, T, C]
    y = model(x)
    print(f"\n=== FORWARD PASS ===")
    print(f"  Input:  {x.shape} -> Output: {y.shape}")
    
    # Test all datasets
    print(f"\n=== PARAMETER COUNTS BY DATASET ===")
    for dataset in ['ucihar', 'motionsense', 'wisdm', 'pamap2', 'skoda', 'daphnet']:
        m = createCiBabyMambaHar(dataset)
        p = m.countParameters()
        c = m.getConfigDict()
        print(f"  {dataset.upper():12s}: {p['total']:,} params "
              f"(in={c['inChannels']}, out={c['numClasses']})")
    
    print("\n" + "=" * 70)
    print("READY FOR TRAINING WITH SIGNAL RESCUE")
    print("=" * 70)

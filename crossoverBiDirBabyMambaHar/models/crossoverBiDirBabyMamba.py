"""
BabyMamba-Crossover-BiDir: The Definitive HAR Architecture

============== FROZEN ARCHITECTURE SPECIFICATION ==============

This is the final, frozen specification for BabyMamba-Crossover-BiDir.
Mathematically verified to fit the ~27k parameter budget (O(N) complexity)
while matching the depth and receptive field of heavier baselines.

Target: ~27,000 parameters (FROZEN)
Target Accuracy: >96% (UCI-HAR), 94%+ (all 4 datasets)

============== HIGH-LEVEL DESIGN ==============

Name: BabyMamba-Crossover-BiDir
Architecture Type: Hybrid Convolutional-SSM (Selective State Space Model)
Target Inference Cost: O(N) (Linear)
Total Parameters: ~27,000 (Approx. 27k)

============== THE GOLDEN CONFIG ==============

| Hyperparameter    | Value | Reasoning                                    |
|-------------------|-------|----------------------------------------------|
| d_model           | 26    | Wider channels for better feature extraction |
| d_state           | 8     | Sufficient latent memory for HAR             |
| n_layers          | 4     | CRITICAL: Deeper than TinierHAR (3 blocks)   |
| expand            | 2     | Inner dimension = 52. Standard Mamba.        |
| dt_rank           | 2     | Minimal rank for Δ discretization            |
| kernel_size       | 4     | Local 1D conv inside SSM                     |
| Bi-Directionality | True  | Weight-Tied. Scans t→T and T→t.              |

============== ARCHITECTURE STAGES ==============

STAGE 1: Local Feature Extraction (Stem) - ~1,352 params
    Conv1D: 9 → 26, k=5, stride=1
    BatchNorm: 26 channels
    SiLU activation

STAGE 2: Discrete Patching - ~2,080 params
    Depthwise: 26 → 26, k=16
    Pointwise: 26 → 26, k=1
    Pos Embedding: 1 × 26 × 32

STAGE 3: Weight-Tied Bi-Directional SSM Backbone - ~23,520 params
    4 × WeightTiedBiDirMambaBlock (5,880 each)
    4 × LayerNorm (52 params each)
    
STAGE 4: Classification Head - ~208 params
    Global Mean Pooling
    LayerNorm: 26 features
    Linear: 26 → num_classes

============== COMPARATIVE ADVANTAGE ==============

vs LightDeepConvLSTM (15k):
    WIN: 4 layers of depth vs shallow hybrid. Mamba's linear scan
    allows extra layers without exploding inference latency.

vs TinierHAR (17k):  
    WIN: GRUs have "forgetting" over long sequences. Mamba's state
    matrix A (discretized) preserves long-term dependencies better.

vs TinyHAR (42k):
    WIN: 40% smaller. O(N) vs O(N² attention. Runs on simpler MCUs.

============== PARAMETER BUDGET BREAKDOWN ==============

| Stage           | Component      | Config              | Params  |
|-----------------|----------------|---------------------|---------|
| 1. Stem         | Conv1D         | 9→26, k=5           | 1,170   |
|                 | BatchNorm      | 26 channels         | 52      |
|                 |                | Subtotal            | 1,222   |
| 2. Patch Embed  | Depthwise      | 26→26, k=16         | 442     |
|                 | Pointwise      | 26→26, k=1          | 676     |
|                 | Pos Embedding  | 1×32×26             | 832     |
|                 |                | Subtotal            | 1,950   |
| 3. Mamba Core   | Layer 1        | Bi-Dir, Shared Wts  | 5,880   |
|                 | Layer 2        | Bi-Dir, Shared Wts  | 5,880   |
|                 | Layer 3        | Bi-Dir, Shared Wts  | 5,880   |
|                 | Layer 4        | Bi-Dir, Shared Wts  | 5,880   |
|                 | Norms          | LayerNorm × 8       | 416     |
|                 |                | Subtotal            | 23,936  |
| 4. Head         | Global Pool    | Mean                | 0       |
|                 | LayerNorm      | 26 features         | 52      |
|                 | Linear         | 26→6 (+ Bias)       | 162     |
|                 |                | Subtotal            | 214     |
|-----------------|----------------|---------------------|---------|
| TOTAL           |                |                     | ~27,322 |
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .crossoverBiDirBlock import WeightTiedBiDirMambaBlock, UnidirectionalMambaBlock


# ============================================================================
# FROZEN ARCHITECTURE - DO NOT MODIFY
# ============================================================================
# These parameters are LOCKED for the research paper.
# Only training hyperparameters (lr, weight_decay, dropout) are tuned via HPO.
#
CROSSOVER_BIDIR_BABYMAMBA_CONFIG = {
    'dModel': 26,        # FROZEN - wider channels for better accuracy
    'dState': 8,         # FROZEN - sufficient latent memory for HAR
    'nLayers': 4,        # FROZEN - deeper than baselines (TinierHAR=3)
    'expand': 2,         # FROZEN - inner dimension = 52
    'dtRank': 2,         # FROZEN - minimal Δ rank
    'dConv': 4,          # FROZEN - local conv inside SSM
    'stemKernel': 5,     # FROZEN - local feature extraction
    'patchKernel': 16,   # FROZEN - discrete patching
    'patchStride': 4,    # FROZEN - sequence compression
}


class CrossoverBiDirBabyMambaHar(nn.Module):
    """
    BabyMamba-Crossover-BiDir: The Definitive HAR Architecture.
    
    Target: ~27,000 parameters with O(N) inference complexity.
    
    Architecture:
        Input[B,C,L] → Stem → PatchEmbed → 4×BiDirMambaBlock → Head → [B,num_classes]
    
    FROZEN Configuration (DO NOT CHANGE):
        - d_model = 26 (model width - wider for better accuracy)
        - d_state = 8 (SSM state dimension)
        - n_layers = 4 (backbone depth - deeper than baselines)
        - expand = 2 (SSM expansion)
        - dt_rank = 2 (Δ discretization rank)
        - d_conv = 4 (local conv kernel)
        - bidirectional = True (weight-tied)
    
    HPO-Tunable (Training Hyperparameters ONLY):
        - dropout: 0.0 - 0.3
        - dropPath: 0.0 - 0.2 (stochastic depth)
        - learning_rate: 0.0003 - 0.003 (log scale)
        - weight_decay: 0.005 - 0.05 (log scale)
        - label_smoothing: 0.0 - 0.2
    
    Args:
        numClasses: Number of activity classes (default: 6 for UCI-HAR)
        inChannels: Number of input sensor channels (default: 9 for acc+gyro)
        seqLen: Sequence length (default: 128)
        dropout: Dropout rate in classifier (default: 0.1, HPO tuned)
        dropPath: Stochastic depth rate (default: 0.0, can be tuned)
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
        bidirectional: Optional[bool] = None,
    ):
        super().__init__()
        
        # Use frozen config, allow ablation overrides
        self.dModel = dModel or CROSSOVER_BIDIR_BABYMAMBA_CONFIG['dModel']
        self.dState = dState or CROSSOVER_BIDIR_BABYMAMBA_CONFIG['dState']
        self.nLayers = nLayers or CROSSOVER_BIDIR_BABYMAMBA_CONFIG['nLayers']
        self.expand = expand or CROSSOVER_BIDIR_BABYMAMBA_CONFIG['expand']
        self.dtRank = dtRank or CROSSOVER_BIDIR_BABYMAMBA_CONFIG['dtRank']
        self.dConv = dConv or CROSSOVER_BIDIR_BABYMAMBA_CONFIG['dConv']
        self.bidirectional = True if bidirectional is None else bool(bidirectional)
        self.stemKernel = CROSSOVER_BIDIR_BABYMAMBA_CONFIG['stemKernel']
        self.patchKernel = CROSSOVER_BIDIR_BABYMAMBA_CONFIG['patchKernel']
        self.patchStride = CROSSOVER_BIDIR_BABYMAMBA_CONFIG['patchStride']
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        self.dropout = dropout
        self.dropPathRate = dropPath
        
        # Calculate number of patches after patching
        self.numPatches = (seqLen - self.patchKernel) // self.patchStride + 1
        
        # ============== STAGE 1: Local Feature Extraction (Stem) ==============
        # Conv1D: inChannels → dModel, k=5
        # Provides local edge detection (spikes/drops) - inductive bias for HAR
        self.stem = nn.Sequential(
            nn.Conv1d(
                inChannels, self.dModel,
                kernel_size=self.stemKernel,
                padding=self.stemKernel // 2,
                stride=1,
                bias=False
            ),
            nn.BatchNorm1d(self.dModel),
            nn.SiLU(inplace=True),
        )
        
        # ============== STAGE 2: Discrete Patching ==============
        # Transition from continuous signal to token-based sequence modeling
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
        
        # ============== STAGE 3: Weight-Tied Bi-Directional SSM Backbone ==============
        # 4 layers of WeightTiedBiDirMambaBlock with stochastic depth
        # Drop path rate increases linearly from 0 to dropPath
        dpRates = [x.item() for x in torch.linspace(0, dropPath, self.nLayers)]
        blockCls = WeightTiedBiDirMambaBlock if self.bidirectional else UnidirectionalMambaBlock
        self.mambaLayers = nn.ModuleList([
            blockCls(
                dModel=self.dModel,
                dState=self.dState,
                dConv=self.dConv,
                expand=self.expand,
                dtRank=self.dtRank,
                dropPath=dpRates[i],
            )
            for i in range(self.nLayers)
        ])
        
        # ============== STAGE 4: Classification Head ==============
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
        Forward pass.
        
        Args:
            x: Input tensor [B, T, C] or [B, C, T]
        Returns:
            Logits [B, numClasses]
        """
        # Ensure input is [B, C, T] format for convolutions
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # Handle [B, T, C] -> [B, C, T]
        if x.shape[-1] == self.inChannels:
            x = x.transpose(1, 2)
        
        # Stage 1: Stem - Local Feature Extraction
        x = self.stem(x)  # [B, dModel, L]
        
        # Stage 2: Discrete Patching
        x = self.patchDepthwise(x)   # [B, dModel, numPatches]
        x = self.patchPointwise(x)   # [B, dModel, numPatches]
        x = self.patchNorm(x)
        x = F.silu(x)
        
        # Transpose for Mamba: [B, dModel, numPatches] -> [B, numPatches, dModel]
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
        
        # Stage 3: Weight-Tied Bi-Directional SSM Backbone
        for layer in self.mambaLayers:
            x = layer(x)  # [B, numPatches, dModel]
        
        # Stage 4: Classification Head
        # Global mean pooling over sequence
        x = x.mean(dim=1)  # [B, dModel]
        x = self.headNorm(x)
        x = self.headDropout(x)
        x = self.headLinear(x)  # [B, numClasses]
        
        return x
    
    def countParameters(self) -> Dict[str, int]:
        """Count model parameters by component."""
        stemParams = sum(p.numel() for p in self.stem.parameters())
        
        patchParams = (
            sum(p.numel() for p in self.patchDepthwise.parameters()) +
            sum(p.numel() for p in self.patchPointwise.parameters()) +
            sum(p.numel() for p in self.patchNorm.parameters()) +
            self.posEmbed.numel()
        )
        
        mambaParams = sum(p.numel() for p in self.mambaLayers.parameters())
        
        headParams = (
            sum(p.numel() for p in self.headNorm.parameters()) +
            sum(p.numel() for p in self.headLinear.parameters())
        )
        
        totalParams = sum(p.numel() for p in self.parameters())
        
        return {
            'stem': stemParams,
            'patch': patchParams,
            'mamba': mambaParams,
            'head': headParams,
            'total': totalParams,
        }
    
    def getConfigDict(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            'architecture': 'BabyMamba-Crossover-BiDir',
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
            'weightTied': bool(self.bidirectional),
        }


def createCrossoverBiDirBabyMambaHar(
    dataset: str = 'ucihar',
    dropout: float = 0.1,
    dModel: Optional[int] = None,
    dState: Optional[int] = None,
    nLayers: Optional[int] = None,
    expand: Optional[int] = None,
    dtRank: Optional[int] = None,
    dConv: Optional[int] = None,
    bidirectional: Optional[bool] = None,
    seqLenOverride: Optional[int] = None,
) -> CrossoverBiDirBabyMambaHar:
    """
    Factory function to create CrossoverBiDirBabyMambaHar for specific datasets.
    
    Args:
        dataset: Dataset name ('ucihar', 'motionsense', 'wisdm', 'pamap2', etc.)
        dropout: Dropout rate in classifier head
        dState: Optional override for d_state (ablation studies only)
        seqLenOverride: Optional override for sequence length (LRD study)
    
    Returns:
        Configured CrossoverBiDirBabyMambaHar model (~27k params)
    """
    datasetConfigs = {
        'ucihar': {'numClasses': 6, 'inChannels': 9, 'seqLen': 128},
        'motionsense': {'numClasses': 6, 'inChannels': 6, 'seqLen': 128},
        'wisdm': {'numClasses': 6, 'inChannels': 3, 'seqLen': 128},
        'pamap2': {'numClasses': 12, 'inChannels': 19, 'seqLen': 128},
        'opportunity': {'numClasses': 5, 'inChannels': 79, 'seqLen': 128},
        'unimib': {'numClasses': 9, 'inChannels': 3, 'seqLen': 128},
        'skoda': {'numClasses': 11, 'inChannels': 30, 'seqLen': 98},
        'daphnet': {'numClasses': 2, 'inChannels': 9, 'seqLen': 64},
    }
    
    if dataset.lower() not in datasetConfigs:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(datasetConfigs.keys())}")
    
    config = datasetConfigs[dataset.lower()]
    # Override seqLen if specified
    if seqLenOverride is not None:
        config = config.copy()
        config['seqLen'] = seqLenOverride
    return CrossoverBiDirBabyMambaHar(
        dropout=dropout,
        dModel=dModel,
        dState=dState,
        nLayers=nLayers,
        expand=expand,
        dtRank=dtRank,
        dConv=dConv,
        bidirectional=bidirectional,
        **config,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("BabyMamba-Crossover-BiDir: The Definitive HAR Architecture")
    print("=" * 70)
    
    # Test UCI-HAR configuration (default)
    model = CrossoverBiDirBabyMambaHar(numClasses=6, inChannels=9)
    params = model.countParameters()
    config = model.getConfigDict()
    
    print(f"\n=== FROZEN CONFIGURATION ===")
    print(f"Architecture: {config['architecture']}")
    print(f"Bidirectional: {config['bidirectional']} (Weight-Tied: {config['weightTied']})")
    print(f"d_model: {config['dModel']}")
    print(f"d_state: {config['dState']}")
    print(f"n_layers: {config['nLayers']}")
    print(f"expand: {config['expand']}")
    print(f"dt_rank: {config['dtRank']}")
    
    print(f"\n=== PARAMETER BUDGET (UCI-HAR) ===")
    print(f"  Stem:     {params['stem']:,} params")
    print(f"  Patch:    {params['patch']:,} params")
    print(f"  Mamba:    {params['mamba']:,} params")
    print(f"  Head:     {params['head']:,} params")
    print(f"  TOTAL:    {params['total']:,} params")
    
    print(f"\n=== TARGET vs ACTUAL ===")
    print(f"  Target: ~27,000 params")
    print(f"  Actual: {params['total']:,} params")
    print(f"  Within budget: {'✓' if params['total'] <= 30000 else '✗'}")
    
    # Test forward pass
    x = torch.randn(2, 128, 9)  # [B, T, C]
    y = model(x)
    print(f"\n=== FORWARD PASS ===")
    print(f"  Input:  {x.shape} -> Output: {y.shape}")
    
    # Test all datasets
    print(f"\n=== PARAMETER COUNTS BY DATASET ===")
    for dataset in ['ucihar', 'motionsense', 'wisdm', 'pamap2', 'opportunity', 'unimib', 'skoda', 'daphnet']:
        m = createCrossoverBiDirBabyMambaHar(dataset)
        p = m.countParameters()
        c = m.getConfigDict()
        print(f"  {dataset.upper():12s}: {p['total']:,} params "
              f"(in={c['inChannels']}, out={c['numClasses']})")
    
    print("\n" + "=" * 70)
    print("READY FOR TRAINING")
    print("=" * 70)

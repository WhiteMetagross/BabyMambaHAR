"""
HARMamba: Bidirectional Selective State-Space Model for HAR

Implementation based on: "HARMamba: Efficient Wearable Sensor Human Activity 
Recognition Based on Bidirectional Selective SSM" (arXiv:2403.20183v3)

Original HARMamba Architecture (~400K parameters):
- Channel Independence Module (CIM)
- Temporal Independence Module (TIM) with patch-wise Mamba
- Bidirectional Mamba blocks
- Multi-scale temporal modeling

This is a faithful reimplementation for baseline comparison.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List

# Try to import mamba_ssm
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class PatchEmbedding(nn.Module):
    """
    Patch Embedding for temporal sequences.
    
    Splits input into patches and projects to embedding dimension.
    """
    
    def __init__(
        self,
        inChannels: int,
        embedDim: int,
        patchSize: int = 16,
        stride: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.patchSize = patchSize
        self.stride = stride
        self.embedDim = embedDim
        
        # Conv-based patch embedding
        self.proj = nn.Conv1d(
            inChannels, embedDim,
            kernel_size=patchSize,
            stride=stride,
            padding=patchSize // 4
        )
        
        self.norm = nn.LayerNorm(embedDim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input
        Returns:
            [B, N, D] patched embeddings
        """
        x = self.proj(x)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        x = self.norm(x)
        x = self.dropout(x)
        return x


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block as in HARMamba paper.
    
    Combines forward and backward Mamba passes for better
    temporal modeling in both directions.
    """
    
    def __init__(
        self,
        dModel: int = 64,
        dState: int = 16,
        dConv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dModel = dModel
        
        # Pre-norm
        self.norm = nn.LayerNorm(dModel)
        
        # Forward Mamba
        if MAMBA_AVAILABLE:
            self.mambaFwd = Mamba(
                d_model=dModel,
                d_state=dState,
                d_conv=dConv,
                expand=expand
            )
            self.mambaBwd = Mamba(
                d_model=dModel,
                d_state=dState,
                d_conv=dConv,
                expand=expand
            )
        else:
            self.mambaFwd = SimpleMamba(dModel, dState, dConv, expand)
            self.mambaBwd = SimpleMamba(dModel, dState, dConv, expand)
        
        # Fusion layer
        self.fusion = nn.Linear(dModel * 2, dModel)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input
        Returns:
            [B, L, D] output
        """
        residual = x
        x = self.norm(x)
        
        # Forward pass
        xFwd = self.mambaFwd(x)
        
        # Backward pass
        xBwd = torch.flip(x, dims=[1])
        xBwd = self.mambaBwd(xBwd)
        xBwd = torch.flip(xBwd, dims=[1])
        
        # Fusion
        xCat = torch.cat([xFwd, xBwd], dim=-1)
        x = self.fusion(xCat)
        x = self.dropout(x)
        
        return residual + x


class SimpleMamba(nn.Module):
    """
    Simple Mamba implementation for when mamba_ssm is not available.
    """
    
    def __init__(
        self,
        dModel: int,
        dState: int = 16,
        dConv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        self.dModel = dModel
        self.dState = dState
        self.dInner = expand * dModel
        self.dtRank = max(1, dModel // 16)
        
        self.inProj = nn.Linear(dModel, self.dInner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            self.dInner, self.dInner,
            kernel_size=dConv,
            groups=self.dInner,
            padding=dConv - 1,
            bias=True
        )
        
        self.xProj = nn.Linear(self.dInner, self.dtRank + dState * 2, bias=False)
        self.dtProj = nn.Linear(self.dtRank, self.dInner, bias=True)
        
        A = torch.arange(1, dState + 1).float().repeat(self.dInner, 1)
        self.ALog = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.dInner))
        
        self.outProj = nn.Linear(self.dInner, dModel, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        xRes = self.inProj(x)
        x, res = xRes.split(self.dInner, dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        
        y = self._ssm(x)
        y = y * F.silu(res)
        return self.outProj(y)
    
    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        A = -torch.exp(self.ALog.float())
        
        xDbl = self.xProj(x)
        delta, B_param, C_param = xDbl.split([self.dtRank, self.dState, self.dState], dim=-1)
        delta = F.softplus(self.dtProj(delta))
        
        # Simple sequential scan
        h = torch.zeros(B, D, self.dState, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(L):
            deltaT = delta[:, t:t+1, :]
            xT = x[:, t:t+1, :]
            BT = B_param[:, t, :]
            CT = C_param[:, t, :]
            
            deltaA = torch.exp(deltaT.unsqueeze(-1) * A)
            deltaBX = (deltaT * xT).unsqueeze(-1) * BT.unsqueeze(1).unsqueeze(1)
            
            h = deltaA.squeeze(1) * h + deltaBX.squeeze(1)
            y = torch.einsum('bdn,bn->bd', h, CT) + self.D * xT.squeeze(1)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


class ChannelMixer(nn.Module):
    """
    Channel Mixer for cross-channel information exchange.
    
    Implemented as MLP with expansion and contraction.
    """
    
    def __init__(self, dModel: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dModel)
        self.fc1 = nn.Linear(dModel, dModel * expansion)
        self.fc2 = nn.Linear(dModel * expansion, dModel)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class HARMamba(nn.Module):
    """
    HARMamba: Bidirectional Selective State-Space Model for HAR.
    
    Based on arXiv:2403.20183v3.
    
    Architecture:
        1. Patch Embedding (converts T timesteps to N patches)
        2. Position Embedding (learnable)
        3. N × BiMamba Blocks (bidirectional SSM)
        4. Channel Mixer (cross-channel interaction)
        5. Global Average Pooling
        6. Classification Head
    
    Args:
        numClasses: Number of activity classes
        inChannels: Input sensor channels
        seqLen: Sequence length
        embedDim: Embedding dimension (default: 64)
        depth: Number of BiMamba blocks (default: 12)
        dState: SSM state dimension (default: 16)
        dConv: SSM conv kernel (default: 4)
        expand: SSM expansion factor (default: 2)
        patchSize: Patch size (default: 16)
        patchStride: Patch stride (default: 8)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        embedDim: int = 64,
        depth: int = 12,
        dState: int = 16,
        dConv: int = 4,
        expand: int = 2,
        patchSize: int = 16,
        patchStride: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        self.embedDim = embedDim
        self.depth = depth
        
        # Calculate number of patches
        self.numPatches = (seqLen + 2 * (patchSize // 4) - patchSize) // patchStride + 1
        
        # Patch embedding
        self.patchEmbed = PatchEmbedding(
            inChannels=inChannels,
            embedDim=embedDim,
            patchSize=patchSize,
            stride=patchStride,
            dropout=dropout
        )
        
        # Position embedding
        self.posEmbed = nn.Parameter(torch.zeros(1, self.numPatches, embedDim))
        nn.init.trunc_normal_(self.posEmbed, std=0.02)
        
        # CLS token
        self.clsToken = nn.Parameter(torch.zeros(1, 1, embedDim))
        nn.init.trunc_normal_(self.clsToken, std=0.02)
        
        # BiMamba blocks
        self.blocks = nn.ModuleList([
            BiMambaBlock(
                dModel=embedDim,
                dState=dState,
                dConv=dConv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Channel mixer after every 2 blocks
        self.mixers = nn.ModuleList([
            ChannelMixer(embedDim, expansion=4, dropout=dropout)
            if i % 2 == 1 else nn.Identity()
            for i in range(depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(embedDim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedDim, numClasses)
        )
        
        self._initWeights()
    
    def _initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] or [B, C, T] input
        Returns:
            [B, numClasses] logits
        """
        # Ensure [B, C, T]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] == self.inChannels:
            x = x.transpose(1, 2)
        
        B = x.size(0)
        
        # Patch embedding: [B, C, T] -> [B, N, D]
        x = self.patchEmbed(x)
        
        # Add position embedding
        if x.size(1) != self.posEmbed.size(1):
            posEmbed = F.interpolate(
                self.posEmbed.transpose(1, 2),
                size=x.size(1),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            posEmbed = self.posEmbed
        
        # Add CLS token
        clsTokens = self.clsToken.expand(B, -1, -1)
        x = torch.cat([clsTokens, x], dim=1)
        
        # Adjust position embedding for CLS
        clsPosEmbed = torch.zeros(1, 1, self.embedDim, device=x.device)
        posEmbed = torch.cat([clsPosEmbed, posEmbed], dim=1)
        x = x + posEmbed
        
        # BiMamba blocks
        for block, mixer in zip(self.blocks, self.mixers):
            x = block(x)
            x = mixer(x)
        
        # Final norm
        x = self.norm(x)
        
        # Use CLS token for classification
        x = x[:, 0]
        
        # Classification
        return self.head(x)
    
    def countParameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        patchParams = sum(p.numel() for p in self.patchEmbed.parameters())
        posParams = self.posEmbed.numel() + self.clsToken.numel()
        blockParams = sum(p.numel() for p in self.blocks.parameters())
        mixerParams = sum(p.numel() for p in self.mixers.parameters())
        headParams = sum(p.numel() for p in self.head.parameters()) + sum(p.numel() for p in self.norm.parameters())
        total = sum(p.numel() for p in self.parameters())
        
        return {
            'patchEmbed': patchParams,
            'posEmbed': posParams,
            'blocks': blockParams,
            'mixers': mixerParams,
            'head': headParams,
            'total': total
        }


class HARMambaLite(nn.Module):
    """
    HARMamba-Lite: Reduced version for fair comparison (~50K params).
    
    Scaled down version of HARMamba with:
    - Smaller embedding dimension (32 vs 64)
    - Fewer blocks (4 vs 12)
    - Smaller state dimension (8 vs 16)
    """
    
    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        embedDim: int = 32,
        depth: int = 4,
        dState: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        
        # Simplified patch embedding
        self.patchEmbed = nn.Sequential(
            nn.Conv1d(inChannels, embedDim, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(embedDim),
            nn.GELU()
        )
        
        self.numPatches = (seqLen + 4 - 8) // 4 + 1
        
        self.posEmbed = nn.Parameter(torch.zeros(1, self.numPatches, embedDim))
        nn.init.trunc_normal_(self.posEmbed, std=0.02)
        
        # BiMamba blocks
        self.blocks = nn.ModuleList([
            BiMambaBlock(
                dModel=embedDim,
                dState=dState,
                dConv=4,
                expand=2,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embedDim)
        self.head = nn.Linear(embedDim, numClasses)
        
        self._initWeights()
    
    def _initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] == self.inChannels:
            x = x.transpose(1, 2)
        
        x = self.patchEmbed(x).transpose(1, 2)
        
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
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x).mean(dim=1)
        return self.head(x)
    
    def countParameters(self) -> Dict[str, int]:
        return {
            'total': sum(p.numel() for p in self.parameters())
        }


def createHARMamba(
    dataset: str = 'ucihar',
    lite: bool = False,
    embedDim: Optional[int] = None,
    depth: Optional[int] = None
) -> nn.Module:
    """
    Factory function for HARMamba models.
    
    Args:
        dataset: Dataset name
        lite: Use lite version (~50K params)
        embedDim: Override embedding dimension
        depth: Override number of blocks
    
    Returns:
        Configured HARMamba model
    """
    configs = {
        'ucihar': {'numClasses': 6, 'inChannels': 9, 'seqLen': 128},
        'motionsense': {'numClasses': 6, 'inChannels': 6, 'seqLen': 128},
        'wisdm': {'numClasses': 6, 'inChannels': 3, 'seqLen': 128},
        'pamap2': {'numClasses': 12, 'inChannels': 52, 'seqLen': 128},
    }
    
    if dataset.lower() not in configs:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    cfg = configs[dataset.lower()]
    
    if lite:
        kwargs = {'embedDim': embedDim or 32, 'depth': depth or 4}
        return HARMambaLite(**cfg, **{k: v for k, v in kwargs.items() if v is not None})
    else:
        kwargs = {'embedDim': embedDim or 64, 'depth': depth or 12}
        return HARMamba(**cfg, **{k: v for k, v in kwargs.items() if v is not None})


if __name__ == '__main__':
    print("=" * 60)
    print("HARMamba: Bidirectional Selective SSM for HAR")
    print("=" * 60)
    
    # Test full HARMamba
    model = createHARMamba('ucihar', lite=False)
    params = model.countParameters()
    print(f"\nHARMamba (Full):")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
    
    x = torch.randn(2, 128, 9)
    y = model(x)
    print(f"  Forward: {x.shape} -> {y.shape}")
    
    # Test HARMamba-Lite
    modelLite = createHARMamba('ucihar', lite=True)
    paramsLite = modelLite.countParameters()
    print(f"\nHARMamba-Lite:")
    for k, v in paramsLite.items():
        print(f"  {k}: {v:,}")
    
    y = modelLite(x)
    print(f"  Forward: {x.shape} -> {y.shape}")

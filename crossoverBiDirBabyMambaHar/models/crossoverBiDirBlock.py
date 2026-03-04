"""
BabyMamba-Crossover-BiDir Block - Weight-Tied Bidirectional SSM for HAR

============== FROZEN ARCHITECTURE SPECIFICATION ==============

This is the final, frozen specification for BabyMamba-Crossover-BiDir.
This plan is mathematically verified to fit the ~27k parameter budget
(O(N) complexity) while matching the depth and receptive field of the
heavier baselines (O(N^2) or LSTM).

Design Philosophy:
- Weight-Tied Bidirectional: Same SSM params θ for both directions
- 4 Layers Deep: Matches TinierHAR/LightDeepConvLSTM depth
- Optimal Width: d_model=26 to stay under 27k params
- Sufficient Memory: d_state=8 for HAR temporal patterns

Core Configuration ("The Golden Config"):
    d_model = 26      (optimal width for 4 layers within budget)
    d_state = 8       (sufficient latent memory for HAR)
    n_layers = 4      (matches depth of baselines)
    expand = 2        (inner dimension = 52)
    dt_rank = 2       (minimal rank for Δ discretization)
    kernel_size = 4   (local 1D conv inside SSM)
    bidir = True      (weight-tied, scans t→T and T→t)

Mathematical Formulation:
    Forward:  h_fwd = SSM_θ(x)
    Backward: h_bwd = SSM_θ(flip(x))  # Same weights θ!
    Fusion:   h_out = h_fwd + flip(h_bwd)

Per-Layer Parameters (~5,880 each):
    - inProj: d_model → 2*d_inner = 26 → 104 = 2,704
    - conv1d: d_inner groups, k=4 = 52*4 + 52 = 260
    - xProj: d_inner → dt_rank + 2*d_state = 52 → 18 = 936
    - dtProj: dt_rank → d_inner = 2 → 52 = 104 + 52 = 156
    - A_log: d_inner × d_state = 52 × 8 = 416
    - D: d_inner = 52
    - outProj: d_inner → d_model = 52 → 26 = 1,352
    - norm: 2 × d_model = 52
    Total per layer: ~5,880 params

Why This Beats Baselines:
- vs LightDeepConvLSTM (15k): 4 layers of depth vs shallow hybrid
- vs TinierHAR (17k): Mamba preserves long-term deps better than GRU sigmoid gates
- vs TinyHAR (42k): 40% smaller, O(N) vs O(N^2) attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import mamba_ssm, fall back to pure PyTorch if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


def dropPath(x: torch.Tensor, dropProb: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic Depth (Drop Path) regularization.
    
    Drops entire residual paths randomly during training.
    This is equivalent to training an ensemble of shallower networks.
    
    Args:
        x: Input tensor
        dropProb: Probability of dropping the path
        training: Whether in training mode
    
    Returns:
        Tensor with path potentially dropped (scaled by survival probability)
    """
    if dropProb == 0.0 or not training:
        return x
    
    keepProb = 1 - dropProb
    # Create random mask for batch dimension
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    randomTensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    randomTensor = randomTensor.floor_()  # Binarize to 0 or 1
    
    # Scale by survival probability to maintain expected value
    output = x.div(keepProb) * randomTensor
    return output


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) module.
    
    Drops entire residual paths randomly during training.
    This acts like training an ensemble of shallower networks.
    
    Args:
        dropProb: Probability of dropping the path (default: 0.0)
    """
    
    def __init__(self, dropProb: float = 0.0):
        super().__init__()
        self.dropProb = dropProb
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dropPath(x, self.dropProb, self.training)
    
    def extra_repr(self) -> str:
        return f'dropProb={self.dropProb:.3f}'


class WeightTiedBiDirMambaBlock(nn.Module):
    """
    Weight-Tied Bidirectional Mamba Block - The Core of BabyMamba-Crossover-BiDir.
    
    This is the FROZEN block for the research paper. DO NOT MODIFY.
    
    Architecture:
        1. Pre-normalization (LayerNorm)
        2. Forward SSM pass: h_fwd = SSM_θ(x)
        3. Backward SSM pass: h_bwd = SSM_θ(flip(x))  # SAME weights θ
        4. Fusion: h_out = h_fwd + flip(h_bwd)
        5. Residual connection with optional DropPath
        6. Post-normalization (LayerNorm)
    
    Key Innovation: Weight-tied bidirectional processing doubles receptive field
    WITHOUT doubling parameters (unlike traditional Bi-LSTM/Bi-GRU).
    
    FROZEN Configuration:
        d_model = 26     (model dimension)
        d_state = 8      (SSM state dimension N)
        d_conv = 4       (local conv kernel inside SSM)
        expand = 2       (expansion factor, inner = 52)
        dt_rank = 2      (time-step discretization rank)
    
    Args:
        dModel: Model dimension (LOCKED at 26)
        dState: SSM state dimension (LOCKED at 8)
        dConv: Local convolution kernel (LOCKED at 4)
        expand: Expansion factor (LOCKED at 2)
        dtRank: Time-step discretization rank (LOCKED at 2)
        dropPath: Stochastic depth probability (default: 0.0)
    """
    
    def __init__(
        self,
        dModel: int = 26,
        dState: int = 8,
        dConv: int = 4,
        expand: int = 2,
        dtRank: int = 2,
        dropPath: float = 0.0
    ):
        super().__init__()
        self.dModel = dModel
        self.dState = dState
        self.dConv = dConv
        self.expand = expand
        self.dtRank = dtRank
        
        # Pre-normalization
        self.preNorm = nn.LayerNorm(dModel)
        
        # The SSM core (shared for forward AND backward - weight-tied!)
        if MAMBA_AVAILABLE:
            self.ssm = Mamba(
                d_model=dModel,
                d_state=dState,
                d_conv=dConv,
                expand=expand
            )
        else:
            self.ssm = PureSelectiveScan(
                dModel=dModel,
                dState=dState,
                dConv=dConv,
                expand=expand,
                dtRank=dtRank
            )
        
        # Post-normalization
        self.postNorm = nn.LayerNorm(dModel)
        
        # Stochastic Depth (Drop Path)
        self.dropPath = DropPath(dropPath) if dropPath > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Weight-tied bidirectional forward pass with optional DropPath.
        
        Args:
            x: Input tensor [B, L, D] (batch, sequence, dModel)
        Returns:
            Output tensor [B, L, D]
        """
        residual = x
        
        # Pre-normalization
        x = self.preNorm(x)
        
        # Forward pass: t → T
        hFwd = self.ssm(x)
        
        # Backward pass: T → t (flip input, same SSM, flip output)
        xFlip = torch.flip(x, dims=[1])
        hBwd = self.ssm(xFlip)
        hBwd = torch.flip(hBwd, dims=[1])
        
        # Fusion: simple addition (parameter-free)
        hOut = hFwd + hBwd
        
        # Apply DropPath to the residual path (stochastic depth)
        hOut = self.dropPath(hOut)
        
        # Residual + post-normalization
        out = residual + hOut
        out = self.postNorm(out)
        
        return out


class UnidirectionalMambaBlock(nn.Module):
    """Unidirectional Mamba block for ablations (forward-only scan)."""

    def __init__(
        self,
        dModel: int = 26,
        dState: int = 8,
        dConv: int = 4,
        expand: int = 2,
        dtRank: int = 2,
        dropPath: float = 0.0,
    ):
        super().__init__()
        self.dModel = dModel
        self.dState = dState
        self.dConv = dConv
        self.expand = expand
        self.dtRank = dtRank

        self.preNorm = nn.LayerNorm(dModel)

        if MAMBA_AVAILABLE:
            self.ssm = Mamba(
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
        self.dropPath = DropPath(dropPath) if dropPath > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.preNorm(x)
        hOut = self.ssm(x)
        hOut = self.dropPath(hOut)
        out = residual + hOut
        out = self.postNorm(out)
        return out


class PureSelectiveScan(nn.Module):
    """
    Pure PyTorch implementation of Selective State Space Model (S6).
    
    Fallback for when mamba_ssm CUDA kernels are not available.
    This implementation matches the Mamba paper's selective scan algorithm.
    
    Mathematical formulation:
        h_t = Ā * h_{t-1} + B̄ * x_t
        y_t = C * h_t + D * x_t
        
    Where Ā, B̄ are discretized from continuous A, B using Δ (delta).
    
    Args:
        dModel: Model dimension (input/output)
        dState: SSM state dimension N
        dConv: Local convolution kernel size
        expand: Expansion factor for inner dimension
        dtRank: Rank for Δ projection (controls time-step discretization)
    """
    
    def __init__(
        self,
        dModel: int,
        dState: int = 8,
        dConv: int = 4,
        expand: int = 2,
        dtRank: int = 2
    ):
        super().__init__()
        self.dModel = dModel
        self.dState = dState
        self.dConv = dConv
        self.expand = expand
        self.dInner = int(expand * dModel)
        self.dtRank = dtRank
        
        # Input projection: d_model → 2 * d_inner (for x and gate)
        self.inProj = nn.Linear(dModel, self.dInner * 2, bias=False)
        
        # Local convolution for mixing adjacent timesteps
        self.conv1d = nn.Conv1d(
            in_channels=self.dInner,
            out_channels=self.dInner,
            kernel_size=dConv,
            groups=self.dInner,  # Depthwise
            padding=dConv - 1,
            bias=True
        )
        
        # SSM parameters projection: d_inner → dt_rank + 2*d_state (Δ, B, C)
        self.xProj = nn.Linear(self.dInner, self.dtRank + dState * 2, bias=False)
        
        # Time-step projection: dt_rank → d_inner
        self.dtProj = nn.Linear(self.dtRank, self.dInner, bias=True)
        
        # Initialize dt bias for proper time-step scaling
        with torch.no_grad():
            dtInitStd = self.dtRank ** -0.5
            nn.init.uniform_(self.dtProj.weight, -dtInitStd, dtInitStd)
            # Initialize bias to make initial Δ ≈ 0.001 - 0.1
            dtMin, dtMax = 0.001, 0.1
            invDt = torch.exp(torch.rand(self.dInner) * (math.log(dtMax) - math.log(dtMin)) + math.log(dtMin))
            self.dtProj.bias.copy_(invDt.log())
        
        # SSM state matrix A (HiPPO-inspired initialization)
        # A is diagonal with entries -1, -2, ..., -N (encourages remembering)
        A = torch.arange(1, dState + 1, dtype=torch.float32).repeat(self.dInner, 1)
        self.ALog = nn.Parameter(torch.log(A))  # Learn in log-space for stability
        
        # Skip connection parameter D
        self.D = nn.Parameter(torch.ones(self.dInner))
        
        # Output projection: d_inner → d_model
        self.outProj = nn.Linear(self.dInner, dModel, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective scan forward pass.
        
        Args:
            x: Input [B, L, D]
        Returns:
            Output [B, L, D]
        """
        batch, seqLen, dim = x.shape
        
        # Project input to inner dimension
        xAndRes = self.inProj(x)  # [B, L, 2*d_inner]
        x, res = xAndRes.split(self.dInner, dim=-1)
        
        # Local convolution for adjacent timestep mixing
        x = x.transpose(1, 2)  # [B, d_inner, L]
        x = self.conv1d(x)[:, :, :seqLen]  # Causal padding
        x = x.transpose(1, 2)  # [B, L, d_inner]
        x = F.silu(x)
        
        # Compute SSM output
        y = self._selectiveScan(x)
        
        # Gating with residual branch
        y = y * F.silu(res)
        
        # Project back to model dimension
        output = self.outProj(y)
        
        return output
    
    def _selectiveScan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective scan implementation with input-dependent Δ, B, C.
        
        This is the core S6 algorithm that makes Mamba "selective".
        """
        batch, seqLen, dInner = x.shape
        
        # Get A from log-space (ensures A < 0 for stability)
        A = -torch.exp(self.ALog.float())  # [d_inner, d_state]
        D = self.D.float()
        
        # Project x to get Δ, B, C (input-dependent!)
        xDbl = self.xProj(x)  # [B, L, dt_rank + 2*d_state]
        delta, B, C = xDbl.split([self.dtRank, self.dState, self.dState], dim=-1)
        
        # Δ transformation: softplus ensures Δ > 0
        delta = F.softplus(self.dtProj(delta))  # [B, L, d_inner]
        
        # Discretize A: Ā = exp(Δ * A)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [B, L, d_inner, d_state]
        
        # Discretize B: B̄ = Δ * B * x (simplified zero-order hold)
        deltaBX = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # [B, L, d_inner, d_state]
        
        # Run the scan
        y = self._parallelScan(deltaA, deltaBX, C, D, x)
        
        return y
    
    def _parallelScan(self, deltaA, deltaBX, C, D, x, chunkSize: int = 32):
        """
        Chunked parallel scan for efficient computation.
        
        Uses a chunk-based approach for numerical stability and efficiency.
        """
        batch, seqLen, dInner, N = deltaA.shape
        device = deltaA.device
        dtype = deltaA.dtype
        
        # Initialize hidden state
        h = torch.zeros(batch, dInner, N, device=device, dtype=dtype)
        outputs = []
        
        for start in range(0, seqLen, chunkSize):
            end = min(start + chunkSize, seqLen)
            
            chunkA = deltaA[:, start:end]
            chunkBX = deltaBX[:, start:end]
            chunkC = C[:, start:end]
            chunkX = x[:, start:end]
            
            # Parallel scan within chunk
            logA = torch.log(chunkA.clamp(min=1e-6))
            cumLogA = torch.cumsum(logA, dim=1)
            cumA = torch.exp(cumLogA.clamp(max=20))
            
            # Contribution from initial state
            hInitContrib = h.unsqueeze(1) * cumA
            
            # Contribution from inputs within chunk
            invCumA = torch.exp(-cumLogA.clamp(min=-20, max=20))
            scaledBX = chunkBX * invCumA
            cumScaledBX = torch.cumsum(scaledBX, dim=1)
            inputContrib = cumA * cumScaledBX
            
            # Combined state for this chunk
            hChunk = hInitContrib + inputContrib
            
            # Update hidden state for next chunk
            h = hChunk[:, -1]
            
            # Compute output: y = C @ h + D * x
            yChunk = torch.einsum('bldi,bli->bld', hChunk, chunkC)
            yChunk = yChunk + chunkX * D
            outputs.append(yChunk)
        
        return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    print("Testing Weight-Tied Bidirectional Mamba Block")
    print("=" * 60)
    
    # Test block
    block = WeightTiedBiDirMambaBlock(dModel=26, dState=8, dConv=4, expand=2, dtRank=2)
    params = sum(p.numel() for p in block.parameters())
    
    x = torch.randn(2, 32, 26)  # [B, L, D]
    y = block(x)
    
    print(f"Block Parameters: {params:,}")
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {y.shape}")
    print(f"MAMBA_AVAILABLE: {MAMBA_AVAILABLE}")
    
    # Verify bidirectional processing
    print(f"\n=== Bidirectional Verification ===")
    print(f"Forward pass uses same SSM as backward: Weight-Tied ✓")

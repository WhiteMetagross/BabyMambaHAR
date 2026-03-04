"""
Stem Modules for CI-BabyMamba-HAR - Simple Conv Stem (TinierHAR Style)

Design Philosophy:
- Steal what works: TinierHAR's kernel=5 provides wide receptive field
- 2-layer Conv: 9 -> 20 -> 28 channels (LOCKED)
- No FFT, no fancy tricks - just convolution

The SimpleStem is the PRIMARY stem for CI-BabyMamba-HAR:
- Layer 1: Conv1d(9->20, k=5) + BatchNorm + SiLU
- Layer 2: Conv1d(20->28, k=5) + BatchNorm + SiLU  
- Total: ~3,400 params
"""

import torch
import torch.nn as nn
import torch.fft


class SimpleStem(nn.Module):
    """
    Simple Convolutional Stem - Stolen from TinierHAR.
    
    The winning formula: kernel=5 provides wide receptive field for HAR.
    No FFT, no depthwise-separable, no tricks - just proven convolutions.
    
    Architecture:
        Layer 1: Conv1d(in→16, k=5) + BN + SiLU  (~736 params)
        Layer 2: Conv1d(16→out, k=5) + BN + SiLU (~1,944 params)
        Total: ~2,700 params
    
    Args:
        inChannels: Number of input sensor channels (e.g., 9 for acc+gyro)
        outChannels: Output dimension (d_model=24 for CI-BabyMamba-HAR)
        kernelSize: Convolution kernel size (default: 5 - from TinierHAR)
    """
    
    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int = 5
    ):
        super().__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        
        # Intermediate channels - LOCKED at 20 for CI-BabyMamba-HAR
        midChannels = 20
        
        self.stem = nn.Sequential(
            # Layer 1: in → 16
            nn.Conv1d(
                inChannels, midChannels,
                kernel_size=kernelSize,
                padding=kernelSize // 2,
                stride=1,
                bias=False
            ),
            nn.BatchNorm1d(midChannels),
            nn.SiLU(inplace=True),
            
            # Layer 2: 16 → out (24)
            nn.Conv1d(
                midChannels, outChannels,
                kernel_size=kernelSize,
                padding=kernelSize // 2,
                stride=1,
                bias=False
            ),
            nn.BatchNorm1d(outChannels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, T] (batch, channels, time)
        Returns:
            Output [B, outChannels, T]
        """
        return self.stem(x)


# ============== LEGACY STEMS (kept for ablation studies) ==============

class WideEyeStem(nn.Module):
    """Legacy: Wide-Eye Stem with depthwise-separable conv. Kept for ablation."""
    
    def __init__(self, inChannels: int, outChannels: int, kernelSize: int = 5):
        super().__init__()
        midChannels = max(outChannels // 2, inChannels * 2)
        
        self.stem = nn.Sequential(
            nn.Conv1d(inChannels, inChannels, kernel_size=kernelSize,
                      padding=kernelSize // 2, groups=inChannels, bias=False),
            nn.BatchNorm1d(inChannels),
            nn.SiLU(inplace=True),
            nn.Conv1d(inChannels, midChannels, kernel_size=1, bias=False),
            nn.BatchNorm1d(midChannels),
            nn.SiLU(inplace=True),
            nn.Conv1d(midChannels, midChannels, kernel_size=kernelSize,
                      padding=kernelSize // 2, groups=1, bias=False),
            nn.BatchNorm1d(midChannels),
            nn.SiLU(inplace=True),
            nn.Conv1d(midChannels, outChannels, kernel_size=1, bias=False),
            nn.BatchNorm1d(outChannels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class SpectralTemporalStem(nn.Module):
    """Legacy: FFT + Conv stem. Kept for ablation."""
    
    def __init__(self, inChannels: int, outChannels: int, kernelSize: int = 3):
        super().__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        
        self.timeBranch = nn.Sequential(
            nn.Conv1d(inChannels, inChannels, kernel_size=kernelSize,
                      padding=kernelSize // 2, groups=inChannels, bias=False),
            nn.BatchNorm1d(inChannels),
            nn.SiLU(inplace=True),
            nn.Conv1d(inChannels, outChannels, kernel_size=1, bias=False),
            nn.BatchNorm1d(outChannels),
            nn.SiLU(inplace=True),
            nn.Conv1d(outChannels, outChannels, kernel_size=kernelSize,
                      padding=kernelSize // 2, bias=False),
            nn.BatchNorm1d(outChannels),
            nn.SiLU(inplace=True)
        )
        
        self.freqProjection = nn.Sequential(
            nn.Linear(inChannels, outChannels, bias=False),
            nn.ReLU(inplace=True)
        )
        self.freqScale = nn.Parameter(torch.ones(1, outChannels, 1) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        xTime = self.timeBranch(x)
        xFft = torch.fft.rfft(x, dim=-1)
        xMag = torch.abs(xFft)
        xSpectralEnergy = xMag.mean(dim=-1)
        xFreq = self.freqProjection(xSpectralEnergy)
        xFreq = xFreq.unsqueeze(-1) * self.freqScale
        xFreq = xFreq.expand(-1, -1, T)
        return xTime + xFreq


class TimeOnlyStem(nn.Module):
    """
    Time-Only Stem (Ablation Variant): No FFT Branch, Kernel Size 3.
    
    Used to demonstrate the importance of wide kernels (WideEyeStem uses k=5).
    Kept for ablation studies.
    """
    
    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int = 3
    ):
        super().__init__()
        
        midChannels = inChannels * 2
        
        self.stem = nn.Sequential(
            nn.Conv1d(
                inChannels, inChannels,
                kernel_size=kernelSize,
                padding=kernelSize // 2,
                groups=inChannels,
                bias=False
            ),
            nn.BatchNorm1d(inChannels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inChannels, midChannels, kernel_size=1, bias=False),
            nn.BatchNorm1d(midChannels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                midChannels, midChannels,
                kernel_size=kernelSize,
                padding=kernelSize // 2,
                groups=midChannels,
                bias=False
            ),
            nn.BatchNorm1d(midChannels),
            nn.ReLU(inplace=True),
            nn.Conv1d(midChannels, outChannels, kernel_size=1, bias=False),
            nn.BatchNorm1d(outChannels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class HollowStem(nn.Module):
    """
    Hollow Stem: Ultra-Lightweight Input Processing.
    
    Uses depthwise-separable convolutions for rapid channel expansion
    with minimal parameters. Key innovation for hitting <15k params.
    
    Architecture:
    1. Depthwise Conv: Spatial mixing per channel
    2. Pointwise Conv: Channel expansion
    3. BatchNorm + ReLU
    
    Args:
        inChannels: Number of input channels (e.g., 9 for acc+gyro)
        outChannels: Output dimension (d_model)
        kernelSize: Convolution kernel size (default: 3)
    """
    
    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int = 3
    ):
        super().__init__()
        
        # Intermediate expansion
        midChannels = inChannels * 2
        
        # Calculate valid groups for the grouped conv
        # Groups must divide both midChannels (input) and be reasonable
        groups = 1
        for g in [4, 2, 1]:
            if midChannels % g == 0:
                groups = g
                break
        
        self.stem = nn.Sequential(
            # Stage 1: Depthwise conv (spatial mixing)
            nn.Conv1d(
                inChannels, inChannels,
                kernel_size=kernelSize,
                padding=kernelSize // 2,
                groups=inChannels,  # Depthwise
                bias=False
            ),
            nn.BatchNorm1d(inChannels),
            nn.ReLU(inplace=True),
            
            # Stage 2: Pointwise expansion to mid
            nn.Conv1d(inChannels, midChannels, kernel_size=1, bias=False),
            nn.BatchNorm1d(midChannels),
            nn.ReLU(inplace=True),
            
            # Stage 3: Final projection to d_model
            nn.Conv1d(
                midChannels, outChannels,
                kernel_size=kernelSize,
                padding=kernelSize // 2,
                groups=groups,  # Grouped conv with valid divisor
                bias=False
            ),
            nn.BatchNorm1d(outChannels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, T]
        Returns:
            Output [B, outChannels, T]
        """
        return self.stem(x)


class SensorStem(nn.Module):
    """
    Standard Sensor Stem.
    
    More expressive but uses more parameters than HollowStem.
    
    Args:
        inChannels: Number of input channels
        outChannels: Output dimension (d_model)
        kernelSize: Convolution kernel size
    """
    
    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int = 3
    ):
        super().__init__()
        
        midChannels = 16
        
        self.stem = nn.Sequential(
            # First conv layer
            nn.Conv1d(
                inChannels, midChannels,
                kernel_size=kernelSize,
                padding=kernelSize // 2,
                bias=True
            ),
            nn.ReLU(inplace=True),
            
            # Second conv layer
            nn.Conv1d(
                midChannels, outChannels,
                kernel_size=kernelSize,
                padding=kernelSize // 2,
                groups=4,  # Grouped for efficiency
                bias=True
            ),
            nn.BatchNorm1d(outChannels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable 1D Convolution.
    
    Splits standard convolution into:
    1. Depthwise: Per-channel spatial filtering
    2. Pointwise: Cross-channel mixing
    
    Reduces parameters by factor of ~kernel_size.
    """
    
    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        
        self.depthwise = nn.Conv1d(
            inChannels, inChannels,
            kernel_size=kernelSize,
            stride=stride,
            padding=padding,
            groups=inChannels,
            bias=False
        )
        self.pointwise = nn.Conv1d(
            inChannels, outChannels,
            kernel_size=1,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

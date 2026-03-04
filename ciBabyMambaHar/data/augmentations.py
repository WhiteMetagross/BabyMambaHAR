"""
Data Augmentations for HAR Sensor Data

CI-BabyMamba-HAR Augmentation Strategy:
- Time Warping (p=0.5): Simulates speed variations
- Magnitude Scaling (p=0.5): Simulates sensor sensitivity
- Gaussian Jitter (p=0.3): Simulates sensor noise
- Channel Dropout (p=0.2): Simulates faulty sensors

This is NON-NEGOTIABLE. TinierHAR's low variance (±0.59%) proves heavy augmentation works.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Optional


class HARaugment:
    """
    HAR Augmentation Pipeline - ALWAYS ENABLED for training.
    
    This is the exact augmentation strategy required for CI-BabyMamba-HAR.
    Do not modify these probabilities without rigorous ablation.
    
    Augmentations:
        1. Time Warping (p=0.5): Stretch/compress segments
        2. Magnitude Scaling (p=0.5): Scale by 0.9-1.1
        3. Gaussian Jitter (p=0.3): Add noise with std=0.05
        4. Channel Dropout (p=0.2): Drop 1-2 channels
    """
    
    def __init__(self):
        pass
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to sensor data.
        
        Args:
            x: Input tensor [C, T] (channels, timesteps)
        
        Returns:
            Augmented tensor [C, T]
        """
        # 1. Time Warping (p=0.5)
        if random.random() < 0.5:
            warp_factor = random.uniform(0.9, 1.1)
            x = self._time_warp(x, warp_factor)
        
        # 2. Magnitude Scaling (p=0.5)
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            x = x * scale
        
        # 3. Gaussian Jitter (p=0.3)
        if random.random() < 0.3:
            noise = torch.randn_like(x) * 0.05
            x = x + noise
        
        # 4. Channel Dropout (p=0.2)
        if random.random() < 0.2:
            num_channels = x.shape[0]
            num_drop = random.randint(1, min(2, num_channels - 1))
            drop_channels = random.sample(range(num_channels), num_drop)
            x[drop_channels] = 0
        
        return x
    
    def _time_warp(self, x: torch.Tensor, factor: float) -> torch.Tensor:
        """
        Apply time warping by interpolation.
        
        Args:
            x: Input tensor [C, T]
            factor: Warp factor (>1 = stretch, <1 = compress)
        
        Returns:
            Warped tensor [C, T]
        """
        C, T = x.shape
        
        # Calculate new length
        new_T = int(T * factor)
        new_T = max(1, new_T)
        
        # Interpolate to new length
        x = x.unsqueeze(0)  # [1, C, T]
        x = F.interpolate(x, size=new_T, mode='linear', align_corners=True)
        x = x.squeeze(0)  # [C, new_T]
        
        # Resize back to original length
        if new_T != T:
            x = x.unsqueeze(0)  # [1, C, new_T]
            x = F.interpolate(x, size=T, mode='linear', align_corners=True)
            x = x.squeeze(0)  # [C, T]
        
        return x


class Compose:
    """
    Compose multiple transforms together.
    
    Args:
        transforms: List of transforms to apply
    """
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


class RandomScaling:
    """
    Random scaling augmentation.
    
    Multiplies signal by random scale factor.
    Simulates variations in sensor sensitivity.
    
    Args:
        scaleRange: Range of scale factors (min, max)
        prob: Probability of applying
    """
    
    def __init__(
        self,
        scaleRange: tuple = (0.9, 1.1),
        prob: float = 0.5
    ):
        self.scaleMin = scaleRange[0]
        self.scaleMax = scaleRange[1]
        self.prob = prob
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.prob:
            scale = self.scaleMin + (self.scaleMax - self.scaleMin) * torch.rand(1)
            x = x * scale
        return x


class RandomNoise:
    """
    Add random Gaussian noise.
    
    Simulates sensor noise and minor measurement errors.
    
    Args:
        noiseLevel: Standard deviation of noise
        prob: Probability of applying
    """
    
    def __init__(
        self,
        noiseLevel: float = 0.02,
        prob: float = 0.5
    ):
        self.noiseLevel = noiseLevel
        self.prob = prob
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.prob:
            noise = self.noiseLevel * torch.randn_like(x)
            x = x + noise
        return x


class RandomRotation:
    """
    Random rotation in 3D space.
    
    Applies random rotation to 3-axis sensor data.
    Simulates different device orientations.
    
    Args:
        maxAngle: Maximum rotation angle in degrees
        prob: Probability of applying
    """
    
    def __init__(
        self,
        maxAngle: float = 15.0,
        prob: float = 0.5
    ):
        self.maxAngle = maxAngle * np.pi / 180  # Convert to radians
        self.prob = prob
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.prob:
            # x shape: [T, C] where C is multiple of 3 (xyz triplets)
            T, C = x.shape
            numTriplets = C // 3
            
            for i in range(numTriplets):
                startIdx = i * 3
                endIdx = startIdx + 3
                
                # Random rotation angles
                angles = (2 * torch.rand(3) - 1) * self.maxAngle
                
                # Rotation matrix (simplified Euler angles)
                Rx = self._rotationMatrix(angles[0], 'x')
                Ry = self._rotationMatrix(angles[1], 'y')
                Rz = self._rotationMatrix(angles[2], 'z')
                R = Rz @ Ry @ Rx
                
                # Apply rotation
                triplet = x[:, startIdx:endIdx]  # [T, 3]
                rotated = (R @ triplet.T).T  # [T, 3]
                x[:, startIdx:endIdx] = rotated
        
        return x
    
    def _rotationMatrix(self, angle: float, axis: str) -> torch.Tensor:
        """Create rotation matrix for given axis and angle."""
        c = torch.cos(angle.clone().detach() if isinstance(angle, torch.Tensor) else torch.tensor(angle))
        s = torch.sin(angle.clone().detach() if isinstance(angle, torch.Tensor) else torch.tensor(angle))
        
        if axis == 'x':
            return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
        elif axis == 'y':
            return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
        else:  # z
            return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


class TimeWarping:
    """
    Time warping augmentation.
    
    Stretches or compresses segments of the time series.
    Simulates variations in movement speed.
    
    Args:
        sigma: Standard deviation of warping
        numKnots: Number of warping control points
        prob: Probability of applying
    """
    
    def __init__(
        self,
        sigma: float = 0.2,
        numKnots: int = 4,
        prob: float = 0.5
    ):
        self.sigma = sigma
        self.numKnots = numKnots
        self.prob = prob
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.prob:
            T, C = x.shape
            
            # Generate warping path
            origSteps = torch.linspace(0, 1, T)
            warpSteps = self._generateWarpPath(T)
            
            # Interpolate
            xWarped = torch.zeros_like(x)
            for c in range(C):
                xWarped[:, c] = self._interp1d(warpSteps, origSteps, x[:, c])
            
            return xWarped
        
        return x
    
    def _generateWarpPath(self, length: int) -> torch.Tensor:
        """Generate a smooth warping path."""
        # Generate random cumulative warp factors
        warpFactors = 1 + self.sigma * torch.randn(self.numKnots)
        warpFactors = torch.clamp(warpFactors, 0.5, 2.0)
        
        # Interpolate to full length
        knotPos = torch.linspace(0, length - 1, self.numKnots)
        fullPos = torch.arange(length, dtype=torch.float32)
        
        # Linear interpolation of warp factors
        interpFactors = torch.ones(length)
        for i in range(self.numKnots - 1):
            startIdx = int(knotPos[i])
            endIdx = int(knotPos[i + 1]) + 1
            for j in range(startIdx, min(endIdx, length)):
                t = (j - knotPos[i]) / (knotPos[i + 1] - knotPos[i])
                interpFactors[j] = warpFactors[i] * (1 - t) + warpFactors[i + 1] * t
        
        # Integrate to get warped positions
        warpPath = torch.cumsum(interpFactors, dim=0)
        warpPath = (warpPath - warpPath[0]) / (warpPath[-1] - warpPath[0])
        
        return warpPath
    
    def _interp1d(
        self,
        xNew: torch.Tensor,
        xOld: torch.Tensor,
        yOld: torch.Tensor
    ) -> torch.Tensor:
        """Simple linear interpolation."""
        yNew = torch.zeros_like(xNew)
        
        for i, x in enumerate(xNew):
            # Find bracketing indices
            idx = torch.searchsorted(xOld, x)
            idx = torch.clamp(idx, 1, len(xOld) - 1)
            
            # Linear interpolation
            x0, x1 = xOld[idx - 1], xOld[idx]
            y0, y1 = yOld[idx - 1], yOld[idx]
            
            t = (x - x0) / (x1 - x0 + 1e-8)
            yNew[i] = y0 + t * (y1 - y0)
        
        return yNew


class ChannelDropout:
    """
    Randomly drop sensor channels.
    
    Simulates missing or faulty sensors.
    
    Args:
        dropProb: Probability of dropping each channel
        prob: Probability of applying augmentation
    """
    
    def __init__(
        self,
        dropProb: float = 0.1,
        prob: float = 0.5
    ):
        self.dropProb = dropProb
        self.prob = prob
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.prob:
            T, C = x.shape
            mask = torch.rand(C) > self.dropProb
            x = x * mask.unsqueeze(0)
        return x


class TemporalCrop:
    """
    Random temporal cropping and padding.
    
    Simulates partial observations.
    
    Args:
        cropRatio: Range of crop ratios (min, max)
        prob: Probability of applying
    """
    
    def __init__(
        self,
        cropRatio: tuple = (0.8, 1.0),
        prob: float = 0.5
    ):
        self.cropMin = cropRatio[0]
        self.cropMax = cropRatio[1]
        self.prob = prob
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.prob:
            T, C = x.shape
            cropLen = int(T * (self.cropMin + (self.cropMax - self.cropMin) * torch.rand(1).item()))
            
            # Random start position
            startIdx = torch.randint(0, T - cropLen + 1, (1,)).item()
            
            # Crop and pad
            cropped = x[startIdx:startIdx + cropLen]
            
            # Pad to original length
            padLen = T - cropLen
            padLeft = padLen // 2
            padRight = padLen - padLeft
            
            x = torch.cat([
                torch.zeros(padLeft, C),
                cropped,
                torch.zeros(padRight, C)
            ], dim=0)
        
        return x


def getTrainAugmentation(strength: str = 'medium'):
    """
    Get standard training augmentation pipeline.
    
    Args:
        strength: 'light', 'medium', or 'strong'
    
    Returns:
        Composed transform
    """
    if strength == 'light':
        return Compose([
            RandomScaling(scaleRange=(0.95, 1.05), prob=0.3),
            RandomNoise(noiseLevel=0.01, prob=0.3),
        ])
    elif strength == 'medium':
        return Compose([
            RandomScaling(scaleRange=(0.9, 1.1), prob=0.5),
            RandomNoise(noiseLevel=0.02, prob=0.5),
            RandomRotation(maxAngle=10, prob=0.3),
        ])
    else:  # strong
        return Compose([
            RandomScaling(scaleRange=(0.8, 1.2), prob=0.5),
            RandomNoise(noiseLevel=0.05, prob=0.5),
            RandomRotation(maxAngle=20, prob=0.5),
            TimeWarping(sigma=0.2, prob=0.3),
            ChannelDropout(dropProb=0.1, prob=0.2),
        ])

"""
TinyHAR Baseline Implementation
A Lightweight Deep Learning Model for Human Activity Recognition

Reference: Zhou et al., "TinyHAR: Multi-Task Learning for HAR," ISWC 2022
GitHub: https://github.com/teco-kit/ISWC22-HAR

Architecture (from paper):
- PART 1: Channel-wise Conv2D feature extraction (4 layers, alternating stride)
- PART 2: Cross-channel self-attention
- PART 3: Cross-channel fusion via FC
- PART 4: Temporal LSTM/GRU
- PART 5: Temporal weighted aggregation
- PART 6: Prediction head

Target: ~25-40k parameters depending on config
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Cross-channel self-attention (from TinyHAR paper).
    Applied at each timestep across sensor channels.
    """
    
    def __init__(self, nChannels: int):
        super().__init__()
        self.query = nn.Linear(nChannels, nChannels, bias=False)
        self.key = nn.Linear(nChannels, nChannels, bias=False)
        self.value = nn.Linear(nChannels, nChannels, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Args:
            x: [B, C, F] where C=sensor_channels, F=filter_num
        Returns:
            [B, C, F]
        """
        f = self.query(x)
        g = self.key(x)
        h = self.value(x)
        
        # Attention: softmax(f @ g^T)
        beta = F.softmax(torch.bmm(f, g.transpose(1, 2)), dim=1)
        
        # Output: gamma * (h^T @ beta) + x
        o = self.gamma * torch.bmm(h.transpose(1, 2), beta) + x.transpose(1, 2)
        return o.transpose(1, 2)


class TemporalWeightedAggregation(nn.Module):
    """
    Temporal weighted aggregation from TinyHAR (tnaive option).
    Uses attention weights to aggregate temporal features.
    """
    
    def __init__(self, hiddenDim: int):
        super().__init__()
        self.fc1 = nn.Linear(hiddenDim, hiddenDim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hiddenDim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Args:
            x: [B, T, F]
        Returns:
            [B, F]
        """
        out = self.activation(self.fc1(x))
        attnWeights = F.softmax(self.fc2(out), dim=1)  # [B, T, 1]
        context = torch.sum(attnWeights * x, dim=1)    # [B, F]
        # Residual: last timestep + gamma * context
        return x[:, -1, :] + self.gamma * context


class TinyHAR(nn.Module):
    """
    TinyHAR: Lightweight HAR Model from ISWC 2022.
    
    Architecture follows the paper:
    - 4 Conv2D layers with temporal downsampling
    - Cross-channel self-attention
    - FC for channel fusion  
    - LSTM for temporal modeling
    - Temporal weighted aggregation
    - Classification head
    
    Args:
        numClasses: Number of activity classes
        inChannels: Number of input sensor channels
        seqLen: Sequence length (timesteps)
        filterNum: Number of filters (F in paper, default 16)
        nbConvLayers: Number of conv layers (default 4)
        dropout: Dropout rate
        
    Target: ~25-40k parameters
    """
    
    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        filterNum: int = 24,       # Verified: 24 gives ~42k params
        nbConvLayers: int = 4,
        filterSize: int = 5,
        dropout: float = 0.5,      # TinyHAR uses heavy dropout
    ):
        super().__init__()
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        
        # PART 1: Channel-wise Conv2D feature extraction
        # Input: [B, 1, T, C] -> Output: [B, F, T', C]
        filterNums = [1] + [filterNum] * nbConvLayers
        self.convLayers = nn.ModuleList()
        
        for i in range(nbConvLayers):
            inC = filterNums[i]
            outC = filterNums[i + 1]
            # Alternating stride: stride=2 on odd layers for temporal downsampling
            stride = (2, 1) if i % 2 == 1 else (1, 1)
            self.convLayers.append(nn.Sequential(
                nn.Conv2d(inC, outC, kernel_size=(filterSize, 1), stride=stride, padding=(filterSize // 2, 0)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(outC)
            ))
        
        # Calculate downsampled length
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seqLen, inChannels)
            for layer in self.convLayers:
                dummy = layer(dummy)
            self.downsampledLen = dummy.shape[2]
            
        # PART 2: Cross-channel self-attention
        self.channelAttention = SelfAttention(filterNum)
        
        # PART 3: Cross-channel fusion (FC)
        self.channelFusion = nn.Linear(inChannels * filterNum, 2 * filterNum)
        self.activation = nn.ReLU()
        
        # PART 4: Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=2 * filterNum,
            hidden_size=2 * filterNum,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # PART 5: Temporal weighted aggregation
        self.temporalAggregation = TemporalWeightedAggregation(2 * filterNum)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # PART 6: Classification head
        self.classifier = nn.Linear(2 * filterNum, numClasses)
        
        self._initWeights()
        
    def _initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, C] 
            
        Returns:
            Logits [B, numClasses]
        """
        # Input: [B, T, C] -> [B, 1, T, C]
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # PART 1: Conv layers
        for layer in self.convLayers:
            x = layer(x)
        # x: [B, F, T', C]
        
        x = x.permute(0, 3, 2, 1)  # [B, C, T', F]
        
        # PART 2: Cross-channel attention at each timestep
        # OPTIMIZED: Vectorized across all timesteps (no Python loop)
        B, C, T, F = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, C, F)  # [B*T, C, F]
        x_attn = self.channelAttention(x_flat)               # [B*T, C, F]
        x = x_attn.reshape(B, T, C, F).permute(0, 2, 1, 3)   # [B, C, T', F]
        
        x = self.dropout(x)
        
        # PART 3: Cross-channel fusion
        x = x.permute(0, 2, 1, 3)  # [B, T', C, F]
        x = x.reshape(B, T, -1)     # [B, T', C*F]
        x = self.activation(self.channelFusion(x))  # [B, T', 2F]
        
        # PART 4: Temporal LSTM
        x, _ = self.lstm(x)  # [B, T', 2F]
        
        # PART 5: Temporal aggregation
        x = self.temporalAggregation(x)  # [B, 2F]
        
        # PART 6: Prediction
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
    def countParameters(self) -> dict:
        """Count parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        return {
            'total': total,
            'convLayers': sum(p.numel() for layer in self.convLayers for p in layer.parameters()),
            'channelAttention': sum(p.numel() for p in self.channelAttention.parameters()),
            'channelFusion': sum(p.numel() for p in self.channelFusion.parameters()),
            'lstm': sum(p.numel() for p in self.lstm.parameters()),
            'temporalAggregation': sum(p.numel() for p in self.temporalAggregation.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
        }


def createTinyHar(dataset: str = 'ucihar') -> TinyHAR:
    """
    Factory function for dataset-specific TinyHAR models.
    
    Args:
        dataset: Dataset name ('ucihar', 'motionsense', 'wisdm')
        
    Returns:
        Configured TinyHAR model
    """
    configs = {
        'ucihar': {'numClasses': 6, 'inChannels': 9, 'seqLen': 128},
        'motionsense': {'numClasses': 6, 'inChannels': 6, 'seqLen': 128},
        'wisdm': {'numClasses': 6, 'inChannels': 3, 'seqLen': 128},
    }
    
    if dataset.lower() not in configs:
        raise ValueError(f"Unknown dataset: {dataset}")
        
    cfg = configs[dataset.lower()]
    return TinyHAR(**cfg)


if __name__ == '__main__':
    # Quick test
    model = createTinyHar('ucihar')
    params = model.countParameters()
    
    print(f"TinyHAR Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
        
    # Test forward pass
    x = torch.randn(2, 128, 9)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")

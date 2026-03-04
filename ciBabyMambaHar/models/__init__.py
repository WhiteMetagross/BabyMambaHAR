"""
CiBabyMambaHar Models Package (CI-BabyMamba-HAR)

Main Model:
- CiBabyMambaHar: CI-BabyMamba-HAR (~27K-29K params)
  - Channel-Independent Stem: Isolates sensor noise
  - Weight-Tied Bidirectional SSM with d_state=16
  - Context-Gated Temporal Attention: Spotlights transient events

Architecture: CI-BabyMamba-HAR (FROZEN)
    d_model = 24, d_state = 16, n_layers = 4, expand = 2
    Bidirectional: True (Weight-Tied)
    Gated Attention: True
    Channel-Independent: True
    Target: 27,000-29,000 parameters, O(N) complexity
    
Core Components:
- ChannelIndependentStem: Shared 1D conv across channels
- GatedTemporalAttention: Context-gated attention pooling
- WeightTiedBiDirMambaBlock: Bidirectional SSM with shared weights
- PureSelectiveScan: Pure PyTorch SSM fallback (no CUDA required)
"""

from ciBabyMambaHar.models.ciBabyMamba import (
    CiBabyMambaHar,
    BabyMamba,  # Legacy alias
    createCiBabyMambaHar,
    createBabyMamba,  # Legacy alias
    CI_BABYMAMBA_HAR_CONFIG,
    ChannelIndependentStem,
    GatedTemporalAttention,
)
from ciBabyMambaHar.models.ciBabyMambaBlock import (
    WeightTiedBiDirMambaBlock,
    PureSelectiveScan,
    # Legacy (for backward compatibility)
    SimpleMambaBlock,
    BabyMambaBlock,
    BiDirectionalMambaBlock,
    RecursiveBiDirectionalBlock,
    SEBlock,
)
from ciBabyMambaHar.models.stems import (
    SimpleStem,
    # Legacy stems (for ablation studies)
    WideEyeStem,
    SpectralTemporalStem,
    TimeOnlyStem,
    HollowStem,
    SensorStem,
    DepthwiseSeparableConv1d,
)
from ciBabyMambaHar.models.heads import ClassificationHead

# Import ablations if they exist
try:
    from ciBabyMambaHar.models.ablations import (
        BabyMambaFull,
        BabyMambaNoSpectral,
        BabyMambaNoRecursion,
        BabyMambaNoSe,
        BabyMambaCnnOnly,
        getAblationModel,
    )
    _ABLATIONS_AVAILABLE = True
except ImportError:
    _ABLATIONS_AVAILABLE = False

__all__ = [
    # Main Model (CiBabyMambaHar - PRIMARY)
    "CiBabyMambaHar",
    "createCiBabyMambaHar",
    "CI_BABYMAMBA_HAR_CONFIG",
    
    # Legacy aliases (backward compatibility)
    "BabyMamba",
    "createBabyMamba",
    
    # Core Blocks
    "WeightTiedBiDirMambaBlock",
    "PureSelectiveScan",
    
    # Primary Stem
    "SimpleStem",
    
    # Classification Head
    "ClassificationHead",
    
    # Legacy (backward compatibility)
    "SimpleMambaBlock",
    "BabyMambaBlock",
    "BiDirectionalMambaBlock",
    "RecursiveBiDirectionalBlock",
    "SEBlock",
    "WideEyeStem",
    "SpectralTemporalStem",
    "TimeOnlyStem",
    "HollowStem",
    "SensorStem",
    "DepthwiseSeparableConv1d",
]

# Add ablation models if available
if _ABLATIONS_AVAILABLE:
    __all__.extend([
        "BabyMambaFull",
        "BabyMambaNoSpectral",
        "BabyMambaNoRecursion",
        "BabyMambaNoSe",
        "BabyMambaCnnOnly",
        "getAblationModel",
    ])

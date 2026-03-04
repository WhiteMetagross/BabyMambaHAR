"""
CiBabyMambaHar - Ultra-Lightweight HAR with State Space Models

BabyMamba-Crossover-BiDir: Weight-Tied Bidirectional SSM for Edge-based Activity Recognition

Architecture (FROZEN):
    d_model = 24, d_state = 8, n_layers = 4, expand = 2
    Bidirectional: True (Weight-Tied)
    Target: ~25,110 parameters, O(N) complexity
"""

__version__ = "0.3.0"
__author__ = "Your Name"

from ciBabyMambaHar.models import (
    CiBabyMambaHar,
    BabyMamba,  # Legacy alias
    createCiBabyMambaHar,
    createBabyMamba,  # Legacy alias
    WeightTiedBiDirMambaBlock,
    CI_BABYMAMBA_HAR_CONFIG,
)
from ciBabyMambaHar.data import (
    UciHarDataset,
    MotionSenseDataset,
    WisdmDataset,
    Pamap2Dataset,
    getUciHarLoaders,
    getMotionSenseLoaders,
    getWisdmLoaders,
    getPamap2Loaders,
)

__all__ = [
    # Main Model (CiBabyMambaHar - BabyMamba-Crossover-BiDir)
    "CiBabyMambaHar",
    "createCiBabyMambaHar",
    "WeightTiedBiDirMambaBlock",
    "CI_BABYMAMBA_HAR_CONFIG",
    # Legacy aliases (backward compatibility)
    "BabyMamba",
    "createBabyMamba",
    # Data
    "UciHarDataset",
    "MotionSenseDataset", 
    "WisdmDataset",
    "Pamap2Dataset",
    "getUciHarLoaders",
    "getMotionSenseLoaders",
    "getWisdmLoaders",
    "getPamap2Loaders",
]

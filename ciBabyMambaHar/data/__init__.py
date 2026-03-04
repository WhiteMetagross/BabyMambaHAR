"""
Data Package for CI-BabyMamba-HAR

Provides data loaders for HAR benchmark datasets:
- UCI-HAR: The Standard (controlled environment)
- MotionSense: In-the-Wild (real iPhone noise)
- WISDM: Scale & Imbalance (1M+ samples)
- PAMAP2: Complex/Multimodal (18+ channels)
- Opportunity: Complex activities with Null class
- UniMiB SHAR: Smartphone-based HAR with falls
- Skoda: Automotive assembly gestures (10 + Null)
- Daphnet: Parkinson's freezing detection (Walk vs Freeze)

Includes HARaugment - the REQUIRED augmentation for training.
"""

from ciBabyMambaHar.data.uciHar import UciHarDataset, getUciHarLoaders
from ciBabyMambaHar.data.motionSense import MotionSenseDataset, getMotionSenseLoaders
from ciBabyMambaHar.data.wisdm import WisdmDataset, getWisdmLoaders
from ciBabyMambaHar.data.pamap2 import Pamap2Dataset, getPamap2Loaders
from ciBabyMambaHar.data.opportunity import OpportunityDataset, getOpportunityLoaders
from ciBabyMambaHar.data.unimib import UniMiBSHARDataset, getUniMiBLoaders
from ciBabyMambaHar.data.skoda import SkodaDataset, getSkodaLoaders
from ciBabyMambaHar.data.daphnet import DaphnetDataset, getDaphnetLoaders
from ciBabyMambaHar.data.augmentations import (
    HARaugment,  # PRIMARY: Required augmentation for CI-BabyMamba-HAR
    RandomScaling,
    RandomNoise,
    RandomRotation,
    TimeWarping,
    Compose,
    getTrainAugmentation,
)

__all__ = [
    # Datasets
    "UciHarDataset",
    "MotionSenseDataset",
    "WisdmDataset",
    "Pamap2Dataset",
    "OpportunityDataset",
    "UniMiBSHARDataset",
    "SkodaDataset",
    "DaphnetDataset",
    # Loaders
    "getUciHarLoaders",
    "getMotionSenseLoaders",
    "getWisdmLoaders",
    "getPamap2Loaders",
    "getOpportunityLoaders",
    "getUniMiBLoaders",
    "getSkodaLoaders",
    "getDaphnetLoaders",
    # Augmentations
    "HARaugment",  # PRIMARY: Always use this for training
    "RandomScaling",
    "RandomNoise",
    "RandomRotation",
    "TimeWarping",
    "Compose",
    "getTrainAugmentation",
]

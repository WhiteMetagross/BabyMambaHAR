#!/usr/bin/env python3
"""
Parallel Training Script for Baseline Models

Features:
- Training for TinierHAR, TinyHAR, LightDeepConvLSTM
- 200 epochs with early stopping (patience=20)
- Random seed generation using RNG for 5 seeds
- Comprehensive metrics: Accuracy, F1, Precision, Recall, Confusion Matrix
- Latency, MACs, and Parameter profiling
- Results stored for comparison
Signal Rescue Strategies (for fairness with CI-BabyMamba-HAR):
- SKODA: 5Hz Butterworth filter + Label Smoothing 0.1 + Batch Size 512
- PAMAP2: 10Hz Butterworth filter + Robust Scaling + Gradient Clip 1.0 + NO label smoothing
- Daphnet: 12Hz Butterworth filter + Class Weights [1.0, 15.0] + Weight Decay 0.05 + NO label smoothing
Usage:
    python scripts/trainBaselines.py --model tinierhar --dataset ucihar
    python scripts/trainBaselines.py --model tinyhar --dataset all --seeds 5
    python scripts/trainBaselines.py --model all --dataset all
"""

import os
import sys
import json
import random
import warnings
import time
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines import TinierHAR, TinyHAR, LightDeepConvLSTM, DeepConvLSTM
from ciBabyMambaHar.utils.profiling import countParameters, computeMacs, benchmarkLatency


# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable TensorFloat-32 for faster training on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions

TRAINING_EPOCHS = 200
N_PARALLEL_JOBS = 4  # Number of parallel workers for training
WARMUP_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10  # Default patience
N_SEEDS = 5

# Dataset-specific patience settings (PAMAP2 needs more patience due to complexity)
DATASET_PATIENCE = {
    'ucihar': 10,
    'motionsense': 10,
    'wisdm': 10,
    'pamap2': 10,  # Reduced from 30 to 10 as per user request
    'opportunity': 10,
    'unimib': 10,
    'skoda': 10,
    'daphnet': 10,
}

# Master seed for RNG seed generation
MASTER_SEED = 17

# Dataset specifications
DATASET_SPECS = {
    'ucihar': {
        'name': 'UCI-HAR', 'numClasses': 6, 'inChannels': 9, 'seqLen': 128,
        'root': './datasets/UCI HAR Dataset'
    },
    'motionsense': {
        'name': 'MotionSense', 'numClasses': 6, 'inChannels': 6, 'seqLen': 128,
        'root': './datasets/motion-sense-master'
    },
    'wisdm': {
        'name': 'WISDM', 'numClasses': 6, 'inChannels': 3, 'seqLen': 128,
        'root': './datasets/WISDM_ar_v1.1'
    },
    'pamap2': {
        'name': 'PAMAP2', 'numClasses': 12, 'inChannels': 19, 'seqLen': 128,
        'root': './datasets/PAMAP2_Dataset'
    },
    'opportunity': {
        'name': 'Opportunity', 'numClasses': 5, 'inChannels': 79, 'seqLen': 128,
        'root': './datasets/Opportunity'
    },
    'unimib': {
        'name': 'UniMiB-SHAR', 'numClasses': 9, 'inChannels': 3, 'seqLen': 128,
        'root': './datasets/UniMiB-SHAR'
    },
    'skoda': {
        'name': 'Skoda', 'numClasses': 11, 'inChannels': 30, 'seqLen': 98,
        'root': './datasets/Skoda'
    },
    'daphnet': {
        'name': 'Daphnet', 'numClasses': 2, 'inChannels': 9, 'seqLen': 64,
        'root': './datasets/Daphnet'
    },
}

# ==========================================
# LOCKED CONFIGURATIONS - NO HPO, NO GUESSING
# Sources:
# - TinierHAR: zhaxidele/TinierHAR (arXiv 2025) - gruUnits=16 for ~17k params
# - TinyHAR: teco-kit/ISWC22-HAR (ISWC 2022) - filterNum=24 for ~42k params
# - LightDeepConvLSTM: Manual downscaling - ~15k params for iso-parameter control
#
# PARAMETER COUNTS (UCI-HAR, 9 channels):
# - TinierHAR:        16,931 params (~17k target)
# - TinyHAR:          42,704 params (~40k target)  
# - LightDeepConvLSTM: 15,286 params (~15k iso-param control)
# - BabyMamba:        13,951 params (~14.8k, ours)
# ==========================================
DEFAULT_HPARAMS = {
    'tinierhar': {
        # TinierHAR (Target: ~17k Params) - gruUnits=16, dropout=0.5 from paper
        'ucihar': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                   'nbFilters': 8, 'nbConvBlocks': 4, 'gruUnits': 16, 'dropout': 0.5},
        'motionsense': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                        'nbFilters': 8, 'nbConvBlocks': 4, 'gruUnits': 16, 'dropout': 0.5},
        'wisdm': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                  'nbFilters': 8, 'nbConvBlocks': 4, 'gruUnits': 16, 'dropout': 0.5},
        # PAMAP2 Signal Rescue: NO label smoothing (extreme outliers need sharp decisions)
        'pamap2': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.0, 'batchSize': 64,
                   'nbFilters': 8, 'nbConvBlocks': 4, 'gruUnits': 16, 'dropout': 0.5},
        # SKODA Signal Rescue: Batch 512 (large dataset, fuzzy gesture boundaries)
        'skoda': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 512,
                  'nbFilters': 8, 'nbConvBlocks': 4, 'gruUnits': 16, 'dropout': 0.5},
        # Daphnet Signal Rescue: Weight decay 0.05, NO label smoothing (binary decisions)
        'daphnet': {'lr': 0.002, 'weightDecay': 0.05, 'labelSmoothing': 0.0, 'batchSize': 512,
                    'nbFilters': 8, 'nbConvBlocks': 4, 'gruUnits': 16, 'dropout': 0.5},
    },
    'tinyhar': {
        # TinyHAR (SOTA: ~42k Params) - filterNum=24, dropout=0.5 from paper
        'ucihar': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                   'filterNum': 24, 'nbConvLayers': 4, 'filterSize': 5, 'dropout': 0.5},
        'motionsense': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                        'filterNum': 24, 'nbConvLayers': 4, 'filterSize': 5, 'dropout': 0.5},
        'wisdm': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                  'filterNum': 24, 'nbConvLayers': 4, 'filterSize': 5, 'dropout': 0.5},
        # PAMAP2 Signal Rescue: NO label smoothing (extreme outliers need sharp decisions)
        'pamap2': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.0, 'batchSize': 64,
                   'filterNum': 24, 'nbConvLayers': 4, 'filterSize': 5, 'dropout': 0.5},
        # SKODA Signal Rescue: Batch 512 (large dataset, fuzzy gesture boundaries)
        'skoda': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 512,
                  'filterNum': 24, 'nbConvLayers': 4, 'filterSize': 5, 'dropout': 0.5},
        # Daphnet Signal Rescue: Weight decay 0.05, NO label smoothing (binary decisions)
        'daphnet': {'lr': 0.002, 'weightDecay': 0.05, 'labelSmoothing': 0.0, 'batchSize': 512,
                    'filterNum': 24, 'nbConvLayers': 4, 'filterSize': 5, 'dropout': 0.5},
    },
    'lightdeepconvlstm': {
        # LightDeepConvLSTM (Control: ~15k Params) - lstmHidden=32 for iso-param control
        'ucihar': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                   'convFilters': 16, 'lstmHidden': 32, 'dropout': 0.5},
        'motionsense': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                        'convFilters': 16, 'lstmHidden': 32, 'dropout': 0.5},
        'wisdm': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                  'convFilters': 16, 'lstmHidden': 32, 'dropout': 0.5},
        # PAMAP2 Signal Rescue: NO label smoothing (extreme outliers need sharp decisions)
        'pamap2': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.0, 'batchSize': 64,
                   'convFilters': 16, 'lstmHidden': 32, 'dropout': 0.5},
        # SKODA Signal Rescue: Batch 512 (large dataset, fuzzy gesture boundaries)
        'skoda': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 512,
                  'convFilters': 16, 'lstmHidden': 32, 'dropout': 0.5},
        # Daphnet Signal Rescue: Weight decay 0.05, NO label smoothing (binary decisions)
        'daphnet': {'lr': 0.002, 'weightDecay': 0.05, 'labelSmoothing': 0.0, 'batchSize': 512,
                    'convFilters': 16, 'lstmHidden': 32, 'dropout': 0.5},
    },
    'deepconvlstm': {
        # DeepConvLSTM (~132K Params) - Traditional deep learning baseline
        # Based on DeepConvLSTM paper (Ordóñez & Roggen 2016): 4 Conv + 2 Unidirectional LSTM
        'ucihar': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                   'convFilters': 64, 'lstmHidden': 64, 'lstmLayers': 2, 'bidirectional': False, 'dropout': 0.5},
        'motionsense': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                        'convFilters': 64, 'lstmHidden': 64, 'lstmLayers': 2, 'bidirectional': False, 'dropout': 0.5},
        'wisdm': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 64,
                  'convFilters': 64, 'lstmHidden': 64, 'lstmLayers': 2, 'bidirectional': False, 'dropout': 0.5},
        'pamap2': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.0, 'batchSize': 64,
                   'convFilters': 64, 'lstmHidden': 64, 'lstmLayers': 2, 'bidirectional': False, 'dropout': 0.5},
        'skoda': {'lr': 0.002, 'weightDecay': 0.01, 'labelSmoothing': 0.1, 'batchSize': 512,
                  'convFilters': 64, 'lstmHidden': 64, 'lstmLayers': 2, 'bidirectional': False, 'dropout': 0.5},
        'daphnet': {'lr': 0.002, 'weightDecay': 0.05, 'labelSmoothing': 0.0, 'batchSize': 512,
                    'convFilters': 64, 'lstmHidden': 64, 'lstmLayers': 2, 'bidirectional': False, 'dropout': 0.5},
    }
}


# ============================================================================
# HPO RESULTS LOADING
# ============================================================================

def loadHpoResults(modelName: str, dataset: str) -> Optional[Dict[str, Any]]:
    """Load HPO results for a model/dataset if available."""
    hpoPath = Path("results/hpo") / f"hpo_{modelName}_{dataset}.json"
    if hpoPath.exists():
        try:
            with open(hpoPath, 'r') as f:
                hpoData = json.load(f)
            print(f"   Loaded HPO results from {hpoPath}")
            return hpoData.get('bestParams', {})
        except Exception as e:
            print(f"   Warning: Could not load HPO results: {e}")
    return None


# ============================================================================
# RANDOM SEED GENERATION
# ============================================================================

def generateRandomSeeds(nSeeds: int, masterSeed: int = MASTER_SEED) -> List[int]:
    """Generate random seeds using RNG from master seed."""
    rng = np.random.default_rng(masterSeed)
    return list(rng.integers(0, 100000, size=nSeeds))


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EpochMetrics:
    epoch: int
    trainLoss: float
    trainAcc: float
    valLoss: float
    valAcc: float
    valF1: float
    valPrecision: float
    valRecall: float
    lr: float


@dataclass
class TrainingResult:
    modelName: str
    dataset: str
    seed: int
    bestAccuracy: float
    bestF1: float
    bestPrecision: float
    bestRecall: float
    finalAccuracy: float
    parameters: int
    macs: int
    sizeMb: float
    latencyMs: float
    throughput: float
    epochsTrained: int
    totalEpochs: int
    trainTime: float
    earlyStopped: bool
    confusionMatrix: List[List[int]] = field(default_factory=list)
    epochMetrics: List[Dict] = field(default_factory=list)


def setSeed(seed: int):
    seed = int(seed)  # Convert numpy.int64 to Python int
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# MODEL CREATION
# ============================================================================

def createModel(
    modelName: str,
    config: Dict[str, Any],
    spec: Dict[str, Any]
) -> nn.Module:
    """Create a baseline model with given config."""
    
    if modelName == 'tinierhar':
        return TinierHAR(
            numClasses=spec['numClasses'],
            inChannels=spec['inChannels'],
            seqLen=spec['seqLen'],
            nbFilters=config.get('nbFilters', 8),
            nbConvBlocks=config.get('nbConvBlocks', 4),
            gruUnits=config.get('gruUnits', 16),
            dropout=config.get('dropout', 0.2)
        )
    elif modelName == 'tinyhar':
        return TinyHAR(
            numClasses=spec['numClasses'],
            inChannels=spec['inChannels'],
            seqLen=spec['seqLen'],
            filterNum=config.get('filterNum', 16),
            nbConvLayers=config.get('nbConvLayers', 4),
            filterSize=config.get('filterSize', 5),
            dropout=config.get('dropout', 0.1)
        )
    elif modelName == 'lightdeepconvlstm':
        return LightDeepConvLSTM(
            numClasses=spec['numClasses'],
            inChannels=spec['inChannels'],
            seqLen=spec['seqLen'],
            convFilters=config.get('convFilters', 32),
            lstmHidden=config.get('lstmHidden', 48),
            dropout=config.get('dropout', 0.3)
        )
    elif modelName == 'deepconvlstm':
        return DeepConvLSTM(
            numClasses=spec['numClasses'],
            inChannels=spec['inChannels'],
            seqLen=spec['seqLen'],
            convFilters=config.get('convFilters', 64),
            lstmHidden=config.get('lstmHidden', 64),
            lstmLayers=config.get('lstmLayers', 2),
            dropout=config.get('dropout', 0.5),
            bidirectional=config.get('bidirectional', False)  # Unidirectional per original paper
        )
    else:
        raise ValueError(f"Unknown model: {modelName}")


# ============================================================================
# DATA LOADING
# ============================================================================

def getDataLoaders(dataset: str, batchSize: int = 64) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
    """Get data loaders for a dataset.
    
    Returns:
        (trainLoader, testLoader, classWeights) where classWeights is None for balanced datasets
    """
    
    spec = DATASET_SPECS[dataset]
    classWeights = None
    
    if dataset == 'ucihar':
        from ciBabyMambaHar.data.uciHar import getUciHarLoaders
        trainLoader, testLoader = getUciHarLoaders(root=spec['root'], batchSize=batchSize, numWorkers=2)
    elif dataset == 'motionsense':
        from ciBabyMambaHar.data.motionSense import getMotionSenseLoaders
        trainLoader, testLoader = getMotionSenseLoaders(root=spec['root'], batchSize=batchSize, numWorkers=2)
    elif dataset == 'wisdm':
        from ciBabyMambaHar.data.wisdm import getWisdmLoaders
        trainLoader, testLoader = getWisdmLoaders(root=spec['root'], batchSize=batchSize, numWorkers=2)
    elif dataset == 'pamap2':
        from ciBabyMambaHar.data.pamap2 import getPamap2Loaders
        # PAMAP2 is imbalanced - get class weights for weighted loss
        trainLoader, testLoader, classWeights = getPamap2Loaders(
            root=spec['root'], batchSize=batchSize, numWorkers=2, returnWeights=True
        )
    elif dataset == 'opportunity':
        from ciBabyMambaHar.data.opportunity import getOpportunityLoaders
        # Opportunity has dominant Null class - get class weights
        trainLoader, testLoader, classWeights = getOpportunityLoaders(
            root=spec['root'], batchSize=batchSize, numWorkers=2, returnWeights=True
        )
    elif dataset == 'unimib':
        from ciBabyMambaHar.data.unimib import getUniMiBLoaders
        trainLoader, testLoader = getUniMiBLoaders(root=spec['root'], batchSize=batchSize, numWorkers=2)
    elif dataset == 'skoda':
        from ciBabyMambaHar.data.skoda import getSkodaLoaders
        trainLoader, testLoader, classWeights = getSkodaLoaders(
            root=spec['root'], batchSize=batchSize, numWorkers=2, returnWeights=True
        )
    elif dataset == 'daphnet':
        from ciBabyMambaHar.data.daphnet import getDaphnetLoaders
        trainLoader, testLoader, classWeights = getDaphnetLoaders(
            root=spec['root'], batchSize=batchSize, numWorkers=2, returnWeights=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return trainLoader, testLoader, classWeights


# ============================================================================
# METRICS
# ============================================================================

def computeMetrics(preds: torch.Tensor, labels: torch.Tensor, numClasses: int) -> Dict[str, float]:
    """Compute macro F1, precision, and recall."""
    precisions, recalls, f1Scores = [], [], []
    
    for c in range(numClasses):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        precisions.append(precision)
        recalls.append(recall)
        f1Scores.append(f1)
    
    return {
        'f1': 100.0 * np.mean(f1Scores),
        'precision': 100.0 * np.mean(precisions),
        'recall': 100.0 * np.mean(recalls)
    }


def computeConfusionMatrix(preds: torch.Tensor, labels: torch.Tensor, numClasses: int) -> List[List[int]]:
    """Compute confusion matrix."""
    matrix = [[0] * numClasses for _ in range(numClasses)]
    for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
        matrix[l][p] += 1
    return matrix


# ============================================================================
# PROFILING (Math-based MAC calculation)
# ============================================================================

def computeModelMacs(model: nn.Module, modelName: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute MACs using math formulas based on model architecture.
    
    MACs formulas:
    - Conv1d: out_channels * kernel_size * in_channels * out_length
    - Conv2d: out_channels * kernel_size_h * kernel_size_w * in_channels * out_h * out_w
    - Linear: in_features * out_features
    - GRU: 3 * (input_size * hidden_size + hidden_size * hidden_size) * seq_len
    - LSTM: 4 * (input_size * hidden_size + hidden_size * hidden_size) * seq_len
    """
    
    totalMacs = 0
    seqLen = spec['seqLen']
    inChannels = spec['inChannels']
    numClasses = spec['numClasses']
    
    # Use thop for accurate calculation if available
    try:
        from thop import profile
        # Move model to CPU for profiling (thop works better on CPU)
        modelCpu = model.cpu()
        x = torch.randn(1, seqLen, inChannels)
        macs, _ = profile(modelCpu, inputs=(x,), verbose=False)
        # Move model back to original device
        model.to(DEVICE)
        return {
            'macs': int(macs),
            'flops': int(macs * 2),
            'method': 'thop'
        }
    except (ImportError, Exception):
        pass
    
    # Fallback: Use formula-based estimation
    if modelName == 'tinierhar':
        # TinierHAR: DepthwiseSeparable Conv2D blocks + Bidirectional GRU
        config = model.countParameters()
        
        # Conv blocks MACs (approximate based on architecture)
        # For depthwise-separable: (DW + PW) per block
        nbFilters = getattr(model, 'convBlocks')[0].main[0].pointwise.out_channels
        currLen = seqLen
        
        for i, block in enumerate(model.convBlocks):
            # Depthwise conv MACs: kernelSize * inChannels * outputLen
            # Pointwise conv MACs: inChannels * outChannels * outputLen
            if hasattr(block, 'useMaxpool') and block.useMaxpool:
                currLen = currLen // 2
        
        # GRU MACs: 3 gates * (input*hidden + hidden*hidden) * 2 (bidirectional) * seqLen
        gruUnits = model.gru.hidden_size
        gruInput = model.gruInputDim
        gruMacs = 3 * (gruInput * gruUnits + gruUnits * gruUnits) * 2 * model.temporalLen
        
        # Attention + classifier MACs
        attnMacs = model.temporalLen * gruUnits * 2
        classifierMacs = gruUnits * 2 * numClasses
        
        totalMacs = config['convBlocks'] * seqLen + gruMacs + attnMacs + classifierMacs
        
    elif modelName == 'tinyhar':
        # TinyHAR: Conv2D + Attention + LSTM
        config = model.countParameters()
        
        filterNum = model.convLayers[0][0].out_channels
        currLen = seqLen
        
        # Conv layers MACs
        for i, layer in enumerate(model.convLayers):
            stride = 2 if i % 2 == 1 else 1
            currLen = currLen // stride
        
        # Self-attention MACs: 3 * (C * F * F) per timestep
        attnMacs = currLen * 3 * filterNum * filterNum
        
        # Channel fusion MACs: (inChannels * filterNum) * (2 * filterNum) * currLen
        fusionMacs = (inChannels * filterNum) * (2 * filterNum) * currLen
        
        # LSTM MACs: 4 * (input*hidden + hidden*hidden) * seqLen
        lstmMacs = 4 * (2 * filterNum * 2 * filterNum + 2 * filterNum * 2 * filterNum) * currLen
        
        # Temporal aggregation + classifier
        aggMacs = currLen * 2 * filterNum * 2 + 2 * filterNum * numClasses
        
        totalMacs = config['convLayers'] * seqLen + attnMacs + fusionMacs + lstmMacs + aggMacs
        
    elif modelName == 'lightdeepconvlstm':
        # LightDeepConvLSTM: Conv1D + BiLSTM
        convFilters = model.convLayers[0].out_channels
        lstmHidden = model.lstm.hidden_size
        
        # Conv layers MACs
        convMacs = 0
        currLen = seqLen
        for layer in model.convLayers:
            if isinstance(layer, nn.Conv1d):
                convMacs += layer.in_channels * layer.out_channels * layer.kernel_size[0] * currLen
            elif isinstance(layer, nn.MaxPool1d):
                currLen = currLen // 2
        
        # Bidirectional LSTM MACs
        lstmMacs = 4 * (convFilters * lstmHidden + lstmHidden * lstmHidden) * 2 * currLen
        
        # Classifier MACs
        classifierMacs = lstmHidden * 2 * numClasses
        
        totalMacs = convMacs + lstmMacs + classifierMacs
    
    return {
        'macs': int(totalMacs),
        'flops': int(totalMacs * 2),
        'method': 'formula'
    }


def profileModelFull(model: nn.Module, modelName: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Full model profiling with parameters, MACs, size, and latency."""
    
    # Parameters
    params = countParameters(model)
    
    # MACs
    macsResult = computeModelMacs(model, modelName, spec)
    
    # Model size
    paramSize = sum(p.nelement() * p.element_size() for p in model.parameters())
    bufferSize = sum(b.nelement() * b.element_size() for b in model.buffers())
    sizeMb = (paramSize + bufferSize) / (1024 ** 2)
    
    # Latency
    latency = {'mean_ms': 0.0, 'throughput': 0.0}
    try:
        model = model.to(DEVICE)
        inputShape = (1, spec['seqLen'], spec['inChannels'])
        latencyResults = benchmarkLatency(model, inputShape, device=str(DEVICE), numRuns=50)
        latency = latencyResults.get('batch_1', latency)
    except Exception as e:
        print(f"   Warning: Latency benchmark failed: {e}")
    
    return {
        'parameters': params['total'],
        'macs': macsResult['macs'],
        'flops': macsResult['flops'],
        'macsMethod': macsResult.get('method', 'unknown'),
        'sizeMb': sizeMb,
        'latencyMs': latency.get('mean_ms', 0.0),
        'throughput': latency.get('throughput', 0.0)
    }


# ============================================================================
# TRAINING
# ============================================================================

def trainModel(
    modelName: str,
    dataset: str,
    seed: int,
    epochs: int = TRAINING_EPOCHS,
    warmupEpochs: int = WARMUP_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE
) -> TrainingResult:
    """Train a single model with full tracking."""
    
    setSeed(seed)
    
    spec = DATASET_SPECS[dataset]
    
    # Try to load HPO results, fall back to defaults
    hpoParams = loadHpoResults(modelName, dataset)
    defaultHparams = DEFAULT_HPARAMS.get(modelName, {}).get(dataset, DEFAULT_HPARAMS['tinierhar']['ucihar'])
    
    if hpoParams:
        # Merge HPO params with defaults (HPO takes priority)
        hparams = {**defaultHparams, **hpoParams}
    else:
        hparams = defaultHparams
    
    # Get data (with class weights for imbalanced datasets like PAMAP2)
    trainLoader, testLoader, classWeights = getDataLoaders(dataset, hparams.get('batchSize', 64))
    
    # Create model
    model = createModel(modelName, hparams, spec)
    model = model.to(DEVICE)
    
    # Profile model (before compile)
    profileResult = profileModelFull(model, modelName, spec)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams['lr'],
        weight_decay=hparams['weightDecay']
    )
    
    # Loss - use class weights for imbalanced datasets (e.g., PAMAP2)
    if classWeights is not None:
        classWeights = classWeights.to(DEVICE)
        criterion = nn.CrossEntropyLoss(
            weight=classWeights, 
            label_smoothing=hparams.get('labelSmoothing', 0.1)
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=hparams.get('labelSmoothing', 0.1))
    
    # Scheduler with warmup
    def lrLambda(epoch):
        if epoch < warmupEpochs:
            return max(1e-6 / hparams['lr'], (epoch + 1) / warmupEpochs)
        else:
            progress = (epoch - warmupEpochs) / max(1, epochs - warmupEpochs)
            return max(1e-6 / hparams['lr'], 0.5 * (1 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrLambda)
    
    # AMP
    scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    # Tracking
    epochMetrics = []
    bestAcc, bestF1, bestPrecision, bestRecall = 0.0, 0.0, 0.0, 0.0
    bestPreds, bestLabels = None, None
    epochsNoImprove = 0
    earlyStopped = False
    
    startTime = time.time()
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        trainLoss, trainCorrect, trainTotal = 0.0, 0, 0
        
        for x, y in trainLoader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    output = model(x)
                    loss = criterion(output, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            trainLoss += loss.item() * x.size(0)
            trainCorrect += (output.argmax(1) == y).sum().item()
            trainTotal += x.size(0)
        
        trainLoss /= trainTotal
        trainAcc = 100.0 * trainCorrect / trainTotal
        
        # Evaluate
        model.eval()
        valLoss = 0.0
        allPreds, allLabels = [], []
        
        with torch.no_grad():
            for x, y in testLoader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                loss = criterion(output, y)
                valLoss += loss.item() * x.size(0)
                allPreds.append(output.argmax(1))
                allLabels.append(y)
        
        allPreds = torch.cat(allPreds)
        allLabels = torch.cat(allLabels)
        
        valLoss /= len(allLabels)
        valAcc = 100.0 * (allPreds == allLabels).float().mean().item()
        metrics = computeMetrics(allPreds, allLabels, spec['numClasses'])
        
        scheduler.step()
        currentLr = optimizer.param_groups[0]['lr']
        
        epochMetrics.append(asdict(EpochMetrics(
            epoch=epoch, trainLoss=trainLoss, trainAcc=trainAcc,
            valLoss=valLoss, valAcc=valAcc,
            valF1=metrics['f1'], valPrecision=metrics['precision'],
            valRecall=metrics['recall'], lr=currentLr
        )))
        
        # Early stopping based on F1 score (better for imbalanced datasets)
        if metrics['f1'] > bestF1:
            bestF1 = metrics['f1']
            bestAcc = valAcc
            bestPrecision = metrics['precision']
            bestRecall = metrics['recall']
            bestPreds = allPreds.clone()
            bestLabels = allLabels.clone()
            epochsNoImprove = 0
        else:
            epochsNoImprove += 1
            if epochsNoImprove >= patience:
                earlyStopped = True
                break
    
    trainTime = time.time() - startTime
    
    confMatrix = computeConfusionMatrix(bestPreds, bestLabels, spec['numClasses']) if bestPreds is not None else []
    
    result = TrainingResult(
        modelName=modelName,
        dataset=dataset,
        seed=seed,
        bestAccuracy=bestAcc,
        bestF1=bestF1,
        bestPrecision=bestPrecision,
        bestRecall=bestRecall,
        finalAccuracy=valAcc,
        parameters=profileResult['parameters'],
        macs=profileResult['macs'],
        sizeMb=profileResult['sizeMb'],
        latencyMs=profileResult['latencyMs'],
        throughput=profileResult['throughput'],
        epochsTrained=epoch,
        totalEpochs=epochs,
        trainTime=trainTime,
        earlyStopped=earlyStopped,
        confusionMatrix=confMatrix,
        epochMetrics=epochMetrics
    )
    
    del model
    torch.cuda.empty_cache()
    
    return result


# ============================================================================
# PARALLEL TRAINING
# ============================================================================

def trainSingleSeed(args: Tuple) -> Optional[TrainingResult]:
    """Worker function for parallel training."""
    modelName, dataset, seed, epochs, patience = args
    try:
        return trainModel(modelName, dataset, seed, epochs, patience=patience)
    except Exception as e:
        print(f"   ❌ Seed {seed} failed: {e}")
        return None


def trainDatasetMultiSeed(
    modelName: str,
    dataset: str,
    nSeeds: int = N_SEEDS,
    epochs: int = TRAINING_EPOCHS,
    nJobs: int = N_PARALLEL_JOBS,
    patience: int = None  # None means use dataset-specific default
) -> Dict[str, Any]:
    """Train on dataset with multiple random seeds using parallel workers."""
    
    # Generate random seeds
    seeds = generateRandomSeeds(nSeeds, MASTER_SEED)
    
    # Use dataset-specific patience if not explicitly provided
    if patience is None:
        patience = DATASET_PATIENCE.get(dataset, EARLY_STOPPING_PATIENCE)
    
    print(f"\nTraining {modelName.upper()} on {dataset.upper()} ({nSeeds} seeds)")
    print(f"   Seeds (RNG generated): {seeds}")
    print(f"   Parallel workers: {nJobs}")
    print(f"   Early stopping patience: {patience}")
    
    results = []
    
    # Use parallel workers if not on GPU (GPU training should be sequential)
    if DEVICE.type == 'cuda' or nJobs == 1:
        # Sequential training for GPU (memory constraints)
        for i, seed in enumerate(seeds, 1):
            print(f"\n   Seed {seed} ({i}/{nSeeds})...")
            
            try:
                result = trainModel(modelName, dataset, seed, epochs, patience=patience)
                results.append(result)
                
                print(f"   Acc: {result.bestAccuracy:.2f}%, F1: {result.bestF1:.2f}%, "
                      f"Params: {result.parameters:,}, Epochs: {result.epochsTrained}/{result.totalEpochs}")
                
            except Exception as e:
                print(f"   Failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Parallel training for CPU
        print(f"\n   Running {nSeeds} seeds in parallel...")
        args = [(modelName, dataset, seed, epochs, patience) for seed in seeds]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=nJobs) as executor:
            futures = list(executor.map(trainSingleSeed, args))
        
        for result in futures:
            if result is not None:
                results.append(result)
                print(f"   Seed {result.seed}: Acc={result.bestAccuracy:.2f}%, "
                      f"F1={result.bestF1:.2f}%")
    
    if not results:
        return {'error': 'All seeds failed'}
    
    # Aggregate
    summary = {
        'model': modelName,
        'dataset': dataset,
        'nSeeds': len(results),
        'seeds': [r.seed for r in results],
        'meanAccuracy': np.mean([r.bestAccuracy for r in results]),
        'stdAccuracy': np.std([r.bestAccuracy for r in results]),
        'meanF1': np.mean([r.bestF1 for r in results]),
        'stdF1': np.std([r.bestF1 for r in results]),
        'meanPrecision': np.mean([r.bestPrecision for r in results]),
        'stdPrecision': np.std([r.bestPrecision for r in results]),
        'meanRecall': np.mean([r.bestRecall for r in results]),
        'stdRecall': np.std([r.bestRecall for r in results]),
        'parameters': results[0].parameters,
        'macs': results[0].macs,
        'sizeMb': results[0].sizeMb,
        'latencyMs': results[0].latencyMs,
        'throughput': results[0].throughput,
        'results': [asdict(r) for r in results]
    }
    
    print(f"\n   {modelName.upper()} on {dataset.upper()} Summary:")
    print(f"      Accuracy:   {summary['meanAccuracy']:.2f}% ± {summary['stdAccuracy']:.2f}%")
    print(f"      F1 Score:   {summary['meanF1']:.2f}% ± {summary['stdF1']:.2f}%")
    print(f"      Parameters: {summary['parameters']:,}")
    print(f"      MACs:       {summary['macs']:,}")
    print(f"      Size:       {summary['sizeMb']:.3f} MB")
    print(f"      Latency:    {summary['latencyMs']:.2f} ms")
    
    return summary


def trainAllDatasets(
    modelName: str,
    nSeeds: int = N_SEEDS,
    epochs: int = TRAINING_EPOCHS,
    nJobs: int = N_PARALLEL_JOBS
) -> Dict[str, Any]:
    """Train on all datasets."""
    
    allResults = {}
    
    for dataset in DATASET_SPECS.keys():
        summary = trainDatasetMultiSeed(modelName, dataset, nSeeds, epochs, nJobs)
        allResults[dataset] = summary
    
    # Save
    resultsDir = Path("results/training")
    resultsDir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outPath = resultsDir / f"{modelName}_all_{timestamp}.json"
    
    def convertTypes(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convertTypes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convertTypes(v) for v in obj]
        return obj
    
    with open(outPath, 'w') as f:
        json.dump(convertTypes(allResults), f, indent=2)
    
    print(f"\nResults saved: {outPath}")
    
    return allResults


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Training for Baseline Models")
    parser.add_argument('--model', '-m', type=str, required=True,
                        choices=['tinierhar', 'tinyhar', 'lightdeepconvlstm', 'deepconvlstm', 'all'],
                        help='Baseline model to train')
    parser.add_argument('--dataset', '-d', type=str, default='all',
                        choices=['ucihar', 'motionsense', 'wisdm', 'pamap2', 'opportunity', 'unimib', 'skoda', 'daphnet', 'all'],
                        help='Dataset (default: all)')
    parser.add_argument('--seeds', '-s', type=int, default=N_SEEDS,
                        help=f'Number of random seeds (default: {N_SEEDS})')
    parser.add_argument('--epochs', '-e', type=int, default=TRAINING_EPOCHS,
                        help=f'Training epochs (default: {TRAINING_EPOCHS})')
    parser.add_argument('--patience', '-p', type=int, default=EARLY_STOPPING_PATIENCE,
                        help=f'Early stopping patience (default: {EARLY_STOPPING_PATIENCE})')
    parser.add_argument('--n-jobs', '-j', type=int, default=N_PARALLEL_JOBS,
                        help=f'Parallel workers (default: {N_PARALLEL_JOBS}, use 1 for GPU)')
    
    args = parser.parse_args()
    
    # Override parallel jobs for GPU (force sequential)
    nJobs = 1 if DEVICE.type == 'cuda' else args.n_jobs
    if nJobs != args.n_jobs:
        print(f"GPU detected: Forcing sequential training (n_jobs=1)")
    
    models = ['tinierhar', 'tinyhar', 'lightdeepconvlstm', 'deepconvlstm'] if args.model == 'all' else [args.model]
    
    for model in models:
        if args.dataset == 'all':
            trainAllDatasets(model, args.seeds, args.epochs, nJobs)
        else:
            summary = trainDatasetMultiSeed(model, args.dataset, args.seeds, args.epochs, nJobs)
            
            resultsDir = Path("results/training")
            resultsDir.mkdir(parents=True, exist_ok=True)
            outPath = resultsDir / f"{model}_{args.dataset}.json"
            with open(outPath, 'w') as f:
                json.dump(summary, f, indent=2, default=float)
            print(f"\nResults saved: {outPath}")


if __name__ == '__main__':
    main()

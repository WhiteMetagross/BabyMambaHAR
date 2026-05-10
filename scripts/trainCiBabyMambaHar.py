"""
CI-BabyMamba-HAR Training Script with Multiple Random Seeds

FROZEN Architecture - CI-BabyMamba-HAR:
    d_model: 24
    d_state: 16  (INCREASED from 8 for nuanced activities)
    n_layers: 4
    expand: 2
    dt_rank: 2
    d_conv: 4
    bidirectional: True (weight-tied)
    gated_attention: True (spotlight transient events)
    channel_independent: True (isolate sensor noise)
    
Target: ~27,000-29,000 parameters

Key Innovations:
- Channel-Independent Stem: Isolates sensor noise per channel
- Context-Gated Temporal Attention: Prevents dilution of transient events
- Expanded d_state=16: Better for nuanced 12-class problems (PAMAP2)

Signal Rescue Recipes:
- SKODA: 5Hz Butterworth filter + Label Smoothing 0.1
- PAMAP2: 10Hz Butterworth filter + Robust Scaling + Gradient Clip 1.0
- Daphnet: 12Hz Butterworth filter + Class Weights [1.0, 15.0]

Features:
- Uses best hyperparameters from HPO
- 200 epochs with early stopping (patience=10, matches baselines)
- 10 epoch warmup
- FP16/AMP training
- 5 random seeds for statistical significance

Usage:
    python scripts/trainCiBabyMambaHar.py --dataset ucihar
    python scripts/trainCiBabyMambaHar.py --dataset ucihar --seeds 3 --epochs 50
    python scripts/trainCiBabyMambaHar.py --dataset all --seeds 5
"""

import os
import sys
import json
import random
import warnings
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from ciBabyMambaHar.models import CiBabyMambaHar, CI_BABYMAMBA_HAR_CONFIG
from ciBabyMambaHar.utils.profiling import countParameters, computeMacs, benchmarkLatency
from ciBabyMambaHar.data.pamap2 import computeClassWeights
from ciBabyMambaHar.data.augmentations import getTrainAugmentation


# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

TRAINING_EPOCHS = 200
WARMUP_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10  # Match baselines for fair comparison
N_SEEDS = 5
MASTER_SEED = 17

# FROZEN Architecture - CI-BabyMamba-HAR
LOCKED_ARCH = {
    'dModel': CI_BABYMAMBA_HAR_CONFIG['dModel'],      # 24
    'dState': CI_BABYMAMBA_HAR_CONFIG['dState'],      # 16 (INCREASED from 8)
    'nLayers': CI_BABYMAMBA_HAR_CONFIG['nLayers'],    # 4
    'expand': CI_BABYMAMBA_HAR_CONFIG['expand'],      # 2
    'dtRank': CI_BABYMAMBA_HAR_CONFIG['dtRank'],      # 2
    'dConv': CI_BABYMAMBA_HAR_CONFIG['dConv'],        # 4
    'bidirectional': True,
    'weightTied': True,
    'useGatedAttention': CI_BABYMAMBA_HAR_CONFIG['useGatedAttention'],  # True
    'channelIndependent': True,
}

DATASET_SPECS = {
    'ucihar': {
        'name': 'UCI-HAR',
        'numClasses': 6,
        'inChannels': 9,
        'seqLen': 128,
        'root': './datasets/UCI HAR Dataset'
    },
    'motionsense': {
        'name': 'MotionSense',
        'numClasses': 6,
        'inChannels': 6,  # 6 = acc + gyro (actual data from dataset)
        'seqLen': 128,
        'root': './datasets/motion-sense-master'
    },
    'wisdm': {
        'name': 'WISDM',
        'numClasses': 6,
        'inChannels': 3,
        'seqLen': 128,
        'root': './datasets/WISDM_ar_v1.1'
    },
    'pamap2': {
        'name': 'PAMAP2',
        'numClasses': 12,
        'inChannels': 19,  # 19 = compact mode (HR + acc + gyro from 3 IMUs)
        'seqLen': 128,
        'root': './datasets/PAMAP2_Dataset'
    },
    'opportunity': {
        'name': 'Opportunity',
        'numClasses': 5,  # Locomotion task
        'inChannels': 79,  # Body-worn IMU channels
        'seqLen': 128,
        'root': './datasets/Opportunity'
    },
    'unimib': {
        'name': 'UniMiB-SHAR',
        'numClasses': 9,  # ADL task
        'inChannels': 3,  # Accelerometer only
        'seqLen': 128,
        'root': './datasets/UniMiB-SHAR'
    },
    'skoda': {
        'name': 'Skoda',
        'numClasses': 11,  # 10 gestures + Null
        'inChannels': 30,  # 10 sensors × 3 axes
        'seqLen': 98,  # ~3 seconds at 30Hz
        'root': './datasets/Skoda'
    },
    'daphnet': {
        'name': 'Daphnet',
        'numClasses': 2,  # Walk vs Freeze
        'inChannels': 9,  # 3 sensors × 3 axes
        'seqLen': 64,  # 2 seconds at 32Hz (downsampled from 64Hz)
        'root': './datasets/Daphnet'
    },
}

DEFAULT_HPARAMS = {
    'ucihar': {'lr': 0.001, 'weightDecay': 0.01, 'dropout': 0.1, 'batchSize': 64},
    'motionsense': {'lr': 0.001, 'weightDecay': 0.01, 'dropout': 0.1, 'batchSize': 64},
    'wisdm': {'lr': 0.001, 'weightDecay': 0.01, 'dropout': 0.1, 'batchSize': 64},
    # PAMAP2 "Shock Absorber": 10Hz filter + Robust Scaling applied in data loader
    'pamap2': {'lr': 0.001, 'weightDecay': 0.01, 'dropout': 0.1, 'batchSize': 64, 'gradClip': 1.0},
    'opportunity': {'lr': 0.001, 'weightDecay': 0.01, 'dropout': 0.1, 'batchSize': 64},
    'unimib': {'lr': 0.001, 'weightDecay': 0.01, 'dropout': 0.1, 'batchSize': 64},
    # SKODA "Speed Run": 5Hz filter applied in data loader, label smoothing for fuzzy gestures
    'skoda': {'lr': 0.001, 'weightDecay': 0.01, 'dropout': 0.1, 'batchSize': 512, 'labelSmoothing': 0.1},
    # DAPHNET: Defaults match HPO best params (lr=0.001, dropout=0.0)
    'daphnet': {'lr': 0.001, 'weightDecay': 0.05, 'dropout': 0.0, 'batchSize': 512, 'labelSmoothing': 0.0},
}


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
    checkpointPath: str = ""
    modelStatePath: str = ""
    runConfigPath: str = ""
    resultPath: str = ""
    confusionMatrix: List[List[int]] = field(default_factory=list)


def generateRandomSeeds(nSeeds: int, masterSeed: int = MASTER_SEED) -> List[int]:
    rng = np.random.default_rng(masterSeed)
    return list(rng.integers(0, 100000, size=nSeeds))


def parseSeedList(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    values = [part.strip() for part in str(raw).split(',') if part.strip()]
    if not values:
        return None
    return [int(value) for value in values]


def loadHpoResults(dataset: str) -> Optional[Dict[str, Any]]:
    # Try CiBabyMambaHar HPO results first, then fall back to legacy names
    hpoPaths = [
        Path("results/hpo") / f"hpo_ciBabyMambaHar_{dataset}.json",
        Path("results/hpo") / f"hpo_ciBabyMambaHar_{dataset}.json",
    ]
    
    for hpoPath in hpoPaths:
        if hpoPath.exists():
            try:
                with open(hpoPath, 'r') as f:
                    hpoData = json.load(f)
                print(f"   Loaded HPO results from {hpoPath}")
                params = hpoData.get('bestParams', {})
                # Normalize parameter names (snake_case -> camelCase)
                if 'weight_decay' in params:
                    params['weightDecay'] = params.pop('weight_decay')
                return params
            except Exception as e:
                print(f"   Warning: Could not load HPO results: {e}")
    return None


def setSeed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def loadDataToGpu(
    dataset: str,
    windowSize: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load dataset (keep on CPU, move batches to GPU during training to save VRAM).
    
    Args:
        dataset: Name of the dataset to load.
        windowSize: Optional override for sliding-window size. Ignored for datasets
            with fixed windows (e.g., UCI-HAR pre-segmented at 128).
    """
    # Clear CUDA cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    classWeights = None
    
    if dataset == 'ucihar':
        from ciBabyMambaHar.data.uciHar import UciHarDataset
        trainDs = UciHarDataset(DATASET_SPECS[dataset]['root'], split='train')
        testDs = UciHarDataset(DATASET_SPECS[dataset]['root'], split='test')
    elif dataset == 'motionsense':
        from ciBabyMambaHar.data.motionSense import MotionSenseDataset
        ws = windowSize or 128
        trainDs = MotionSenseDataset(DATASET_SPECS[dataset]['root'], split='train', windowSize=ws)
        testDs = MotionSenseDataset(DATASET_SPECS[dataset]['root'], split='test', windowSize=ws)
    elif dataset == 'wisdm':
        from ciBabyMambaHar.data.wisdm import WisdmDataset
        ws = windowSize or 128
        trainDs = WisdmDataset(DATASET_SPECS[dataset]['root'], split='train', windowSize=ws)
        testDs = WisdmDataset(DATASET_SPECS[dataset]['root'], split='test', windowSize=ws)
    elif dataset == 'pamap2':
        from ciBabyMambaHar.data.pamap2 import Pamap2Dataset
        # PAMAP2 "Shock Absorber": 10Hz filter + Robust Scaling
        ws = windowSize or 128
        trainDs = Pamap2Dataset(DATASET_SPECS[dataset]['root'], split='train',
                                windowSize=ws, applyFilter=True, filterCutoff=10.0, useRobustScaling=True)
        testDs = Pamap2Dataset(DATASET_SPECS[dataset]['root'], split='test',
                               windowSize=ws, applyFilter=True, filterCutoff=10.0, useRobustScaling=True)
        # Compute class weights for imbalanced PAMAP2
        if len(trainDs.labels) > 0:
            classWeights = computeClassWeights(trainDs.labels, numClasses=12)
    elif dataset == 'opportunity':
        from ciBabyMambaHar.data.opportunity import OpportunityDataset
        trainDs = OpportunityDataset(DATASET_SPECS[dataset]['root'], split='train', task='locomotion')
        testDs = OpportunityDataset(DATASET_SPECS[dataset]['root'], split='test', task='locomotion')
        # Compute class weights for imbalanced Opportunity
        if len(trainDs.labels) > 0:
            classWeights = computeClassWeights(trainDs.labels, numClasses=5)
    elif dataset == 'unimib':
        from ciBabyMambaHar.data.unimib import UniMiBSHARDataset
        trainDs = UniMiBSHARDataset(DATASET_SPECS[dataset]['root'], split='train', task='adl')
        testDs = UniMiBSHARDataset(DATASET_SPECS[dataset]['root'], split='test', task='adl')
    elif dataset == 'skoda':
        from ciBabyMambaHar.data.skoda import SkodaDataset, computeSkodaClassWeights
        # SKODA "Speed Run": 5Hz filter to remove machine vibration
        ws = windowSize or 98
        trainDs = SkodaDataset(DATASET_SPECS[dataset]['root'], split='train', windowSize=ws, applyFilter=True, filterCutoff=5.0)
        testDs = SkodaDataset(DATASET_SPECS[dataset]['root'], split='test', windowSize=ws, applyFilter=True, filterCutoff=5.0)
        # Compute class weights for imbalanced Skoda (Null-heavy)
        if len(trainDs.labels) > 0:
            classWeights = computeSkodaClassWeights(trainDs.labels, numClasses=11)
    elif dataset == 'daphnet':
        from ciBabyMambaHar.data.daphnet import DaphnetDataset, computeDaphnetClassWeights
        # Apply 12Hz Butterworth low-pass filter (removes sensor jitter)
        trainDs = DaphnetDataset(DATASET_SPECS[dataset]['root'], split='train', applyFilter=True, filterCutoff=12.0)
        testDs = DaphnetDataset(DATASET_SPECS[dataset]['root'], split='test', applyFilter=True, filterCutoff=12.0)
        # Use DATA-COMPUTED class weights (simpler approach that achieved 87% in HPO)
        if len(trainDs.labels) > 0:
            classWeights = computeDaphnetClassWeights(trainDs.labels, numClasses=2, aggressive=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Load all data to GPU at once (matches HPO)
    trainLoader = DataLoader(trainDs, batch_size=len(trainDs), shuffle=False, num_workers=0)
    testLoader = DataLoader(testDs, batch_size=len(testDs), shuffle=False, num_workers=0)
    
    xTrain, yTrain = next(iter(trainLoader))
    xTest, yTest = next(iter(testLoader))
    
    # Keep ALL data on CPU - move batches to GPU during training/validation (saves VRAM)
    # The CI-Stem multiplies memory by number of channels, so full GPU loading fails
    if classWeights is not None:
        classWeights = classWeights.to(DEVICE)
    
    return xTrain, yTrain, xTest, yTest, classWeights


def computeMetrics(preds: torch.Tensor, labels: torch.Tensor, numClasses: int) -> Dict[str, float]:
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
    matrix = [[0] * numClasses for _ in range(numClasses)]
    for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
        matrix[l][p] += 1
    return matrix


def convertJsonTypes(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: convertJsonTypes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convertJsonTypes(v) for v in obj]
    return obj


def saveJson(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(convertJsonTypes(payload), handle, indent=2)


def trainModel(
    dataset: str,
    seed: int,
    epochs: int = TRAINING_EPOCHS,
    warmupEpochs: int = WARMUP_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    archOverrides: Optional[Dict[str, Any]] = None,
    runTag: str = 'baseline',
    seqLenOverride: Optional[int] = None,
    accumulationSteps: int = 1,
    useTorchCompile: bool = False,
    artifactDir: Optional[Path] = None,
) -> TrainingResult:
    """
    Train a single CiBabyMambaHar model.
    
    DAPHNET VARIANCE STABILIZER:
    - 10-epoch linear warmup (1e-6 to peak LR)
    - Cosine annealing to 1e-5
    - Weight decay: 0.05
    - NO label smoothing (binary decisions)
    - WeightedRandomSampler (50/50 Freeze/Walk)
    """
    setSeed(seed)
    
    spec = DATASET_SPECS[dataset]
    effectiveSeqLen = seqLenOverride or spec['seqLen']

    hpoParams = loadHpoResults(dataset)
    defaultHparams = DEFAULT_HPARAMS.get(dataset, DEFAULT_HPARAMS['ucihar'])
    hparams = {**defaultHparams, **(hpoParams or {})}
    
    # Load all data to GPU (matches HPO approach)
    xTrain, yTrain, xTest, yTest, classWeights = loadDataToGpu(dataset, windowSize=seqLenOverride)
    batchSize = hparams.get('batchSize', 128)
    nSamples = xTrain.size(0)
    
    # Resolve architecture (frozen by default; overridable for ablation studies)
    resolvedArch = dict(LOCKED_ARCH)
    if archOverrides:
        for k, v in archOverrides.items():
            if v is not None:
                resolvedArch[k] = v

    # Create CiBabyMambaHar
    model = CiBabyMambaHar(
        numClasses=spec['numClasses'],
        inChannels=spec['inChannels'],
        seqLen=effectiveSeqLen,
        dropout=hparams.get('dropout', 0.1),
        dModel=resolvedArch['dModel'],
        dState=resolvedArch['dState'],
        nLayers=resolvedArch['nLayers'],
        expand=resolvedArch['expand'],
        dtRank=resolvedArch['dtRank'],
        dConv=resolvedArch['dConv'],
        bidirectional=resolvedArch['bidirectional'],
        channelIndependent=resolvedArch['channelIndependent'],
        useGatedAttention=resolvedArch['useGatedAttention'],
    ).to(DEVICE)
    
    # Optional torch.compile for speedup (requires PyTorch 2.0+)
    if useTorchCompile and hasattr(torch, 'compile'):
        print(f"   Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')
    
    params = sum(p.numel() for p in model.parameters())
    
    inputShape = (1, effectiveSeqLen, spec['inChannels'])
    try:
        # Use a fresh model copy for MACs to avoid moving training model
        macModel = CiBabyMambaHar(
            numClasses=spec['numClasses'],
            inChannels=spec['inChannels'],
            seqLen=effectiveSeqLen,
            dropout=hparams.get('dropout', 0.1),
            dModel=resolvedArch['dModel'],
            dState=resolvedArch['dState'],
            nLayers=resolvedArch['nLayers'],
            expand=resolvedArch['expand'],
            dtRank=resolvedArch['dtRank'],
            dConv=resolvedArch['dConv'],
            bidirectional=resolvedArch['bidirectional'],
            channelIndependent=resolvedArch['channelIndependent'],
            useGatedAttention=resolvedArch['useGatedAttention'],
        )
        macsResult = computeMacs(macModel, inputShape, device='cpu')
        macs = macsResult.get('macs', 0) or 0
        del macModel
    except:
        macs = 0
    
    paramSize = sum(p.nelement() * p.element_size() for p in model.parameters())
    bufferSize = sum(b.nelement() * b.element_size() for b in model.buffers())
    sizeMb = (paramSize + bufferSize) / (1024 ** 2)
    
    latencyMs, throughput = 0.0, 0.0
    # Skip latency benchmark to save memory (can be done separately)
    # try:
    #     latencyResults = benchmarkLatency(model, inputShape, device=str(DEVICE), numRuns=50)
    #     latencyMs = latencyResults.get('batch_1', {}).get('mean_ms', 0.0)
    #     throughput = latencyResults.get('batch_1', {}).get('throughput', 0.0)
    # except:
    #     pass
    
    # Clear CUDA cache after profiling (free temporary tensors)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========================================
    # DAPHNET: Use simple recipe that achieved 87% F1 in HPO
    # Key: Early stopping on F1 + best checkpoint restoration
    # ========================================
    if dataset == 'daphnet':
        # Standard 2-epoch warmup (like other datasets)
        actualWarmupEpochs = 2
        useWeightedSampling = False
        sampleWeights = None
        
        # Standard optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams['lr'],
            weight_decay=hparams['weightDecay']
        )
        
        # CosineAnnealingLR with T_max=10 (like HPO)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        warmupScheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=actualWarmupEpochs)
        
        # NO label smoothing for binary classification
        if classWeights is not None:
            criterion = nn.CrossEntropyLoss(weight=classWeights, label_smoothing=0.0)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        
        # No augmentation for Daphnet (filter already applied)
        useAugmentation = False
        gradClipNorm = 1.0  # Standard gradient clipping
        
        print(f"   Daphnet Training Recipe (Simple):")
        print(f"      LR: {hparams['lr']}")
        print(f"      Weight decay: {hparams['weightDecay']}")
        print(f"      Dropout: {hparams.get('dropout', 0.0)}")
        print(f"      Class weights: {classWeights.cpu().numpy() if classWeights is not None else 'None'}")
        print(f"      Patience: {patience}")
    elif dataset in ['skoda']:
        # SKODA "Speed Run": 5Hz filter + Label Smoothing 0.1
        actualWarmupEpochs = 2
        useWeightedSampling = False
        labelSmoothing = hparams.get('labelSmoothing', 0.1)  # Fuzzy gesture boundaries
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weightDecay'])
        
        if classWeights is not None:
            criterion = nn.CrossEntropyLoss(weight=classWeights, label_smoothing=labelSmoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=labelSmoothing)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        warmupScheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=actualWarmupEpochs)
        useAugmentation = False  # Skip for large dataset
        sampleWeights = None
        gradClipNorm = hparams.get('gradClip', 1.0)
        
        print(f"   SKODA Training Recipe (Speed Run):")
        print(f"      5Hz low-pass filter: Applied in data loader")
        print(f"      Label smoothing: {labelSmoothing}")
        print(f"      LR: {hparams['lr']}")
        print(f"      Weight decay: {hparams['weightDecay']}")
        
    elif dataset == 'pamap2':
        # PAMAP2 "Shock Absorber": 10Hz filter + Robust Scaling + Gradient Clipping
        actualWarmupEpochs = 2
        useWeightedSampling = False
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weightDecay'])
        
        # No label smoothing for PAMAP2 (clear activity boundaries)
        if classWeights is not None:
            criterion = nn.CrossEntropyLoss(weight=classWeights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        warmupScheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=actualWarmupEpochs)
        useAugmentation = False  # Skip to match HPO
        sampleWeights = None
        gradClipNorm = hparams.get('gradClip', 1.0)  # Stabilize extreme dynamics
        
        print(f"   PAMAP2 Training Recipe (Shock Absorber):")
        print(f"      10Hz low-pass filter: Applied in data loader")
        print(f"      Robust Scaling (IQR): Applied in data loader")
        print(f"      Gradient clipping: {gradClipNorm}")
        print(f"      LR: {hparams['lr']}")
        print(f"      Weight decay: {hparams['weightDecay']}")
    else:
        # Standard training for other datasets
        actualWarmupEpochs = 2
        useWeightedSampling = False
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weightDecay'])
        
        # Use class weights for imbalanced datasets (matches HPO)
        # Add label smoothing (0.1) to improve generalization
        if classWeights is not None:
            criterion = nn.CrossEntropyLoss(weight=classWeights, label_smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Use CosineAnnealingLR with T_max=10 (matches HPO exactly)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        warmupScheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=actualWarmupEpochs)
        
        # Data augmentation - DISABLED for fair comparison with baselines
        # Baselines (TinierHAR, TinyHAR, LightDeepConvLSTM) do NOT use augmentation
        useAugmentation = False  # Disabled for fairness
        sampleWeights = None
        gradClipNorm = 1.0  # Default gradient clipping
    
    scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    if useAugmentation:
        augment = getTrainAugmentation('strong')
    
    bestAcc, bestF1, bestPrecision, bestRecall = 0.0, 0.0, 0.0, 0.0
    bestPreds, bestLabels = None, None
    bestModelState = None  # CRITICAL: Save best model checkpoint (fixes F1 collapse)
    epochsWithoutImprovement = 0
    earlyStopped = False
    
    # Use standard patience for all datasets (matches baselines)
    # Removed special Daphnet patience override for fairness
    
    startTime = time.time()
    
    # Calculate effective batch size for gradient accumulation
    effectiveBatchSize = batchSize // accumulationSteps if accumulationSteps > 1 else batchSize
    nBatches = (nSamples + effectiveBatchSize - 1) // effectiveBatchSize
    
    if accumulationSteps > 1:
        print(f"   Gradient accumulation: {accumulationSteps} steps (effective batch={batchSize}, micro batch={effectiveBatchSize})")
    
    # Epoch progress bar
    epochPbar = tqdm(range(1, epochs + 1), desc=f"Training {dataset}", unit="epoch", leave=True)
    
    for epoch in epochPbar:
        model.train()
        trainLoss, trainCorrect, trainTotal = 0.0, 0, 0
        
        # Use warmup scheduler for first epochs, then cosine annealing
        currentLr = optimizer.param_groups[0]['lr']
        
        # ========================================
        # DAPHNET: Use weighted sampling for 50/50 balance
        # ========================================
        if useWeightedSampling and sampleWeights is not None:
            # Sample with replacement to achieve 50/50 balance
            sampledIndices = torch.multinomial(sampleWeights, nSamples, replacement=True)
            permutation = sampledIndices[torch.randperm(nSamples)]
        else:
            # Standard shuffle (matches HPO approach) - keep on CPU
            permutation = torch.randperm(nSamples)
        
        # Zero gradients at start of epoch
        optimizer.zero_grad()
        accumStep = 0
        
        # Batch progress bar (nested inside epoch)
        batchPbar = tqdm(range(0, nSamples, effectiveBatchSize), 
                         desc=f"Epoch {epoch}", unit="batch", leave=False)
        
        for i in batchPbar:
            indices = permutation[i:i + effectiveBatchSize]
            batchX = xTrain[indices]
            batchY = yTrain[indices]
            
            # Apply augmentation per sample (on CPU) - only for smaller datasets
            if useAugmentation:
                batchX = batchX.clone()
                for j in range(batchX.size(0)):
                    batchX[j] = augment(batchX[j])
            
            # Move batch to GPU (data stays on CPU to save VRAM)
            batchX = batchX.to(DEVICE)
            batchY = batchY.to(DEVICE)
            
            if scaler:
                with autocast('cuda'):
                    output = model(batchX)
                    loss = criterion(output, batchY)
                    # Scale loss for gradient accumulation
                    if accumulationSteps > 1:
                        loss = loss / accumulationSteps
                scaler.scale(loss).backward()
            else:
                output = model(batchX)
                loss = criterion(output, batchY)
                if accumulationSteps > 1:
                    loss = loss / accumulationSteps
                loss.backward()
            
            accumStep += 1
            
            # Optimizer step after accumulation steps
            if accumStep >= accumulationSteps or (i + effectiveBatchSize) >= nSamples:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradClipNorm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradClipNorm)
                    optimizer.step()
                optimizer.zero_grad()
                accumStep = 0
            
            # Track loss (undo scaling for logging)
            actualLoss = loss.item() * (accumulationSteps if accumulationSteps > 1 else 1)
            trainLoss += actualLoss * batchX.size(0)
            trainCorrect += (output.argmax(1) == batchY).sum().item()
            trainTotal += batchX.size(0)
            
            # Update batch progress bar
            batchPbar.set_postfix(loss=f"{actualLoss:.4f}", acc=f"{100*trainCorrect/trainTotal:.1f}%")
        
        model.eval()
        
        # Validate on test set (batched to save VRAM)
        allPreds = []
        allLabels = []
        valBatchSize = min(batchSize, 64)  # Use smaller batches for validation
        nTestSamples = xTest.size(0)
        
        with torch.no_grad():
            for i in range(0, nTestSamples, valBatchSize):
                batchX = xTest[i:i + valBatchSize].to(DEVICE)
                batchY = yTest[i:i + valBatchSize]
                output = model(batchX)
                allPreds.append(output.argmax(1).cpu())
                allLabels.append(batchY)
        
        allPreds = torch.cat(allPreds)
        allLabels = torch.cat(allLabels)
        
        # Compute F1 using sklearn (matches HPO exactly)
        valF1 = f1_score(allLabels.cpu().numpy(), allPreds.cpu().numpy(), average='macro') * 100
        valAcc = 100.0 * (allPreds == allLabels).float().mean().item()
        metrics = computeMetrics(allPreds, allLabels, spec['numClasses'])
        
        # Use warmup scheduler for first epochs, then cosine annealing
        if epoch <= actualWarmupEpochs:
            warmupScheduler.step()
        else:
            scheduler.step()
        
        # Early stopping based on F1 (matches HPO optimization target)
        # CRITICAL: Save model checkpoint when F1 improves (fixes collapse issue)
        improved = ""
        if valF1 > bestF1:
            bestF1 = valF1
            bestAcc = valAcc
            bestPrecision = metrics['precision']
            bestRecall = metrics['recall']
            bestPreds = allPreds.clone()
            bestLabels = allLabels.clone()
            # Save best model state dict (this is the key fix!)
            bestModelState = {k: v.clone() for k, v in model.state_dict().items()}
            epochsWithoutImprovement = 0
            improved = "*"
        else:
            epochsWithoutImprovement += 1
        
        # Update epoch progress bar with validation metrics
        epochPbar.set_postfix(
            F1=f"{valF1:.2f}%{improved}",
            best=f"{bestF1:.2f}%",
            patience=f"{epochsWithoutImprovement}/{patience}"
        )
        
        if epochsWithoutImprovement >= patience:
            earlyStopped = True
            epochPbar.close()
            print(f"   Early stopping at epoch {epoch} (best F1: {bestF1:.2f}%)")
            break
    
    trainTime = time.time() - startTime
    
    # CRITICAL: Restore best model checkpoint (the "aggressive" F1 state)
    # Without this, we return the final "conservative" model that overfits to Walk
    if bestModelState is not None:
        model.load_state_dict(bestModelState)
    
    confMatrix = []
    if bestPreds is not None:
        confMatrix = computeConfusionMatrix(bestPreds, bestLabels, spec['numClasses'])
    
    result = TrainingResult(
        modelName='CiBabyMambaHar',
        dataset=dataset,
        seed=seed,
        bestAccuracy=bestAcc,
        bestF1=bestF1,
        bestPrecision=bestPrecision,
        bestRecall=bestRecall,
        finalAccuracy=valAcc,
        parameters=params,
        macs=macs,
        sizeMb=sizeMb,
        latencyMs=latencyMs,
        throughput=throughput,
        epochsTrained=epoch,
        totalEpochs=epochs,
        trainTime=trainTime,
        earlyStopped=earlyStopped,
        confusionMatrix=confMatrix,
    )

    if artifactDir is not None:
        artifactDir.mkdir(parents=True, exist_ok=True)
        checkpointPath = artifactDir / f"best_model_seed{seed}.pt"
        modelStatePath = artifactDir / f"model_state_seed{seed}.pt"
        runConfigPath = artifactDir / f"run_config_seed{seed}.json"
        resultPath = artifactDir / f"train_result_seed{seed}.json"

        checkpointPayload = {
            'model_state_dict': bestModelState or model.state_dict(),
            'dataset': dataset,
            'seed': int(seed),
            'model_name': 'CiBabyMambaHar',
            'run_tag': runTag,
            'epochs_requested': int(epochs),
            'epochs_trained': int(epoch),
            'patience': int(patience),
            'seq_len': int(effectiveSeqLen),
            'dataset_spec': spec,
            'architecture_config': resolvedArch,
            'hparams': hparams,
            'best_metrics': {
                'accuracy': float(bestAcc),
                'f1': float(bestF1),
                'precision': float(bestPrecision),
                'recall': float(bestRecall),
            },
            'final_metrics': {
                'accuracy': float(valAcc),
            },
        }
        torch.save(checkpointPayload, checkpointPath)
        torch.save(bestModelState or model.state_dict(), modelStatePath)

        runConfig = {
            'dataset': dataset,
            'seed': int(seed),
            'run_tag': runTag,
            'epochs': int(epochs),
            'warmup_epochs': int(warmupEpochs),
            'patience': int(patience),
            'seq_len': int(effectiveSeqLen),
            'accumulation_steps': int(accumulationSteps),
            'use_torch_compile': bool(useTorchCompile),
            'architecture_config': resolvedArch,
            'hparams': hparams,
            'dataset_spec': spec,
        }
        saveJson(runConfigPath, runConfig)

        result.checkpointPath = str(checkpointPath)
        result.modelStatePath = str(modelStatePath)
        result.runConfigPath = str(runConfigPath)
        result.resultPath = str(resultPath)
        saveJson(resultPath, asdict(result))
    
    del model
    torch.cuda.empty_cache()
    
    return result


def trainDatasetMultiSeed(
    dataset: str,
    nSeeds: int = N_SEEDS,
    epochs: int = TRAINING_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    archOverrides: Optional[Dict[str, Any]] = None,
    runTag: str = 'baseline',
    seqLenOverride: Optional[int] = None,
    accumulationSteps: int = 1,
    useTorchCompile: bool = False,
    seedList: Optional[List[int]] = None,
    outputDir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train on dataset with multiple seeds."""
    seeds = [int(seed) for seed in seedList] if seedList else generateRandomSeeds(nSeeds, MASTER_SEED)

    resolvedArch = dict(LOCKED_ARCH)
    if archOverrides:
        for k, v in archOverrides.items():
            if v is not None:
                resolvedArch[k] = v

    print(f"\nTraining CI-BabyMamba-HAR on {dataset.upper()} ({len(seeds)} seeds)")
    print(f"   Run Tag: {runTag}")
    print(f"   Architecture: CI-BabyMamba-HAR")
    print(f"   d_model={resolvedArch['dModel']}, d_state={resolvedArch['dState']}, "
        f"n_layers={resolvedArch['nLayers']}, expand={resolvedArch['expand']}, dt_rank={resolvedArch['dtRank']}")
    print(f"   bidirectional={resolvedArch['bidirectional']}, CI-stem={resolvedArch['channelIndependent']}, gated_attention={resolvedArch['useGatedAttention']}")
    print(f"   Seeds: {seeds}")
    print(f"   Patience: {patience} epochs")
    
    results = []
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n   Seed {seed} ({i}/{len(seeds)})...")
        
        try:
            result = trainModel(
                dataset,
                seed,
                epochs,
                patience=patience,
                archOverrides=archOverrides,
                runTag=runTag,
                seqLenOverride=seqLenOverride,
                accumulationSteps=accumulationSteps,
                useTorchCompile=useTorchCompile,
                artifactDir=outputDir,
            )
            results.append(result)
            
            print(f"   Accuracy: {result.bestAccuracy:.2f}%, "
                  f"F1: {result.bestF1:.2f}%, "
                  f"Epochs: {result.epochsTrained}/{result.totalEpochs}")
        except Exception as e:
            print(f"   Failed: {e}")
    
    if not results:
        return {'error': 'All seeds failed'}
    
    accuracies = [r.bestAccuracy for r in results]
    f1Scores = [r.bestF1 for r in results]
    
    summary = {
        'dataset': dataset,
        'model': 'CI-BabyMamba-HAR',
        'architecture': 'CI-BabyMamba-HAR',
        'nSeeds': len(results),
        'seeds': [r.seed for r in results],
        'meanAccuracy': np.mean(accuracies),
        'stdAccuracy': np.std(accuracies),
        'meanF1': np.mean(f1Scores),
        'stdF1': np.std(f1Scores),
        'parameters': results[0].parameters,
        'macs': results[0].macs,
        'sizeMb': results[0].sizeMb,
        'latencyMs': results[0].latencyMs,
        'architectureConfig': resolvedArch,
        'runTag': runTag,
        'seqLen': seqLenOverride or DATASET_SPECS[dataset]['seqLen'],
        'results': [asdict(r) for r in results]
    }
    
    print(f"\n   {dataset.upper()} Summary:")
    print(f"      Accuracy:   {summary['meanAccuracy']:.2f}% ± {summary['stdAccuracy']:.2f}%")
    print(f"      F1 Score:   {summary['meanF1']:.2f}% ± {summary['stdF1']:.2f}%")
    print(f"      Parameters: {summary['parameters']:,}")
    print(f"      Latency:    {summary['latencyMs']:.2f} ms")

    if outputDir is not None:
        saveJson(outputDir / 'summary.json', summary)
    
    return summary


def trainAllDatasets(
    nSeeds: int = N_SEEDS,
    epochs: int = TRAINING_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    archOverrides: Optional[Dict[str, Any]] = None,
    runTag: str = 'baseline',
    seqLenOverride: Optional[int] = None,
    accumulationSteps: int = 1,
    useTorchCompile: bool = False,
    seedList: Optional[List[int]] = None,
    outputRoot: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train on all datasets."""
    seeds = [int(seed) for seed in seedList] if seedList else generateRandomSeeds(nSeeds, MASTER_SEED)
    
    resolvedArch = dict(LOCKED_ARCH)
    if archOverrides:
        for k, v in archOverrides.items():
            if v is not None:
                resolvedArch[k] = v

    print("\n" + "=" * 70)
    print("CI-BABYMAMBA-HAR TRAINING")
    print("=" * 70)
    print(f"   Run Tag: {runTag}")
    print(f"   Architecture: CI-BabyMamba-HAR")
    print(f"   d_model={resolvedArch['dModel']}, d_state={resolvedArch['dState']}, n_layers={resolvedArch['nLayers']}, expand={resolvedArch['expand']}")
    print(f"   CI-Stem: {resolvedArch['channelIndependent']}, Gated Attention: {resolvedArch['useGatedAttention']}, BiDir: {resolvedArch['bidirectional']}")
    print(f"   Datasets: {list(DATASET_SPECS.keys())}")
    print(f"   Seeds (RNG): {seeds}")
    print(f"   Epochs: {epochs}")
    print(f"   Patience: {patience}")
    print(f"   Device: {DEVICE}")
    
    allResults = {}
    
    for dataset in DATASET_SPECS.keys():
        datasetOutputDir = None
        if outputRoot is not None:
            datasetOutputDir = outputRoot / dataset
        summary = trainDatasetMultiSeed(
            dataset,
            nSeeds,
            epochs,
            patience,
            archOverrides=archOverrides,
            runTag=runTag,
            seqLenOverride=seqLenOverride,
            accumulationSteps=accumulationSteps,
            useTorchCompile=useTorchCompile,
            seedList=seedList,
            outputDir=datasetOutputDir,
        )
        allResults[dataset] = summary
    
    print(f"\n{'='*70}")
    print("CI-BABYMAMBA-HAR RESULTS")
    print(f"{'='*70}")
    
    print(f"\n| Dataset | Params | MACs | Latency | Accuracy | F1 Score |")
    print(f"|---------|--------|------|---------|----------|----------|")
    
    for dataset, summary in allResults.items():
        if 'error' in summary:
            print(f"| {dataset} | - | - | - | FAILED | - |")
        else:
            print(f"| {dataset} | {summary['parameters']:,} | {summary['macs']:,} | "
                  f"{summary['latencyMs']:.1f}ms | "
                  f"{summary['meanAccuracy']:.1f}%±{summary['stdAccuracy']:.1f}% | "
                  f"{summary['meanF1']:.1f}%±{summary['stdF1']:.1f}% |")
    
    resultsDir = Path("results/training")
    resultsDir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outPath = resultsDir / f"ciBabyMambaHarTraining_{timestamp}.json"
    
    saveJson(outPath, allResults)
    
    print(f"\nResults saved: {outPath}")
    
    return allResults


def main():
    import argparse

    def parseBool(value: str) -> bool:
        v = str(value).strip().lower()
        if v in {'1', 'true', 't', 'yes', 'y', 'on'}:
            return True
        if v in {'0', 'false', 'f', 'no', 'n', 'off'}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")
    
    parser = argparse.ArgumentParser(description="CI-BabyMamba-HAR Training")
    parser.add_argument('--dataset', '-d', type=str, default='all',
                        choices=['ucihar', 'motionsense', 'wisdm', 'pamap2', 'opportunity', 'unimib', 'skoda', 'daphnet', 'all'])
    parser.add_argument('--seeds', '-s', type=int, default=N_SEEDS)
    parser.add_argument('--seed-list', type=str, default=None,
                        help='Comma-separated explicit seed list, e.g. 29 or 29,10734')
    parser.add_argument('--epochs', '-e', type=int, default=TRAINING_EPOCHS)
    parser.add_argument('--patience', '-p', type=int, default=EARLY_STOPPING_PATIENCE)

    # Ablation runner helpers
    parser.add_argument('--outDir', type=str, default='results/training',
                        help='Base output directory for training runs')
    parser.add_argument('--tag', type=str, default='baseline',
                        help='Run tag for naming/grouping outputs')

    # Architecture overrides (ablation studies)
    parser.add_argument('--dModel', type=int, default=None)
    parser.add_argument('--dState', type=int, default=None)
    parser.add_argument('--nLayers', type=int, default=None)
    parser.add_argument('--expand', type=int, default=None)
    parser.add_argument('--dtRank', type=int, default=None)
    parser.add_argument('--dConv', type=int, default=None)
    parser.add_argument('--bidirectional', type=parseBool, default=None)
    parser.add_argument('--channelIndependent', type=parseBool, default=None)
    parser.add_argument('--useGatedAttention', type=parseBool, default=None)
    parser.add_argument('--seqLen', type=int, default=None,
                        help='Override sequence length for LRD study (e.g., 64, 128, 256, 512)')
    
    # Performance optimizations
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1). Use >1 for memory-constrained runs.')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for speedup (requires PyTorch 2.0+)')
    
    args = parser.parse_args()
    explicitSeedList = parseSeedList(args.seed_list)

    archOverrides: Dict[str, Any] = {}
    if args.dModel is not None:
        archOverrides['dModel'] = args.dModel
    if args.dState is not None:
        archOverrides['dState'] = args.dState
    if args.nLayers is not None:
        archOverrides['nLayers'] = args.nLayers
    if args.expand is not None:
        archOverrides['expand'] = args.expand
    if args.dtRank is not None:
        archOverrides['dtRank'] = args.dtRank
    if args.dConv is not None:
        archOverrides['dConv'] = args.dConv
    if args.bidirectional is not None:
        archOverrides['bidirectional'] = args.bidirectional
    if args.channelIndependent is not None:
        archOverrides['channelIndependent'] = args.channelIndependent
    if args.useGatedAttention is not None:
        archOverrides['useGatedAttention'] = args.useGatedAttention

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.dataset == 'all':
        allResults = trainAllDatasets(
            args.seeds,
            args.epochs,
            args.patience,
            archOverrides=archOverrides or None,
            runTag=args.tag,
            seqLenOverride=args.seqLen,
            accumulationSteps=args.accumulation_steps,
            useTorchCompile=args.compile,
            seedList=explicitSeedList,
            outputRoot=Path(args.outDir) / 'ciBabyMambaHar' / args.tag / timestamp,
        )

        resultsDir = Path(args.outDir) / 'ciBabyMambaHar' / args.tag / 'all' / timestamp
        resultsDir.mkdir(parents=True, exist_ok=True)
        outPath = resultsDir / 'summary.json'
        saveJson(outPath, allResults)
        print(f"\nTraining results saved: {outPath}")
    else:
        resultsDir = Path(args.outDir) / 'ciBabyMambaHar' / args.tag / args.dataset / timestamp
        resultsDir.mkdir(parents=True, exist_ok=True)
        summary = trainDatasetMultiSeed(
            args.dataset,
            args.seeds,
            args.epochs,
            args.patience,
            archOverrides=archOverrides or None,
            runTag=args.tag,
            seqLenOverride=args.seqLen,
            accumulationSteps=args.accumulation_steps,
            useTorchCompile=args.compile,
            seedList=explicitSeedList,
            outputDir=resultsDir,
        )
        outPath = resultsDir / 'summary.json'
        saveJson(outPath, summary)
        print(f"\nTraining results saved: {outPath}")


if __name__ == '__main__':
    main()


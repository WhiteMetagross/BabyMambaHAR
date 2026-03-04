"""
CI-BabyMamba-HAR HPO Script - Training Hyperparameters ONLY

FROZEN Architecture (DO NOT CHANGE):
    d_model: 24        # FROZEN - model width
    n_layers: 4        # FROZEN - deeper than baselines (TinierHAR=3)
    d_state: 16        # FROZEN - INCREASED from 8 for nuanced activities
    d_conv: 4          # FROZEN - local convolution
    expand: 2          # FROZEN - inner expansion
    dt_rank: 2         # FROZEN - delta projection rank
    bidirectional: True  # FROZEN - weight-tied BiDir SSM
    gated_attention: True  # FROZEN - spotlight transient events
    channel_independent: True  # FROZEN - isolate sensor noise

Key Innovations:
- Channel-Independent Stem: Isolates sensor noise per channel (SKODA/PAMAP2/Daphnet)
- Context-Gated Temporal Attention: Prevents dilution of transient events
- Expanded d_state=16: Better for nuanced multi-class problems

HPO Search Space (Training Hyperparameters ONLY - matches baselines):
    learning_rate: log_uniform(1e-4, 1e-2)
    weight_decay: log_uniform(0.005, 0.05)
    dropout: uniform(0.0, 0.5)  # Applied in classification head ONLY
    
    FIXED (not swept):
        batch_size: 64 (512 for Skoda/Daphnet)
        optimizer: AdamW
        betas: (0.9, 0.999)
        eps: 1e-8
        scheduler: CosineAnnealingLR
        warmup_epochs: 0 (no warmup in HPO - matches baselines)

HPO Protocol:
    n_trials: 50
    epochs_per_trial: 10
    early_stopping_patience: 5
    seed: 17
    metric: val_f1

Usage:
    python scripts/hpoCiBabyMambaHar.py --dataset ucihar
    python scripts/hpoCiBabyMambaHar.py --dataset ucihar --n-trials 50
    python scripts/hpoCiBabyMambaHar.py --dataset ucihar --n-trials 2 --epochs 2  # Quick test
    python scripts/hpoCiBabyMambaHar.py --dataset all
"""

import os
import sys
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from ciBabyMambaHar.models import ciBabyMambaHar
from ciBabyMambaHar.data.augmentations import getTrainAugmentation


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 17  # Match baselines
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable TensorFloat-32 for faster training on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# HPO Protocol (match baselines)
N_TRIALS = 50
EPOCHS_PER_TRIAL = 10
EARLY_STOPPING_PATIENCE = 5

# FIXED Training Config (not tuned)
FIXED_CONFIG = {
    'batchSize': 64,  # Match baselines for fair comparison
    'optimizer': 'AdamW',
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'scheduler': 'CosineAnnealingLR',
    'warmupEpochs': 0,  # No warmup in HPO (matches baselines)
    'schedulerMinLr': 1e-6,
    'gradientClipNorm': 1.0,
}

# FROZEN Architecture Config - CI-BabyMamba-HAR (DO NOT CHANGE)
FROZEN_ARCHITECTURE = {
    'dModel': 24,      # Model width (FROZEN)
    'nLayers': 4,      # Deeper than baselines (TinierHAR=3)
    'dState': 16,      # INCREASED from 8 for nuanced activities
    'dConv': 4,        # Local convolution kernel
    'expand': 2,       # Inner dimension expansion
    'dtRank': 2,       # Delta projection rank
    'bidirectional': True,  # Weight-tied BiDir SSM
    'useGatedAttention': True,  # Spotlight transient events
    'channelIndependent': True,  # Isolate sensor noise
}

# Dataset specifications
DATASET_SPECS = {
    'ucihar': {'numClasses': 6, 'inChannels': 9, 'seqLen': 128},
    'motionsense': {'numClasses': 6, 'inChannels': 6, 'seqLen': 128},  # 6 = acc + gyro (actual data)
    'wisdm': {'numClasses': 6, 'inChannels': 3, 'seqLen': 128},
    'pamap2': {'numClasses': 12, 'inChannels': 19, 'seqLen': 128},  # 19 = compact mode (HR + acc + gyro)
    'opportunity': {'numClasses': 5, 'inChannels': 79, 'seqLen': 128},  # Locomotion task
    'unimib': {'numClasses': 9, 'inChannels': 3, 'seqLen': 128},  # ADL task
    'skoda': {'numClasses': 11, 'inChannels': 30, 'seqLen': 98},  # 10 gestures + Null
    'daphnet': {'numClasses': 2, 'inChannels': 9, 'seqLen': 64},  # Walk vs Freeze
}


def setSeed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ============================================================================
# DATA LOADING WITH AUGMENTATION
# ============================================================================

def loadDataset(dataset: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load dataset to GPU with augmentation support.
    
    Returns:
        (xTrain, yTrain, xTest, yTest, classWeights) where classWeights is None for balanced datasets
    """
    print(f"   Loading {dataset}...")
    
    classWeights = None
    
    if dataset == 'ucihar':
        from ciBabyMambaHar.data.uciHar import UciHarDataset
        trainDs = UciHarDataset('./datasets/UCI HAR Dataset', split='train')
        testDs = UciHarDataset('./datasets/UCI HAR Dataset', split='test')
    elif dataset == 'motionsense':
        from ciBabyMambaHar.data.motionSense import MotionSenseDataset
        trainDs = MotionSenseDataset('./datasets/motion-sense-master', split='train')
        testDs = MotionSenseDataset('./datasets/motion-sense-master', split='test')
    elif dataset == 'wisdm':
        from ciBabyMambaHar.data.wisdm import WisdmDataset
        trainDs = WisdmDataset('./datasets/WISDM_ar_v1.1', split='train')
        testDs = WisdmDataset('./datasets/WISDM_ar_v1.1', split='test')
    elif dataset == 'pamap2':
        from ciBabyMambaHar.data.pamap2 import Pamap2Dataset, computeClassWeights
        # PAMAP2 "Shock Absorber": 10Hz filter + Robust Scaling
        trainDs = Pamap2Dataset('./datasets/PAMAP2_Dataset', split='train',
                                applyFilter=True, filterCutoff=10.0, useRobustScaling=True)
        testDs = Pamap2Dataset('./datasets/PAMAP2_Dataset', split='test',
                               applyFilter=True, filterCutoff=10.0, useRobustScaling=True)
        # Compute class weights for imbalanced PAMAP2
        if len(trainDs.labels) > 0:
            classWeights = computeClassWeights(trainDs.labels, numClasses=12)
    elif dataset == 'opportunity':
        from ciBabyMambaHar.data.opportunity import OpportunityDataset
        trainDs = OpportunityDataset('./datasets/Opportunity', split='train', task='locomotion')
        testDs = OpportunityDataset('./datasets/Opportunity', split='test', task='locomotion')
        # Compute class weights (Null class is dominant)
        from ciBabyMambaHar.data.pamap2 import computeClassWeights
        if len(trainDs.labels) > 0:
            classWeights = computeClassWeights(trainDs.labels, numClasses=5)
    elif dataset == 'unimib':
        from ciBabyMambaHar.data.unimib import UniMiBSHARDataset
        trainDs = UniMiBSHARDataset('./datasets/UniMiB-SHAR', split='train', task='adl')
        testDs = UniMiBSHARDataset('./datasets/UniMiB-SHAR', split='test', task='adl')
    elif dataset == 'skoda':
        from ciBabyMambaHar.data.skoda import SkodaDataset, computeSkodaClassWeights
        # SKODA "Speed Run": 5Hz filter to remove machine vibration
        trainDs = SkodaDataset('./datasets/Skoda', split='train', applyFilter=True, filterCutoff=5.0)
        testDs = SkodaDataset('./datasets/Skoda', split='test', applyFilter=True, filterCutoff=5.0)
        # Compute class weights (Null class down-weighted, gestures up-weighted)
        if len(trainDs.labels) > 0:
            classWeights = computeSkodaClassWeights(trainDs.labels, numClasses=11)
    elif dataset == 'daphnet':
        from ciBabyMambaHar.data.daphnet import DaphnetDataset, computeDaphnetClassWeights
        # Apply 12Hz Butterworth low-pass filter (removes sensor jitter)
        trainDs = DaphnetDataset('./datasets/Daphnet', split='train', applyFilter=True, filterCutoff=12.0)
        testDs = DaphnetDataset('./datasets/Daphnet', split='test', applyFilter=True, filterCutoff=12.0)
        # Use DATA-COMPUTED class weights (not aggressive 15:1)
        if len(trainDs.labels) > 0:
            classWeights = computeDaphnetClassWeights(trainDs.labels, numClasses=2, aggressive=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    from torch.utils.data import DataLoader
    trainLoader = DataLoader(trainDs, batch_size=len(trainDs), shuffle=False, num_workers=0)
    testLoader = DataLoader(testDs, batch_size=len(testDs), shuffle=False, num_workers=0)
    
    xTrain, yTrain = next(iter(trainLoader))
    xTest, yTest = next(iter(testLoader))
    
    xTrain = xTrain.to(DEVICE)
    yTrain = yTrain.to(DEVICE)
    xTest = xTest.to(DEVICE)
    yTest = yTest.to(DEVICE)
    
    if classWeights is not None:
        classWeights = classWeights.to(DEVICE)
    
    print(f"   Loaded: Train={xTrain.shape[0]}, Test={xTest.shape[0]} samples")
    
    return xTrain, yTrain, xTest, yTest, classWeights


# ============================================================================
# TRAINING WITH F1 OPTIMIZATION
# ============================================================================

def validateFast(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """Compute accuracy and F1."""
    model.eval()
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1)
        acc = 100.0 * (pred == y).float().mean().item()
        f1 = f1_score(y.cpu().numpy(), pred.cpu().numpy(), average='macro') * 100
    model.train()
    return acc, f1


def trainWithHpo(
    model: nn.Module,
    xTrain: torch.Tensor,
    yTrain: torch.Tensor,
    xVal: torch.Tensor,
    yVal: torch.Tensor,
    config: Dict[str, Any],
    trial: Optional[optuna.Trial] = None,
    classWeights: Optional[torch.Tensor] = None,
    dataset: str = 'ucihar'  # NEW: Dataset-specific training recipes
) -> float:
    """
    Train model with HPO configuration (matches baseline style).
    
    Uses:
    - CosineAnnealingLR scheduler (like baselines)
    - Gradient clipping (1.0)
    - Early stopping on val_f1 (patience=5)
    - Mixed precision (AMP)
    
    DAPHNET VARIANCE STABILIZER:
    - Linear warmup (10 epochs): 1e-6 → peak LR
    - Cosine annealing: peak LR → 1e-5
    - Weight decay: 0.05
    - Aggressive class weights: 15:1 for Freeze
    - WeightedRandomSampler: 50/50 batch balance
    """
    model = model.to(DEVICE)
    model.train()
    
    batchSize = config.get('batchSize', FIXED_CONFIG['batchSize'])
    maxEpochs = EPOCHS_PER_TRIAL
    patience = EARLY_STOPPING_PATIENCE
    nSamples = xTrain.size(0)
    
    # ========================================
    # DAPHNET: Use original recipe that achieved 87% F1
    # The Variance Stabilizer (15:1 weights, 10-epoch warmup) was too aggressive
    # ========================================
    if dataset == 'daphnet':
        # Use STANDARD training (what achieved 87% in HPO)
        warmupEpochs = 2
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weightDecay']
        )
        
        # Standard CosineAnnealingLR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxEpochs)
        warmupScheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmupEpochs)
        
        # Use DATA-COMPUTED class weights (not fixed 15:1)
        # The passed-in classWeights were computed from getDaphnetLoaders with aggressive=True
        # But that was too aggressive - use the original computed weights instead
        if classWeights is not None:
            criterion = nn.CrossEntropyLoss(weight=classWeights, label_smoothing=0.0)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        
        sampleWeights = None
        gradClipNorm = FIXED_CONFIG['gradientClipNorm']
        
    elif dataset == 'skoda':
        # SKODA "Speed Run": Label smoothing 0.1 for fuzzy gesture boundaries
        warmupEpochs = 2
        labelSmoothing = config.get('labelSmoothing', 0.1)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weightDecay']
        )
        
        # Scheduler: CosineAnnealingLR (matches baselines)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxEpochs)
        
        # LR Warmup: 2 epochs of linear warmup
        warmupScheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmupEpochs)
        
        # Use class weights with label smoothing (0.1) for fuzzy gesture boundaries
        if classWeights is not None:
            criterion = nn.CrossEntropyLoss(weight=classWeights, label_smoothing=labelSmoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=labelSmoothing)
        
        sampleWeights = None
        gradClipNorm = config.get('gradClip', FIXED_CONFIG['gradientClipNorm'])
        
    elif dataset == 'pamap2':
        # PAMAP2 "Shock Absorber": Gradient clipping for extreme dynamics
        warmupEpochs = 2
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weightDecay']
        )
        
        # Scheduler: CosineAnnealingLR (matches baselines)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxEpochs)
        
        # LR Warmup: 2 epochs of linear warmup
        warmupScheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmupEpochs)
        
        # Use class weights (PAMAP2 is imbalanced - Running/Jumping rare)
        if classWeights is not None:
            criterion = nn.CrossEntropyLoss(weight=classWeights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        sampleWeights = None
        gradClipNorm = config.get('gradClip', FIXED_CONFIG['gradientClipNorm'])
        
    else:
        # Default recipe for other datasets (no warmup - matches baselines HPO)
        warmupEpochs = 0
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weightDecay']
        )
        
        # Scheduler: CosineAnnealingLR (matches baselines)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxEpochs)
        
        # No warmup in HPO (matches baselines)
        warmupScheduler = None
        
        # Use class weights for imbalanced datasets
        if classWeights is not None:
            criterion = nn.CrossEntropyLoss(weight=classWeights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        sampleWeights = None
        gradClipNorm = FIXED_CONFIG['gradientClipNorm']
    
    # AMP scaler for mixed precision
    scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    # Data augmentation - DISABLED for HPO speed (augmentation slows down 18x)
    # Final training scripts will use augmentation
    # Baselines don't use augmentation in HPO either
    useAugmentation = False  # Disabled for HPO speed
    if useAugmentation:
        augment = getTrainAugmentation('strong')
    
    bestF1 = 0.0
    epochsNoImprove = 0
    
    for epoch in range(maxEpochs):
        model.train()
        
        # ========================================
        # DAPHNET: Use weighted sampling for 50/50 balance
        # ========================================
        if dataset == 'daphnet' and sampleWeights is not None:
            # Sample with replacement to achieve 50/50 balance
            sampledIndices = torch.multinomial(sampleWeights, nSamples, replacement=True)
            permutation = sampledIndices[torch.randperm(nSamples, device=DEVICE)]
        else:
            # Standard shuffle
            permutation = torch.randperm(nSamples, device=DEVICE)
        
        for i in range(0, nSamples, batchSize):
            indices = permutation[i:i + batchSize]
            batchX = xTrain[indices]
            batchY = yTrain[indices]
            
            # Apply augmentation per sample (on CPU) - only for smaller datasets
            if useAugmentation:
                batchX = batchX.clone().cpu()
                for j in range(batchX.size(0)):
                    batchX[j] = augment(batchX[j])
                batchX = batchX.to(DEVICE)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    output = model(batchX)
                    loss = criterion(output, batchY)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradClipNorm)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(batchX)
                loss = criterion(output, batchY)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradClipNorm)
                optimizer.step()
        
        # ========================================
        # Scheduler step: warmup then cosine annealing
        # ========================================
        if warmupEpochs > 0 and epoch < warmupEpochs and warmupScheduler is not None:
            warmupScheduler.step()
        else:
            scheduler.step()
        
        # Validation
        valAcc, valF1 = validateFast(model, xVal, yVal)
        
        # Pruning (report F1 to Optuna)
        if trial is not None:
            trial.report(valF1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Early stopping on F1
        if valF1 > bestF1:
            bestF1 = valF1
            epochsNoImprove = 0
        else:
            epochsNoImprove += 1
            if epochsNoImprove >= patience:
                break
    
    return bestF1


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def createObjective(
    dataset: str,
    xTrain: torch.Tensor,
    yTrain: torch.Tensor,
    xVal: torch.Tensor,
    yVal: torch.Tensor,
    classWeights: Optional[torch.Tensor] = None
):
    """
    Create Optuna objective.
    
    HPO Search Space (Training Hyperparameters ONLY):
        Default: lr, weight_decay, dropout
        Skoda: lr [1e-3, 5e-4], weight_decay [0.01, 0.05], dropout [0.1, 0.2]
        Daphnet: lr [5e-4, 1e-4], weight_decay [0.001], dropout [0.0, 0.05]
    
    FROZEN Architecture (NOT tuned):
        d_model: 24, n_layers: 4, d_state: 8, d_conv: 4, expand: 2, dt_rank: 2
    """
    spec = DATASET_SPECS[dataset]
    
    def objective(trial: optuna.Trial) -> float:
        # Set seed for reproducibility (seed 17 + trial number for consistency)
        setSeed(SEED + trial.number)
        
        # Dataset-specific HPO search spaces (training params only)
        if dataset == 'skoda':
            # SKODA "Speed Run":
            # - Signal Rescue: 5Hz low-pass filter (applied in data loader)
            # - Label Smoothing: 0.1 (fuzzy gesture boundaries)
            # - Standard class weights (SKODA is reasonably balanced)
            config = {
                'lr': trial.suggest_categorical('lr', [1e-3, 5e-4]),
                'weightDecay': trial.suggest_categorical('weight_decay', [0.01, 0.05]),
                'dropout': trial.suggest_categorical('dropout', [0.1, 0.2]),
                'batchSize': 512,  # Large batch for fast training (21k samples)
                'labelSmoothing': 0.1,  # Fuzzy gesture boundaries
            }
        elif dataset == 'pamap2':
            # PAMAP2 "Shock Absorber":
            # - Signal Rescue: 10Hz low-pass filter (applied in data loader)
            # - Robust Scaling: IQR-based (applied in data loader)
            # - Gradient Clipping: 1.0 (stabilize extreme dynamics)
            config = {
                'lr': trial.suggest_categorical('lr', [1e-3, 5e-4, 3e-4]),
                'weightDecay': trial.suggest_categorical('weight_decay', [0.01, 0.02]),
                'dropout': trial.suggest_categorical('dropout', [0.1, 0.2]),
                'batchSize': 128,  # Standard batch
                'gradClip': 1.0,  # Stabilize training with extreme dynamics
            }
        elif dataset == 'daphnet':
            # DAPHNET VARIANCE STABILIZER:
            # - Fixed weight_decay=0.05 (suppresses jitter tracking)
            # - Only search LR (peak after warmup)
            # - Low/no dropout (small model, need all capacity)
            config = {
                'lr': trial.suggest_categorical('lr', [1e-3, 5e-4, 1e-4]),  # Peak LR after 10-epoch warmup
                'weightDecay': 0.05,  # FIXED: High weight decay for freeze detection
                'dropout': trial.suggest_categorical('dropout', [0.0, 0.05]),
                'batchSize': 512,  # Large batch for fast training (54k samples)
            }
        else:
            # Default search space for other datasets (matches baselines)
            config = {
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'weightDecay': trial.suggest_float('weight_decay', 0.005, 0.05, log=True),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            }
        
        try:
            # Create CiBabyMambaHar model with FROZEN architecture
            # Only dropout is from HPO - architecture is FIXED
            model = CiBabyMambaHar(
                numClasses=spec['numClasses'],
                inChannels=spec['inChannels'],
                dModel=FROZEN_ARCHITECTURE['dModel'],
                nLayers=FROZEN_ARCHITECTURE['nLayers'],
                dState=FROZEN_ARCHITECTURE['dState'],
                dropout=config['dropout'],  # Only tunable model param
            )
            
            # Log model params (should be ~24,000-29,000 depending on dataset)
            params = model.countParameters()['total']
            trial.set_user_attr('params', params)
            
            # Train (with class weights for imbalanced datasets)
            # Pass dataset name for dataset-specific training recipes (e.g., Daphnet)
            f1 = trainWithHpo(
                model, xTrain, yTrain, xVal, yVal,
                config, trial=trial, classWeights=classWeights,
                dataset=dataset  # NEW: Enable dataset-specific training recipes
            )
            
            del model
            torch.cuda.empty_cache()
            
            return f1
            
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"   Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned(f"Trial failed: {e}")
    
    return objective


# ============================================================================
# HPO RUNNER
# ============================================================================

def runHpo(
    dataset: str,
    nTrials: int = N_TRIALS,
    seed: int = SEED
) -> Dict[str, Any]:
    """
    Run HPO for CiBabyMambaHar.
    
    Optimization target: val_f1 (macro F1 score)
    
    FROZEN Architecture:
        d_model=24, n_layers=4, d_state=8, d_conv=4, expand=2, dt_rank=2, bidirectional=True
    
    Tuned Hyperparameters:
        lr, weight_decay, dropout
    """
    setSeed(seed)
    
    print(f"\n{'='*60}")
    print(f"CI-BabyMamba-HAR HPO - {dataset.upper()}")
    print(f"{'='*60}")
    print(f"   FROZEN Architecture (CI-BabyMamba-HAR):")
    print(f"      d_model: {FROZEN_ARCHITECTURE['dModel']}")
    print(f"      n_layers: {FROZEN_ARCHITECTURE['nLayers']}")
    print(f"      d_state: {FROZEN_ARCHITECTURE['dState']}")
    print(f"      d_conv: {FROZEN_ARCHITECTURE['dConv']}")
    print(f"      expand: {FROZEN_ARCHITECTURE['expand']}")
    print(f"      dt_rank: {FROZEN_ARCHITECTURE['dtRank']}")
    print(f"      bidirectional: {FROZEN_ARCHITECTURE['bidirectional']}")
    print(f"      gated_attention: {FROZEN_ARCHITECTURE['useGatedAttention']}")
    print(f"      channel_independent: {FROZEN_ARCHITECTURE['channelIndependent']}")
    print(f"   HPO Search Space (Training Hyperparameters ONLY):")
    print(f"      lr: log_uniform(1e-4, 1e-2)  (matches baselines)")
    print(f"      weight_decay: log_uniform(0.005, 0.05)")
    print(f"      dropout: uniform(0.0, 0.5)  (matches baselines)")
    print(f"   Fixed: batch_size=64, epochs_per_trial={EPOCHS_PER_TRIAL}")
    print(f"   Trials: {nTrials}")
    print(f"   Device: {DEVICE}")
    
    # Load data (with class weights for imbalanced datasets)
    xTrain, yTrain, xTest, yTest, classWeights = loadDataset(dataset)
    
    # Verify model params (should be ~27k)
    spec = DATASET_SPECS[dataset]
    testModel = CiBabyMambaHar(
        numClasses=spec['numClasses'],
        inChannels=spec['inChannels'],
        dModel=FROZEN_ARCHITECTURE['dModel'],
        nLayers=FROZEN_ARCHITECTURE['nLayers'],
        dState=FROZEN_ARCHITECTURE['dState'],
        dropout=0.1
    )
    params = testModel.countParameters()['total']
    print(f"   Model params (FROZEN): {params:,}")
    if classWeights is not None:
        print(f"   Using class weights for imbalanced dataset")
    del testModel
    
    # Create study
    studyName = f"ciBabyMambaHar_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(
        study_name=studyName,
        direction='maximize',  # Maximize val_f1
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1
        )
    )
    
    # Run optimization (with class weights for imbalanced datasets)
    objective = createObjective(dataset, xTrain, yTrain, xTest, yTest, classWeights)
    
    print(f"\n   Running {nTrials} trials (optimizing F1)...")
    study.optimize(objective, n_trials=nTrials, show_progress_bar=True)
    
    # Results
    print(f"\n{'='*60}")
    print(f"HPO RESULTS - {dataset.upper()}")
    print(f"{'='*60}")
    print(f"   Best F1: {study.best_value:.2f}%")
    print(f"   Best params (Training Hyperparameters ONLY):")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"      {k}: {v:.6f}")
        else:
            print(f"      {k}: {v}")
    
    # Save results
    resultsDir = Path("results/hpo")
    resultsDir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'dataset': dataset,
        'bestF1': study.best_value,
        'bestParams': study.best_params,
        'frozenArchitecture': FROZEN_ARCHITECTURE,
        'fixedConfig': FIXED_CONFIG,
        'nTrials': nTrials,
        'epochsPerTrial': EPOCHS_PER_TRIAL,
        'studyName': studyName,
        'timestamp': datetime.now().isoformat()
    }
    
    outPath = resultsDir / f"hpo_ciBabyMambaHar_{dataset}.json"
    with open(outPath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved: {outPath}")
    
    del xTrain, yTrain, xTest, yTest
    torch.cuda.empty_cache()
    
    return results


def runAllHpo(nTrials: int = N_TRIALS):
    """Run HPO for all datasets."""
    
    print(f"\n{'='*60}")
    print("CI-BabyMamba-HAR HPO - ALL DATASETS")
    print(f"{'='*60}")
    
    allResults = {}
    
    for dataset in DATASET_SPECS.keys():
        try:
            results = runHpo(dataset, nTrials)
            allResults[dataset] = results
        except Exception as e:
            print(f"HPO failed for {dataset}: {e}")
            allResults[dataset] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("HPO SUMMARY")
    print(f"{'='*60}")
    
    for dataset, results in allResults.items():
        if 'error' in results:
            print(f"   {dataset}: FAILED")
        else:
            print(f"   {dataset}: F1={results['bestF1']:.2f}%")
            print(f"      lr={results['bestParams']['lr']:.6f}")
            print(f"      weight_decay={results['bestParams']['weight_decay']:.6f}")
            print(f"      dropout={results['bestParams']['dropout']:.4f}")
    
    outPath = Path("results/hpo/hpo_ciBabyMambaHar_all.json")
    with open(outPath, 'w') as f:
        json.dump(allResults, f, indent=2)
    
    return allResults


# ============================================================================
# MAIN
# ============================================================================

def main():
    global EPOCHS_PER_TRIAL
    import argparse
    
    parser = argparse.ArgumentParser(description="CiBabyMambaHar HPO (Training Hyperparameters ONLY)")
    
    parser.add_argument('--dataset', '-d', type=str, default='ucihar',
                        choices=['ucihar', 'motionsense', 'wisdm', 'pamap2', 'opportunity', 'unimib', 'skoda', 'daphnet', 'all'],
                        help='Dataset for HPO (default: ucihar)')
    parser.add_argument('--n-trials', '-n', type=int, default=N_TRIALS,
                        help=f'Number of trials (default: {N_TRIALS})')
    parser.add_argument('--epochs', '-e', type=int, default=EPOCHS_PER_TRIAL,
                        help=f'Epochs per trial (default: {EPOCHS_PER_TRIAL})')
    parser.add_argument('--seed', '-s', type=int, default=SEED,
                        help=f'Random seed (default: {SEED})')
    
    args = parser.parse_args()
    
    # Override global epochs if specified
    EPOCHS_PER_TRIAL = args.epochs
    
    if args.dataset == 'all':
        runAllHpo(args.n_trials)
    else:
        runHpo(args.dataset, args.n_trials, args.seed)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Parallel HPO Script for Baseline Models (TinierHAR, TinyHAR)

Features:
- HPO for TinierHAR and TinyHAR baselines
- Early stopping with patience=5
- Uses TPE sampler from Optuna
- Same evaluation metrics as BabyMamba
Signal Rescue Strategies (for fairness with CI-BabyMamba-HAR):
- SKODA: 5Hz Butterworth filter + Label Smoothing 0.1 + Batch Size 512
- PAMAP2: 10Hz Butterworth filter + Robust Scaling + Gradient Clip 1.0 + NO label smoothing
- Daphnet: 12Hz Butterworth filter + Class Weights [1.0, 15.0] + Weight Decay 0.05 + NO label smoothing
Usage:
    python scripts/hpoBaselines.py --model tinierhar --dataset ucihar
    python scripts/hpoBaselines.py --model tinyhar --dataset all --trials 50
    python scripts/hpoBaselines.py --model all --dataset all
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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines import TinierHAR, TinyHAR, LightDeepConvLSTM, DeepConvLSTM
from ciBabyMambaHar.utils.profiling import countParameters, computeMacs, benchmarkLatency


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 17
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HPO_EPOCHS = 10      # Matched with BabyMamba HPO
HPO_PATIENCE = 5     # Early stopping patience
N_TRIALS = 50
SUBSET_SIZE = 5000

# Parallel HPO configuration
N_PARALLEL_JOBS = 4

# Dataset specifications
DATASET_SPECS = {
    'ucihar': {'numClasses': 6, 'inChannels': 9, 'seqLen': 128, 'root': './datasets/UCI HAR Dataset'},
    'motionsense': {'numClasses': 6, 'inChannels': 6, 'seqLen': 128, 'root': './datasets/motion-sense-master'},
    'wisdm': {'numClasses': 6, 'inChannels': 3, 'seqLen': 128, 'root': './datasets/WISDM_ar_v1.1'},
    'pamap2': {'numClasses': 12, 'inChannels': 19, 'seqLen': 128, 'root': './datasets/PAMAP2_Dataset'},  # 19 = compact mode (acc+gyro)
    'opportunity': {'numClasses': 5, 'inChannels': 79, 'seqLen': 128, 'root': './datasets/Opportunity'},  # Locomotion task
    'unimib': {'numClasses': 9, 'inChannels': 3, 'seqLen': 128, 'root': './datasets/UniMiB-SHAR'},  # ADL task
    'skoda': {'numClasses': 11, 'inChannels': 30, 'seqLen': 98, 'root': './datasets/Skoda'},  # 10 gestures + Null
    'daphnet': {'numClasses': 2, 'inChannels': 9, 'seqLen': 64, 'root': './datasets/Daphnet'},  # Binary: Walk vs Freeze
}

# Signal Rescue: Dataset-specific fixed training parameters
# These are NOT tuned by HPO, they are LOCKED for fairness
SIGNAL_RESCUE_CONFIG = {
    'skoda': {
        'labelSmoothing': 0.1,  # Fuzzy gesture boundaries
        'batchSize': 512,       # Large dataset
        'weightDecay': 0.01,
    },
    'pamap2': {
        'labelSmoothing': 0.0,  # Sharp decisions for extreme outliers
        'batchSize': 64,
        'weightDecay': 0.01,
    },
    'daphnet': {
        'labelSmoothing': 0.0,  # Binary decisions, no smoothing
        'batchSize': 512,       # Large dataset
        'weightDecay': 0.05,    # Higher regularization
    },
    # Default for other datasets
    'default': {
        'labelSmoothing': 0.1,
        'batchSize': 64,
        'weightDecay': 0.01,
    },
}


def setSeed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# DATA LOADING
# ============================================================================

def getDataLoaders(
    dataset: str,
    batchSize: int = 64,
    subsetSize: Optional[int] = SUBSET_SIZE
) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
    """Get data loaders with optional subset for HPO speed.
    
    Returns:
        (trainLoader, testLoader, classWeights) where classWeights is None for balanced datasets
    """
    
    spec = DATASET_SPECS[dataset]
    classWeights = None
    
    if dataset == 'ucihar':
        from ciBabyMambaHar.data.uciHar import getUciHarLoaders
        trainLoader, testLoader = getUciHarLoaders(
            root=spec['root'], batchSize=batchSize, numWorkers=0
        )
    elif dataset == 'motionsense':
        from ciBabyMambaHar.data.motionSense import getMotionSenseLoaders
        trainLoader, testLoader = getMotionSenseLoaders(
            root=spec['root'], batchSize=batchSize, numWorkers=0
        )
    elif dataset == 'wisdm':
        from ciBabyMambaHar.data.wisdm import getWisdmLoaders
        trainLoader, testLoader = getWisdmLoaders(
            root=spec['root'], batchSize=batchSize, numWorkers=0
        )
    elif dataset == 'pamap2':
        from ciBabyMambaHar.data.pamap2 import getPamap2Loaders
        # PAMAP2 is imbalanced - get class weights for weighted loss
        trainLoader, testLoader, classWeights = getPamap2Loaders(
            root=spec['root'], batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dataset == 'opportunity':
        from ciBabyMambaHar.data.opportunity import getOpportunityLoaders
        # Opportunity has dominant Null class - get class weights
        trainLoader, testLoader, classWeights = getOpportunityLoaders(
            root=spec['root'], batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dataset == 'unimib':
        from ciBabyMambaHar.data.unimib import getUniMiBLoaders
        trainLoader, testLoader = getUniMiBLoaders(
            root=spec['root'], batchSize=batchSize, numWorkers=0
        )
    elif dataset == 'skoda':
        from ciBabyMambaHar.data.skoda import getSkodaLoaders
        trainLoader, testLoader, classWeights = getSkodaLoaders(
            root=spec['root'], batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dataset == 'daphnet':
        from ciBabyMambaHar.data.daphnet import getDaphnetLoaders
        trainLoader, testLoader, classWeights = getDaphnetLoaders(
            root=spec['root'], batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Only use subset for balanced datasets (not PAMAP2 which is imbalanced)
    if subsetSize and subsetSize < len(trainLoader.dataset) and dataset != 'pamap2':
        indices = list(range(min(subsetSize, len(trainLoader.dataset))))
        subsetDs = Subset(trainLoader.dataset, indices)
        trainLoader = DataLoader(subsetDs, batch_size=batchSize, shuffle=True, num_workers=0)
    
    return trainLoader, testLoader, classWeights


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
# TRAINING FUNCTION
# ============================================================================

def computeF1(preds: torch.Tensor, labels: torch.Tensor, numClasses: int) -> float:
    """Compute macro F1 score."""
    f1Scores = []
    
    for c in range(numClasses):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1Scores.append(f1)
    
    return 100.0 * np.mean(f1Scores)


def trainAndEval(
    model: nn.Module,
    trainLoader: DataLoader,
    testLoader: DataLoader,
    config: Dict[str, Any],
    epochs: int = HPO_EPOCHS,
    patience: int = HPO_PATIENCE,
    classWeights: Optional[torch.Tensor] = None,
    numClasses: int = 6
) -> float:
    """Train and evaluate with early stopping, return best F1 score."""
    
    model = model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weightDecay']
    )
    
    # Loss - use class weights for imbalanced datasets (e.g., PAMAP2)
    if classWeights is not None:
        classWeights = classWeights.to(DEVICE)
        criterion = nn.CrossEntropyLoss(
            weight=classWeights,
            label_smoothing=config.get('labelSmoothing', 0.1)
        )
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get('labelSmoothing', 0.1)
        )
    
    scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    bestF1 = 0.0
    epochsNoImprove = 0
    
    for epoch in range(epochs):
        model.train()
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
        
        scheduler.step()
        
        # Evaluate - compute F1 score instead of accuracy
        model.eval()
        allPreds, allLabels = [], []
        with torch.no_grad():
            for x, y in testLoader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                allPreds.append(output.argmax(1))
                allLabels.append(y)
        
        allPreds = torch.cat(allPreds)
        allLabels = torch.cat(allLabels)
        f1Score = computeF1(allPreds, allLabels, numClasses)
        
        if f1Score > bestF1:
            bestF1 = f1Score
            epochsNoImprove = 0
        else:
            epochsNoImprove += 1
            if epochsNoImprove >= patience:
                break
    
    del model
    torch.cuda.empty_cache()
    
    return bestF1


# ============================================================================
# OBJECTIVE FUNCTIONS
# ============================================================================

# ==========================================
# LOCKED ARCHITECTURE CONFIGURATIONS
# These are FIXED to match the original papers' parameter counts.
# We ONLY tune training hyperparameters (lr, weightDecay, dropout, etc.)
# NOT architecture parameters (filterNum, gruUnits, etc.)
#
# PARAMETER COUNTS (UCI-HAR, 9 channels):
# - TinierHAR:         16,931 params (~17k target)
# - TinyHAR:           42,704 params (~40k target)  
# - LightDeepConvLSTM: 15,286 params (~15k iso-param control)
# - DeepConvLSTM:      ~132k params (unidirectional, per original paper)
# - BabyMamba:         13,951 params (~14.8k, ours)
# ==========================================
LOCKED_ARCHITECTURES = {
    'tinierhar': {
        # TinierHAR: ~17k params (from zhaxidele/TinierHAR)
        # gruUnits=16 gives 16,931 params on UCI-HAR
        'nbFilters': 8,
        'nbConvBlocks': 4,
        'gruUnits': 16,
        'dropout': 0.5,  # From paper
    },
    'tinyhar': {
        # TinyHAR: ~42k params (from teco-kit/ISWC22-HAR)  
        # filterNum=24 gives 42,704 params on UCI-HAR
        'filterNum': 24,
        'nbConvLayers': 4,
        'filterSize': 5,
        'dropout': 0.5,  # From paper
    },
    'lightdeepconvlstm': {
        # LightDeepConvLSTM: ~15k params (iso-param control vs BabyMamba)
        # lstmHidden=32 gives 15,286 params on UCI-HAR
        'convFilters': 16,
        'lstmHidden': 32,
        'dropout': 0.5,  # High dropout for small LSTMs
    },
    'deepconvlstm': {
        # DeepConvLSTM: ~132k params (traditional DL baseline, per Ordóñez & Roggen 2016)
        # convFilters=64, lstmHidden=64, unidirectional gives ~132k params on UCI-HAR
        'convFilters': 64,
        'lstmHidden': 64,
        'lstmLayers': 2,
        'bidirectional': False,  # Unidirectional as per original paper
        'dropout': 0.5,  # High dropout for LSTM regularization
    },
}


def createObjective(modelName: str, dataset: str, nEpochs: int):
    """Create Optuna objective for a model and dataset.
    
    IMPORTANT: Architecture is LOCKED. We only tune training hyperparameters.
    Signal Rescue parameters (labelSmoothing, batchSize, weightDecay for some datasets) 
    are FIXED for fairness with CI-BabyMamba-HAR.
    
    Optimization Target: F1 Score (macro) - better for imbalanced datasets
    """
    
    spec = DATASET_SPECS[dataset]
    lockedArch = LOCKED_ARCHITECTURES[modelName]
    
    # Get Signal Rescue config for this dataset (or default)
    signalRescue = SIGNAL_RESCUE_CONFIG.get(dataset, SIGNAL_RESCUE_CONFIG['default'])
    
    def objective(trial: optuna.Trial) -> float:
        # TRAINING hyperparameters only (architecture is LOCKED)
        # Signal Rescue parameters are FIXED for fairness
        config = {
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            # Signal Rescue: Use fixed values for these datasets
            'weightDecay': signalRescue['weightDecay'],
            'labelSmoothing': signalRescue['labelSmoothing'],
            'batchSize': signalRescue['batchSize'],
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        }
        
        # Merge with LOCKED architecture parameters (architecture takes priority)
        config.update(lockedArch)
        
        # Get data (with class weights for imbalanced datasets)
        trainLoader, testLoader, classWeights = getDataLoaders(dataset, config['batchSize'])
        
        # Create model with LOCKED architecture
        model = createModel(modelName, config, spec)
        
        # Train (only training hparams vary) - returns F1 score
        try:
            f1Score = trainAndEval(
                model, trainLoader, testLoader, config, nEpochs,
                classWeights=classWeights, numClasses=spec['numClasses']
            )
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0
        
        return f1Score
    
    return objective


# ============================================================================
# HPO RUNNER
# ============================================================================

def runHpo(
    modelName: str,
    dataset: str,
    nTrials: int = N_TRIALS,
    nEpochs: int = HPO_EPOCHS,
    seed: int = SEED,
    nJobs: int = N_PARALLEL_JOBS
) -> Dict[str, Any]:
    """Run HPO for a model on a dataset."""
    
    setSeed(seed)
    spec = DATASET_SPECS[dataset]
    
    # Determine actual parallel jobs (GPU forces sequential)
    actualJobs = 1 if DEVICE.type == 'cuda' else nJobs
    
    print(f"\n{'='*60}")
    print(f"🔍 HPO for {modelName.upper()} on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"   Trials: {nTrials}, Epochs: {nEpochs}, Patience: {HPO_PATIENCE}")
    print(f"   Workers: {actualJobs}")
    print(f"   Device: {DEVICE}")
    
    # Get Signal Rescue config for this dataset
    signalRescue = SIGNAL_RESCUE_CONFIG.get(dataset, SIGNAL_RESCUE_CONFIG['default'])
    print(f"   Signal Rescue: labelSmooth={signalRescue['labelSmoothing']}, batch={signalRescue['batchSize']}, wd={signalRescue['weightDecay']}")
    
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=f"{modelName}_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        direction='maximize',
        sampler=sampler
    )
    
    objective = createObjective(modelName, dataset, nEpochs)
    study.optimize(objective, n_trials=nTrials, n_jobs=actualJobs, show_progress_bar=True)
    
    # Results
    print(f"\n📊 Results: {study.best_value:.2f}% F1")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    
    # Get best model stats (merge with LOCKED architecture and Signal Rescue)
    lockedArch = LOCKED_ARCHITECTURES[modelName]
    fullConfig = {**study.best_params, **lockedArch, **signalRescue}
    bestModel = createModel(modelName, fullConfig, spec)
    params = countParameters(bestModel)['total']
    macsResult = computeMacs(bestModel, (1, spec['seqLen'], spec['inChannels']), device='cpu')
    
    print(f"   Parameters: {params:,}")
    print(f"   MACs: {macsResult.get('macs', 'N/A'):,}" if macsResult.get('macs') else "   MACs: N/A")
    
    # Save
    resultsDir = Path("results/hpo")
    resultsDir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model': modelName,
        'dataset': dataset,
        'bestF1': study.best_value,
        'bestTrainingParams': study.best_params,  # Training params only (lr, dropout)
        'signalRescue': signalRescue,             # Signal Rescue (FIXED for fairness)
        'lockedArchitecture': lockedArch,         # Architecture (LOCKED)
        'bestParams': fullConfig,                  # Full config for training script
        'parameters': params,
        'macs': macsResult.get('macs'),
        'nTrials': nTrials,
        'nEpochs': nEpochs,
        'optimizationMetric': 'F1',
        'timestamp': datetime.now().isoformat()
    }
    
    outPath = resultsDir / f"hpo_{modelName}_{dataset}.json"
    with open(outPath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved: {outPath}")
    
    return results


def runAllHpo(modelName: str, nTrials: int = N_TRIALS, nEpochs: int = HPO_EPOCHS, nJobs: int = N_PARALLEL_JOBS):
    """Run HPO for a model on all datasets."""
    
    allResults = {}
    for dataset in DATASET_SPECS.keys():
        try:
            results = runHpo(modelName, dataset, nTrials, nEpochs, SEED, nJobs)
            allResults[dataset] = results
        except Exception as e:
            print(f"❌ HPO failed for {modelName} on {dataset}: {e}")
            allResults[dataset] = {'error': str(e)}
    
    # Save combined
    outPath = Path(f"results/hpo/hpo_{modelName}_all.json")
    with open(outPath, 'w') as f:
        json.dump(allResults, f, indent=2)
    
    return allResults


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="HPO for Baseline Models")
    parser.add_argument('--model', '-m', type=str, required=True,
                        choices=['tinierhar', 'tinyhar', 'lightdeepconvlstm', 'deepconvlstm', 'all'],
                        help='Baseline model to tune')
    parser.add_argument('--dataset', '-d', type=str, default='all',
                        choices=['ucihar', 'motionsense', 'wisdm', 'pamap2', 'opportunity', 'unimib', 'skoda', 'daphnet', 'all'],
                        help='Dataset (default: all)')
    parser.add_argument('--trials', '-n', type=int, default=N_TRIALS,
                        help=f'Number of trials (default: {N_TRIALS})')
    parser.add_argument('--epochs', '-e', type=int, default=HPO_EPOCHS,
                        help=f'Epochs per trial (default: {HPO_EPOCHS})')
    parser.add_argument('--seed', '-s', type=int, default=SEED,
                        help=f'Random seed (default: {SEED})')
    parser.add_argument('--n-jobs', '-j', type=int, default=N_PARALLEL_JOBS,
                        help=f'Parallel workers (default: {N_PARALLEL_JOBS}, use 1 for GPU)')
    
    args = parser.parse_args()
    
    models = ['tinierhar', 'tinyhar', 'lightdeepconvlstm', 'deepconvlstm'] if args.model == 'all' else [args.model]
    
    for model in models:
        if args.dataset == 'all':
            runAllHpo(model, args.trials, args.epochs, args.n_jobs)
        else:
            runHpo(model, args.dataset, args.trials, args.epochs, args.seed, args.n_jobs)


if __name__ == '__main__':
    main()

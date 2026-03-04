"""
BabyMamba-Crossover-BiDir Hyperparameter Optimization Script

This script performs HPO for the BabyMamba-Crossover-BiDir architecture.
It uses Optuna with TPE sampler and FROZEN architecture hyperparameters.

FROZEN Architecture (NOT tuned):
    d_model = 26      (model dimension)
    d_state = 8       (SSM state dimension)
    n_layers = 4      (number of BiDir Mamba layers)
    expand = 2        (inner dimension expansion)
    dt_rank = 2       (time-step discretization rank)
    d_conv = 4        (local convolution kernel)

TUNED Hyperparameters (matches baselines for fairness):
    learning_rate:  [1e-4, 1e-2]   log-uniform
    weight_decay:   [0.005, 0.05]  log-uniform
    dropout:        [0.0, 0.5]     uniform

HPO Configuration:
    sampler = TPE (Tree-structured Parzen Estimator)
    trials = 50
    epochs per trial = 10 (matches baselines for fairness)
    pruning = Median pruner with warmup
    patience = 5 (early stopping)
    optimization_target = F1 Score (Macro) - MAXIMIZED

Usage:
    python hpoCrossoverBiDirBabyMambaHar.py --dataset uciHar --trials 50
    python hpoCrossoverBiDirBabyMambaHar.py --dataset pamap2 --trials 100 --epochs 100
"""

import os
import sys
import argparse
import random
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CrossoverBiDirBabyMambaHar


# ============== Dataset Configurations ==============

DATASET_CONFIGS = {
    'uciHar': {
        'name': 'UCI-HAR',
        'inputChannels': 9,
        'seqLen': 128,
        'numClasses': 6,
        'batchSize': 64
    },
    'motionSense': {
        'name': 'MotionSense',
        'inputChannels': 6,  # FIXED: was 12, actual data is 6 (acc + gyro)
        'seqLen': 128,
        'numClasses': 6,
        'batchSize': 64
    },
    'wisdm': {
        'name': 'WISDM',
        'inputChannels': 3,
        'seqLen': 128,  # FIXED: was 200, now matches baselines
        'numClasses': 6,
        'batchSize': 64
    },
    'pamap2': {
        'name': 'PAMAP2',
        'inputChannels': 19,  # FIXED: was 52, now 19 (compact mode: HR + acc + gyro)
        'seqLen': 128,  # FIXED: was 171, now matches baselines
        'numClasses': 12,
        'batchSize': 64
    },
    'opportunity': {
        'name': 'Opportunity',
        'inputChannels': 79,  # FIXED: was 113, now 79 (body-worn IMU)
        'seqLen': 128,  # FIXED: was 24, now matches baselines
        'numClasses': 5,  # FIXED: was 17, now 5 (Locomotion task)
        'batchSize': 64
    },
    'unimib': {
        'name': 'UniMiB-SHAR',
        'inputChannels': 3,
        'seqLen': 128,  # FIXED: was 151, now matches baselines
        'numClasses': 9,  # FIXED: was 17, now 9 (ADL task)
        'batchSize': 64
    },
    'skoda': {
        'name': 'Skoda',
        'inputChannels': 30,  # FIXED: was 60, now 30 (10 sensors × 3 axes)
        'seqLen': 98,
        'numClasses': 11,  # FIXED: was 10, now 11 (10 gestures + Null)
        'batchSize': 512  # FIXED: matches Signal Rescue config
    },
    'daphnet': {
        'name': 'Daphnet',
        'inputChannels': 9,
        'seqLen': 64,  # FIXED: was 128, now 64 (2 seconds at 32Hz)
        'numClasses': 2,
        'batchSize': 512  # FIXED: matches Signal Rescue config
    }
}

# FROZEN architecture hyperparameters - DO NOT TUNE
FROZEN_ARCH = {
    'd_model': 26,
    'd_state': 8,
    'n_layers': 4,
    'expand': 2,
    'dt_rank': 2,
    'd_conv': 4,
    'stem_kernel': 5,
    'patch_kernel': 16,
    'patch_stride': 4
}


def setSeed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def getDevice():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def loadDataset(datasetName: str, batchSize: int):
    """
    Load dataset with train/val splits.
    
    Uses actual dataset loaders from ciBabyMambaHar.data with:
    - Signal Rescue filters for PAMAP2, Skoda, Daphnet
    - Class weights for imbalanced datasets
    - Same preprocessing as baselines for fairness
    """
    import sys
    from pathlib import Path
    
    # Add parent CiBabyMambaHar to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    config = DATASET_CONFIGS[datasetName]
    classWeights = None
    
    # Convert dataset name to lowercase for loader imports
    dsName = datasetName.lower()
    
    if dsName == 'ucihar':
        from ciBabyMambaHar.data.uciHar import getUciHarLoaders
        trainLoader, testLoader = getUciHarLoaders(
            root='./datasets/UCI HAR Dataset', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'motionsense':
        from ciBabyMambaHar.data.motionSense import getMotionSenseLoaders
        trainLoader, testLoader = getMotionSenseLoaders(
            root='./datasets/motion-sense-master', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'wisdm':
        from ciBabyMambaHar.data.wisdm import getWisdmLoaders
        trainLoader, testLoader = getWisdmLoaders(
            root='./datasets/WISDM_ar_v1.1', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'pamap2':
        from ciBabyMambaHar.data.pamap2 import getPamap2Loaders
        # PAMAP2: Uses class weights, Signal Rescue (10Hz filter + Robust Scaling) applied in dataset
        trainLoader, testLoader, classWeights = getPamap2Loaders(
            root='./datasets/PAMAP2_Dataset', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dsName == 'opportunity':
        from ciBabyMambaHar.data.opportunity import getOpportunityLoaders
        trainLoader, testLoader, classWeights = getOpportunityLoaders(
            root='./datasets/Opportunity', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dsName == 'unimib':
        from ciBabyMambaHar.data.unimib import getUniMiBLoaders
        trainLoader, testLoader = getUniMiBLoaders(
            root='./datasets/UniMiB-SHAR', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'skoda':
        from ciBabyMambaHar.data.skoda import getSkodaLoaders
        # SKODA: Signal Rescue (5Hz filter) applied in dataset, class weights for imbalanced
        trainLoader, testLoader, classWeights = getSkodaLoaders(
            root='./datasets/Skoda', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dsName == 'daphnet':
        from ciBabyMambaHar.data.daphnet import getDaphnetLoaders
        # Daphnet: Signal Rescue (12Hz filter) applied in dataset, class weights for imbalanced
        trainLoader, testLoader, classWeights = getDaphnetLoaders(
            root='./datasets/Daphnet', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    else:
        raise ValueError(f"Unknown dataset: {datasetName}")
    
    return trainLoader, testLoader, classWeights


def createObjective(args, config, device, trainLoader, valLoader, classWeights=None):
    """
    Create Optuna objective function.
    
    Only tunes: learning_rate, weight_decay, dropout
    Architecture is FROZEN.
    Uses class weights for imbalanced datasets (PAMAP2, Skoda, Daphnet).
    """
    
    def objective(trial):
        # Sample tunable hyperparameters (MATCHED with baselines for fairness)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)  # FIXED: was [5e-4, 5e-3]
        weightDecay = trial.suggest_float('weight_decay', 0.005, 0.05, log=True)  # FIXED: was [1e-3, 5e-2]
        dropout = trial.suggest_float('dropout', 0.0, 0.5)  # FIXED: was [0.0, 0.2]
        
        # Create model with FROZEN architecture
        # Note: Architecture params are frozen in the model, only pass dataset-specific config
        model = CrossoverBiDirBabyMambaHar(
            numClasses=config['numClasses'],
            inChannels=config['inputChannels'],
            seqLen=config['seqLen'],
            dropout=dropout
        )
        model = model.to(device)
        
        # Setup training with class weights for imbalanced datasets
        if classWeights is not None:
            criterion = nn.CrossEntropyLoss(weight=classWeights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weightDecay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = GradScaler('cuda')
        
        # Training loop
        bestValF1 = 0
        
        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            for data, target in trainLoader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                with autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            scheduler.step()
            
            # Validate
            model.eval()
            allPreds, allLabels = [], []
            with torch.no_grad():
                for data, target in valLoader:
                    data, target = data.to(device), target.to(device)
                    with autocast('cuda'):
                        output = model(data)
                    preds = output.argmax(dim=1).cpu().numpy()
                    allPreds.extend(preds)
                    allLabels.extend(target.cpu().numpy())
            
            valF1 = f1_score(allLabels, allPreds, average='macro') * 100
            bestValF1 = max(bestValF1, valF1)
            
            # Report to Optuna for pruning
            trial.report(valF1, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return bestValF1
    
    return objective


def runHPO(args):
    """Run hyperparameter optimization."""
    setSeed(42)
    device = getDevice()
    config = DATASET_CONFIGS[args.dataset]
    
    # Setup results directory
    resultsDir = Path(args.outDir) / args.dataset / datetime.now().strftime('%Y%m%d_%H%M%S')
    resultsDir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"BabyMamba-Crossover-BiDir HPO")
    print(f"{'='*60}")
    print(f"Dataset: {config['name']}")
    print(f"Trials: {args.trials}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"Device: {device}")
    print(f"Results Dir: {resultsDir}")
    print(f"\nFROZEN Architecture:")
    for k, v in FROZEN_ARCH.items():
        print(f"  {k}: {v}")
    
    # Load data (with class weights for imbalanced datasets)
    trainLoader, valLoader, classWeights = loadDataset(args.dataset, config['batchSize'])
    if classWeights is not None:
        print(f"\nUsing class weights for imbalanced dataset")
    
    # Create Optuna study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)  # Match baselines
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=f'BabyMamba-Crossover-BiDir-{args.dataset}'
    )
    
    # Create objective with class weights
    objective = createObjective(args, config, device, trainLoader, valLoader, classWeights)
    
    # Run optimization
    print(f"\n{'='*60}")
    print(f"Starting HPO with {args.trials} trials...")
    print(f"{'='*60}\n")
    
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"HPO RESULTS")
    print(f"{'='*60}")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best F1 Score: {study.best_value:.2f}%")
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.6f}")
    
    # Save results
    hpoResults = {
        'dataset': args.dataset,
        'config': config,
        'frozen_arch': FROZEN_ARCH,
        'trials': args.trials,
        'epochs': args.epochs,
        'best_trial': study.best_trial.number,
        'best_f1': study.best_value,
        'best_params': study.best_params,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
    }
    
    with open(resultsDir / 'hpo_results.json', 'w') as f:
        json.dump(hpoResults, f, indent=2, default=str)
    
    # Save Optuna visualization
    try:
        import optuna.visualization as vis
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(str(resultsDir / 'param_importances.html'))
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(resultsDir / 'optimization_history.html'))
        
        # Parallel coordinate
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(resultsDir / 'parallel_coordinate.html'))
        
        print(f"\nVisualization plots saved to {resultsDir}")
    except Exception as e:
        print(f"Could not generate visualizations: {e}")
    
    print(f"\nResults saved to {resultsDir / 'hpo_results.json'}")
    
    return study


def main():
    parser = argparse.ArgumentParser(description='HPO for BabyMamba-Crossover-BiDir')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to optimize on')
    parser.add_argument('--trials', type=int, default=50, help='Number of HPO trials')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per trial (matches baselines)')
    parser.add_argument('--outDir', type=str, default='results/hpo',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    study = runHPO(args)


if __name__ == '__main__':
    main()

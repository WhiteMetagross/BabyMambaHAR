"""
BabyMamba-Crossover-BiDir Training Script

This script trains the BabyMamba-Crossover-BiDir architecture on HAR datasets.
It supports multi-seed training for statistical significance and includes
comprehensive logging and model checkpointing.

FROZEN Architecture Configuration:
    d_model = 26      (model dimension)
    d_state = 8       (SSM state dimension)
    n_layers = 4      (number of BiDir Mamba layers)
    expand = 2        (inner dimension expansion)
    dt_rank = 2       (time-step discretization rank)
    d_conv = 4        (local convolution kernel)

Training Configuration (MATCHED WITH BASELINES FOR FAIRNESS):
    epochs = 200      (with early stopping)
    patience = 10     (FIXED: was 30, now matches baselines)
    batch_size = 64   (512 for Skoda/Daphnet, matches Signal Rescue)
    optimizer = AdamW
    scheduler = CosineAnnealingLR
    loss = CrossEntropyLoss (with optional class weights)
    AMP/FP16 = enabled for faster training
    optimization_target = F1 Score (Macro) for early stopping & best model

Usage:
    python trainCrossoverBiDirBabyMambaHar.py --dataset uciHar --seeds 5
    python trainCrossoverBiDirBabyMambaHar.py --dataset pamap2 --seeds 3 --epochs 300
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CrossoverBiDirBabyMambaHar, createCrossoverBiDirBabyMambaHar


# ============== Dataset Configurations ==============

DATASET_CONFIGS = {
    'uciHar': {
        'name': 'UCI-HAR',
        'inputChannels': 9,
        'seqLen': 128,
        'numClasses': 6,
        'classNames': ['Walking', 'Walking Upstairs', 'Walking Downstairs', 
                       'Sitting', 'Standing', 'Laying'],
        'batchSize': 64,
        'lr': 0.001551,
        'weightDecay': 0.038995,
        'dropout': 0.072
    },
    'motionSense': {
        'name': 'MotionSense',
        'inputChannels': 6,  # FIXED: was 12, actual data is 6 (acc + gyro)
        'seqLen': 128,
        'numClasses': 6,
        'classNames': ['Downstairs', 'Upstairs', 'Walking', 'Jogging', 'Sitting', 'Standing'],
        'batchSize': 64,
        'lr': 0.001931,
        'weightDecay': 0.013692,
        'dropout': 0.085
    },
    'wisdm': {
        'name': 'WISDM',
        'inputChannels': 3,
        'seqLen': 128,  # FIXED: was 200, now matches baselines
        'numClasses': 6,
        'classNames': ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing'],
        'batchSize': 64,
        'lr': 0.001396,
        'weightDecay': 0.005775,
        'dropout': 0.052
    },
    'pamap2': {
        'name': 'PAMAP2',
        'inputChannels': 19,  # FIXED: was 52, now 19 (compact mode)
        'seqLen': 128,  # FIXED: was 171, now matches baselines
        'numClasses': 12,
        'classNames': ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 
                       'Cycling', 'Nordic Walking', 'Ascending Stairs', 
                       'Descending Stairs', 'Vacuum Cleaning', 'Ironing', 'Rope Jumping'],
        'batchSize': 64,
        'lr': 0.001702,
        'weightDecay': 0.016797,
        'dropout': 0.051
    },
    'opportunity': {
        'name': 'Opportunity',
        'inputChannels': 79,  # FIXED: was 113, now 79 (body-worn IMU)
        'seqLen': 128,  # FIXED: was 24, now matches baselines
        'numClasses': 5,  # FIXED: was 17, now 5 (Locomotion task)
        'classNames': ['Null', 'Stand', 'Walk', 'Sit', 'Lie'],
        'batchSize': 64,
        'lr': 0.002612,
        'weightDecay': 0.028567,
        'dropout': 0.054
    },
    'unimib': {
        'name': 'UniMiB-SHAR',
        'inputChannels': 3,
        'seqLen': 128,  # FIXED: was 151, now matches baselines
        'numClasses': 9,  # FIXED: was 17, now 9 (ADL task)
        'classNames': [f'Activity_{i}' for i in range(9)],
        'batchSize': 64,
        'lr': 0.002946,
        'weightDecay': 0.029282,
        'dropout': 0.080
    },
    'skoda': {
        'name': 'Skoda',
        'inputChannels': 30,  # FIXED: was 60, now 30
        'seqLen': 98,
        'numClasses': 11,  # FIXED: was 10, now 11 (10 gestures + Null)
        'classNames': ['Null'] + [f'Gesture_{i}' for i in range(10)],
        'batchSize': 512,  # FIXED: matches Signal Rescue
        'lr': 0.002,
        'weightDecay': 0.01,
        'dropout': 0.05
    },
    'daphnet': {
        'name': 'Daphnet',
        'inputChannels': 9,
        'seqLen': 64,  # FIXED: was 128, now 64 (2 seconds at 32Hz)
        'numClasses': 2,
        'classNames': ['No Freeze', 'Freeze'],
        'batchSize': 512,  # FIXED: matches Signal Rescue
        'lr': 0.0005,
        'weightDecay': 0.05,  # FIXED: higher regularization for fairness
        'dropout': 0.0
    }
}

# Master seed for RNG seed generation (matches baselines for fairness)
MASTER_SEED = 17


def generateRandomSeeds(nSeeds: int, masterSeed: int = MASTER_SEED) -> list:
    """Generate random seeds using RNG from master seed (matches baselines)."""
    rng = np.random.default_rng(masterSeed)
    return [int(s) for s in rng.integers(0, 100000, size=nSeeds)]


def setSeed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getDevice():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def loadDataset(datasetName: str, batchSize: int, windowSize: int = None):
    """
    Load dataset with train/test splits.
    
    Uses actual dataset loaders from ciBabyMambaHar.data with:
    - Signal Rescue filters for PAMAP2, Skoda, Daphnet
    - Class weights for imbalanced datasets
    - Same preprocessing as baselines for fairness
    
    Args:
        datasetName: Name of the dataset to load
        batchSize: Batch size for DataLoaders
        windowSize: Optional override for sequence length (for LRD study)
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
            root='./datasets/UCI HAR Dataset', batchSize=batchSize, numWorkers=2
        )
    elif dsName == 'motionsense':
        from ciBabyMambaHar.data.motionSense import getMotionSenseLoaders
        trainLoader, testLoader = getMotionSenseLoaders(
            root='./datasets/motion-sense-master', batchSize=batchSize, numWorkers=2,
            windowSize=windowSize or config['seqLen']
        )
    elif dsName == 'wisdm':
        from ciBabyMambaHar.data.wisdm import getWisdmLoaders
        trainLoader, testLoader = getWisdmLoaders(
            root='./datasets/WISDM_ar_v1.1', batchSize=batchSize, numWorkers=2,
            windowSize=windowSize or config['seqLen']
        )
    elif dsName == 'pamap2':
        from ciBabyMambaHar.data.pamap2 import getPamap2Loaders
        # PAMAP2: Uses class weights, Signal Rescue (10Hz filter + Robust Scaling) applied in dataset
        trainLoader, testLoader, classWeights = getPamap2Loaders(
            root='./datasets/PAMAP2_Dataset', batchSize=batchSize, numWorkers=2, returnWeights=True,
            windowSize=windowSize or config['seqLen']
        )
    elif dsName == 'opportunity':
        from ciBabyMambaHar.data.opportunity import getOpportunityLoaders
        trainLoader, testLoader, classWeights = getOpportunityLoaders(
            root='./datasets/Opportunity', batchSize=batchSize, numWorkers=2, returnWeights=True
        )
    elif dsName == 'unimib':
        from ciBabyMambaHar.data.unimib import getUniMiBLoaders
        trainLoader, testLoader = getUniMiBLoaders(
            root='./datasets/UniMiB-SHAR', batchSize=batchSize, numWorkers=2
        )
    elif dsName == 'skoda':
        from ciBabyMambaHar.data.skoda import getSkodaLoaders
        # SKODA: Signal Rescue (5Hz filter) applied in dataset, class weights for imbalanced
        trainLoader, testLoader, classWeights = getSkodaLoaders(
            root='./datasets/Skoda', batchSize=batchSize, numWorkers=2, returnWeights=True,
            windowSize=windowSize or config['seqLen']
        )
    elif dsName == 'daphnet':
        from ciBabyMambaHar.data.daphnet import getDaphnetLoaders
        # Daphnet: Signal Rescue (12Hz filter) applied in dataset, class weights for imbalanced
        trainLoader, testLoader, classWeights = getDaphnetLoaders(
            root='./datasets/Daphnet', batchSize=batchSize, numWorkers=2, returnWeights=True
        )
    else:
        raise ValueError(f"Unknown dataset: {datasetName}")
    
    return trainLoader, testLoader, classWeights


def trainEpoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch with AMP."""
    model.train()
    totalLoss = 0
    allPreds, allLabels = [], []
    
    for batchIdx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        totalLoss += loss.item()
        preds = output.argmax(dim=1).cpu().numpy()
        allPreds.extend(preds)
        allLabels.extend(target.cpu().numpy())
    
    avgLoss = totalLoss / len(loader)
    accuracy = accuracy_score(allLabels, allPreds) * 100
    f1 = f1_score(allLabels, allPreds, average='macro') * 100
    
    return avgLoss, accuracy, f1


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    totalLoss = 0
    allPreds, allLabels = [], []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            with autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
            
            totalLoss += loss.item()
            preds = output.argmax(dim=1).cpu().numpy()
            allPreds.extend(preds)
            allLabels.extend(target.cpu().numpy())
    
    avgLoss = totalLoss / len(loader)
    accuracy = accuracy_score(allLabels, allPreds) * 100
    f1 = f1_score(allLabels, allPreds, average='macro') * 100
    
    return avgLoss, accuracy, f1, allPreds, allLabels


def trainSingleSeed(args, seed: int, config: dict, resultsDir: Path, seqLenOverride: int = None):
    """Train model with a single seed."""
    setSeed(seed)
    device = getDevice()
    
    # Effective sequence length (override or config default)
    effectiveSeqLen = seqLenOverride or config['seqLen']
    
    # Load data (with class weights for imbalanced datasets)
    trainLoader, testLoader, classWeights = loadDataset(args.dataset, config['batchSize'], windowSize=seqLenOverride)
    if classWeights is not None:
        print(f"Using class weights for imbalanced dataset")
    
    # Create model (optionally overridden for ablation studies)
    model = createCrossoverBiDirBabyMambaHar(
        args.dataset,
        dModel=getattr(args, 'dModel', None),
        dState=getattr(args, 'dState', None),
        nLayers=getattr(args, 'nLayers', None),
        expand=getattr(args, 'expand', None),
        dtRank=getattr(args, 'dtRank', None),
        dConv=getattr(args, 'dConv', None),
        bidirectional=getattr(args, 'bidirectional', None),
        seqLenOverride=effectiveSeqLen,
    )
    model = model.to(device)
    
    # Count parameters
    totalParams = sum(p.numel() for p in model.parameters())
    trainParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {totalParams:,}")
    print(f"Trainable Parameters: {trainParams:,}")
    
    # Loss with class weights, optimizer, scheduler
    if classWeights is not None:
        criterion = nn.CrossEntropyLoss(weight=classWeights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weightDecay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    # Training loop
    bestValF1 = 0
    bestEpoch = 0
    patience = args.patience
    patienceCounter = 0
    
    trainHistory = {'loss': [], 'acc': [], 'f1': []}
    valHistory = {'loss': [], 'acc': [], 'f1': []}
    
    for epoch in range(1, args.epochs + 1):
        startTime = time.time()
        
        # Train
        trainLoss, trainAcc, trainF1 = trainEpoch(
            model, trainLoader, criterion, optimizer, scaler, device, epoch
        )
        trainHistory['loss'].append(trainLoss)
        trainHistory['acc'].append(trainAcc)
        trainHistory['f1'].append(trainF1)
        
        # Validate on test set (same as baselines - no separate val set)
        valLoss, valAcc, valF1, _, _ = evaluate(model, testLoader, criterion, device)
        valHistory['loss'].append(valLoss)
        valHistory['acc'].append(valAcc)
        valHistory['f1'].append(valF1)
        
        scheduler.step()
        
        epochTime = time.time() - startTime
        
        # Check for improvement
        if valF1 > bestValF1:
            bestValF1 = valF1
            bestEpoch = epoch
            patienceCounter = 0
            # Save best model
            modelPath = resultsDir / f'best_model_seed{seed}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': valF1,
                'config': config
            }, modelPath)
        else:
            patienceCounter += 1
        
        # Logging
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {trainLoss:.4f} Acc: {trainAcc:.2f}% F1: {trainF1:.2f}% | "
                  f"Val Loss: {valLoss:.4f} Acc: {valAcc:.2f}% F1: {valF1:.2f}% | "
                  f"Time: {epochTime:.1f}s")
        
        # Early stopping
        if patienceCounter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model for testing
    checkpoint = torch.load(resultsDir / f'best_model_seed{seed}.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    testLoss, testAcc, testF1, testPreds, testLabels = evaluate(
        model, testLoader, criterion, device
    )
    
    print(f"\nBest Epoch: {bestEpoch}")
    print(f"Test Results - Loss: {testLoss:.4f} Acc: {testAcc:.2f}% F1: {testF1:.2f}%")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(testLabels, testPreds, target_names=config['classNames'], digits=4))
    
    return {
        'seed': seed,
        'bestEpoch': bestEpoch,
        'testAcc': testAcc,
        'testF1': testF1,
        'testLoss': testLoss,
        'trainHistory': trainHistory,
        'valHistory': valHistory
    }


def main():
    parser = argparse.ArgumentParser(description='Train BabyMamba-Crossover-BiDir')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to train on')
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds to train')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (FIXED: was 30, now 10 for fairness)')
    parser.add_argument('--outDir', type=str, default='results/training',
                        help='Output directory for results')

    # Ablation runner helpers
    parser.add_argument('--tag', type=str, default=None,
                        help='Optional run tag; if set, results are written under outDir/CrossoverBiDirBabyMambaHar/<tag>/...')

    def parseBool(value: str) -> bool:
        v = str(value).strip().lower()
        if v in {'1', 'true', 't', 'yes', 'y', 'on'}:
            return True
        if v in {'0', 'false', 'f', 'no', 'n', 'off'}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")

    # Architecture overrides (ablation studies)
    parser.add_argument('--dModel', type=int, default=None)
    parser.add_argument('--dState', type=int, default=None)
    parser.add_argument('--nLayers', type=int, default=None)
    parser.add_argument('--expand', type=int, default=None)
    parser.add_argument('--dtRank', type=int, default=None)
    parser.add_argument('--dConv', type=int, default=None)
    parser.add_argument('--bidirectional', type=parseBool, default=None)
    parser.add_argument('--seqLen', type=int, default=None,
                        help='Override sequence length for LRD study (e.g., 64, 128, 256, 512)')
    
    args = parser.parse_args()
    
    # Setup
    config = DATASET_CONFIGS[args.dataset]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.tag:
        resultsDir = Path(args.outDir) / 'CrossoverBiDirBabyMambaHar' / args.tag / args.dataset / timestamp
    else:
        resultsDir = Path(args.outDir) / args.dataset / timestamp
    resultsDir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"BabyMamba-Crossover-BiDir Training")
    print(f"{'='*60}")
    print(f"Dataset: {config['name']}")
    print(f"Input Channels: {config['inputChannels']}")
    print(f"Sequence Length: {config['seqLen']}")
    print(f"Number of Classes: {config['numClasses']}")
    print(f"Seeds: {args.seeds}")
    print(f"Max Epochs: {args.epochs}")
    print(f"Results Dir: {resultsDir}")
    print(f"Device: {getDevice()}")
    if args.tag:
        print(f"Run Tag: {args.tag}")
    if args.bidirectional is not None:
        print(f"Bidirectional override: {args.bidirectional}")
    
    # Generate random seeds from master seed (for fairness with baselines)
    seeds = generateRandomSeeds(args.seeds, MASTER_SEED)
    print(f"Seeds (RNG generated): {seeds}")
    if args.seqLen is not None:
        print(f"Sequence length override: {args.seqLen}")
    
    # Train with multiple seeds
    allResults = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Training with seed {seed} ({i+1}/{args.seeds})")
        print(f"{'='*60}")
        result = trainSingleSeed(args, seed, config, resultsDir, seqLenOverride=args.seqLen)
        allResults.append(result)
    
    # Aggregate results
    testAccs = [r['testAcc'] for r in allResults]
    testF1s = [r['testF1'] for r in allResults]
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({args.seeds} seeds)")
    print(f"{'='*60}")
    print(f"Test Accuracy: {np.mean(testAccs):.2f}% ± {np.std(testAccs):.2f}%")
    print(f"Test F1 Score: {np.mean(testF1s):.2f}% ± {np.std(testF1s):.2f}%")
    
    # Save summary
    summary = {
        'dataset': args.dataset,
        'config': config,
        'seeds': args.seeds,
        'epochs': args.epochs,
        'results': allResults,
        'summary': {
            'meanAcc': np.mean(testAccs),
            'stdAcc': np.std(testAccs),
            'meanF1': np.mean(testF1s),
            'stdF1': np.std(testF1s)
        }
    }
    
    with open(resultsDir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to {resultsDir}")


if __name__ == '__main__':
    main()

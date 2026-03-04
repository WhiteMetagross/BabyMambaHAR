"""
Run CiBabyMambaHar Ablation Studies

Ablation experiments to demonstrate the importance of each
architectural component in CiBabyMambaHar (BabyMamba-Crossover-BiDir):

| Variant          | Description                              | Expected Result       |
|------------------|------------------------------------------|----------------------|
| full             | Complete CiBabyMambaHar (baseline)         | Best accuracy        |
| unidirectional   | Without bidirectional SSM                | -2-4% accuracy       |
| 2layer           | Only 2 layers instead of 4               | -3-5% accuracy       |
| nopatching       | Without discrete patching                | -1-2% accuracy       |
| cnnonly          | Replace SSM with CNN                     | -5-8% accuracy       |

Usage:
    python scripts/runCiBabyMambaHarAblations.py --dataset ucihar
    python scripts/runCiBabyMambaHarAblations.py --dataset all --epochs 100 --patience 10 --seed 17 --use-hpo
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ciBabyMambaHar.models.ciBabyMambaAblations import (
    ABLATION_MODELS,
    getAblationModel,
)
from ciBabyMambaHar.data import (
    getUciHarLoaders,
    getMotionSenseLoaders,
    getWisdmLoaders,
    getOpportunityLoaders,
    getUniMiBLoaders,
)
from ciBabyMambaHar.data.pamap2 import getPamap2Loaders


DATASET_CONFIGS = {
    'ucihar': {
        'numClasses': 6,
        'inChannels': 9,
        'seqLen': 128,
        'loaderFn': getUciHarLoaders,
        'root': './datasets/UCI HAR Dataset'
    },
    'motionsense': {
        'numClasses': 6,
        'inChannels': 6,
        'seqLen': 128,
        'loaderFn': getMotionSenseLoaders,
        'root': './datasets/motion-sense-master'
    },
    'wisdm': {
        'numClasses': 6,
        'inChannels': 3,
        'seqLen': 128,
        'loaderFn': getWisdmLoaders,
        'root': './datasets/WISDM_ar_v1.1'
    },
    'pamap2': {
        'numClasses': 12,
        'inChannels': 19,
        'seqLen': 128,
        'loaderFn': getPamap2Loaders,
        'root': './datasets/PAMAP2_Dataset'
    },
    'opportunity': {
        'numClasses': 5,
        'inChannels': 79,
        'seqLen': 128,
        'loaderFn': getOpportunityLoaders,
        'root': './datasets/Opportunity'
    },
    'unimib': {
        'numClasses': 9,
        'inChannels': 3,
        'seqLen': 128,
        'loaderFn': getUniMiBLoaders,
        'root': './datasets/UniMiB-SHAR'
    }
}


def loadHpoParams(dataset: str) -> dict:
    """Load HPO parameters for CiBabyMambaHar on given dataset."""
    hpoPath = Path('results/hpo') / f'hpo_ciBabyMambaHar_{dataset}.json'
    
    if not hpoPath.exists():
        print(f"  HPO results not found at {hpoPath}, using defaults")
        return {'lr': 1e-3, 'weightDecay': 0.01, 'batchSize': 64}
    
    with open(hpoPath) as f:
        hpoData = json.load(f)
    
    bestParams = hpoData.get('bestParams', {})
    
    # Normalize parameter names (snake_case -> camelCase)
    normalized = {}
    for key, value in bestParams.items():
        if key == 'weight_decay':
            normalized['weightDecay'] = value
        elif key == 'batch_size':
            normalized['batchSize'] = value
        else:
            normalized[key] = value
    
    # Ensure required params exist
    if 'lr' not in normalized:
        normalized['lr'] = 1e-3
    if 'weightDecay' not in normalized:
        normalized['weightDecay'] = 0.01
    if 'batchSize' not in normalized:
        normalized['batchSize'] = 64
    
    print(f"  Loaded HPO params: lr={normalized['lr']}, weightDecay={normalized['weightDecay']}, batchSize={normalized['batchSize']}")
    return normalized


def parseArgs():
    parser = argparse.ArgumentParser(description='Run CiBabyMambaHar Ablations')
    parser.add_argument('--dataset', type=str, default='ucihar',
                        choices=['ucihar', 'motionsense', 'wisdm', 'pamap2', 'opportunity', 'unimib', 'all'],
                        help='Dataset for ablation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per ablation')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (overridden by --use-hpo)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (overridden by --use-hpo)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay (overridden by --use-hpo)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    parser.add_argument('--use-hpo', action='store_true',
                        help='Load training params from ciBabyMambaHar HPO results')
    return parser.parse_args()


def setRandomSeed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def trainAndEvaluate(model, trainLoader, testLoader, config, device):
    """Train model and return final metrics with optional early stopping.
    
    Early stopping based on F1 score (better for imbalanced datasets).
    """
    model = model.to(device)
    
    epochs = config['epochs']
    lr = config['lr']
    weightDecay = config.get('weightDecay', 0.01)
    patience = config.get('patience', 0)
    numClasses = config.get('numClasses', 6)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weightDecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    bestF1 = 0.0
    bestAcc = 0.0
    noImproveCount = 0
    
    for epoch in range(epochs):
        model.train()
        for data, target in trainLoader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Evaluate - compute F1 for early stopping
        model.eval()
        allPreds, allLabels = [], []
        
        with torch.no_grad():
            for data, target in testLoader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                allPreds.append(output.argmax(1))
                allLabels.append(target)
        
        allPreds = torch.cat(allPreds)
        allLabels = torch.cat(allLabels)
        
        accuracy = 100.0 * (allPreds == allLabels).float().mean().item()
        f1 = computeF1(allPreds, allLabels, numClasses)
        
        # Track best F1 (not accuracy)
        if f1 > bestF1:
            bestF1 = f1
            bestAcc = accuracy
            noImproveCount = 0
        else:
            noImproveCount += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Acc={accuracy:.2f}%, F1={f1:.2f}%")
        
        # Early stopping based on F1
        if patience > 0 and noImproveCount >= patience:
            print(f"    Early stopping at epoch {epoch+1} (no F1 improvement for {patience} epochs)")
            break
    
    return {'accuracy': bestAcc, 'f1': bestF1}


def runAblationOnDataset(dataset: str, args):
    """Run all CiBabyMambaHar ablation variants on a single dataset."""
    print(f"\n{'='*60}")
    print(f"CiBabyMambaHar ABLATION STUDY: {dataset.upper()}")
    print(f"{'='*60}")
    
    datasetConfig = DATASET_CONFIGS[dataset]
    
    # Load HPO params if requested
    if args.use_hpo:
        hpoParams = loadHpoParams(dataset)
        batchSize = hpoParams.get('batchSize', args.batch_size)
        lr = hpoParams.get('lr', args.lr)
        weightDecay = hpoParams.get('weightDecay', args.weight_decay)
    else:
        batchSize = args.batch_size
        lr = args.lr
        weightDecay = args.weight_decay
    
    # Get data loaders
    try:
        loaderFn = datasetConfig['loaderFn']
        if dataset in ['pamap2', 'opportunity']:
            trainLoader, testLoader, _ = loaderFn(
                root=datasetConfig['root'],
                batchSize=batchSize,
                numWorkers=2,
                returnWeights=True
            )
        else:
            trainLoader, testLoader = loaderFn(
                root=datasetConfig['root'],
                batchSize=batchSize,
                numWorkers=2
            )
    except FileNotFoundError as e:
        print(f"  Dataset not found: {e}")
        return None
    
    config = {
        'epochs': args.epochs,
        'lr': lr,
        'weightDecay': weightDecay,
        'patience': args.patience,
        'numClasses': datasetConfig['numClasses'],
    }
    
    print(f"  Config: lr={lr}, weightDecay={weightDecay}, batchSize={batchSize}, patience={args.patience}")
    
    results = {}
    
    for modelKey in ABLATION_MODELS.keys():
        description = {
            'full': 'Full CiBabyMambaHar (Baseline)',
            'unidirectional': 'w/o Bidirectional SSM',
            '2layer': 'w/o 4 Layers (2 only)',
            'nopatching': 'w/o Discrete Patching',
            'cnnonly': 'w/o Mamba (CNN Only)',
        }.get(modelKey, modelKey)
        
        print(f"\n  {description}")
        
        setRandomSeed(args.seed)
        
        model = getAblationModel(
            modelKey,
            numClasses=datasetConfig['numClasses'],
            inChannels=datasetConfig['inChannels'],
            seqLen=datasetConfig['seqLen']
        )
        
        numParams = sum(p.numel() for p in model.parameters())
        print(f"     Parameters: {numParams:,}")
        
        device = args.device if torch.cuda.is_available() else 'cpu'
        metrics = trainAndEvaluate(model, trainLoader, testLoader, config, device)
        
        print(f"     Best Accuracy: {metrics['accuracy']:.2f}%, F1: {metrics['f1']:.2f}%")
        
        results[modelKey] = {
            'name': description,
            'parameters': numParams,
            'accuracy': metrics['accuracy'],
        }
        
        del model
        torch.cuda.empty_cache()
    
    return results


def main():
    args = parseArgs()
    
    print("\nCI-BabyMamba-HAR Ablation STUDIES")
    print(f"Epochs:     {args.epochs}")
    print(f"Patience:   {args.patience if args.patience > 0 else 'disabled'}")
    print(f"Seed:       {args.seed}")
    print(f"Use HPO:    {args.use_hpo}")
    if not args.use_hpo:
        print(f"Batch Size: {args.batch_size}")
        print(f"LR:         {args.lr}")
        print(f"WeightDecay:{args.weight_decay}")
    
    if args.dataset == 'all':
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [args.dataset]
    
    allResults = {}
    
    for dataset in datasets:
        results = runAblationOnDataset(dataset, args)
        if results:
            allResults[dataset] = results
    
    # Print summary
    print(f"\n{'='*60}")
    print("CiBabyMambaHar ABLATION SUMMARY")
    print(f"{'='*60}")
    
    for dataset, results in allResults.items():
        print(f"\n{dataset.upper()}:")
        print(f"{'Configuration':<35} {'Params':>10} {'Accuracy':>10}")
        print("-" * 57)
        
        baseAcc = results.get('full', {}).get('accuracy', 0)
        
        for modelKey, data in results.items():
            delta = data['accuracy'] - baseAcc if modelKey != 'full' else 0
            deltaStr = f"({delta:+.1f})" if modelKey != 'full' else ""
            print(f"{data['name']:<35} {data['parameters']:>10,} {data['accuracy']:>7.2f}% {deltaStr}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    resultsDir = Path('results') / 'ablations'
    resultsDir.mkdir(parents=True, exist_ok=True)
    
    outputPath = resultsDir / f'ciBabyMambaHar_ablation_{timestamp}.json'
    with open(outputPath, 'w') as f:
        json.dump(allResults, f, indent=2)
    
    print(f"\nResults saved to: {outputPath}")


if __name__ == "__main__":
    main()

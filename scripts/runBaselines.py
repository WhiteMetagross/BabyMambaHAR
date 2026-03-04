"""
Run Baselines for Comparison

Trains baseline models (TinyHAR, DeepConvLSTM) on Big 4 datasets
for fair comparison against BabyMamba.

Usage:
    python scripts/runBaselines.py --dataset ucihar
    python scripts/runBaselines.py --dataset all --model tinyhar
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.tinyHar import TinyHAR
from baselines.deepConvLstm import DeepConvLSTM, LightDeepConvLSTM, TinierHAR


# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS = 100
PATIENCE = 15

DATASET_SPECS = {
    'ucihar': {'numClasses': 6, 'inChannels': 9, 'seqLen': 128, 'root': './datasets/UCI HAR Dataset'},
    'motionsense': {'numClasses': 6, 'inChannels': 6, 'seqLen': 128, 'root': './datasets/motion-sense-master'},
    'wisdm': {'numClasses': 6, 'inChannels': 3, 'seqLen': 128, 'root': './datasets/WISDM_ar_v1.1'},
    'pamap2': {'numClasses': 12, 'inChannels': 19, 'seqLen': 128, 'root': './datasets/PAMAP2_Dataset'}
}


def setSeed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def getDataLoaders(dataset: str, batchSize: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Get data loaders."""
    if dataset == 'ucihar':
        from ciBabyMambaHar.data.uciHar import getUciHarLoaders
        return getUciHarLoaders(root=DATASET_SPECS[dataset]['root'], batchSize=batchSize)
    elif dataset == 'motionsense':
        from ciBabyMambaHar.data.motionSense import getMotionSenseLoaders
        return getMotionSenseLoaders(root=DATASET_SPECS[dataset]['root'], batchSize=batchSize)
    elif dataset == 'wisdm':
        from ciBabyMambaHar.data.wisdm import getWisdmLoaders
        return getWisdmLoaders(root=DATASET_SPECS[dataset]['root'], batchSize=batchSize)
    elif dataset == 'pamap2':
        from ciBabyMambaHar.data.pamap2 import getPamap2Loaders
        return getPamap2Loaders(root=DATASET_SPECS[dataset]['root'], batchSize=batchSize)
    raise ValueError(f"Unknown dataset: {dataset}")


def createModel(modelName: str, numClasses: int, inChannels: int) -> nn.Module:
    """Create baseline model."""
    if modelName == 'tinyhar':
        return TinyHAR(numClasses=numClasses, inChannels=inChannels)
    elif modelName == 'tinierhar':
        return TinierHAR(numClasses=numClasses, inChannels=inChannels)
    elif modelName == 'deepconvlstm':
        return DeepConvLSTM(numClasses=numClasses, inChannels=inChannels)
    elif modelName == 'lightdeepconvlstm':
        return LightDeepConvLSTM(numClasses=numClasses, inChannels=inChannels)
    raise ValueError(f"Unknown model: {modelName}")


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


def trainBaseline(
    modelName: str,
    dataset: str,
    epochs: int = EPOCHS,
    patience: int = PATIENCE
) -> Dict[str, Any]:
    """Train a baseline model."""
    
    setSeed(SEED)
    
    spec = DATASET_SPECS[dataset]
    
    print(f"\nTraining {modelName.upper()} on {dataset.upper()}")
    
    # Get data
    trainLoader, testLoader = getDataLoaders(dataset)
    
    # Create model
    model = createModel(modelName, spec['numClasses'], spec['inChannels']).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # AMP
    scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    bestAcc = 0.0
    bestF1 = 0.0
    epochsNoImprove = 0
    
    startTime = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for x, y in trainLoader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    output = model(x)
                    loss = criterion(output, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
        
        scheduler.step()
        
        # Evaluate - compute F1 for early stopping
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
        
        acc = 100.0 * (allPreds == allLabels).float().mean().item()
        f1 = computeF1(allPreds, allLabels, spec['numClasses'])
        
        # Early stopping based on F1 score
        if f1 > bestF1:
            bestF1 = f1
            bestAcc = acc
            epochsNoImprove = 0
            marker = "*"
        else:
            epochsNoImprove += 1
            marker = " "
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  {marker} Epoch {epoch:3d}/{epochs}: Acc = {acc:.2f}%, F1 = {f1:.2f}% (best F1: {bestF1:.2f}%)")
        
        if epochsNoImprove >= patience:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    trainTime = time.time() - startTime
    
    print(f"\n   Best Accuracy: {bestAcc:.2f}%")
    print(f"   Best F1 Score: {bestF1:.2f}%")
    print(f"   Training Time: {trainTime:.1f}s")
    
    return {
        'model': modelName,
        'dataset': dataset,
        'accuracy': bestAcc,
        'f1': bestF1,
        'parameters': params,
        'trainTime': trainTime,
        'epochs': epoch
    }


def runAllBaselines(datasets: List[str] = None, models: List[str] = None):
    """Run all baselines on all datasets."""
    
    if datasets is None:
        datasets = list(DATASET_SPECS.keys())
    if models is None:
        models = ['tinyhar', 'lightdeepconvlstm']
    
    results = {}
    
    for dataset in datasets:
        results[dataset] = {}
        for modelName in models:
            try:
                result = trainBaseline(modelName, dataset)
                results[dataset][modelName] = result
            except Exception as e:
                print(f"{modelName} on {dataset} failed: {e}")
                results[dataset][modelName] = {'error': str(e)}
    
    # Summary table
    print(f"\nBASELINE RESULTS SUMMARY")
    print(f"\n| Model | Dataset | Params | Accuracy | F1 Score |")
    print(f"|-------|---------|--------|----------|----------|")
    
    for dataset, dataResults in results.items():
        for modelName, result in dataResults.items():
            if 'error' in result:
                print(f"| {modelName} | {dataset} | - | FAILED | - |")
            else:
                print(f"| {modelName} | {dataset} | {result['parameters']:,} | {result['accuracy']:.1f}% | {result.get('f1', 0.0):.1f}% |")
    
    # Save results
    outDir = Path("results/baselines")
    outDir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outPath = outDir / f"baselines_{timestamp}.json"
    
    with open(outPath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved: {outPath}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline models")
    
    parser.add_argument('--dataset', '-d', type=str, default='all',
                        choices=['ucihar', 'motionsense', 'wisdm', 'pamap2', 'all'])
    parser.add_argument('--model', '-m', type=str, default='all',
                        choices=['tinyhar', 'tinierhar', 'deepconvlstm', 'lightdeepconvlstm', 'all'])
    parser.add_argument('--epochs', '-e', type=int, default=EPOCHS)
    
    args = parser.parse_args()
    
    datasets = list(DATASET_SPECS.keys()) if args.dataset == 'all' else [args.dataset]
    models = ['tinyhar', 'tinierhar'] if args.model == 'all' else [args.model]
    
    runAllBaselines(datasets, models)


if __name__ == '__main__':
    main()

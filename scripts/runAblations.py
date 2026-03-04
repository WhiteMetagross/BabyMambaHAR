"""
Run Ablation Studies

Runs the ablation experiments to demonstrate the importance of each
architectural component:
1. w/o SE Block: Importance of channel attention
2. w/o Residual: Importance of gradient flow  
3. w/o Mamba (CNN Only): Importance of SSM for temporal modeling

Usage:
    python scripts/runAblations.py --dataset ucihar
    python scripts/runAblations.py --dataset all --epochs 100
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ciBabyMambaHar.models import (
    BabyMamba,
    BabyMambaNoSe,
    BabyMambaNoResidual,
    BabyMambaCnnOnly
)
from ciBabyMambaHar.data import (
    getUciHarLoaders,
    getMotionSenseLoaders,
    getWisdmLoaders
)
from ciBabyMambaHar.utils import (
    getOptimizer,
    getScheduler,
    Accuracy,
    F1Score,
    AverageMeter,
    countParameters
)


# Model configurations for ablation
ABLATION_MODELS = {
    'BabyMamba': {
        'class': BabyMamba,
        'name': 'BabyMamba (Full)',
        'description': 'Complete model with SE-SSM'
    },
    'noSe': {
        'class': BabyMambaNoSe,
        'name': 'w/o SE Block',
        'description': 'Without Squeeze-and-Excitation'
    },
    'noResidual': {
        'class': BabyMambaNoResidual,
        'name': 'w/o Residual',
        'description': 'Without residual connections'
    },
    'cnnOnly': {
        'class': BabyMambaCnnOnly,
        'name': 'w/o Mamba (CNN)',
        'description': 'Pure CNN instead of SSM'
    }
}

DATASET_CONFIGS = {
    'ucihar': {
        'numClasses': 6,
        'inChannels': 9,
        'loaderFn': getUciHarLoaders,
        'root': './datasets/UCI HAR Dataset'
    },
    'motionsense': {
        'numClasses': 6,
        'inChannels': 6,
        'loaderFn': getMotionSenseLoaders,
        'root': './datasets/motion-sense-master'
    },
    'wisdm': {
        'numClasses': 6,
        'inChannels': 3,
        'loaderFn': getWisdmLoaders,
        'root': './datasets/WISDM_ar_v1.1'
    }
}


def parseArgs():
    parser = argparse.ArgumentParser(description='Run BabyMamba Ablations')
    parser.add_argument('--dataset', type=str, default='ucihar',
                        choices=['ucihar', 'motionsense', 'wisdm', 'all'],
                        help='Dataset for ablation')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    return parser.parse_args()


def setRandomSeed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainAndEvaluate(model, trainLoader, testLoader, config, device):
    """Train model and return final metrics. Early stopping based on F1 score."""
    model = model.to(device)
    
    epochs = config['epochs']
    lr = config['lr']
    numClasses = config['numClasses']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = getOptimizer(model, name='adamw', lr=lr, weightDecay=0.01)
    scheduler = getScheduler(optimizer, name='cosine', epochs=epochs)
    
    bestF1 = 0.0
    bestAcc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for data, target in trainLoader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Evaluate
        model.eval()
        accuracy = Accuracy()
        f1 = F1Score(numClasses)
        
        with torch.no_grad():
            for data, target in testLoader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                accuracy.update(output, target)
                f1.update(output, target)
        
        # Track best F1 (not accuracy)
        currentF1 = f1.compute()['f1']
        if currentF1 > bestF1:
            bestF1 = currentF1
            bestAcc = accuracy.value
        
        # Progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Acc={accuracy.value:.2f}%, F1={currentF1:.2f}%")
    
    # Final evaluation
    model.eval()
    accuracy = Accuracy()
    f1 = F1Score(numClasses)
    
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            accuracy.update(output, target)
            f1.update(output, target)
    
    return {
        'accuracy': accuracy.value,
        'f1': f1.compute()['f1'],
        'bestAcc': bestAcc,
        'bestF1': bestF1
    }


def runAblationOnDataset(dataset: str, args):
    """Run all ablation models on a single dataset."""
    print(f"\nABLATION STUDY: {dataset.upper()}")
    
    datasetConfig = DATASET_CONFIGS[dataset]
    
    # Get data loaders
    try:
        loaderFn = datasetConfig['loaderFn']
        trainLoader, testLoader = loaderFn(
            root=datasetConfig['root'],
            batchSize=args.batch_size,
            numWorkers=2
        )
    except FileNotFoundError as e:
        print(f"  Dataset not found: {e}")
        print(f"  Run: python scripts/downloadBenchmarkDatasets.py")
        return None
    
    config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'numClasses': datasetConfig['numClasses']
    }
    
    results = {}
    
    for modelKey, modelInfo in ABLATION_MODELS.items():
        print(f"\n  {modelInfo['name']}: {modelInfo['description']}")
        
        setRandomSeed(args.seed)
        
        # Create model
        ModelClass = modelInfo['class']
        model = ModelClass(
            numClasses=datasetConfig['numClasses'],
            inChannels=datasetConfig['inChannels']
        )
        
        numParams = sum(p.numel() for p in model.parameters())
        print(f"     Parameters: {numParams:,}")
        
        # Train and evaluate
        device = args.device if torch.cuda.is_available() else 'cpu'
        metrics = trainAndEvaluate(model, trainLoader, testLoader, config, device)
        
        print(f"     Accuracy: {metrics['accuracy']:.2f}%, F1: {metrics['f1']:.2f}%")
        
        results[modelKey] = {
            'name': modelInfo['name'],
            'parameters': numParams,
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1']
        }
    
    return results


def main():
    args = parseArgs()
    
    print("\\nBabyMamba-HAR Ablation Studies")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LR:         {args.lr}")
    print(f"Seed:       {args.seed}")
    
    # Determine datasets
    if args.dataset == 'all':
        datasets = ['ucihar', 'motionsense', 'wisdm']
    else:
        datasets = [args.dataset]
    
    allResults = {}
    
    for dataset in datasets:
        results = runAblationOnDataset(dataset, args)
        if results:
            allResults[dataset] = results
    
    # Print summary table
    print("\nABLATION SUMMARY (Table 2)")
    
    for dataset, results in allResults.items():
        print(f"\n{dataset.upper()}:")
        print(f"{'Configuration':<25} {'Params':>10} {'Accuracy':>10}")
        
        for modelKey, data in results.items():
            print(f"{data['name']:<25} {data['parameters']:>10,} {data['accuracy']:>9.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    resultsDir = Path('results') / 'ablations'
    resultsDir.mkdir(parents=True, exist_ok=True)
    
    outputPath = resultsDir / f'ablation_results_{timestamp}.json'
    with open(outputPath, 'w') as f:
        json.dump(allResults, f, indent=2)
    
    print(f"\nResults saved to: {outputPath}")


if __name__ == "__main__":
    main()

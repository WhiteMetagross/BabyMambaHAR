"""
Evaluate BabyMamba Model

Evaluate trained model with detailed metrics and profiling.

Usage:
    python scripts/evaluate.py --config configs/uciHar.yaml --checkpoint results/checkpoints/best.pt
"""

import os
import sys
import argparse
import json
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ciBabyMambaHar.models import BabyMamba
from ciBabyMambaHar.data import (
    getUciHarLoaders,
    getMotionSenseLoaders,
    getWisdmLoaders
)
from ciBabyMambaHar.utils import (
    loadCheckpoint,
    Accuracy,
    F1Score,
    ConfusionMatrix,
    profileModel,
    benchmarkLatency,
    computeMacs
)


def parseArgs():
    parser = argparse.ArgumentParser(description='Evaluate BabyMamba')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results')
    parser.add_argument('--profile', action='store_true',
                        help='Run profiling (MACs, latency)')
    return parser.parse_args()


def loadConfig(configPath: str) -> dict:
    with open(configPath, 'r') as f:
        return yaml.safe_load(f)


def getDataLoaders(config: dict):
    """Get test data loader based on config."""
    dataCfg = config['data']
    dataset = dataCfg['dataset'].lower()
    batchSize = dataCfg['batchSize']
    numWorkers = dataCfg.get('numWorkers', 2)
    
    if dataset == 'ucihar':
        _, testLoader = getUciHarLoaders(
            root=dataCfg.get('root', './datasets/UCI HAR Dataset'),
            batchSize=batchSize,
            numWorkers=numWorkers
        )
    elif dataset == 'motionsense':
        _, testLoader = getMotionSenseLoaders(
            root=dataCfg.get('root', './datasets/motion-sense-master'),
            batchSize=batchSize,
            numWorkers=numWorkers
        )
    elif dataset == 'wisdm':
        _, testLoader = getWisdmLoaders(
            root=dataCfg.get('root', './datasets/WISDM_ar_v1.1'),
            batchSize=batchSize,
            numWorkers=numWorkers
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return testLoader


def buildModel(config: dict) -> nn.Module:
    """Build model from config."""
    modelCfg = config['model']
    
    return BabyMamba(
        numClasses=modelCfg['numClasses'],
        inChannels=modelCfg['inChannels'],
        seqLen=modelCfg.get('seqLen', 128),
        variant=modelCfg.get('variant', 'nano'),
        dModel=modelCfg.get('dModel'),
        nLayers=modelCfg.get('nLayers'),
        dState=modelCfg.get('dState'),
        expand=modelCfg.get('expand', 2),
        stemType=modelCfg.get('stemType', 'hollow')
    )


@torch.no_grad()
def evaluate(model: nn.Module, testLoader, device: str, config: dict):
    """Comprehensive evaluation."""
    model.eval()
    
    numClasses = config['model']['numClasses']
    
    accuracy = Accuracy()
    f1 = F1Score(numClasses=numClasses)
    confMatrix = ConfusionMatrix(numClasses=numClasses)
    
    allPreds = []
    allTargets = []
    
    for data, target in tqdm(testLoader, desc='Evaluating'):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        
        accuracy.update(output, target)
        f1.update(output, target)
        confMatrix.update(output, target)
        
        allPreds.extend(output.argmax(dim=-1).cpu().tolist())
        allTargets.extend(target.cpu().tolist())
    
    f1Results = f1.compute()
    
    return {
        'accuracy': accuracy.value,
        'f1Macro': f1Results['f1'],
        'precision': f1Results['precision'],
        'recall': f1Results['recall'],
        'f1PerClass': f1Results['f1PerClass'],
        'confusionMatrix': confMatrix.compute().tolist(),
        'predictions': allPreds,
        'targets': allTargets
    }


def profileModelFull(model: nn.Module, config: dict, device: str):
    """Full model profiling with MACs and latency."""
    modelCfg = config['model']
    inputShape = (1, modelCfg.get('seqLen', 128), modelCfg['inChannels'])
    
    # Count parameters
    numParams = sum(p.numel() for p in model.parameters())
    paramsByComponent = model.countParameters()
    
    # Compute MACs
    macsResult = computeMacs(model, inputShape, device='cpu')
    
    # Benchmark latency
    latencyResult = benchmarkLatency(
        model, inputShape, device=device,
        numWarmup=20, numRuns=100, batchSizes=[1, 8, 32]
    )
    
    return {
        'parameters': {
            'total': numParams,
            'byComponent': paramsByComponent
        },
        'macs': macsResult.get('macs'),
        'flops': macsResult.get('flops'),
        'latency': latencyResult
    }


def main():
    args = parseArgs()
    
    # Load config
    config = loadConfig(args.config)
    
    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = buildModel(config)
    loadCheckpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    
    numParams = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {numParams:,}")
    
    # Load test data
    print("Loading test data...")
    testLoader = getDataLoaders(config)
    
    # Evaluate
    print("Running evaluation...")
    evalResults = evaluate(model, testLoader, device, config)
    
    # Profile if requested
    profileResults = None
    if args.profile:
        print("Running profiling...")
        profileResults = profileModelFull(model, config, device)
    
    # Print results
    print(f"\nEVALUATION RESULTS")
    print(f"Dataset:    {config['data']['dataset']}")
    print(f"Parameters: {numParams:,}")
    print(f"Accuracy:   {evalResults['accuracy']:.2f}%")
    print(f"F1 (Macro): {evalResults['f1Macro']:.2f}%")
    print(f"Precision:  {evalResults['precision']:.2f}%")
    print(f"Recall:     {evalResults['recall']:.2f}%")
    
    if profileResults:
        print(f"\nPROFILING RESULTS")
        if profileResults['macs']:
            print(f"MACs:       {profileResults['macs']:,}")
            print(f"FLOPs:      {profileResults['flops']:,}")
        
        for batchKey, latency in profileResults['latency'].items():
            print(f"Latency ({batchKey}): {latency['mean_ms']:.2f} ms "
                  f"(±{latency['std_ms']:.2f}), "
                  f"Throughput: {latency['throughput']:.1f} samples/sec")
    
    # Save results
    outputPath = args.output
    if outputPath is None:
        checkpointDir = Path(args.checkpoint).parent
        outputPath = checkpointDir / 'evaluation.json'
    
    results = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'dataset': config['data']['dataset'],
        'parameters': numParams,
        'evaluation': {
            'accuracy': evalResults['accuracy'],
            'f1Macro': evalResults['f1Macro'],
            'precision': evalResults['precision'],
            'recall': evalResults['recall'],
            'f1PerClass': evalResults['f1PerClass']
        }
    }
    
    if profileResults:
        results['profiling'] = profileResults
    
    with open(outputPath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {outputPath}")


if __name__ == "__main__":
    main()

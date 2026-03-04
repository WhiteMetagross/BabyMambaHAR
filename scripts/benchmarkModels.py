#!/usr/bin/env python3
"""
Benchmark Models - Latency, MACs, FLOPs, Parameters Comparison

Compares CiBabyMambaHar against baseline models (TinierHAR, TinyHAR, LightDeepConvLSTM)
on any dataset.

Usage:
    python scripts/benchmarkModels.py --dataset ucihar
    python scripts/benchmarkModels.py --dataset daphnet
    python scripts/benchmarkModels.py --dataset all
    python scripts/benchmarkModels.py --dataset ucihar --experiments 10 --runs 200
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ciBabyMambaHar.models import ciBabyMambaHar
from baselines import TinierHAR, TinyHAR, LightDeepConvLSTM

# Dataset specifications
DATASET_SPECS = {
    'ucihar': {'numClasses': 6, 'inChannels': 9, 'seqLen': 128, 'name': 'UCI-HAR'},
    'motionsense': {'numClasses': 6, 'inChannels': 6, 'seqLen': 128, 'name': 'MotionSense'},
    'wisdm': {'numClasses': 6, 'inChannels': 3, 'seqLen': 128, 'name': 'WISDM'},
    'pamap2': {'numClasses': 12, 'inChannels': 19, 'seqLen': 128, 'name': 'PAMAP2'},
    'opportunity': {'numClasses': 5, 'inChannels': 79, 'seqLen': 128, 'name': 'Opportunity'},
    'unimib': {'numClasses': 9, 'inChannels': 3, 'seqLen': 128, 'name': 'UniMiB-SHAR'},
    'skoda': {'numClasses': 11, 'inChannels': 30, 'seqLen': 98, 'name': 'Skoda'},
    'daphnet': {'numClasses': 2, 'inChannels': 9, 'seqLen': 64, 'name': 'Daphnet'},
}

# Model configurations (locked architectures from papers)
MODEL_CONFIGS = {
    'CiBabyMambaHar': {'dropout': 0.0},
    'tinierhar': {'nbFilters': 8, 'nbConvBlocks': 4, 'gruUnits': 16, 'dropout': 0.5},
    'tinyhar': {'filterNum': 24, 'nbConvLayers': 4, 'filterSize': 5, 'dropout': 0.5},
    'lightdeepconvlstm': {'convFilters': 16, 'lstmHidden': 32, 'dropout': 0.5},
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model(model_name: str, spec: Dict[str, Any]) -> torch.nn.Module:
    """Create a model with the given specification."""
    config = MODEL_CONFIGS[model_name]
    
    if model_name == 'CiBabyMambaHar':
        return CiBabyMambaHar(
            numClasses=spec['numClasses'],
            inChannels=spec['inChannels'],
            seqLen=spec['seqLen'],
            dropout=config['dropout']
        )
    elif model_name == 'tinierhar':
        return TinierHAR(
            numClasses=spec['numClasses'],
            inChannels=spec['inChannels'],
            seqLen=spec['seqLen'],
            **config
        )
    elif model_name == 'tinyhar':
        return TinyHAR(
            numClasses=spec['numClasses'],
            inChannels=spec['inChannels'],
            seqLen=spec['seqLen'],
            **config
        )
    elif model_name == 'lightdeepconvlstm':
        return LightDeepConvLSTM(
            numClasses=spec['numClasses'],
            inChannels=spec['inChannels'],
            seqLen=spec['seqLen'],
            **config
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def benchmark_model_single(
    model: torch.nn.Module,
    input_shape: tuple,
    warmup_runs: int = 50,
    benchmark_runs: int = 100,
    use_compile: bool = False
) -> Dict[str, Any]:
    """Single benchmark run for latency, MACs, FLOPs, and parameters."""
    
    model = model.to(DEVICE).eval()
    
    # Parameters
    params = sum(p.numel() for p in model.parameters())
    
    # Actual model size in MB (save to temp file and measure)
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 * 1024)
        os.unlink(f.name)
    
    # MACs/FLOPs using thop (before compile)
    try:
        from thop import profile
        x = torch.randn(1, input_shape[1], input_shape[2]).to(DEVICE)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        macs = int(macs)
        flops = macs * 2
    except Exception as e:
        print(f"   Warning: thop failed: {e}")
        macs, flops = 0, 0
    
    # Apply torch.compile if requested
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"   Warning: torch.compile failed: {e}")
    
    # Latency benchmark
    x = torch.randn(1, input_shape[1], input_shape[2]).to(DEVICE)
    
    # Warmup (more warmup needed for compiled models)
    actual_warmup = warmup_runs * 2 if use_compile else warmup_runs
    for _ in range(actual_warmup):
        with torch.no_grad():
            _ = model(x)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    latency_ms = sum(times) / len(times)
    throughput = 1000 / latency_ms
    
    return {
        'parameters': params,
        'size_mb': size_mb,
        'macs': macs,
        'flops': flops,
        'latency_ms': latency_ms,
        'throughput': throughput,
    }


def benchmark_model(
    model_name: str,
    spec: Dict[str, Any],
    input_shape: tuple,
    n_experiments: int = 5,
    warmup_runs: int = 50,
    benchmark_runs: int = 100,
    use_compile: bool = False
) -> Dict[str, Any]:
    """Run multiple benchmark experiments and compute mean ± std."""
    
    latencies = []
    throughputs = []
    params, macs, flops, size_mb = 0, 0, 0, 0.0
    
    for exp in range(n_experiments):
        # Create fresh model for each experiment
        model = create_model(model_name, spec)
        result = benchmark_model_single(model, input_shape, warmup_runs, benchmark_runs, use_compile)
        
        latencies.append(result['latency_ms'])
        throughputs.append(result['throughput'])
        params = result['parameters']
        size_mb = result['size_mb']
        macs = result['macs']
        flops = result['flops']
        
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Compute mean and std across experiments
    import numpy as np
    latency_mean = np.mean(latencies)
    latency_std = np.std(latencies)
    throughput_mean = np.mean(throughputs)
    throughput_std = np.std(throughputs)
    
    return {
        'parameters': params,
        'size_mb': size_mb,
        'macs': macs,
        'flops': flops,
        'latency_ms': latency_mean,
        'latency_std_ms': latency_std,
        'throughput': throughput_mean,
        'throughput_std': throughput_std,
        'n_experiments': n_experiments,
        'latencies': latencies,
        'compiled': use_compile,
    }


def benchmark_dataset(dataset: str, n_experiments: int = 5, warmup_runs: int = 50, benchmark_runs: int = 100, use_compile: bool = False) -> Dict[str, Any]:
    """Benchmark all models on a dataset with multiple experiments."""
    
    spec = DATASET_SPECS[dataset]
    input_shape = (1, spec['seqLen'], spec['inChannels'])
    
    compile_str = " + torch.compile" if use_compile else ""
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK - {spec['name']} (seqLen={spec['seqLen']}, channels={spec['inChannels']}, classes={spec['numClasses']}){compile_str}")
    print(f"{'=' * 80}")
    print(f"Device: {DEVICE}")
    print(f"torch.compile: {'ENABLED' if use_compile else 'DISABLED'}")
    print(f"Experiments: {n_experiments}, Warmup: {warmup_runs} runs, Benchmark: {benchmark_runs} runs each")
    
    results = {}
    models = ['CiBabyMambaHar', 'tinierhar', 'tinyhar', 'lightdeepconvlstm']
    
    for model_name in models:
        print(f"\n--- {model_name.upper()} ({n_experiments} experiments){compile_str} ---")
        
        try:
            result = benchmark_model(model_name, spec, input_shape, n_experiments, warmup_runs, benchmark_runs, use_compile)
            results[model_name] = result
            
            print(f"Parameters: {result['parameters']:,}")
            print(f"Size: {result['size_mb']:.3f} MB")
            print(f"MACs: {result['macs']:,}")
            print(f"FLOPs: {result['flops']:,}")
            print(f"Latency: {result['latency_ms']:.3f} ± {result['latency_std_ms']:.3f} ms")
            print(f"Throughput: {result['throughput']:.1f} ± {result['throughput_std']:.1f} samples/sec")
            print(f"  (individual runs: {[f'{l:.3f}' for l in result['latencies']]})")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
    
    # Print summary table
    print(f"\n{'=' * 80}")
    print(f"SUMMARY - {spec['name']} ({n_experiments} experiments){compile_str}")
    print(f"{'=' * 80}")
    print(f"| {'Model':<18} | {'Params':>10} | {'Size (MB)':>10} | {'MACs':>12} | {'FLOPs':>12} | {'Latency (ms)':>16} | {'Throughput (/s)':>18} |")
    print(f"|{'-' * 20}|{'-' * 12}|{'-' * 12}|{'-' * 14}|{'-' * 14}|{'-' * 18}|{'-' * 20}|")
    
    for model_name in models:
        if 'error' in results.get(model_name, {}):
            print(f"| {model_name:<18} | {'ERROR':>10} | {'-':>10} | {'-':>12} | {'-':>12} | {'-':>16} | {'-':>18} |")
        else:
            r = results[model_name]
            lat_str = f"{r['latency_ms']:.3f}±{r['latency_std_ms']:.3f}"
            thr_str = f"{r['throughput']:.0f}±{r['throughput_std']:.0f}"
            print(f"| {model_name:<18} | {r['parameters']:>10,} | {r['size_mb']:>10.3f} | {r['macs']:>12,} | {r['flops']:>12,} | {lat_str:>16} | {thr_str:>18} |")
    
    # Comparison with CiBabyMambaHar
    if 'CiBabyMambaHar' in results and 'error' not in results['CiBabyMambaHar']:
        nano = results['CiBabyMambaHar']
        print(f"\n--- Comparison vs CiBabyMambaHar ---")
        for model_name in ['tinierhar', 'tinyhar', 'lightdeepconvlstm']:
            if model_name in results and 'error' not in results[model_name]:
                r = results[model_name]
                param_ratio = r['parameters'] / nano['parameters']
                macs_ratio = r['macs'] / nano['macs'] if nano['macs'] > 0 else 0
                latency_ratio = r['latency_ms'] / nano['latency_ms']
                print(f"{model_name}: {param_ratio:.2f}x params, {macs_ratio:.2f}x MACs, {latency_ratio:.2f}x latency")
    
    return {
        'dataset': dataset,
        'spec': spec,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Models - Latency, MACs, FLOPs")
    parser.add_argument('--dataset', '-d', type=str, default='ucihar',
                        choices=list(DATASET_SPECS.keys()) + ['all'],
                        help='Dataset to benchmark (default: ucihar)')
    parser.add_argument('--experiments', '-e', type=int, default=5,
                        help='Number of experiments to average (default: 5)')
    parser.add_argument('--warmup', '-w', type=int, default=50,
                        help='Warmup runs per experiment (default: 50)')
    parser.add_argument('--runs', '-r', type=int, default=100,
                        help='Benchmark runs per experiment (default: 100)')
    parser.add_argument('--compile', '-c', action='store_true',
                        help='Use torch.compile for optimized inference')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        datasets = list(DATASET_SPECS.keys())
    else:
        datasets = [args.dataset]
    
    all_results = {}
    
    for dataset in datasets:
        result = benchmark_dataset(dataset, args.experiments, args.warmup, args.runs, args.compile)
        all_results[dataset] = result
    
    # Save results if requested
    if args.save:
        results_dir = Path("results/benchmarks")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        if args.dataset == 'all':
            out_path = results_dir / "benchmark_all.json"
        else:
            out_path = results_dir / f"benchmark_{args.dataset}.json"
        
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved: {out_path}")
    
    return all_results


if __name__ == '__main__':
    main()

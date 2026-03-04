"""
Inference Latency and Theoretical MACs Benchmark Script

This script measures CPU inference latency for all models and calculates
theoretical MAC (Multiply-Accumulate) operations.

Simulates two edge deployment scenarios:
1. Single-Core (Cortex-M7 style): Pure algorithmic efficiency test
2. Quad-Core (Raspberry Pi 4 style): Parallelization scaling test

Usage:
    python scripts/benchmarkLatency.py
    python scripts/benchmarkLatency.py --dataset ucihar --runs 1000
    python scripts/benchmarkLatency.py --all-datasets

Author: BabyMamba Team
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ciBabyMambaHar.models.ciBabyMamba import BabyMamba
from baselines import TinierHAR, TinyHAR, LightDeepConvLSTM


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_CONFIGS = {
    'ucihar': {'numClasses': 6, 'inChannels': 9, 'seqLen': 128},
    'motionsense': {'numClasses': 6, 'inChannels': 12, 'seqLen': 128},
    'wisdm': {'numClasses': 6, 'inChannels': 3, 'seqLen': 128},
    'pamap2': {'numClasses': 12, 'inChannels': 19, 'seqLen': 128},
}

# Golden Config (Locked Architectures)
BABYMAMBA_CONFIG = {
    'dModel': 24,
    'nLayers': 2,
    'dState': 16,
    'expand': 2,
    'variant': None,  # Use custom params
}

TINIER_HAR_CONFIG = {
    'nbFilters': 8,
    'nbConvBlocks': 4,
    'gruUnits': 16,
    'dropout': 0.5,
}

TINY_HAR_CONFIG = {
    'filterNum': 24,
    'nbConvLayers': 4,
    'filterSize': 5,
    'dropout': 0.5,
}

LIGHT_DCNN_CONFIG = {
    'convFilters': 16,
    'lstmHidden': 32,
    'dropout': 0.5,
}


# ============================================================================
# MAC CALCULATION FUNCTIONS
# ============================================================================

def calculateBabyMambaMacs(T: int, C: int, d: int, N: int, expand: int, n_layers: int) -> Dict[str, int]:
    """
    Calculate theoretical MACs for BabyMamba.
    
    Architecture breakdown:
    1. SpectralTemporalStem:
       - Time Branch: DW-Conv + PW-Conv
       - Freq Branch: FFT (not counted as MACs) + Linear
    
    2. BiDirectionalMambaBlock (×n_layers):
       - Input Projection: d → 2*expand*d
       - DW-Conv1D: k=4 on expanded dimension
       - SSM scan: O(T * expand * d * N) - but parallelizable
       - Output Projection: expand*d → d
       - SE Block: GlobalPool + FC bottleneck
       - Bidirectional: 2× forward pass
    
    3. Classification Head: GlobalPool + Linear
    
    Args:
        T: Sequence length (timesteps)
        C: Input channels
        d: Model dimension (d_model)
        N: State dimension (d_state)
        expand: Expansion factor
        n_layers: Number of SSM layers
    
    Returns:
        Dict with MAC breakdown by component
    """
    d_inner = expand * d  # Expanded dimension
    k_conv = 4  # SSM conv kernel size
    k_stem = 3  # Stem conv kernel size
    se_reduction = 4
    
    macs = {}
    
    # ===== STEM =====
    # Time Branch: DW-Conv(k=3) + PW-Conv
    stem_dw = T * C * k_stem  # Depthwise conv
    stem_pw = T * C * d  # Pointwise conv
    # Freq Branch: Linear(C → d) applied once per sample
    stem_freq = C * d
    macs['stem'] = stem_dw + stem_pw + stem_freq
    
    # ===== SSM BLOCKS (×n_layers, ×2 for bidirectional) =====
    per_direction = 0
    
    # Input projection: d → 2*d_inner (for x and residual gate)
    in_proj = T * d * (2 * d_inner)
    
    # DW-Conv1D: k=4 on d_inner channels
    dw_conv = T * d_inner * k_conv
    
    # SSM Projections: x_proj (d_inner → dt_rank + 2*N)
    dt_rank = max(1, d // 16)
    x_proj = T * d_inner * (dt_rank + 2 * N)
    dt_proj = T * dt_rank * d_inner
    
    # SSM Scan: Per timestep, update N states for d_inner channels
    # Each state update: A*h + B*x → N multiplications
    # Output: C @ h → N multiplications
    # Total per timestep: 2*N*d_inner
    ssm_scan = T * d_inner * N * 2
    
    # Output projection: d_inner → d
    out_proj = T * d_inner * d
    
    # Gating: element-wise multiply with residual (counted as MACs)
    gating = T * d_inner
    
    per_direction = in_proj + dw_conv + x_proj + dt_proj + ssm_scan + out_proj + gating
    
    # SE Block (per layer)
    se_fc1 = d * (d // se_reduction)
    se_fc2 = (d // se_reduction) * d
    se_scale = T * d  # Element-wise scaling
    se_total = se_fc1 + se_fc2 + se_scale
    
    # Per layer: 2 directions × SSM + SE
    per_layer = 2 * per_direction + se_total
    macs['ssm_blocks'] = per_layer * n_layers
    
    # ===== CLASSIFICATION HEAD =====
    # Global average pool is not counted (just division)
    # Final linear: d → num_classes (but varies by dataset, use 6 as default)
    macs['head'] = d * 6  # Conservative estimate
    
    macs['total'] = macs['stem'] + macs['ssm_blocks'] + macs['head']
    
    return macs


def calculate_tinierhar_macs(T: int, C: int, nb_filters: int, nb_blocks: int, gru_units: int) -> Dict[str, int]:
    """
    Calculate theoretical MACs for TinierHAR.
    
    Architecture:
    1. DepthwiseSeparableConv2D blocks (with MaxPool on first 2)
    2. Bidirectional GRU
    3. Temporal Attention
    4. Classifier
    
    Key formula for GRU (3 gates: reset, update, new):
        MACs_GRU = T × 3 × (H² + H×D)
        where H=hidden_size, D=input_dim
        For BiGRU: multiply by 2
    """
    macs = {}
    
    # ===== CONV BLOCKS =====
    # Block 1: 1 → nb_filters, with maxpool (T/2)
    # DW-Sep = DW + PW
    conv_block1_dw = T * 1 * 5  # kernel=5, groups=1
    conv_block1_pw = T * 1 * nb_filters
    
    # After maxpool: T' = T/2
    T1 = T // 2
    
    # Block 2: nb_filters → 2*nb_filters, with maxpool (T/4)
    conv_block2_dw = T1 * nb_filters * 5
    conv_block2_pw = T1 * nb_filters * (2 * nb_filters)
    
    T2 = T1 // 2  # T/4
    
    # Blocks 3-4: 2*nb_filters → 2*nb_filters, no maxpool
    conv_blocks_34 = 0
    for _ in range(nb_blocks - 2):
        conv_blocks_34 += T2 * (2 * nb_filters) * 5  # DW
        conv_blocks_34 += T2 * (2 * nb_filters) * (2 * nb_filters)  # PW
    
    # Shortcut convs (1x1 for channel mismatch)
    shortcut1 = T * 1 * nb_filters  # Block 1 shortcut
    shortcut2 = T1 * nb_filters * (2 * nb_filters)  # Block 2 shortcut
    
    macs['conv_blocks'] = (conv_block1_dw + conv_block1_pw + 
                           conv_block2_dw + conv_block2_pw + 
                           conv_blocks_34 + shortcut1 + shortcut2)
    
    # ===== BIDIRECTIONAL GRU =====
    # Input dim to GRU: 2*nb_filters * C (flattened)
    gru_input_dim = 2 * nb_filters * C
    H = gru_units
    
    # GRU has 3 gates (reset, update, new)
    # Each gate: input→hidden (D×H) + hidden→hidden (H×H)
    # Per direction, per timestep: 3 × (D×H + H×H)
    gru_per_step = 3 * (gru_input_dim * H + H * H)
    
    # BiGRU: 2 directions × T2 timesteps
    macs['gru'] = 2 * T2 * gru_per_step
    
    # ===== TEMPORAL ATTENTION =====
    # Linear(2H → 1) for attention weights
    attn_dim = 2 * H
    macs['attention'] = T2 * attn_dim * 1 + T2 * attn_dim  # Score + weighted sum
    
    # ===== CLASSIFIER =====
    macs['classifier'] = 2 * H * 6  # 2H → num_classes
    
    macs['total'] = macs['conv_blocks'] + macs['gru'] + macs['attention'] + macs['classifier']
    
    return macs


def calculate_tinyhar_macs(T: int, C: int, filter_num: int, nb_conv: int, filter_size: int) -> Dict[str, int]:
    """
    Calculate theoretical MACs for TinyHAR.
    
    Architecture:
    1. 4 Conv2D layers with stride (temporal downsampling)
    2. Cross-channel Self-Attention
    3. FC fusion (C*F → 2F)
    4. LSTM
    5. Temporal Weighted Aggregation
    6. Classifier
    
    Key formula for LSTM (4 gates: input, forget, cell, output):
        MACs_LSTM = T × 4 × (H² + H×D)
        where H=hidden_size, D=input_dim
    """
    macs = {}
    F = filter_num
    
    # ===== CONV LAYERS =====
    # Layer 1: 1 → F, stride=(1,1)
    conv1 = T * C * 1 * filter_size * F
    
    # Layer 2: F → F, stride=(2,1) → T/2
    T2 = T // 2
    conv2 = T * C * F * filter_size * F  # Before stride
    
    # Layer 3: F → F, stride=(1,1)
    conv3 = T2 * C * F * filter_size * F
    
    # Layer 4: F → F, stride=(2,1) → T/4
    T4 = T2 // 2
    conv4 = T2 * C * F * filter_size * F  # Before stride
    
    macs['conv_layers'] = conv1 + conv2 + conv3 + conv4
    
    # ===== CROSS-CHANNEL ATTENTION =====
    # Self-attention at each timestep: Q, K, V projections + attention
    # Q, K, V: F → F each
    attn_proj = T4 * 3 * F * F  # 3 projections
    # Attention: (C × C) × F for each timestep
    attn_scores = T4 * C * C * F  # QK^T
    attn_values = T4 * C * C * F  # softmax(QK^T) × V
    
    macs['attention'] = attn_proj + attn_scores + attn_values
    
    # ===== FC FUSION =====
    # C*F → 2F
    macs['fc_fusion'] = T4 * (C * F) * (2 * F)
    
    # ===== LSTM =====
    # Input: 2F, Hidden: 2F
    # 4 gates: input, forget, cell, output
    H = 2 * F
    D = 2 * F
    lstm_per_step = 4 * (H * H + H * D)
    macs['lstm'] = T4 * lstm_per_step
    
    # ===== TEMPORAL AGGREGATION =====
    # FC(H → H) + FC(H → 1) + weighted sum
    macs['temporal_agg'] = T4 * H * H + T4 * H * 1 + T4 * H
    
    # ===== CLASSIFIER =====
    macs['classifier'] = H * 6
    
    macs['total'] = (macs['conv_layers'] + macs['attention'] + 
                     macs['fc_fusion'] + macs['lstm'] + 
                     macs['temporal_agg'] + macs['classifier'])
    
    return macs


def calculate_lightdcnn_macs(T: int, C: int, conv_filters: int, lstm_hidden: int) -> Dict[str, int]:
    """
    Calculate theoretical MACs for LightDeepConvLSTM.
    
    Architecture:
    1. 2 Conv1D layers with MaxPool
    2. Bidirectional LSTM
    3. Classifier
    """
    macs = {}
    
    # ===== CONV LAYERS =====
    # Conv1: C → F, k=5
    conv1 = T * C * 5 * conv_filters
    T1 = T // 2  # After MaxPool
    
    # Conv2: F → F, k=5
    conv2 = T1 * conv_filters * 5 * conv_filters
    T2 = T1 // 2  # After MaxPool
    
    macs['conv_layers'] = conv1 + conv2
    
    # ===== BIDIRECTIONAL LSTM =====
    H = lstm_hidden
    D = conv_filters
    # 4 gates per direction
    lstm_per_step = 4 * (H * H + H * D)
    macs['lstm'] = 2 * T2 * lstm_per_step  # BiLSTM
    
    # ===== CLASSIFIER =====
    macs['classifier'] = 2 * H * 6  # 2H for bidirectional
    
    macs['total'] = macs['conv_layers'] + macs['lstm'] + macs['classifier']
    
    return macs


# ============================================================================
# LATENCY BENCHMARK ENGINE
# ============================================================================

def benchmark_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_threads: int,
    runs: int = 1000,
    warmup: int = 50
) -> Dict[str, float]:
    """
    Benchmark inference latency on CPU.
    
    Args:
        model: PyTorch model (should be in eval mode)
        input_shape: Shape of input tensor (B, T, C)
        num_threads: Number of CPU threads to use
        runs: Number of inference runs
        warmup: Number of warmup runs
    
    Returns:
        Dict with latency statistics (mean, std, p99, min, max)
    """
    # Set CPU threading (only set num_threads, interop_threads can only be set once)
    torch.set_num_threads(num_threads)
    
    # Ensure model is on CPU and in eval mode
    model = model.cpu().eval()
    
    # Create dummy input
    x = torch.randn(*input_shape, dtype=torch.float32)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    # Measurement
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'median_ms': float(np.median(latencies)),
    }


def run_full_benchmark(
    dataset: str,
    runs: int = 1000,
    warmup: int = 50
) -> Dict[str, Any]:
    """
    Run full benchmark for all models on a specific dataset.
    
    Args:
        dataset: Dataset name
        runs: Number of inference runs per model
        warmup: Warmup runs
    
    Returns:
        Complete benchmark results
    """
    cfg = DATASET_CONFIGS[dataset]
    T, C, num_classes = cfg['seqLen'], cfg['inChannels'], cfg['numClasses']
    
    results = {
        'dataset': dataset,
        'config': cfg,
        'timestamp': datetime.now().isoformat(),
        'runs': runs,
        'warmup': warmup,
        'models': {}
    }
    
    input_shape = (1, T, C)  # Batch=1 for real-time inference
    
    # ===== BabyMamba =====
    print(f"\n{'='*60}")
    print(f"Benchmarking BabyMamba on {dataset}")
    print(f"{'='*60}")
    
    model = BabyMamba(
        numClasses=num_classes,
        inChannels=C,
        seqLen=T,
        **BABYMAMBA_CONFIG
    )
    params = sum(p.numel() for p in model.parameters())
    
    # Calculate MACs
    macs = calculateBabyMambaMacs(
        T=T, C=C, 
        d=BABYMAMBA_CONFIG['dModel'],
        N=BABYMAMBA_CONFIG['dState'],
        expand=BABYMAMBA_CONFIG['expand'],
        n_layers=BABYMAMBA_CONFIG['nLayers']
    )
    
    # Latency benchmarks
    latency_1core = benchmark_latency(model, input_shape, num_threads=1, runs=runs, warmup=warmup)
    latency_4core = benchmark_latency(model, input_shape, num_threads=4, runs=runs, warmup=warmup)
    
    results['models']['BabyMamba'] = {
        'params': params,
        'macs': macs,
        'latency_1core': latency_1core,
        'latency_4core': latency_4core,
        'config': BABYMAMBA_CONFIG
    }
    
    print(f"Parameters: {params:,}")
    print(f"Total MACs: {macs['total']:,}")
    print(f"1-Core Latency: {latency_1core['mean_ms']:.3f} ± {latency_1core['std_ms']:.3f} ms")
    print(f"4-Core Latency: {latency_4core['mean_ms']:.3f} ± {latency_4core['std_ms']:.3f} ms")
    
    # ===== TinierHAR =====
    print(f"\n{'='*60}")
    print(f"Benchmarking TinierHAR on {dataset}")
    print(f"{'='*60}")
    
    model = TinierHAR(
        numClasses=num_classes,
        inChannels=C,
        seqLen=T,
        **TINIER_HAR_CONFIG
    )
    params = sum(p.numel() for p in model.parameters())
    
    macs = calculate_tinierhar_macs(
        T=T, C=C,
        nb_filters=TINIER_HAR_CONFIG['nbFilters'],
        nb_blocks=TINIER_HAR_CONFIG['nbConvBlocks'],
        gru_units=TINIER_HAR_CONFIG['gruUnits']
    )
    
    latency_1core = benchmark_latency(model, input_shape, num_threads=1, runs=runs, warmup=warmup)
    latency_4core = benchmark_latency(model, input_shape, num_threads=4, runs=runs, warmup=warmup)
    
    results['models']['TinierHAR'] = {
        'params': params,
        'macs': macs,
        'latency_1core': latency_1core,
        'latency_4core': latency_4core,
        'config': TINIER_HAR_CONFIG
    }
    
    print(f"Parameters: {params:,}")
    print(f"Total MACs: {macs['total']:,}")
    print(f"1-Core Latency: {latency_1core['mean_ms']:.3f} ± {latency_1core['std_ms']:.3f} ms")
    print(f"4-Core Latency: {latency_4core['mean_ms']:.3f} ± {latency_4core['std_ms']:.3f} ms")
    
    # ===== TinyHAR =====
    print(f"\n{'='*60}")
    print(f"Benchmarking TinyHAR on {dataset}")
    print(f"{'='*60}")
    
    model = TinyHAR(
        numClasses=num_classes,
        inChannels=C,
        seqLen=T,
        **TINY_HAR_CONFIG
    )
    params = sum(p.numel() for p in model.parameters())
    
    macs = calculate_tinyhar_macs(
        T=T, C=C,
        filter_num=TINY_HAR_CONFIG['filterNum'],
        nb_conv=TINY_HAR_CONFIG['nbConvLayers'],
        filter_size=TINY_HAR_CONFIG['filterSize']
    )
    
    latency_1core = benchmark_latency(model, input_shape, num_threads=1, runs=runs, warmup=warmup)
    latency_4core = benchmark_latency(model, input_shape, num_threads=4, runs=runs, warmup=warmup)
    
    results['models']['TinyHAR'] = {
        'params': params,
        'macs': macs,
        'latency_1core': latency_1core,
        'latency_4core': latency_4core,
        'config': TINY_HAR_CONFIG
    }
    
    print(f"Parameters: {params:,}")
    print(f"Total MACs: {macs['total']:,}")
    print(f"1-Core Latency: {latency_1core['mean_ms']:.3f} ± {latency_1core['std_ms']:.3f} ms")
    print(f"4-Core Latency: {latency_4core['mean_ms']:.3f} ± {latency_4core['std_ms']:.3f} ms")
    
    # ===== LightDeepConvLSTM =====
    print(f"\n{'='*60}")
    print(f"Benchmarking LightDeepConvLSTM on {dataset}")
    print(f"{'='*60}")
    
    model = LightDeepConvLSTM(
        numClasses=num_classes,
        inChannels=C,
        seqLen=T,
        **LIGHT_DCNN_CONFIG
    )
    params = sum(p.numel() for p in model.parameters())
    
    macs = calculate_lightdcnn_macs(
        T=T, C=C,
        conv_filters=LIGHT_DCNN_CONFIG['convFilters'],
        lstm_hidden=LIGHT_DCNN_CONFIG['lstmHidden']
    )
    
    latency_1core = benchmark_latency(model, input_shape, num_threads=1, runs=runs, warmup=warmup)
    latency_4core = benchmark_latency(model, input_shape, num_threads=4, runs=runs, warmup=warmup)
    
    results['models']['LightDeepConvLSTM'] = {
        'params': params,
        'macs': macs,
        'latency_1core': latency_1core,
        'latency_4core': latency_4core,
        'config': LIGHT_DCNN_CONFIG
    }
    
    print(f"Parameters: {params:,}")
    print(f"Total MACs: {macs['total']:,}")
    print(f"1-Core Latency: {latency_1core['mean_ms']:.3f} ± {latency_1core['std_ms']:.3f} ms")
    print(f"4-Core Latency: {latency_4core['mean_ms']:.3f} ± {latency_4core['std_ms']:.3f} ms")
    
    return results


def print_summary_table(results: Dict[str, Any]):
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY: {results['dataset'].upper()}")
    print(f"{'='*80}")
    
    # Header
    print(f"\n{'Model':<20} | {'Params':<10} | {'MACs':<12} | {'1-Core (ms)':<15} | {'4-Core (ms)':<15}")
    print("-" * 80)
    
    for name, data in results['models'].items():
        params = data['params']
        macs = data['macs']['total']
        lat1 = f"{data['latency_1core']['mean_ms']:.3f}±{data['latency_1core']['std_ms']:.3f}"
        lat4 = f"{data['latency_4core']['mean_ms']:.3f}±{data['latency_4core']['std_ms']:.3f}"
        
        print(f"{name:<20} | {params:<10,} | {macs:<12,} | {lat1:<15} | {lat4:<15}")
    
    print("-" * 80)
    
    # Efficiency comparison
    nano = results['models']['BabyMamba']
    print(f"\n--- Efficiency Comparison (vs BabyMamba) ---")
    
    for name, data in results['models'].items():
        if name == 'BabyMamba':
            continue
        
        mac_ratio = data['macs']['total'] / nano['macs']['total']
        lat1_ratio = data['latency_1core']['mean_ms'] / nano['latency_1core']['mean_ms']
        lat4_ratio = data['latency_4core']['mean_ms'] / nano['latency_4core']['mean_ms']
        
        print(f"{name}:")
        print(f"  MACs: {mac_ratio:.2f}x BabyMamba")
        print(f"  1-Core Latency: {lat1_ratio:.2f}x BabyMamba")
        print(f"  4-Core Latency: {lat4_ratio:.2f}x BabyMamba")


def print_mac_breakdown(results: Dict[str, Any]):
    """Print detailed MAC breakdown for each model."""
    print(f"\n{'='*80}")
    print("MAC BREAKDOWN BY COMPONENT")
    print(f"{'='*80}")
    
    for name, data in results['models'].items():
        macs = data['macs']
        total = macs['total']
        
        print(f"\n{name}:")
        for component, value in macs.items():
            if component != 'total':
                pct = 100 * value / total
                print(f"  {component:<15}: {value:>12,} MACs ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<15}: {total:>12,} MACs")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark inference latency and MACs')
    parser.add_argument('--dataset', type=str, default='ucihar', 
                       choices=list(DATASET_CONFIGS.keys()),
                       help='Dataset to benchmark')
    parser.add_argument('--all-datasets', action='store_true',
                       help='Benchmark all datasets')
    parser.add_argument('--runs', type=int, default=1000,
                       help='Number of inference runs')
    parser.add_argument('--warmup', type=int, default=50,
                       help='Number of warmup runs')
    parser.add_argument('--output-dir', type=str, default='results/latency',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    datasets = list(DATASET_CONFIGS.keys()) if args.all_datasets else [args.dataset]
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'#'*80}")
        print(f"# BENCHMARKING: {dataset.upper()}")
        print(f"{'#'*80}")
        
        results = run_full_benchmark(dataset, runs=args.runs, warmup=args.warmup)
        all_results[dataset] = results
        
        # Print summary
        print_summary_table(results)
        print_mac_breakdown(results)
        
        # Save individual results
        output_file = os.path.join(args.output_dir, f'latency_{dataset}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    # Save combined results if multiple datasets
    if len(datasets) > 1:
        combined_file = os.path.join(args.output_dir, 'latency_all_datasets.json')
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to: {combined_file}")
    
    # Print final cross-dataset summary
    if len(datasets) > 1:
        print(f"\n{'='*80}")
        print("CROSS-DATASET SUMMARY")
        print(f"{'='*80}")
        
        for dataset in datasets:
            results = all_results[dataset]
            nano_macs = results['models']['BabyMamba']['macs']['total']
            nano_lat = results['models']['BabyMamba']['latency_1core']['mean_ms']
            print(f"\n{dataset}:")
            print(f"  BabyMamba: {nano_macs:,} MACs, {nano_lat:.3f}ms (1-core)")
            
            for name in ['TinierHAR', 'TinyHAR', 'LightDeepConvLSTM']:
                data = results['models'][name]
                mac_ratio = data['macs']['total'] / nano_macs
                lat_ratio = data['latency_1core']['mean_ms'] / nano_lat
                print(f"  {name}: {mac_ratio:.2f}x MACs, {lat_ratio:.2f}x latency")


if __name__ == '__main__':
    main()

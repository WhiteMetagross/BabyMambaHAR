"""
Profiling Utilities

Measure FLOPs, MACs, parameters, latency, and memory usage.
"""

import time
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


def countParameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dict with total, trainable, and non-trainable counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'nonTrainable': total - trainable
    }


def countParametersByLayer(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters for each named layer.
    
    Returns:
        Dict mapping layer name to parameter count
    """
    result = {}
    for name, param in model.named_parameters():
        result[name] = param.numel()
    return result


def computeMacs(
    model: nn.Module,
    inputShape: Tuple[int, ...],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Compute MACs (Multiply-Accumulate Operations).
    
    Args:
        model: PyTorch model
        inputShape: Input tensor shape (B, ...)
        device: Device to run on
    
    Returns:
        Dict with MACs, FLOPs, and per-layer breakdown
    """
    model = model.to(device)
    model.eval()
    
    result = {'macs': None, 'flops': None, 'perLayer': {}}
    
    # Try using thop
    try:
        from thop import profile, clever_format
        
        x = torch.randn(*inputShape, device=device)
        macs, params = profile(model, inputs=(x,), verbose=False)
        
        result['macs'] = int(macs)
        result['flops'] = int(macs * 2)  # FLOPs ≈ 2 * MACs
        result['macsFormatted'], _ = clever_format([macs, params], "%.3f")
        
        return result
    except ImportError:
        pass
    
    # Try using fvcore
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        
        x = torch.randn(*inputShape, device=device)
        flops = FlopCountAnalysis(model, x)
        
        result['flops'] = flops.total()
        result['macs'] = result['flops'] // 2
        
        return result
    except ImportError:
        pass
    
    # Fallback: Estimate based on parameters
    params = countParameters(model)['total']
    # Rough estimate: 2 * params * sequence_length (very approximate)
    seqLen = inputShape[-2] if len(inputShape) > 2 else 1
    result['macs'] = params * seqLen
    result['flops'] = result['macs'] * 2
    result['note'] = 'Estimated (install thop or fvcore for accurate count)'
    
    return result


def profileModel(
    model: nn.Module,
    inputShape: Tuple[int, ...],
    device: str = 'cuda',
    detailed: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive model profiling.
    
    Args:
        model: PyTorch model
        inputShape: Input tensor shape (B, ...)
        device: Device to run on
        detailed: Include per-layer breakdown
    
    Returns:
        Dict with profiling results
    """
    model = model.to(device)
    model.eval()
    
    # Count parameters
    params = countParameters(model)
    
    # Compute MACs
    macs = computeMacs(model, inputShape, device)
    
    # Model size in MB
    paramSize = sum(p.nelement() * p.element_size() for p in model.parameters())
    bufferSize = sum(b.nelement() * b.element_size() for b in model.buffers())
    sizeMb = (paramSize + bufferSize) / (1024 ** 2)
    
    result = {
        'parameters': params,
        'macs': macs['macs'],
        'flops': macs['flops'],
        'sizeMb': sizeMb,
    }
    
    if detailed:
        result['paramsByLayer'] = countParametersByLayer(model)
    
    return result


def benchmarkLatency(
    model: nn.Module,
    inputShape: Tuple[int, ...],
    device: str = 'cuda',
    numWarmup: int = 10,
    numRuns: int = 100,
    batchSizes: Optional[list] = None
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark model inference latency.
    
    Args:
        model: PyTorch model
        inputShape: Input tensor shape (B, ...)
        device: Device to run on
        numWarmup: Number of warmup iterations
        numRuns: Number of timed iterations
        batchSizes: List of batch sizes to test (default: [1, 8, 16, 32])
    
    Returns:
        Dict mapping batch size to latency stats
    """
    model = model.to(device)
    model.eval()
    
    if batchSizes is None:
        batchSizes = [1, 8, 16, 32]
    
    results = {}
    
    for batchSize in batchSizes:
        currentShape = (batchSize,) + inputShape[1:]
        x = torch.randn(*currentShape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(numWarmup):
                _ = model(x)
        
        # Synchronize for CUDA
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(numRuns):
                start = time.perf_counter()
                _ = model(x)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
        
        results[f'batch_{batchSize}'] = {
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
            'throughput': batchSize / (sum(times) / len(times) / 1000)  # samples/sec
        }
    
    return results


def estimateEdgeLatency(
    macs: int,
    device: str = 'cortex-m4'
) -> Dict[str, float]:
    """
    Estimate latency on edge devices (rough approximation).
    
    Args:
        macs: Number of MACs
        device: Target device type
    
    Returns:
        Estimated latency in milliseconds
    """
    # Rough estimates of MACS/sec for different devices
    deviceSpecs = {
        'cortex-m4': 10e6,    # ~10 MMACS/sec
        'cortex-m7': 50e6,    # ~50 MMACS/sec
        'esp32': 5e6,         # ~5 MMACS/sec
        'rpi-zero': 100e6,    # ~100 MMACS/sec
        'rpi-4': 1e9,         # ~1 GMACS/sec
    }
    
    macsPerSec = deviceSpecs.get(device, 10e6)
    latencyMs = (macs / macsPerSec) * 1000
    
    return {
        'device': device,
        'estimatedMs': latencyMs,
        'estimatedHz': 1000 / latencyMs if latencyMs > 0 else 0,
        'note': 'Rough estimate - actual may vary significantly'
    }


def getModelSummary(
    model: nn.Module,
    inputShape: Tuple[int, ...],
    device: str = 'cpu'
) -> str:
    """
    Get a formatted model summary string.
    
    Args:
        model: PyTorch model
        inputShape: Input tensor shape
        device: Device to run on
    
    Returns:
        Formatted summary string
    """
    profile = profileModel(model, inputShape, device, detailed=True)
    latency = benchmarkLatency(model, inputShape, device, numWarmup=5, numRuns=20, batchSizes=[1])
    
    lines = [
        "=" * 60,
        "BabyMamba Model Summary",
        "=" * 60,
        f"Total Parameters:     {profile['parameters']['total']:,}",
        f"Trainable Parameters: {profile['parameters']['trainable']:,}",
        f"Model Size:           {profile['sizeMb']:.3f} MB",
        f"MACs:                 {profile['macs']:,}" if profile['macs'] else "MACs: N/A",
        f"FLOPs:                {profile['flops']:,}" if profile['flops'] else "FLOPs: N/A",
        "-" * 60,
        f"Latency (batch=1):    {latency['batch_1']['mean_ms']:.2f} ms",
        f"Throughput:           {latency['batch_1']['throughput']:.1f} samples/sec",
        "=" * 60,
    ]
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test profiling
    from ciBabyMambaHar.models import BabyMamba
    
    model = BabyMamba(numClasses=6, inChannels=9, variant='nano')
    inputShape = (1, 128, 9)
    
    print(getModelSummary(model, inputShape, device='cpu'))

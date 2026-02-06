#!/usr/bin/env python
"""
MLP Empirical Benchmark Sweep

Runs parameter sweeps across MLP configurations to:
1. Measure real execution time on CPU/GPU
2. Compare against analytical estimates from FXGraphWalker
3. Calibrate efficiency_factor coefficients
4. Identify memory hierarchy bottlenecks

Usage:
    # Full sweep (slow - hours!)
    python validation/empirical/sweep_mlp.py --full --device cpu

    # Quick smoke test (5 configs, 1 minute)
    python validation/empirical/sweep_mlp.py --quick

    # Specific memory scenario
    python validation/empirical/sweep_mlp.py --scenario L3_fit
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import time
import argparse
import itertools
import csv
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Add repo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.graphs.transform.partitioning import FusionBasedPartitioner
from src.graphs.hardware.mappers.cpu import create_intel_cpu_mapper, create_i7_12700k_mapper
from src.graphs.hardware.mappers.gpu import (
    create_h100_pcie_80gb_mapper,
    create_jetson_orin_agx_64gb_mapper,
    create_jetson_orin_nano_8gb_mapper,
    create_jetson_orin_nx_16gb_mapper,
)
from src.graphs.hardware.resource_model import Precision

# Import existing MLP models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../workloads/pytorch/mlp'))
from multi_layer_perceptrons import OneLayerMLP, TwoLayerMLP, ThreeLayerMLP, FourLayerMLP


# ============================================================================
# SWEEP PARAMETER DEFINITIONS
# ============================================================================

# Quick smoke test (5 configs, ~1 minute)
QUICK_SWEEP = {
    'input_dim': [512],
    'hidden_configs': [
        ([128], 64),           # 1-layer, small
        ([512, 512], 128),     # 2-layer, medium
    ],
    'batch_size': [1, 32],
    'precision': ['fp32'],
}

# Full sweep (150+ configs, hours)
FULL_SWEEP = {
    'input_dim': [256, 512, 1024, 2048],
    'hidden_configs': [
        # (hidden_dims, output_dim)
        ([128], 64),                      # 1-layer small
        ([512], 128),                     # 1-layer medium
        ([2048], 256),                    # 1-layer large
        ([512, 512], 128),                # 2-layer medium
        ([1024, 1024], 256),              # 2-layer large
        ([1024, 1024, 1024], 256),        # 3-layer large
        ([2048, 2048, 2048, 2048], 512),  # 4-layer xlarge
    ],
    'batch_size': [1, 4, 16, 32, 64, 128],
    'precision': ['fp32', 'fp16'],
}

# Memory scenario sweep (designed to trigger cache spills)
MEMORY_SCENARIOS = {
    'L1_fit': {
        # Target: < 32 KB total working set
        'input_dim': [64],
        'hidden_configs': [([32], 16)],
        'batch_size': [1],
        'precision': ['fp32'],
    },
    'L2_fit': {
        # Target: 32 KB - 1 MB
        'input_dim': [512],
        'hidden_configs': [([256], 128)],
        'batch_size': [1, 4],
        'precision': ['fp32'],
    },
    'L3_fit': {
        # Target: 1-32 MB (fits in L3, spills from L2)
        'input_dim': [2048],
        'hidden_configs': [([2048, 2048], 1024)],
        'batch_size': [1, 4, 16],
        'precision': ['fp32'],
    },
    'DRAM_spill': {
        # Target: > 32 MB (spills to DRAM)
        'input_dim': [4096],
        'hidden_configs': [([4096, 4096, 4096], 2048)],
        'batch_size': [1, 16, 64],
        'precision': ['fp32'],
    },
}


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_mlp(hidden_dims: List[int], input_dim: int, output_dim: int) -> nn.Module:
    """Build MLP from configuration"""
    num_layers = len(hidden_dims)

    if num_layers == 1:
        # Use OneLayerMLP (Linear → Softmax)
        # But OneLayerMLP doesn't have hidden dim, so use generic builder
        return build_generic_mlp(input_dim, hidden_dims, output_dim)

    elif num_layers == 2:
        return TwoLayerMLP(input_dim, hidden_dims[0], output_dim)

    elif num_layers == 3:
        return ThreeLayerMLP(input_dim, hidden_dims[0], hidden_dims[1], output_dim)

    elif num_layers == 4:
        return FourLayerMLP(input_dim, hidden_dims[0], hidden_dims[1], hidden_dims[2], output_dim)

    else:
        # Generic builder for 5+ layers
        return build_generic_mlp(input_dim, hidden_dims, output_dim)


def build_generic_mlp(input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Module:
    """Generic MLP builder for any number of layers"""
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:  # No activation after last layer
            layers.append(nn.Tanh())

    layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)


# ============================================================================
# EMPIRICAL BENCHMARKING
# ============================================================================

def get_memory_usage_cpu() -> float:
    """Get current CPU memory usage in MB (simple RSS)"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024**2
    except ImportError:
        # psutil not available, return 0
        return 0.0


def get_memory_usage_cuda() -> float:
    """Get current CUDA memory usage in MB"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**2


def run_empirical_benchmark(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cpu',
    precision: str = 'fp32',
    num_warmup: int = 10,
    num_runs: int = 50
) -> Dict[str, float]:
    """
    Run empirical benchmark measuring actual execution time and memory

    Returns:
        {
            'time_ms': Mean execution time in milliseconds,
            'time_std_ms': Std deviation,
            'memory_mb': Peak memory usage,
            'throughput_samples_per_sec': Samples processed per second,
        }
    """
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Set precision
    if precision == 'fp16':
        model = model.half()
        input_tensor = input_tensor.half()
    # fp32 is default

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    # Synchronize for accurate timing
    if device == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    peak_memory = 0.0

    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            output = model(input_tensor)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

            # Track peak memory
            if device == 'cuda':
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1024**2)
            else:
                peak_memory = max(peak_memory, get_memory_usage_cpu())

    times_tensor = torch.tensor(times)
    batch_size = input_tensor.shape[0]

    return {
        'time_ms': float(times_tensor.mean()),
        'time_std_ms': float(times_tensor.std()),
        'memory_mb': peak_memory,
        'throughput_samples_per_sec': batch_size / (times_tensor.mean() / 1000),
    }


# ============================================================================
# CPU DETECTION
# ============================================================================

def detect_cpu_model() -> str:
    """
    Detect CPU model to auto-select appropriate mapper.

    Returns:
        'i7-12700k', 'generic', etc.
    """
    try:
        import subprocess
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        output = result.stdout

        # Check for i7-12700K
        if 'i7-12700K' in output or '12th Gen Intel(R) Core(TM) i7-12700K' in output:
            return 'i7-12700k'

        # Add more CPU detection here in the future
        # if 'i9-13900K' in output:
        #     return 'i9-13900k'

        return 'generic'
    except:
        return 'generic'


def detect_gpu_model() -> str:
    """
    Detect GPU model to auto-select appropriate mapper.

    Returns:
        'jetson-orin-agx', 'jetson-orin-nx', 'jetson-orin-nano', 'h100', 'generic', etc.
    """
    if not torch.cuda.is_available():
        return 'generic'

    try:
        gpu_name = torch.cuda.get_device_name(0).lower()

        # Jetson devices
        if 'orin' in gpu_name:
            if 'agx' in gpu_name:
                return 'jetson-orin-agx'
            elif 'nx' in gpu_name:
                return 'jetson-orin-nx'
            elif 'nano' in gpu_name:
                return 'jetson-orin-nano'
            else:
                # Default to AGX for unspecified Orin
                return 'jetson-orin-agx'

        # NVIDIA datacenter GPUs
        if 'h100' in gpu_name:
            return 'h100'
        if 'a100' in gpu_name:
            return 'a100'

        return 'generic'
    except:
        return 'generic'


# ============================================================================
# ANALYTICAL ESTIMATION
# ============================================================================

def extract_execution_stages(fusion_report: 'FusionReport') -> List[List[int]]:
    """
    Extract execution stages from fusion report.

    TEMPORARY WORKAROUND: Creates simple sequential stages since fusion partitioner
    doesn't populate dependencies yet.

    Returns:
        List of stages, each stage is a list of subgraph indices
    """
    subgraphs = fusion_report.fused_subgraphs
    n = len(subgraphs)

    if n == 0:
        return []

    # Create stages with limited parallelism (1-3 ops per stage)
    stages = []
    i = 0
    while i < n:
        stage_size = min(3, n - i)
        stages.append(list(range(i, i + stage_size)))
        i += stage_size

    return stages


def run_analytical_estimate(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cpu',
    precision: str = 'fp32'
) -> Dict[str, float]:
    """
    Run analytical estimate using FXGraphWalker + HardwareMapper

    Returns:
        {
            'time_ms': Estimated latency,
            'memory_mb': Estimated memory traffic,
            'flops': Total FLOPs,
            'arithmetic_intensity': FLOPs/Byte,
            'bottleneck': 'compute' or 'memory',
        }
    """
    # Ensure model and input are on the same device for tracing
    # (empirical benchmark may have moved model to CUDA in-place)
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # For CUDA with reduced precision, apply precision before tracing
    if device == 'cuda' and precision in ('fp16', 'bf16'):
        model = model.half() if precision == 'fp16' else model.bfloat16()
        input_tensor = input_tensor.half() if precision == 'fp16' else input_tensor.bfloat16()

    # Trace model
    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(input_tensor)

    # Partition graph
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(traced)

    # Extract execution stages
    execution_stages = extract_execution_stages(fusion_report)

    # Map to hardware (auto-detect for accurate mapping)
    if device == 'cpu':
        cpu_model = detect_cpu_model()
        if cpu_model == 'i7-12700k':
            mapper = create_i7_12700k_mapper()
        else:
            mapper = create_intel_cpu_mapper()
    elif device == 'cuda':
        gpu_model = detect_gpu_model()
        if gpu_model == 'jetson-orin-agx':
            mapper = create_jetson_orin_agx_64gb_mapper()
        elif gpu_model == 'jetson-orin-nx':
            mapper = create_jetson_orin_nx_16gb_mapper()
        elif gpu_model == 'jetson-orin-nano':
            mapper = create_jetson_orin_nano_8gb_mapper()
        else:
            # Default to H100 for datacenter GPUs
            mapper = create_h100_pcie_80gb_mapper()
    else:
        raise ValueError(f"Unsupported device: {device}")

    # Convert precision string to Precision enum
    precision_map = {
        'fp32': Precision.FP32,
        'fp16': Precision.FP16,
        'int8': Precision.INT8,
    }
    prec = precision_map.get(precision, Precision.FP32)

    # Get hardware-mapped report
    batch_size = input_tensor.shape[0]
    hw_allocation = mapper.map_graph(
        fusion_report=fusion_report,
        execution_stages=execution_stages,
        batch_size=batch_size,
        precision=prec
    )

    # Compute arithmetic intensity (using fused memory traffic)
    memory_traffic = fusion_report.total_memory_traffic
    ai = fusion_report.total_flops / memory_traffic if memory_traffic > 0 else 0
    bottleneck = 'compute' if ai > 10 else 'memory'

    return {
        'time_ms': hw_allocation.total_latency * 1000,  # Convert to ms
        'memory_mb': memory_traffic / 1024**2,
        'flops': fusion_report.total_flops,
        'arithmetic_intensity': ai,
        'bottleneck': bottleneck,
    }


# ============================================================================
# SWEEP RUNNER
# ============================================================================

def run_sweep(
    sweep_params: Dict,
    device: str = 'cpu',
    output_file: str = 'mlp_sweep_results.csv'
) -> List[Dict]:
    """
    Run parameter sweep across all combinations

    Args:
        sweep_params: Dictionary of parameter lists
        device: 'cpu' or 'cuda'
        output_file: CSV output file path

    Returns:
        List of result dictionaries
    """
    results = []

    # Generate all combinations
    param_names = ['input_dim', 'hidden_configs', 'batch_size', 'precision']
    param_values = [
        sweep_params['input_dim'],
        sweep_params['hidden_configs'],
        sweep_params['batch_size'],
        sweep_params['precision'],
    ]

    total_configs = 1
    for values in param_values:
        total_configs *= len(values)

    print(f"Running sweep with {total_configs} configurations on {device}...")
    print(f"Results will be saved to: {output_file}")

    # Show detected hardware mapper
    if device == 'cpu':
        cpu_model = detect_cpu_model()
        if cpu_model == 'i7-12700k':
            print(f"Detected CPU: Intel i7-12700K -> Using calibrated mapper")
        else:
            print(f"Using generic Intel CPU mapper")
    elif device == 'cuda':
        gpu_model = detect_gpu_model()
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        if gpu_model.startswith('jetson'):
            print(f"Detected GPU: {gpu_name} -> Using {gpu_model} mapper")
        elif gpu_model == 'h100':
            print(f"Detected GPU: {gpu_name} -> Using H100 mapper")
        else:
            print(f"Detected GPU: {gpu_name} -> Using H100 mapper (default)")

    print("=" * 80)

    config_num = 0
    for params in itertools.product(*param_values):
        config_num += 1
        input_dim, (hidden_dims, output_dim), batch_size, precision = params

        # Build model
        model = build_mlp(hidden_dims, input_dim, output_dim)
        input_tensor = torch.randn(batch_size, input_dim)

        # Model description
        model_name = f"mlp_{len(hidden_dims)}layer"
        hidden_str = str(hidden_dims)

        print(f"\n[{config_num}/{total_configs}] {model_name}")
        print(f"  Config: input={input_dim}, hidden={hidden_str}, output={output_dim}")
        print(f"  Batch={batch_size}, precision={precision}, device={device}")

        try:
            # Run empirical benchmark
            print("  Running empirical benchmark...", end=" ", flush=True)
            empirical = run_empirical_benchmark(model, input_tensor, device, precision)
            print(f"✓ {empirical['time_ms']:.3f} ms")

            # Run analytical estimate
            print("  Running analytical estimate...", end=" ", flush=True)
            analytical = run_analytical_estimate(model, input_tensor, device, precision)
            print(f"✓ {analytical['time_ms']:.3f} ms")

            # Compute errors
            time_error = abs(empirical['time_ms'] - analytical['time_ms']) / empirical['time_ms'] * 100
            memory_error = abs(empirical['memory_mb'] - analytical['memory_mb']) / max(empirical['memory_mb'], 0.01) * 100

            print(f"  Errors: time={time_error:.1f}%, memory={memory_error:.1f}%")

            # Store results
            result = {
                'model': model_name,
                'input_dim': input_dim,
                'hidden_dims': hidden_str,
                'output_dim': output_dim,
                'batch_size': batch_size,
                'device': device,
                'precision': precision,

                # Empirical measurements
                'empirical_time_ms': empirical['time_ms'],
                'empirical_time_std_ms': empirical['time_std_ms'],
                'empirical_memory_mb': empirical['memory_mb'],
                'empirical_throughput': empirical['throughput_samples_per_sec'],

                # Analytical estimates
                'analytical_time_ms': analytical['time_ms'],
                'analytical_memory_mb': analytical['memory_mb'],
                'analytical_flops': analytical['flops'],
                'analytical_ai': analytical['arithmetic_intensity'],
                'analytical_bottleneck': analytical['bottleneck'],

                # Errors
                'time_error_pct': time_error,
                'memory_error_pct': memory_error,
            }

            results.append(result)

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue

    # Write results to CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print("\n" + "=" * 80)
        print(f"✓ Sweep complete! {len(results)}/{total_configs} configs succeeded")
        print(f"  Results saved to: {output_file}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MLP Empirical Benchmark Sweep")
    parser.add_argument('--quick', action='store_true', help="Quick smoke test (5 configs)")
    parser.add_argument('--full', action='store_true', help="Full parameter sweep (hours!)")
    parser.add_argument('--scenario', choices=['L1_fit', 'L2_fit', 'L3_fit', 'DRAM_spill'],
                        help="Run specific memory scenario")
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help="Device to benchmark on")
    parser.add_argument('--output', default=None, help="Output CSV file")

    args = parser.parse_args()

    # Determine sweep parameters
    if args.scenario:
        sweep_params = MEMORY_SCENARIOS[args.scenario]
        output_file = args.output or f'mlp_sweep_{args.scenario}_{args.device}.csv'
    elif args.full:
        sweep_params = FULL_SWEEP
        output_file = args.output or f'mlp_sweep_full_{args.device}.csv'
    else:  # Default to quick
        sweep_params = QUICK_SWEEP
        output_file = args.output or f'mlp_sweep_quick_{args.device}.csv'

    # Ensure output directory exists
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_file

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA not available. Falling back to CPU.")
        args.device = 'cpu'

    # Run sweep
    results = run_sweep(sweep_params, device=args.device, output_file=str(output_path))

    # Print summary
    if results:
        times_emp = [r['empirical_time_ms'] for r in results]
        times_ana = [r['analytical_time_ms'] for r in results]
        errors = [r['time_error_pct'] for r in results]

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Configurations tested: {len(results)}")
        print(f"Device: {args.device}")
        print(f"\nEmpirical timing:")
        print(f"  Mean: {sum(times_emp)/len(times_emp):.3f} ms")
        print(f"  Min:  {min(times_emp):.3f} ms")
        print(f"  Max:  {max(times_emp):.3f} ms")
        print(f"\nAnalytical timing:")
        print(f"  Mean: {sum(times_ana)/len(times_ana):.3f} ms")
        print(f"  Min:  {min(times_ana):.3f} ms")
        print(f"  Max:  {max(times_ana):.3f} ms")
        print(f"\nMean Absolute Percentage Error (MAPE): {sum(errors)/len(errors):.1f}%")
        print(f"  Best:  {min(errors):.1f}%")
        print(f"  Worst: {max(errors):.1f}%")


if __name__ == "__main__":
    main()

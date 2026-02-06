#!/usr/bin/env python
"""
GEMM (General Matrix Multiply) Empirical Benchmark Sweep

Runs parameter sweeps across GEMM configurations to:
1. Measure real execution time on CPU/GPU
2. Compare against analytical estimates
3. Validate calibration accuracy for pure matmul operations
4. Build calibration data for isolated matrix multiplications

This complements sweep_mlp.py by testing pure GEMM without fusion patterns.

Usage:
    # Quick sweep (few configs, ~1 minute)
    python validation/empirical/sweep_gemm.py --quick --device cuda

    # Full sweep (many configs, longer)
    python validation/empirical/sweep_gemm.py --full --device cuda

    # Square matrices only
    python validation/empirical/sweep_gemm.py --square --device cuda
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import time
import argparse
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
from src.graphs.calibration.gpu_calibration import GPUCalibration
from src.graphs.calibration.gpu_clock import get_jetson_power_mode


# ============================================================================
# SWEEP PARAMETER DEFINITIONS
# ============================================================================

# Quick sweep (~20 configs)
QUICK_SWEEP = {
    'dimensions': [
        # (M, N, K) - output is MxN, inputs are MxK and KxN
        (256, 256, 256),      # Small square
        (1024, 1024, 1024),   # Medium square
        (512, 128, 512),      # Tall output (like MLP hidden->small)
        (128, 512, 128),      # Wide output (like MLP small->hidden)
    ],
    'batch_size': [1, 32],
    'precision': ['fp32'],
}

# Full sweep (~200 configs)
FULL_SWEEP = {
    'dimensions': [
        # Square matrices (typical GEMM benchmarks)
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),

        # Rectangular: tall output (M > N)
        (1024, 128, 512),
        (2048, 256, 1024),
        (4096, 512, 2048),

        # Rectangular: wide output (M < N)
        (128, 1024, 512),
        (256, 2048, 1024),
        (512, 4096, 2048),

        # MLP-like dimensions (hidden layer transitions)
        (512, 512, 768),      # Like transformer hidden
        (768, 768, 3072),     # Like transformer FFN
        (3072, 3072, 768),    # Like transformer FFN back

        # Attention-like dimensions
        (512, 64, 512),       # Like attention head
        (64, 512, 64),        # Like attention output
    ],
    'batch_size': [1, 4, 16, 32, 64, 128],
    'precision': ['fp32', 'fp16'],
}

# Square matrices only (for roofline analysis)
SQUARE_SWEEP = {
    'dimensions': [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (768, 768, 768),
        (1024, 1024, 1024),
        (1536, 1536, 1536),
        (2048, 2048, 2048),
        (3072, 3072, 3072),
        (4096, 4096, 4096),
    ],
    'batch_size': [1, 8, 32, 128],
    'precision': ['fp32', 'fp16'],
}


# ============================================================================
# GEMM MODULE
# ============================================================================

class GEMM(nn.Module):
    """
    Simple GEMM module for FX tracing.

    Computes: output = input @ weight.T
    Where input is (batch, M, K) and weight is (N, K)
    Output is (batch, M, N)
    """

    def __init__(self, m: int, n: int, k: int):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        # Weight matrix: (N, K) so output is (batch, M, N)
        self.weight = nn.Parameter(torch.randn(n, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, M, K) or (M, K)
        # weight: (N, K)
        # output: (batch, M, N) or (M, N)
        return torch.matmul(x, self.weight.t())


class BatchedGEMM(nn.Module):
    """
    Batched GEMM for higher throughput testing.

    Computes: output = input @ weight
    Where input is (batch, M, K) and weight is (K, N)
    """

    def __init__(self, m: int, n: int, k: int):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.weight = nn.Parameter(torch.randn(k, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


# ============================================================================
# EMPIRICAL BENCHMARKING
# ============================================================================

def run_empirical_benchmark(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cpu',
    precision: str = 'fp32',
    num_warmup: int = 20,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Run empirical benchmark measuring actual execution time.

    Returns:
        {
            'time_ms': Mean execution time in milliseconds,
            'time_std_ms': Std deviation,
            'throughput_gflops': Achieved GFLOPS,
        }
    """
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    if precision == 'fp16':
        model = model.half()
        input_tensor = input_tensor.half()

    model.eval()

    # Calculate FLOPs for GEMM: 2 * M * N * K (multiply-add)
    batch_size = input_tensor.shape[0] if input_tensor.dim() == 3 else 1
    m = model.m
    n = model.n
    k = model.k
    flops_per_gemm = 2 * m * n * k
    total_flops = flops_per_gemm * batch_size

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_tensor)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    times_tensor = torch.tensor(times)
    mean_time_ms = float(times_tensor.mean())
    mean_time_s = mean_time_ms / 1000

    throughput_gflops = (total_flops / mean_time_s) / 1e9 if mean_time_s > 0 else 0

    return {
        'time_ms': mean_time_ms,
        'time_std_ms': float(times_tensor.std()),
        'throughput_gflops': throughput_gflops,
        'total_flops': total_flops,
    }


# ============================================================================
# HARDWARE DETECTION
# ============================================================================

def detect_cpu_model() -> str:
    """Detect CPU model for mapper selection."""
    try:
        import subprocess
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        if 'i7-12700K' in result.stdout:
            return 'i7-12700k'
        return 'generic'
    except:
        return 'generic'


def detect_gpu_model() -> str:
    """Detect GPU model for mapper selection."""
    if not torch.cuda.is_available():
        return 'generic'

    try:
        gpu_name = torch.cuda.get_device_name(0).lower()

        if 'orin' in gpu_name:
            if 'agx' in gpu_name:
                return 'jetson-orin-agx'
            elif 'nx' in gpu_name:
                return 'jetson-orin-nx'
            elif 'nano' in gpu_name:
                return 'jetson-orin-nano'
            return 'jetson-orin-agx'

        if 'h100' in gpu_name:
            return 'h100'
        if 'a100' in gpu_name:
            return 'a100'

        return 'generic'
    except:
        return 'generic'


def normalize_power_mode(power_mode: str) -> str:
    """Normalize power mode by stripping MODE_ prefix."""
    if power_mode:
        pm = power_mode.upper()
        if pm.startswith('MODE_'):
            pm = pm[5:]
        return pm
    return None


# ============================================================================
# ANALYTICAL ESTIMATION
# ============================================================================

def run_analytical_estimate(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cpu',
    precision: str = 'fp32'
) -> Dict[str, float]:
    """
    Run analytical estimate using FX tracing + hardware mapper.

    Returns:
        {
            'time_ms': Estimated latency,
            'flops': Total FLOPs,
            'efficiency': Achieved efficiency (if calibration available),
        }
    """
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    if device == 'cuda' and precision == 'fp16':
        model = model.half()
        input_tensor = input_tensor.half()

    # Trace model
    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(input_tensor)

    # Partition graph
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(traced)

    # Create execution stages (simple sequential)
    execution_stages = [[i] for i in range(len(fusion_report.fused_subgraphs))]

    # Select mapper
    if device == 'cpu':
        cpu_model = detect_cpu_model()
        if cpu_model == 'i7-12700k':
            mapper = create_i7_12700k_mapper()
        else:
            mapper = create_intel_cpu_mapper()
    elif device == 'cuda':
        gpu_model = detect_gpu_model()

        # Load calibration
        calibration = None
        detected_power_mode = get_jetson_power_mode()
        thermal_profile = normalize_power_mode(detected_power_mode)

        if gpu_model == 'jetson-orin-agx' and thermal_profile:
            cal_id = f"jetson_orin_agx_{thermal_profile.lower()}"
            calibration = GPUCalibration.load(cal_id, precision)

        if gpu_model == 'jetson-orin-agx':
            mapper = create_jetson_orin_agx_64gb_mapper(
                thermal_profile=thermal_profile,
                calibration=calibration
            )
        elif gpu_model == 'jetson-orin-nx':
            mapper = create_jetson_orin_nx_16gb_mapper(
                thermal_profile=thermal_profile,
                calibration=calibration
            )
        elif gpu_model == 'jetson-orin-nano':
            mapper = create_jetson_orin_nano_8gb_mapper(
                thermal_profile=thermal_profile,
                calibration=calibration
            )
        else:
            mapper = create_h100_pcie_80gb_mapper()
    else:
        raise ValueError(f"Unsupported device: {device}")

    # Map to hardware
    prec = Precision.FP16 if precision == 'fp16' else Precision.FP32
    batch_size = input_tensor.shape[0] if input_tensor.dim() == 3 else 1

    hw_allocation = mapper.map_graph(
        fusion_report=fusion_report,
        execution_stages=execution_stages,
        batch_size=batch_size,
        precision=prec
    )

    return {
        'time_ms': hw_allocation.total_latency * 1000,
        'flops': fusion_report.total_flops,
    }


# ============================================================================
# SWEEP RUNNER
# ============================================================================

def run_sweep(
    sweep_params: Dict,
    device: str = 'cpu',
    output_file: str = 'gemm_sweep_results.csv'
) -> List[Dict]:
    """Run GEMM parameter sweep."""
    results = []

    # Generate all combinations
    total_configs = (len(sweep_params['dimensions']) *
                     len(sweep_params['batch_size']) *
                     len(sweep_params['precision']))

    print(f"Running GEMM sweep with {total_configs} configurations on {device}...")
    print(f"Results will be saved to: {output_file}")

    # Show hardware info
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        power_mode = get_jetson_power_mode()
        thermal_profile = normalize_power_mode(power_mode)

        print(f"  Hardware:    {gpu_name}")
        if power_mode:
            print(f"  Power Mode:  {power_mode}")

        # Check calibration
        gpu_model = detect_gpu_model()
        if gpu_model.startswith('jetson') and thermal_profile:
            cal_id = f"jetson_orin_agx_{thermal_profile.lower()}"
            cal = GPUCalibration.load(cal_id, 'fp32')
            if cal:
                print(f"  Calibration: {cal_id} (loaded)")
            else:
                print(f"  Calibration: {cal_id} (NOT FOUND)")

    print("=" * 70)

    config_num = 0
    success_count = 0

    for m, n, k in sweep_params['dimensions']:
        for batch_size in sweep_params['batch_size']:
            for precision in sweep_params['precision']:
                config_num += 1

                # Create model and input
                model = BatchedGEMM(m, n, k)
                input_tensor = torch.randn(batch_size, m, k)

                # Calculate theoretical FLOPs
                flops = 2 * m * n * k * batch_size

                print(f"\n[{config_num}/{total_configs}] GEMM {m}x{n}x{k}")
                print(f"  Batch={batch_size}, precision={precision}, FLOPs={flops/1e6:.1f}M")

                try:
                    # Run empirical benchmark
                    print("  Running empirical benchmark...", end=" ", flush=True)
                    empirical = run_empirical_benchmark(
                        model, input_tensor, device, precision
                    )
                    print(f"OK {empirical['time_ms']:.3f} ms, {empirical['throughput_gflops']:.1f} GFLOPS")

                    # Run analytical estimate
                    print("  Running analytical estimate...", end=" ", flush=True)
                    analytical = run_analytical_estimate(
                        model, input_tensor, device, precision
                    )
                    print(f"OK {analytical['time_ms']:.3f} ms")

                    # Compute error
                    time_error = abs(empirical['time_ms'] - analytical['time_ms']) / empirical['time_ms'] * 100
                    print(f"  Error: {time_error:.1f}%")

                    # Store result
                    result = {
                        'M': m,
                        'N': n,
                        'K': k,
                        'batch_size': batch_size,
                        'precision': precision,
                        'device': device,
                        'flops': flops,
                        'empirical_time_ms': empirical['time_ms'],
                        'empirical_time_std_ms': empirical['time_std_ms'],
                        'empirical_gflops': empirical['throughput_gflops'],
                        'analytical_time_ms': analytical['time_ms'],
                        'time_error_pct': time_error,
                    }
                    results.append(result)
                    success_count += 1

                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

    # Write CSV
    if results:
        output_path = Path(__file__).parent / 'results' / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'=' * 70}")
        print(f"Sweep complete! {success_count}/{total_configs} configs succeeded")
        print(f"Results saved to: {output_path}")

    # Summary statistics
    if results:
        times_emp = [r['empirical_time_ms'] for r in results]
        times_ana = [r['analytical_time_ms'] for r in results]
        errors = [r['time_error_pct'] for r in results]
        gflops = [r['empirical_gflops'] for r in results]

        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print("=" * 70)
        print(f"Configurations tested: {len(results)}")
        print(f"Device: {device}")
        print()
        print(f"Empirical timing: {min(times_emp):.3f} - {max(times_emp):.3f} ms")
        print(f"Analytical timing: {min(times_ana):.3f} - {max(times_ana):.3f} ms")
        print(f"Throughput: {min(gflops):.1f} - {max(gflops):.1f} GFLOPS")
        print()
        print(f"Mean Absolute Percentage Error (MAPE): {sum(errors)/len(errors):.1f}%")
        print(f"  Best:  {min(errors):.1f}%")
        print(f"  Worst: {max(errors):.1f}%")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GEMM Empirical Benchmark Sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick --device cuda
  %(prog)s --full --device cuda
  %(prog)s --square --device cuda
"""
    )

    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument('--quick', action='store_true',
                             help='Quick sweep (few configs)')
    sweep_group.add_argument('--full', action='store_true',
                             help='Full sweep (many configs)')
    sweep_group.add_argument('--square', action='store_true',
                             help='Square matrices only')

    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                        help='Device to benchmark on')
    parser.add_argument('--output', default=None, help='Output CSV file')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA not available. Falling back to CPU.")
        args.device = 'cpu'

    # Build device suffix
    device_suffix = args.device
    if args.device == 'cuda':
        power_mode = get_jetson_power_mode()
        if power_mode:
            pm = normalize_power_mode(power_mode)
            device_suffix = f"cuda_{pm.lower()}"

    # Select sweep and output file
    if args.full:
        sweep_params = FULL_SWEEP
        output_file = args.output or f'gemm_sweep_full_{device_suffix}.csv'
    elif args.square:
        sweep_params = SQUARE_SWEEP
        output_file = args.output or f'gemm_sweep_square_{device_suffix}.csv'
    else:
        sweep_params = QUICK_SWEEP
        output_file = args.output or f'gemm_sweep_quick_{device_suffix}.csv'

    # Run sweep
    run_sweep(sweep_params, args.device, output_file)


if __name__ == '__main__':
    main()

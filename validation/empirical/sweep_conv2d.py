#!/usr/bin/env python
"""
Conv2D Empirical Benchmark Sweep

Runs parameter sweeps across Conv2D configurations to:
1. Measure real execution time on CPU/GPU
2. Compare against analytical estimates
3. Validate calibration accuracy for convolution operations
4. Cover different convolution types (standard, depthwise, pointwise)

Usage:
    # Quick sweep (~20 configs)
    python validation/empirical/sweep_conv2d.py --quick --device cuda

    # Full sweep (~300+ configs)
    python validation/empirical/sweep_conv2d.py --full --device cuda

    # Depthwise only (memory-bound ops)
    python validation/empirical/sweep_conv2d.py --depthwise --device cuda
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
from typing import Dict, List, Tuple, Optional

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

# Config format: (in_channels, out_channels, spatial_size, kernel_size, stride, groups)
# groups=1 for standard conv, groups=in_channels for depthwise

# Quick sweep (~24 configs)
QUICK_SWEEP = {
    'configs': [
        # Standard 3x3 convolutions
        (64, 64, 56, 3, 1, 1),      # ResNet-like
        (128, 128, 28, 3, 1, 1),    # ResNet-like
        (256, 256, 14, 3, 1, 1),    # ResNet-like

        # Pointwise 1x1 (channel mixing, compute-bound)
        (64, 256, 56, 1, 1, 1),     # ResNet bottleneck expand
        (256, 64, 56, 1, 1, 1),     # ResNet bottleneck reduce

        # Depthwise 3x3 (memory-bound)
        (64, 64, 56, 3, 1, 64),     # MobileNet-like
    ],
    'batch_size': [1, 16, 64],
    'precision': ['fp32'],
}

# Full sweep (~300+ configs)
FULL_SWEEP = {
    'configs': [
        # =====================================================
        # Standard 3x3 convolutions (most common)
        # =====================================================
        # Early layers (large spatial, few channels)
        (3, 64, 224, 3, 1, 1),       # First conv (like ResNet)
        (64, 64, 112, 3, 1, 1),
        (64, 64, 56, 3, 1, 1),

        # Middle layers
        (64, 128, 56, 3, 2, 1),      # Stride-2 downsample
        (128, 128, 28, 3, 1, 1),
        (128, 256, 28, 3, 2, 1),
        (256, 256, 14, 3, 1, 1),
        (256, 512, 14, 3, 2, 1),
        (512, 512, 7, 3, 1, 1),

        # Late layers (small spatial, many channels)
        (512, 512, 7, 3, 1, 1),
        (1024, 1024, 7, 3, 1, 1),

        # =====================================================
        # Pointwise 1x1 convolutions (compute-bound)
        # =====================================================
        (64, 64, 56, 1, 1, 1),
        (64, 256, 56, 1, 1, 1),      # Bottleneck expand
        (256, 64, 56, 1, 1, 1),      # Bottleneck reduce
        (256, 256, 28, 1, 1, 1),
        (512, 128, 14, 1, 1, 1),
        (128, 512, 14, 1, 1, 1),
        (1024, 256, 7, 1, 1, 1),

        # =====================================================
        # 5x5 convolutions
        # =====================================================
        (32, 32, 112, 5, 1, 1),
        (64, 64, 56, 5, 1, 1),
        (128, 128, 28, 5, 1, 1),

        # =====================================================
        # 7x7 convolutions (first layer typical)
        # =====================================================
        (3, 64, 224, 7, 2, 1),       # ResNet first conv

        # =====================================================
        # Depthwise convolutions (memory-bound)
        # =====================================================
        (32, 32, 112, 3, 1, 32),
        (64, 64, 56, 3, 1, 64),
        (128, 128, 28, 3, 1, 128),
        (256, 256, 14, 3, 1, 256),
        (512, 512, 7, 3, 1, 512),

        # Depthwise 5x5
        (64, 64, 56, 5, 1, 64),
        (128, 128, 28, 5, 1, 128),
    ],
    'batch_size': [1, 4, 16, 32, 64],
    'precision': ['fp32', 'fp16'],
}

# Depthwise only (for memory-bound analysis)
DEPTHWISE_SWEEP = {
    'configs': [
        # Depthwise 3x3 at various scales
        (16, 16, 224, 3, 1, 16),
        (32, 32, 112, 3, 1, 32),
        (64, 64, 56, 3, 1, 64),
        (96, 96, 56, 3, 1, 96),
        (128, 128, 28, 3, 1, 128),
        (192, 192, 28, 3, 1, 192),
        (256, 256, 14, 3, 1, 256),
        (384, 384, 14, 3, 1, 384),
        (512, 512, 7, 3, 1, 512),
        (768, 768, 7, 3, 1, 768),

        # Depthwise 5x5
        (32, 32, 112, 5, 1, 32),
        (64, 64, 56, 5, 1, 64),
        (128, 128, 28, 5, 1, 128),
        (256, 256, 14, 5, 1, 256),

        # Depthwise with stride
        (64, 64, 112, 3, 2, 64),
        (128, 128, 56, 3, 2, 128),
        (256, 256, 28, 3, 2, 256),
    ],
    'batch_size': [1, 8, 32, 128],
    'precision': ['fp32', 'fp16'],
}

# Pointwise only (for compute-bound analysis)
POINTWISE_SWEEP = {
    'configs': [
        # Channel expansion
        (32, 128, 112, 1, 1, 1),
        (64, 256, 56, 1, 1, 1),
        (128, 512, 28, 1, 1, 1),
        (256, 1024, 14, 1, 1, 1),

        # Channel reduction
        (128, 32, 112, 1, 1, 1),
        (256, 64, 56, 1, 1, 1),
        (512, 128, 28, 1, 1, 1),
        (1024, 256, 14, 1, 1, 1),

        # Same channels
        (64, 64, 56, 1, 1, 1),
        (128, 128, 28, 1, 1, 1),
        (256, 256, 14, 1, 1, 1),
        (512, 512, 7, 1, 1, 1),
    ],
    'batch_size': [1, 8, 32, 128],
    'precision': ['fp32', 'fp16'],
}


# ============================================================================
# CONV2D MODULE
# ============================================================================

class Conv2DModule(nn.Module):
    """Simple Conv2D wrapper for FX tracing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1
    ):
        super().__init__()
        padding = kernel_size // 2  # Same padding
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
        )
        # Store for FLOP calculation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ============================================================================
# EMPIRICAL BENCHMARKING
# ============================================================================

def calculate_conv2d_flops(
    batch: int,
    in_channels: int,
    out_channels: int,
    spatial_h: int,
    spatial_w: int,
    kernel_size: int,
    stride: int,
    groups: int
) -> int:
    """Calculate FLOPs for a Conv2D operation."""
    # Output spatial dimensions
    out_h = spatial_h // stride
    out_w = spatial_w // stride

    # FLOPs per output element: 2 * (kernel_size^2) * (in_channels / groups)
    # (multiply-add = 2 ops)
    flops_per_output = 2 * (kernel_size ** 2) * (in_channels // groups)

    # Total FLOPs
    total_flops = batch * out_channels * out_h * out_w * flops_per_output

    return total_flops


def run_empirical_benchmark(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cpu',
    precision: str = 'fp32',
    num_warmup: int = 20,
    num_runs: int = 100
) -> Dict[str, float]:
    """Run empirical benchmark measuring actual execution time."""
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    if precision == 'fp16':
        model = model.half()
        input_tensor = input_tensor.half()

    model.eval()

    # Calculate FLOPs
    batch_size = input_tensor.shape[0]
    in_channels = input_tensor.shape[1]
    spatial_h = input_tensor.shape[2]
    spatial_w = input_tensor.shape[3]

    total_flops = calculate_conv2d_flops(
        batch_size,
        model.in_channels,
        model.out_channels,
        spatial_h, spatial_w,
        model.kernel_size,
        model.stride,
        model.groups
    )

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
            times.append((end - start) * 1000)

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


def normalize_power_mode(power_mode: str) -> Optional[str]:
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
    """Run analytical estimate using FX tracing + hardware mapper."""
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

    # Create execution stages
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
    batch_size = input_tensor.shape[0]

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

def get_conv_type(groups: int, in_channels: int, kernel_size: int) -> str:
    """Classify convolution type."""
    if groups == in_channels:
        return "depthwise"
    elif kernel_size == 1:
        return "pointwise"
    else:
        return "standard"


def run_sweep(
    sweep_params: Dict,
    device: str = 'cpu',
    output_file: str = 'conv2d_sweep_results.csv'
) -> List[Dict]:
    """Run Conv2D parameter sweep."""
    results = []

    total_configs = (len(sweep_params['configs']) *
                     len(sweep_params['batch_size']) *
                     len(sweep_params['precision']))

    print(f"Running Conv2D sweep with {total_configs} configurations on {device}...")
    print(f"Results will be saved to: {output_file}")

    # Show hardware info
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        power_mode = get_jetson_power_mode()
        thermal_profile = normalize_power_mode(power_mode)

        print(f"  Hardware:    {gpu_name}")
        if power_mode:
            print(f"  Power Mode:  {power_mode}")

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

    for in_ch, out_ch, spatial, kernel, stride, groups in sweep_params['configs']:
        for batch_size in sweep_params['batch_size']:
            for precision in sweep_params['precision']:
                config_num += 1

                conv_type = get_conv_type(groups, in_ch, kernel)

                # Create model and input
                model = Conv2DModule(in_ch, out_ch, kernel, stride, groups)
                input_tensor = torch.randn(batch_size, in_ch, spatial, spatial)

                # Calculate FLOPs
                flops = calculate_conv2d_flops(
                    batch_size, in_ch, out_ch, spatial, spatial, kernel, stride, groups
                )

                print(f"\n[{config_num}/{total_configs}] Conv2D {in_ch}->{out_ch} {kernel}x{kernel} ({conv_type})")
                print(f"  Spatial={spatial}x{spatial}, stride={stride}, batch={batch_size}, {precision}")
                print(f"  FLOPs={flops/1e6:.1f}M")

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
                        'conv_type': conv_type,
                        'in_channels': in_ch,
                        'out_channels': out_ch,
                        'spatial_size': spatial,
                        'kernel_size': kernel,
                        'stride': stride,
                        'groups': groups,
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
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print("=" * 70)

        # Overall stats
        errors = [r['time_error_pct'] for r in results]
        gflops = [r['empirical_gflops'] for r in results]

        print(f"Configurations tested: {len(results)}")
        print(f"Device: {device}")
        print()
        print(f"Throughput: {min(gflops):.1f} - {max(gflops):.1f} GFLOPS")
        print()
        print(f"Mean Absolute Percentage Error (MAPE): {sum(errors)/len(errors):.1f}%")
        print(f"  Best:  {min(errors):.1f}%")
        print(f"  Worst: {max(errors):.1f}%")

        # Stats by conv type
        print()
        print("MAPE by convolution type:")
        for conv_type in ['standard', 'pointwise', 'depthwise']:
            type_results = [r for r in results if r['conv_type'] == conv_type]
            if type_results:
                type_errors = [r['time_error_pct'] for r in type_results]
                type_gflops = [r['empirical_gflops'] for r in type_results]
                print(f"  {conv_type:10}: MAPE={sum(type_errors)/len(type_errors):6.1f}%, "
                      f"GFLOPS={sum(type_gflops)/len(type_gflops):6.1f} ({len(type_results)} configs)")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Conv2D Empirical Benchmark Sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick --device cuda
  %(prog)s --full --device cuda
  %(prog)s --depthwise --device cuda
  %(prog)s --pointwise --device cuda
"""
    )

    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument('--quick', action='store_true',
                             help='Quick sweep (few configs)')
    sweep_group.add_argument('--full', action='store_true',
                             help='Full sweep (many configs)')
    sweep_group.add_argument('--depthwise', action='store_true',
                             help='Depthwise convolutions only')
    sweep_group.add_argument('--pointwise', action='store_true',
                             help='Pointwise (1x1) convolutions only')

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
        output_file = args.output or f'conv2d_sweep_full_{device_suffix}.csv'
    elif args.depthwise:
        sweep_params = DEPTHWISE_SWEEP
        output_file = args.output or f'conv2d_sweep_depthwise_{device_suffix}.csv'
    elif args.pointwise:
        sweep_params = POINTWISE_SWEEP
        output_file = args.output or f'conv2d_sweep_pointwise_{device_suffix}.csv'
    else:
        sweep_params = QUICK_SWEEP
        output_file = args.output or f'conv2d_sweep_quick_{device_suffix}.csv'

    # Run sweep
    run_sweep(sweep_params, args.device, output_file)


if __name__ == '__main__':
    main()

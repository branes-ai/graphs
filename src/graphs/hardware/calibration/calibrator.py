"""
Hardware calibration orchestrator.

Runs calibration benchmarks and produces a complete HardwareCalibration profile.
"""

import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .schema import HardwareCalibration, CalibrationMetadata, OperationCalibration, FusionCalibration
from .benchmarks import calibrate_matmul, calibrate_memory_bandwidth


def detect_hardware_info() -> dict:
    """Detect system hardware information"""
    import psutil

    cpu_info = {
        'cpu_model': platform.processor() or 'Unknown',
        'cpu_count': psutil.cpu_count(logical=False),  # Physical cores
        'logical_cpu_count': psutil.cpu_count(logical=True),
        'total_memory_gb': psutil.virtual_memory().total / (1024**3),
    }

    return cpu_info


def detect_software_versions() -> dict:
    """Detect software versions"""
    versions = {
        'python_version': platform.python_version(),
    }

    try:
        import numpy as np
        versions['numpy_version'] = np.__version__
    except ImportError:
        versions['numpy_version'] = None

    try:
        import torch
        versions['pytorch_version'] = torch.__version__
    except ImportError:
        versions['pytorch_version'] = None

    return versions


def calibrate_hardware(
    hardware_name: str,
    theoretical_peak_gflops: float,
    theoretical_bandwidth_gbps: float,
    output_path: Optional[Path] = None,
    operations: Optional[List[str]] = None,
    fusion_patterns: Optional[List[str]] = None,
    quick: bool = False
) -> HardwareCalibration:
    """
    Run full hardware calibration.

    Args:
        hardware_name: Name of hardware being calibrated (e.g., "i7-12700K")
        theoretical_peak_gflops: Theoretical peak GFLOPS from datasheet
        theoretical_bandwidth_gbps: Theoretical memory bandwidth from datasheet
        output_path: Optional path to save calibration JSON
        operations: List of operations to calibrate (None = all)
        fusion_patterns: List of fusion patterns to benchmark (e.g., ['linear', 'conv', 'attention'] or ['all'])
        quick: If True, run faster but less comprehensive calibration

    Returns:
        Complete HardwareCalibration object
    """
    print("=" * 80)
    print(f"Hardware Calibration: {hardware_name}")
    print("=" * 80)
    print()

    # Gather system information
    hw_info = detect_hardware_info()
    sw_versions = detect_software_versions()

    print("System Information:")
    print(f"  CPU: {hw_info['cpu_model']}")
    print(f"  Cores: {hw_info['cpu_count']} physical, {hw_info['logical_cpu_count']} logical")
    print(f"  Memory: {hw_info['total_memory_gb']:.1f} GB")
    print(f"  Python: {sw_versions['python_version']}")
    if sw_versions['numpy_version']:
        print(f"  NumPy: {sw_versions['numpy_version']}")
    if sw_versions['pytorch_version']:
        print(f"  PyTorch: {sw_versions['pytorch_version']}")
    print()

    # Create metadata
    metadata = CalibrationMetadata(
        hardware_name=hardware_name,
        calibration_date=datetime.now().isoformat(),
        calibration_tool_version="1.0.0",
        cpu_model=hw_info['cpu_model'],
        cpu_count=hw_info['cpu_count'],
        total_memory_gb=hw_info['total_memory_gb'],
        python_version=sw_versions['python_version'],
        numpy_version=sw_versions['numpy_version'],
        pytorch_version=sw_versions['pytorch_version'],
        num_warmup_runs=2 if quick else 3,
        num_measurement_runs=5 if quick else 10,
    )

    # Initialize calibration object
    calibration = HardwareCalibration(
        metadata=metadata,
        theoretical_peak_gflops=theoretical_peak_gflops,
        theoretical_bandwidth_gbps=theoretical_bandwidth_gbps,
        best_measured_gflops=0.0,
        avg_measured_gflops=0.0,
        worst_measured_gflops=0.0,
        measured_bandwidth_gbps=0.0,
        bandwidth_efficiency=0.0,
    )

    # Determine which operations to calibrate
    if operations is None:
        operations = ['matmul', 'memory']

    # Run calibrations
    print("Running calibration benchmarks...")
    print()

    if 'memory' in operations:
        print("1. Memory Bandwidth")
        print("-" * 80)
        sizes = [128, 256] if quick else [64, 128, 256, 512]
        mem_calibrations = calibrate_memory_bandwidth(
            sizes_mb=sizes,
            theoretical_bandwidth_gbps=theoretical_bandwidth_gbps,
            num_trials=metadata.num_measurement_runs
        )

        for cal in mem_calibrations:
            calibration.add_operation(cal)

        # Update bandwidth measurement
        calibration.measured_bandwidth_gbps = max(
            c.achieved_bandwidth_gbps for c in mem_calibrations
        )
        calibration.bandwidth_efficiency = (
            calibration.measured_bandwidth_gbps / theoretical_bandwidth_gbps
        )
        print()

    if 'matmul' in operations:
        print("2. Matrix Multiplication")
        print("-" * 80)
        sizes = [1024, 2048] if quick else [512, 1024, 2048, 4096]
        matmul_calibrations = calibrate_matmul(
            sizes=sizes,
            theoretical_peak_gflops=theoretical_peak_gflops,
            theoretical_bandwidth_gbps=theoretical_bandwidth_gbps,
            num_trials=metadata.num_measurement_runs
        )

        for cal in matmul_calibrations:
            calibration.add_operation(cal)
        print()

    # Fusion pattern calibration
    if fusion_patterns:
        print("3. Fused Kernel Patterns")
        print("-" * 80)

        # Expand 'all' to all patterns
        if 'all' in fusion_patterns:
            fusion_patterns = ['linear', 'conv', 'attention']

        for pattern in fusion_patterns:
            if pattern == 'linear':
                from .benchmarks.fused_linear_bench import calibrate_linear_fusion_patterns

                print("  Linear fusion patterns...")
                fusion_results = calibrate_linear_fusion_patterns(quick=quick)

                for result in fusion_results:
                    fusion_cal = FusionCalibration(
                        fusion_pattern=result['fusion_pattern'],
                        operators=result['fusion_pattern'].split('_'),
                        num_operators=len(result['fusion_pattern'].split('_')),
                        unfused_latency_ms=result['unfused_latency_ms'],
                        unfused_gflops=result['unfused_gflops'],
                        unfused_memory_bytes=result['unfused_bytes'],
                        fused_latency_ms=result['fused_latency_ms'],
                        fused_gflops=result['fused_gflops'],
                        fused_memory_bytes=result['fused_bytes'],
                        speedup_factor=result['speedup_factor'],
                        memory_reduction=result['memory_reduction'],
                        gflops_improvement=(result['fused_gflops'] - result['unfused_gflops']) / result['unfused_gflops'] if result['unfused_gflops'] > 0 else 0.0,
                        input_shape=result['input_shape'],
                        num_trials=50 if quick else 100,
                    )
                    calibration.add_fusion_pattern(fusion_cal)

            elif pattern == 'conv':
                from .benchmarks.fused_conv_bench import calibrate_conv_fusion_patterns

                print("  Conv fusion patterns...")
                fusion_results = calibrate_conv_fusion_patterns(quick=quick)

                for result in fusion_results:
                    fusion_cal = FusionCalibration(
                        fusion_pattern=result['fusion_pattern'],
                        operators=result['fusion_pattern'].split('_'),
                        num_operators=len(result['fusion_pattern'].split('_')),
                        unfused_latency_ms=result['unfused_latency_ms'],
                        unfused_gflops=result['unfused_gflops'],
                        unfused_memory_bytes=result['unfused_bytes'],
                        fused_latency_ms=result['fused_latency_ms'],
                        fused_gflops=result['fused_gflops'],
                        fused_memory_bytes=result['fused_bytes'],
                        speedup_factor=result['speedup_factor'],
                        memory_reduction=result['memory_reduction'],
                        gflops_improvement=(result['fused_gflops'] - result['unfused_gflops']) / result['unfused_gflops'] if result['unfused_gflops'] > 0 else 0.0,
                        input_shape=result['input_shape'],
                        extra_params=result.get('extra_params', {}),
                        num_trials=50 if quick else 100,
                    )
                    calibration.add_fusion_pattern(fusion_cal)

            elif pattern == 'attention':
                from .benchmarks.fused_attention_bench import calibrate_attention_fusion_patterns

                print("  Attention fusion patterns...")
                fusion_results = calibrate_attention_fusion_patterns(quick=quick)

                for result in fusion_results:
                    fusion_cal = FusionCalibration(
                        fusion_pattern=result['fusion_pattern'],
                        operators=result['fusion_pattern'].split('_'),
                        num_operators=len(result['fusion_pattern'].split('_')),
                        unfused_latency_ms=result['unfused_latency_ms'],
                        unfused_gflops=result['unfused_gflops'],
                        unfused_memory_bytes=result['unfused_bytes'],
                        fused_latency_ms=result['fused_latency_ms'],
                        fused_gflops=result['fused_gflops'],
                        fused_memory_bytes=result['fused_bytes'],
                        speedup_factor=result['speedup_factor'],
                        memory_reduction=result['memory_reduction'],
                        gflops_improvement=(result['fused_gflops'] - result['unfused_gflops']) / result['unfused_gflops'] if result['unfused_gflops'] > 0 else 0.0,
                        input_shape=result['input_shape'],
                        num_trials=50 if quick else 100,
                    )
                    calibration.add_fusion_pattern(fusion_cal)

        print()

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        calibration.save(output_path)
        print(f"Calibration saved to: {output_path}")
        print()

    # Print summary
    calibration.print_summary()

    return calibration


def load_calibration(filepath: Path) -> HardwareCalibration:
    """Load calibration from JSON file"""
    return HardwareCalibration.load(filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hardware Performance Calibration")
    parser.add_argument("--hardware", type=str, required=True,
                       help="Hardware name (e.g., i7-12700K)")
    parser.add_argument("--peak-gflops", type=float, required=True,
                       help="Theoretical peak GFLOPS")
    parser.add_argument("--peak-bandwidth", type=float, required=True,
                       help="Theoretical peak bandwidth (GB/s)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick calibration (fewer sizes/trials)")
    parser.add_argument("--operations", type=str, default=None,
                       help="Comma-separated list of operations (matmul,memory)")
    parser.add_argument("--fusion-patterns", type=str, default=None,
                       help="Comma-separated list of fusion patterns (linear,conv,attention,all)")

    args = parser.parse_args()

    operations = None
    if args.operations:
        operations = [op.strip() for op in args.operations.split(',')]

    fusion_patterns = None
    if args.fusion_patterns:
        fusion_patterns = [p.strip() for p in args.fusion_patterns.split(',')]

    calibration = calibrate_hardware(
        hardware_name=args.hardware,
        theoretical_peak_gflops=args.peak_gflops,
        theoretical_bandwidth_gbps=args.peak_bandwidth,
        output_path=args.output,
        operations=operations,
        fusion_patterns=fusion_patterns,
        quick=args.quick
    )

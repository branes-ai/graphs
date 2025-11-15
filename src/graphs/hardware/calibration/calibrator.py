"""
Hardware calibration orchestrator.

Runs calibration benchmarks and produces a complete HardwareCalibration profile.
"""

import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from .schema import HardwareCalibration, CalibrationMetadata, OperationCalibration, FusionCalibration, PrecisionCapabilityMatrix
from .benchmarks import calibrate_matmul, calibrate_memory_bandwidth
from .benchmarks.matmul_bench_multi import calibrate_matmul_all_precisions
from .precision_detector import get_precision_capabilities
from ..resource_model import Precision


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
    theoretical_peaks: Optional[Dict[str, float]] = None,  # NEW: per-precision theoretical peaks
    device: str = 'cpu',  # NEW: 'cpu' or 'cuda'
    output_path: Optional[Path] = None,
    operations: Optional[List[str]] = None,
    fusion_patterns: Optional[List[str]] = None,
    quick: bool = False
) -> HardwareCalibration:
    """
    Run full hardware calibration.

    Args:
        hardware_name: Name of hardware being calibrated (e.g., "i7-12700K")
        theoretical_peak_gflops: Theoretical peak GFLOPS from datasheet (FP32 or default precision)
        theoretical_bandwidth_gbps: Theoretical memory bandwidth from datasheet
        theoretical_peaks: Per-precision theoretical peaks (dict: precision -> GFLOPS)
        device: Device type ('cpu' or 'cuda')
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
        device_type=device,  # NEW: record device type
        platform_architecture=platform.machine().lower(),  # NEW: record platform
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
        print("2. Matrix Multiplication (Multi-Precision)")
        print("-" * 80)

        # Determine which precisions to test from theoretical_peaks
        if theoretical_peaks:
            precisions_to_test = [
                Precision(prec_name) for prec_name in theoretical_peaks.keys()
                if prec_name in [p.value for p in Precision]
            ]
        else:
            # Fallback: test common precisions
            precisions_to_test = [Precision.FP64, Precision.FP32, Precision.INT32, Precision.INT16, Precision.INT8]

        sizes = [1024, 2048] if quick else [1024, 2048, 4096]

        # Run multi-precision calibration
        multi_prec_results = calibrate_matmul_all_precisions(
            sizes=sizes,
            precisions=precisions_to_test,
            theoretical_peaks=theoretical_peaks or {},
            device=device,
            num_trials=metadata.num_measurement_runs,
            min_useful_throughput=50.0  # Skip precisions with <50 GOPS (not useful for Embodied AI)
        )

        # Convert to OperationCalibration objects with precision_results populated
        for size, precision_results in multi_prec_results.items():
            # Use FP32 as the primary result for backward compatibility
            fp32_result = precision_results.get('fp32')
            if not fp32_result or not fp32_result.supported:
                # Fallback to first supported precision
                fp32_result = next((r for r in precision_results.values() if r.supported), None)

            if fp32_result and fp32_result.supported:
                # Create OperationCalibration
                op_cal = OperationCalibration(
                    operation_type='matmul',
                    measured_gflops=fp32_result.measured_gops,  # Use FP32 result for backward compat
                    efficiency=fp32_result.efficiency or 0.0,
                    achieved_bandwidth_gbps=0.0,  # Computed separately
                    memory_bound=fp32_result.arithmetic_intensity < 10.0 if fp32_result.arithmetic_intensity else False,
                    compute_bound=fp32_result.arithmetic_intensity >= 10.0 if fp32_result.arithmetic_intensity else True,
                    arithmetic_intensity=fp32_result.arithmetic_intensity or 0.0,
                    batch_size=1,
                    input_shape=(size, size),
                    output_shape=(size, size),
                    mean_latency_ms=fp32_result.mean_latency_ms,
                    std_latency_ms=fp32_result.std_latency_ms or 0.0,
                    min_latency_ms=fp32_result.min_latency_ms or 0.0,
                    max_latency_ms=fp32_result.max_latency_ms or 0.0,
                    num_trials=fp32_result.num_trials,
                    extra_params={'matrix_size': size, 'device': device},
                    precision_results=precision_results  # NEW: all precision results
                )

                calibration.add_operation(op_cal)

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

    # Build precision capability matrix
    print("Building precision capability matrix...")
    supported, unsupported = get_precision_capabilities(device)

    precision_matrix = PrecisionCapabilityMatrix(
        hardware_name=hardware_name,
        supported_precisions=[p.value for p in supported],
        unsupported_precisions=[p.value for p in unsupported],
        peak_gflops_by_precision={},
        speedup_vs_fp32={},
        theoretical_peaks=theoretical_peaks or {}
    )

    # Extract peak GOPS (GFLOPS/GIOPS) per precision from matmul results
    for op_cal in calibration.operation_profiles.values():
        if op_cal.operation_type == 'matmul' and op_cal.precision_results:
            for prec_name, prec_result in op_cal.precision_results.items():
                if prec_result.supported and prec_result.measured_gops:
                    # Track best GOPS for each precision (GFLOPS for float, GIOPS for int)
                    current_best = precision_matrix.peak_gflops_by_precision.get(prec_name, 0.0)
                    precision_matrix.peak_gflops_by_precision[prec_name] = max(
                        current_best,
                        prec_result.measured_gops
                    )

                    # Track speedup vs FP32
                    if prec_result.speedup_vs_fp32:
                        precision_matrix.speedup_vs_fp32[prec_name] = prec_result.speedup_vs_fp32

    calibration.precision_matrix = precision_matrix
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

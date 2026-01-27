"""
Hardware calibration orchestrator.

Runs calibration benchmarks and produces a complete HardwareCalibration profile.
"""

import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from .schema import (
    HardwareCalibration, CalibrationMetadata, OperationCalibration, FusionCalibration,
    PrecisionCapabilityMatrix, GPUClockData, CPUClockData, PreflightData, PreflightCheckResult,
    CANONICAL_PRECISION_ORDER
)
# Note: Benchmark imports moved to functions to avoid circular imports
# from graphs.benchmarks import calibrate_matmul, calibrate_memory_bandwidth
# from graphs.benchmarks.matmul_bench_multi import calibrate_matmul_all_precisions
from .precision_detector import get_precision_capabilities
from .gpu_clock import get_gpu_clock_info, get_gpu_clock_under_load, GPUClockInfo
from .cpu_clock import get_cpu_clock_info, CPUClockInfo
from .preflight import run_preflight_checks, PreflightReport
from graphs.hardware.resource_model import Precision

# Framework-specific benchmark imports
try:
    from graphs.benchmarks.numpy_benchmarks import calibrate_matmul_numpy, calibrate_memory_bandwidth_numpy
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from graphs.benchmarks.pytorch_benchmarks import calibrate_matmul_pytorch, calibrate_memory_bandwidth_pytorch
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


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


def select_framework(device: str, framework_override: Optional[str] = None) -> str:
    """
    Select which framework to use for benchmarks.

    Args:
        device: Device type ('cpu' or 'cuda')
        framework_override: Optional framework override ('numpy' or 'pytorch')

    Returns:
        Framework name ('numpy' or 'pytorch')

    Raises:
        RuntimeError if requirements not met
    """
    # Handle explicit override
    if framework_override:
        if framework_override == 'numpy':
            if not NUMPY_AVAILABLE:
                raise RuntimeError("NumPy framework requested but NumPy is not installed")
            if device == 'cuda':
                raise RuntimeError("NumPy cannot use CUDA. Use PyTorch for GPU benchmarks.")
            return 'numpy'
        elif framework_override == 'pytorch':
            if not PYTORCH_AVAILABLE:
                raise RuntimeError("PyTorch framework requested but PyTorch is not installed")
            return 'pytorch'
        else:
            raise ValueError(f"Unknown framework: {framework_override}")

    # Auto-select based on device
    if device == 'cuda':
        # GPU requires PyTorch
        if not PYTORCH_AVAILABLE:
            raise RuntimeError(
                "CUDA device requires PyTorch, but PyTorch is not installed. "
                "Install PyTorch with CUDA support."
            )
        return 'pytorch'
    else:
        # CPU: prefer NumPy (represents real-world Embodied AI signal processing)
        if NUMPY_AVAILABLE:
            return 'numpy'
        elif PYTORCH_AVAILABLE:
            print("[!] NumPy not available, falling back to PyTorch for CPU benchmarks")
            return 'pytorch'
        else:
            raise RuntimeError("Neither NumPy nor PyTorch is installed")


def calibrate_hardware(
    hardware_name: str,
    theoretical_peak_gflops: float,
    theoretical_bandwidth_gbps: float,
    theoretical_peaks: Optional[Dict[str, float]] = None,  # NEW: per-precision theoretical peaks
    device: str = 'cpu',  # NEW: 'cpu' or 'cuda'
    actual_device_info: Optional[Dict] = None,  # NEW: actual device being used (for reporting)
    framework: Optional[str] = None,  # NEW: framework override ('numpy' or 'pytorch')
    output_path: Optional[Path] = None,
    operations: Optional[List[str]] = None,
    fusion_patterns: Optional[List[str]] = None,
    quick: bool = False,
    min_useful_gflops: float = 1.0,  # NEW: minimum GFLOPS threshold for early termination
    force: bool = False,  # NEW: force calibration despite failed pre-flight checks
    skip_preflight: bool = False,  # NEW: skip pre-flight checks entirely
) -> HardwareCalibration:
    """
    Run full hardware calibration.

    Args:
        hardware_name: Name of hardware being calibrated (e.g., "i7-12700K")
        theoretical_peak_gflops: Theoretical peak GFLOPS from datasheet (FP32 or default precision)
        theoretical_bandwidth_gbps: Theoretical memory bandwidth from datasheet
        theoretical_peaks: Per-precision theoretical peaks (dict: precision -> GFLOPS)
        device: Device type ('cpu' or 'cuda')
        actual_device_info: Actual device being used (from detect_actual_device), optional
        framework: Framework override ('numpy' or 'pytorch', default: auto-select)
        output_path: Optional path to save calibration JSON
        operations: List of operations to calibrate (None = all)
        fusion_patterns: List of fusion patterns to benchmark (e.g., ['linear', 'conv', 'attention'] or ['all'])
        quick: If True, run faster but less comprehensive calibration
        force: If True, continue calibration even if pre-flight checks fail
        skip_preflight: If True, skip pre-flight checks entirely

    Returns:
        Complete HardwareCalibration object

    Raises:
        RuntimeError: If pre-flight checks fail and force=False
    """
    print("=" * 80)
    print(f"Hardware Calibration: {hardware_name}")
    print("=" * 80)
    print()

    # Run pre-flight checks (unless skipped)
    preflight_data: Optional[PreflightData] = None
    if not skip_preflight:
        print("Running pre-flight checks...")
        preflight_report = run_preflight_checks(device)
        print(preflight_report.format_report())

        # Convert preflight report to storage format
        preflight_data = PreflightData(
            timestamp=preflight_report.timestamp,
            passed=preflight_report.passed,
            forced=force if not preflight_report.passed else False,
            checks=[
                PreflightCheckResult(
                    name=c.name,
                    status=c.status.value,
                    message=c.message,
                    current_value=c.current_value,
                    expected_value=c.expected_value,
                )
                for c in preflight_report.checks
            ]
        )

        # Abort if pre-flight checks failed and not forcing
        if not preflight_report.passed and not force:
            raise RuntimeError(
                "Pre-flight checks failed. System is not in performance mode.\n"
                "Calibration results would not represent peak hardware capability.\n"
                "Use --force to override (results will be flagged as non-representative)."
            )

        if not preflight_report.passed and force:
            print()
            print("[!] WARNING: Proceeding with calibration despite failed pre-flight checks.")
            print("  Results will be flagged as non-representative of peak performance.")
            print()
        elif preflight_report.has_warnings:
            print()
            print("[!] Note: Some pre-flight checks have warnings.")
            print("  Results may not represent absolute peak performance.")
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

    # Select framework
    selected_framework = select_framework(device, framework)

    # Display execution device and framework information prominently
    if actual_device_info:
        print("Execution Device:")
        print(f"  Running on: {actual_device_info['device_name']}")
        if actual_device_info['fallback_occurred']:
            print(f"  [!] FALLBACK from requested '{device}' to '{actual_device_info['actual_device']}'")
            print(f"  Reason: {actual_device_info['fallback_reason']}")
        print(f"  Framework:  {selected_framework.upper()}")
        if selected_framework == 'numpy':
            print(f"              (CPU-only, real-world signal processing performance)")
        else:
            print(f"              (PyTorch DL framework, GPU-accelerated)")
        print()
    else:
        # Fallback if device info not provided (for backward compatibility)
        print(f"Target Device: {device.upper()}")
        print(f"Framework:     {selected_framework.upper()}")
        print()

    # Query CPU clock frequencies (required for all calibrations)
    print("Querying CPU clock frequencies...")
    cpu_clock_info = get_cpu_clock_info()
    if not cpu_clock_info.query_success:
        raise RuntimeError(
            f"Cannot calibrate without CPU clock frequency data.\n"
            f"Query method: {cpu_clock_info.query_method}\n"
            f"Error: {cpu_clock_info.error_message}\n"
            f"Ensure cpufreq sysfs is available or install psutil."
        )

    if not cpu_clock_info.current_freq_mhz:
        raise RuntimeError(
            f"CPU clock query succeeded but no frequency data available.\n"
            f"Query method: {cpu_clock_info.query_method}\n"
            f"Cannot calibrate without knowing the CPU clock frequency."
        )

    cpu_clock_data = CPUClockData(
        current_freq_mhz=cpu_clock_info.current_freq_mhz,  # Required
        query_method=cpu_clock_info.query_method,  # Required
        min_freq_mhz=cpu_clock_info.min_freq_mhz,
        max_freq_mhz=cpu_clock_info.max_freq_mhz,
        base_freq_mhz=cpu_clock_info.base_freq_mhz,
        per_core_freq_mhz=cpu_clock_info.per_core_freq_mhz,
        governor=cpu_clock_info.governor,
        driver=cpu_clock_info.driver,
        turbo_enabled=cpu_clock_info.turbo_enabled,
    )

    print(f"  CPU Freq: {cpu_clock_info.current_freq_mhz:.0f} MHz", end="")
    if cpu_clock_info.max_freq_mhz:
        pct = cpu_clock_info.current_freq_mhz / cpu_clock_info.max_freq_mhz * 100
        print(f" ({pct:.0f}% of max {cpu_clock_info.max_freq_mhz:.0f} MHz)")
    else:
        print()
    if cpu_clock_info.governor:
        print(f"  Governor: {cpu_clock_info.governor}")
    if cpu_clock_info.turbo_enabled is not None:
        print(f"  Turbo:    {'Enabled' if cpu_clock_info.turbo_enabled else 'Disabled'}")
    print()

    # Query GPU clock if using CUDA (required for GPU calibration)
    gpu_clock_data = None
    if device == 'cuda':
        print("Querying GPU clock frequencies...")

        # First get idle clock for reference
        idle_clock_info = get_gpu_clock_info()
        idle_sm_clock = idle_clock_info.sm_clock_mhz if idle_clock_info.query_success else None

        # Run warmup and query clock under load (captures actual operating frequency)
        print("  Running warmup to capture clock under load...")
        gpu_clock_info = get_gpu_clock_under_load(warmup_duration_ms=500, matrix_size=2048)

        if not gpu_clock_info.query_success:
            raise RuntimeError(
                f"Cannot calibrate GPU without clock frequency data.\n"
                f"Query method: {gpu_clock_info.query_method}\n"
                f"Error: {gpu_clock_info.error_message}\n"
                f"Ensure nvidia-smi is available or check GPU driver installation."
            )

        if not gpu_clock_info.sm_clock_mhz:
            raise RuntimeError(
                f"GPU clock query succeeded but no SM clock frequency available.\n"
                f"Query method: {gpu_clock_info.query_method}\n"
                f"Cannot calibrate GPU without knowing the clock frequency."
            )

        gpu_clock_data = GPUClockData(
            sm_clock_mhz=gpu_clock_info.sm_clock_mhz,  # Required - now captured under load
            query_method=gpu_clock_info.query_method,  # Required
            mem_clock_mhz=gpu_clock_info.mem_clock_mhz,
            max_sm_clock_mhz=gpu_clock_info.max_sm_clock_mhz,
            max_mem_clock_mhz=gpu_clock_info.max_mem_clock_mhz,
            power_draw_watts=gpu_clock_info.power_draw_watts,
            power_limit_watts=gpu_clock_info.power_limit_watts,
            temperature_c=gpu_clock_info.temperature_c,
            nvpmodel_mode=gpu_clock_info.nvpmodel_mode,
            power_mode_name=gpu_clock_info.power_mode_name,
        )

        # Show both idle and load clocks for transparency
        if idle_sm_clock and idle_sm_clock != gpu_clock_info.sm_clock_mhz:
            print(f"  SM Clock (idle): {idle_sm_clock} MHz")
            print(f"  SM Clock (load): {gpu_clock_info.sm_clock_mhz} MHz", end="")
        else:
            print(f"  SM Clock: {gpu_clock_info.sm_clock_mhz} MHz", end="")

        if gpu_clock_info.max_sm_clock_mhz:
            pct = gpu_clock_info.sm_clock_mhz / gpu_clock_info.max_sm_clock_mhz * 100
            print(f" ({pct:.0f}% of max {gpu_clock_info.max_sm_clock_mhz} MHz)")
        else:
            print()
        if gpu_clock_info.power_mode_name:
            print(f"  Power Mode: {gpu_clock_info.power_mode_name}")
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
        device_type=device,
        platform_architecture=platform.machine().lower(),
        framework=selected_framework,
        cpu_clock=cpu_clock_data,  # CPU clock data
        gpu_clock=gpu_clock_data,  # GPU clock data for CUDA devices
        preflight=preflight_data,  # Pre-flight check results
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
        operations = ['blas', 'stream']  # Default to BLAS suite + STREAM

    # Run calibrations
    print("Running calibration benchmarks...")
    print()

    # Handle STREAM memory bandwidth benchmarks
    # 'stream' = all 4 kernels, 'stream_copy', 'stream_scale', etc. for individual kernels
    stream_kernels_requested = []
    if 'stream' in operations:
        stream_kernels_requested = ['copy', 'scale', 'add', 'triad']
    else:
        # Check for individual STREAM kernels
        for op in operations:
            if op.startswith('stream_'):
                kernel_name = op[7:]  # Remove 'stream_' prefix
                if kernel_name in ['copy', 'scale', 'add', 'triad']:
                    stream_kernels_requested.append(kernel_name)

    # Backward compatibility: 'memory' means 'stream' (all kernels)
    if 'memory' in operations and not stream_kernels_requested:
        stream_kernels_requested = ['copy', 'scale', 'add', 'triad']

    if stream_kernels_requested:
        print("1. STREAM Memory Bandwidth Benchmark")
        print("-" * 80)
        # Smaller sizes (8-32MB) capture L3 cache effects, larger sizes (64-512MB) measure DRAM bandwidth
        sizes = [8, 16, 32, 64] if quick else [8, 16, 32, 64, 128, 256, 512]

        # Dispatch to framework-specific STREAM benchmark
        if selected_framework == 'numpy':
            from graphs.benchmarks.numpy_benchmarks import calibrate_stream_bandwidth_numpy
            mem_calibrations = calibrate_stream_bandwidth_numpy(
                kernels=stream_kernels_requested,
                sizes_mb=sizes,
                theoretical_bandwidth_gbps=theoretical_bandwidth_gbps,
                num_trials=metadata.num_measurement_runs
            )
        else:  # pytorch
            from graphs.benchmarks.pytorch_benchmarks import calibrate_stream_bandwidth_pytorch
            mem_calibrations = calibrate_stream_bandwidth_pytorch(
                kernels=stream_kernels_requested,
                sizes_mb=sizes,
                theoretical_bandwidth_gbps=theoretical_bandwidth_gbps,
                device=device,
                num_trials=metadata.num_measurement_runs
            )

        for cal in mem_calibrations:
            calibration.add_operation(cal)

        # Update bandwidth measurement (use maximum across all STREAM kernels)
        calibration.measured_bandwidth_gbps = max(
            c.achieved_bandwidth_gbps for c in mem_calibrations
        )
        calibration.bandwidth_efficiency = (
            calibration.measured_bandwidth_gbps / theoretical_bandwidth_gbps
        )

        # Print STREAM score (minimum bandwidth across all kernels)
        stream_score = min(c.achieved_bandwidth_gbps for c in mem_calibrations)
        print(f"STREAM Score (minimum bandwidth): {stream_score:.1f} GB/s")
        print()

    # Handle BLAS compute benchmarks
    # 'blas' = all levels, 'blas1', 'blas2', 'blas3' for individual levels
    blas_ops_requested = []
    if 'blas' in operations:
        blas_ops_requested = ['dot', 'axpy', 'gemv', 'gemm']
    elif 'blas1' in operations:
        blas_ops_requested.extend(['dot', 'axpy'])
    elif 'blas2' in operations:
        blas_ops_requested.append('gemv')
    elif 'blas3' in operations:
        blas_ops_requested.append('gemm')
    # Check for individual BLAS operations
    for op in ['dot', 'axpy', 'gemv', 'gemm']:
        if op in operations and op not in blas_ops_requested:
            blas_ops_requested.append(op)

    if blas_ops_requested:
        print("2. BLAS Compute Benchmark Suite")
        print("-" * 80)

        # Define sizes per BLAS level
        if quick:
            blas_sizes = {
                'dot':  [1000, 10000, 100000, 1000000],
                'axpy': [1000, 10000, 100000, 1000000],
                'gemv': [64, 128, 256, 512, 1024],
                'gemm': [64, 128, 256, 512, 1024],
            }
        else:
            blas_sizes = {
                'dot':  [1000, 10000, 100000, 1000000, 10000000],
                'axpy': [1000, 10000, 100000, 1000000, 10000000],
                'gemv': [32, 64, 128, 256, 512, 1024, 2048],
                'gemm': [32, 64, 128, 256, 512, 1024, 2048],
            }

        # Determine which precisions to test
        # Default precisions based on framework and device
        # NumPy has poor support for fp16/bf16/fp8/fp4 on CPU (falls back to slow emulation)
        # PyTorch has better support, especially on GPU
        if selected_framework == 'numpy':
            # NumPy: Only test native dtypes (fast)
            default_precisions = ['fp64', 'fp32', 'int64', 'int32', 'int16', 'int8']
        else:
            # PyTorch: Can test more precisions (especially on GPU)
            if device == 'cuda':
                # GPU: Test all supported precisions including TF32
                # TF32 uses FP32 dtype but with Tensor Core truncation (19-bit mantissa)
                default_precisions = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'int64', 'int32', 'int16', 'int8']
            else:
                # CPU: Skip poorly-supported fp16/bf16 unless explicitly requested
                default_precisions = ['fp64', 'fp32', 'int64', 'int32', 'int16', 'int8']

        # Merge with theoretical_peaks if provided (allows sparse specs)
        # This ensures we test default precisions even if spec only lists a few
        if theoretical_peaks:
            # Start with default precisions, add any extras from theoretical_peaks
            precisions_from_spec = [
                prec_name for prec_name in theoretical_peaks.keys()
                if prec_name in [p.value for p in Precision]
            ]
            # Use set to merge, then sort by canonical order
            all_precisions = set(default_precisions) | set(precisions_from_spec)
            precisions_to_test = sorted(
                all_precisions,
                key=lambda p: CANONICAL_PRECISION_ORDER.index(p) if p in CANONICAL_PRECISION_ORDER else 999
            )
        else:
            precisions_to_test = default_precisions

        print(f"Testing precisions: {', '.join(precisions_to_test)}")
        print()

        # Dispatch to framework-specific BLAS benchmark
        if selected_framework == 'numpy':
            from graphs.benchmarks.numpy_benchmarks import calibrate_blas_suite_numpy
            blas_calibrations = calibrate_blas_suite_numpy(
                operations=blas_ops_requested,
                sizes=blas_sizes,
                precisions=precisions_to_test,
                theoretical_peak_gflops=theoretical_peak_gflops,
                precision_peaks=theoretical_peaks or {},
                num_trials=metadata.num_measurement_runs,
                min_useful_gflops=min_useful_gflops
            )
        else:  # pytorch
            from graphs.benchmarks.pytorch_benchmarks import calibrate_blas_suite_pytorch
            blas_calibrations = calibrate_blas_suite_pytorch(
                operations=blas_ops_requested,
                sizes=blas_sizes,
                precisions=precisions_to_test,
                theoretical_peak_gflops=theoretical_peak_gflops,
                precision_peaks=theoretical_peaks or {},
                device=device,
                num_trials=metadata.num_measurement_runs,
                min_useful_gflops=min_useful_gflops
            )

        for cal in blas_calibrations:
            calibration.add_operation(cal)

        print()

    # Legacy matmul support (kept for backward compatibility)
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

        # Always start with 32 as a probe to quickly identify unusable precisions
        sizes = [32, 256, 1024, 2048] if quick else [32, 256, 1024, 2048, 4096]

        # Dispatch to framework-specific benchmark
        if selected_framework == 'numpy':
            multi_prec_results = calibrate_matmul_numpy(
                sizes=sizes,
                precisions=precisions_to_test,
                theoretical_peaks=theoretical_peaks or {},
                num_trials=metadata.num_measurement_runs,
                min_useful_throughput=50.0
            )
        else:  # pytorch
            multi_prec_results = calibrate_matmul_pytorch(
                sizes=sizes,
                precisions=precisions_to_test,
                theoretical_peaks=theoretical_peaks or {},
                device=device,
                num_trials=metadata.num_measurement_runs,
                min_useful_throughput=50.0
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
                from graphs.benchmarks.fused_linear_bench import calibrate_linear_fusion_patterns

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
                from graphs.benchmarks.fused_conv_bench import calibrate_conv_fusion_patterns

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
                from graphs.benchmarks.fused_attention_bench import calibrate_attention_fusion_patterns

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

    # Build precision capability matrix based on ACTUAL benchmark results
    print("Building precision capability matrix...")

    # Collect precisions from actual benchmark results
    actually_supported = set()
    actually_unsupported = set()
    peak_gflops_by_precision = {}
    speedup_vs_fp32 = {}

    # Extract precision results from all BLAS operations (not just matmul)
    for op_cal in calibration.operation_profiles.values():
        if op_cal.precision_results:
            for prec_name, prec_result in op_cal.precision_results.items():
                if prec_result.supported and prec_result.measured_gops:
                    actually_supported.add(prec_name)
                    # Track best GOPS for each precision (GFLOPS for float, GIOPS for int)
                    current_best = peak_gflops_by_precision.get(prec_name, 0.0)
                    peak_gflops_by_precision[prec_name] = max(
                        current_best,
                        prec_result.measured_gops
                    )
                    # Track speedup vs FP32
                    if prec_result.speedup_vs_fp32:
                        speedup_vs_fp32[prec_name] = prec_result.speedup_vs_fp32
                elif not prec_result.supported:
                    # Only add to unsupported if not already in supported
                    if prec_name not in actually_supported:
                        actually_unsupported.add(prec_name)

    # Final cleanup: remove any precision from unsupported if it's in supported
    # This handles edge cases where order of operations causes both to be set
    actually_unsupported -= actually_supported

    precision_matrix = PrecisionCapabilityMatrix(
        hardware_name=hardware_name,
        supported_precisions=sorted(actually_supported, key=lambda p: CANONICAL_PRECISION_ORDER.index(p) if p in CANONICAL_PRECISION_ORDER else 999),
        unsupported_precisions=sorted(actually_unsupported, key=lambda p: CANONICAL_PRECISION_ORDER.index(p) if p in CANONICAL_PRECISION_ORDER else 999),
        peak_gflops_by_precision=peak_gflops_by_precision,
        speedup_vs_fp32=speedup_vs_fp32,
        theoretical_peaks=theoretical_peaks or {}
    )

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

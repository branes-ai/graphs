#!/usr/bin/env python3
"""
Hardware Calibration CLI Tool

Runs calibration benchmarks and generates performance profiles for hardware mappers.

Usage:
    # Auto-detect and calibrate current hardware (runs BLAS + STREAM by default)
    ./cli/calibrate_hardware.py

    # Calibrate specific hardware from database
    ./cli/calibrate_hardware.py --id i7_12700k
    ./cli/calibrate_hardware.py --id jetson_orin_nano_gpu

    # Quick calibration (fewer sizes/trials)
    ./cli/calibrate_hardware.py --quick

    # BLAS suite only (all 3 levels: vector-vector, matrix-vector, matrix-matrix)
    ./cli/calibrate_hardware.py --operations blas

    # Individual BLAS levels
    ./cli/calibrate_hardware.py --operations blas1,blas2,blas3

    # Individual BLAS operations
    ./cli/calibrate_hardware.py --operations dot,axpy,gemv,gemm

    # STREAM benchmark only (all 4 kernels)
    ./cli/calibrate_hardware.py --operations stream

    # Individual STREAM kernels
    ./cli/calibrate_hardware.py --operations stream_copy,stream_triad

    # Combined BLAS + STREAM
    ./cli/calibrate_hardware.py --operations blas,stream

    # Load and view existing calibration
    ./cli/calibrate_hardware.py --load profiles/jetson_orin_nano.json

    # Legacy: Use preset (deprecated, will use database)
    ./cli/calibrate_hardware.py --preset i7-12700k
"""

import argparse
import sys
import platform
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.calibration.calibrator import calibrate_hardware, load_calibration
from graphs.hardware.database import HardwareDatabase, HardwareDetector, get_database


# Hardware presets with multi-precision theoretical peaks
# Format: precision_name -> theoretical GFLOPS/GOPS
PRESETS = {
    'i7-12700k': {
        'name': 'Intel-i7-12700K',
        'device': 'cpu',
        'platform': 'x86_64',
        'peak_bandwidth': 75.0,  # DDR5 dual-channel
        'theoretical_peaks': {
            # AVX2 (256-bit): 8 P-cores + 4 E-cores (Golden Cove + Gracemont)
            # P-cores: 2 FMA × 8 FP32/vec × up to 5.0 GHz boost
            # Effective: ~8 cores × 16 FP32/cycle × ~3.0 GHz sustained = ~384 GFLOPS
            # Measured: 747 GFLOPS (likely boost + Turbo Boost Max 3.0)
            'fp64': 360.0,      # 8 FP64/cycle × ~3.0 GHz sustained (GFLOPS)
            'fp32': 720.0,      # 16 FP32/cycle × ~3.0 GHz sustained (GFLOPS)
            # FP16 NOT included - emulated in software via FP32, 800× slower

            # Integer precisions: Theoretical peaks assume VNNI (Vector Neural Network Instructions)
            # WARNING: NumPy/PyTorch do NOT use VNNI for integer matmul!
            # Expected efficiency: 0.2-0.3% (unoptimized generic integer ops)
            # For VNNI performance, use oneDNN, TensorFlow, or PyTorch with MKL-DNN backend
            'int64': 360.0,     # Same as FP64 (GIOPS) - no SIMD/VNNI support for 64-bit
            'int32': 360.0,     # Same as FP64 (GIOPS) - requires VNNI
            'int16': 720.0,     # VNNI DP2A: 2× INT16 throughput (GIOPS)
            'int8': 1440.0,     # VNNI DP4A: 4× INT8 throughput (GIOPS)
        }
    },
    'h100-sxm5': {
        'name': 'NVIDIA-H100-SXM5-80GB',
        'device': 'cuda',
        'platform': 'x86_64',  # Datacenter x86 host
        'peak_bandwidth': 3352.0,  # HBM3
        'theoretical_peaks': {
            'fp64': 34000.0,    # 34 TFLOPS
            'fp32': 67000.0,    # 67 TFLOPS (with sparsity)
            'fp16': 1979000.0,  # 1979 TFLOPS (Tensor Cores)
            'bf16': 1979000.0,  # 1979 TFLOPS (Tensor Cores)
            'fp8_e4m3': 3958000.0,  # 3958 TFLOPS (Tensor Cores, FP8)
            'fp8_e5m2': 3958000.0,  # 3958 TFLOPS (Tensor Cores, FP8)
            'int8': 3958000.0,  # 3958 TOPS (Tensor Cores)
        }
    },
    'jetson-orin-agx': {
        'name': 'NVIDIA-Jetson-Orin-AGX-64GB',
        'device': 'cuda',
        'platform': 'aarch64',
        'peak_bandwidth': 204.8,  # LPDDR5
        'theoretical_peaks': {
            # 2048 CUDA cores @ 1.3 GHz
            # FMA as 2 FLOPS: 2048 × 2 × 1.3 = 5324.8 GFLOPS
            'fp32': 5325.0,     # 2048 CUDA cores × 2 FP32 ops/cycle × 1.3 GHz
            'fp16': 10650.0,    # 2× FP32 (Tensor Cores)
            'int8': 21300.0,    # 4× FP32 (Tensor Cores)
        }
    },
    'jetson-orin-nano': {
        'name': 'NVIDIA-Jetson-Orin-Nano-8GB',
        'device': 'cuda',
        'platform': 'aarch64',
        'peak_bandwidth': 68.0,  # LPDDR5 (64-bit bus)
        'theoretical_peaks': {
            # 1024 CUDA cores @ 625 MHz
            # FMA as 2 FLOPS: 1024 × 2 × 0.625 = 1280 GFLOPS
            'fp32': 1280.0,     # 1024 CUDA cores × 2 FP32 ops/cycle × 625 MHz
            'fp16': 2560.0,     # 2× FP32 (Tensor Cores)
            'int8': 5120.0,     # 4× FP32 (Tensor Cores)
        }
    },
    'ampere-altra-max': {
        'name': 'Ampere-Altra-Max-128',
        'device': 'cpu',
        'platform': 'aarch64',
        'peak_bandwidth': 204.8,  # DDR4-3200 8-channel
        'theoretical_peaks': {
            'fp64': 1024.0,     # 128 cores × 2 NEON lanes × 2 FMA × 2.0 GHz
            'fp32': 2048.0,     # 2× FP64
            'fp16': 4096.0,     # 4× FP64 (NEON FP16)
            'int64': 1024.0,    # Same as FP64
            'int32': 1024.0,    # Same as FP64
            'int16': 2048.0,    # 2× INT32
            'int8': 4096.0,     # 4× INT32
        }
    },
    # Jetson CPU-only presets (for environments without CUDA)
    'jetson-orin-agx-cpu': {
        'name': 'NVIDIA-Jetson-Orin-AGX-CPU',
        'device': 'cpu',
        'platform': 'aarch64',
        'peak_bandwidth': 204.8,  # LPDDR5
        'theoretical_peaks': {
            # Cortex-A78AE: 2× 128-bit NEON units/core, 16 FP32 ops/cycle/core
            # 12 cores (8P+4E all A78AE) @ 2.2 GHz max boost
            'fp64': 211.2,      # 12 cores × 8 FP64/cycle × 2.2 GHz
            'fp32': 422.4,      # 12 cores × 16 FP32/cycle × 2.2 GHz
            'fp16': 844.8,      # 2× FP32 throughput
            'int64': 211.2,     # Same as FP64
            'int32': 211.2,     # Same as FP64
            'int16': 422.4,     # 2× INT32
            'int8': 844.8,      # 4× INT32
        }
    },
    'jetson-orin-agx-gpu': {
        'name': 'NVIDIA-Jetson-Orin-AGX-GPU',
        'device': 'cuda',
        'platform': 'aarch64',
        'peak_bandwidth': 204.8,  # LPDDR5
        'theoretical_peaks': {
            # IMPORTANT: FP16/INT8 use Tensor Cores (specialized matrix multiply units)
            # FP32 uses CUDA cores (general-purpose compute)

            # FP32: CUDA Cores only
            # 2048 CUDA cores @ 1.3 GHz, FMA = 2 FLOPS: 2048 × 2 × 1.3 = 5325 GFLOPS
            'fp32': 5325.0,     # CUDA cores (general purpose)

            # FP16: Tensor Cores for matmul/conv (automatically used by cuBLAS/cuDNN)
            # Ampere Tensor Cores: 16 Tensor Cores, 256 FP16 FMA ops/cycle @ 1.3 GHz
            # Estimated: 16 × 256 × 2 × 1.3 = 10650 GFLOPS theoretical
            # Conservative (based on Nano ratio): 5325 × 6 = ~32000 GFLOPS real
            # Using middle ground until calibrated:
            'fp16': 30000.0,    # Tensor Cores (matrix ops only, ~6× faster than FP32)

            # INT8: Tensor Cores for matmul/conv
            # 2× FP16 throughput with INT8 Tensor Cores
            'int8': 60000.0,    # Tensor Cores INT8 (matrix ops only)
        }
    },
    'jetson-orin-nano-cpu': {
        'name': 'NVIDIA-Jetson-Orin-Nano-CPU',
        'device': 'cpu',
        'platform': 'aarch64',
        'peak_bandwidth': 68.0,  # LPDDR5 (64-bit bus)
        'theoretical_peaks': {
            # Cortex-A78AE: 2× 128-bit NEON units/core, 16 FP32 ops/cycle/core
            # 6 cores @ 1.9 GHz max boost (25W mode)
            # Note: 15W mode = 1.5 GHz, 7W mode = 1.0 GHz
            'fp64': 91.2,       # 6 cores × 8 FP64/cycle × 1.9 GHz
            'fp32': 182.4,      # 6 cores × 16 FP32/cycle × 1.9 GHz
            'fp16': 364.8,      # 2× FP32 throughput
            'int64': 91.2,      # Same as FP64
            'int32': 91.2,      # Same as FP64
            'int16': 182.4,     # 2× INT32
            'int8': 364.8,      # 4× INT32
        }
    },
    'jetson-orin-nano-gpu': {
        'name': 'NVIDIA-Jetson-Orin-Nano-GPU',
        'device': 'cuda',
        'platform': 'aarch64',
        'peak_bandwidth': 68.0,  # LPDDR5 (64-bit bus)
        'theoretical_peaks': {
            # IMPORTANT: FP16/INT8 use Tensor Cores (specialized matrix multiply units)
            # FP32 uses CUDA cores (general-purpose compute)
            # This creates a large performance gap between precisions!

            # FP32: CUDA Cores only
            # 1024 CUDA cores @ 625 MHz, FMA = 2 FLOPS: 1024 × 2 × 0.625 = 1280 GFLOPS
            'fp32': 1280.0,     # CUDA cores (general purpose)

            # FP16: Tensor Cores for matmul/conv (automatically used by cuBLAS/cuDNN)
            # Ampere Tensor Cores: 8 Tensor Cores, 256 FP16 FMA ops/cycle @ 625 MHz
            # Real measured performance: ~7600 GFLOPS for large matmul
            # Conservative estimate based on calibration data:
            'fp16': 7600.0,     # Tensor Cores (matrix ops only, 6× faster than FP32!)

            # INT8: Tensor Cores for matmul/conv
            # 2× FP16 throughput with INT8 Tensor Cores
            'int8': 15200.0,    # Tensor Cores INT8 (matrix ops only)
        }
    },
}


def auto_detect_hardware(db: HardwareDatabase):
    """
    Auto-detect current hardware and match to database.

    Returns:
        tuple: (matched_spec, confidence) or (None, 0.0) if no match
    """
    print()
    print("=" * 80)
    print("Auto-Detecting Hardware")
    print("=" * 80)

    detector = HardwareDetector()
    results = detector.auto_detect(db)

    # Show detected hardware
    if results['cpu']:
        cpu = results['cpu']
        print(f"CPU:      {cpu.model_name}")
        print(f"Vendor:   {cpu.vendor}")
        print(f"Cores:    {cpu.cores} cores, {cpu.threads} threads")

        # Show E-cores if hybrid
        if cpu.e_cores:
            p_cores = cpu.cores - cpu.e_cores
            print(f"          ({p_cores}P + {cpu.e_cores}E cores)")

    if results['gpus']:
        for i, gpu in enumerate(results['gpus']):
            print(f"GPU #{i+1}:  {gpu.model_name}")
            if gpu.memory_gb:
                print(f"Memory:   {gpu.memory_gb} GB")

    print()

    # Check for CPU match
    if results['cpu_matches']:
        best_match = results['cpu_matches'][0]
        conf_pct = best_match.confidence * 100
        spec = best_match.matched_spec

        print(f"✓ Matched to database: {spec.id}")
        print(f"  Confidence: {conf_pct:.0f}%")
        print(f"  Device: {spec.device_type.upper()}")
        print()

        return spec, best_match.confidence

    # Check for GPU match
    if results['gpu_matches'] and results['gpu_matches'][0]:
        best_match = results['gpu_matches'][0][0]
        conf_pct = best_match.confidence * 100
        spec = best_match.matched_spec

        print(f"✓ Matched to database: {spec.id}")
        print(f"  Confidence: {conf_pct:.0f}%")
        print(f"  Device: {spec.device_type.upper()}")
        print()

        return spec, best_match.confidence

    # No match found
    print("✗ No hardware match found in database")
    print()
    print("Options:")
    print("  1. Add your hardware: python scripts/hardware_db/add_hardware.py")
    print("  2. Use specific hardware: --id <hardware_id>")
    print("  3. List available hardware: python scripts/hardware_db/list_hardware.py")
    print()

    return None, 0.0


def detect_platform() -> dict:
    """
    Detect current platform characteristics.

    Returns:
        dict with 'architecture', 'has_cuda', 'cuda_device_name'
    """
    arch = platform.machine().lower()  # 'x86_64', 'aarch64', 'arm64', etc.

    # Normalize architecture names
    if arch in ['aarch64', 'arm64']:
        arch = 'aarch64'
    elif arch in ['x86_64', 'amd64']:
        arch = 'x86_64'

    has_cuda = False
    cuda_device_name = None

    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            cuda_device_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        'architecture': arch,
        'has_cuda': has_cuda,
        'cuda_device_name': cuda_device_name,
    }


def detect_actual_device(requested_device: str) -> dict:
    """
    Detect which device will actually be used for benchmarks.

    This can differ from the requested device if PyTorch/CUDA is not available.

    Args:
        requested_device: Device user requested ('cpu' or 'cuda')

    Returns:
        dict with 'actual_device', 'device_name', 'fallback_occurred', 'fallback_reason'
    """
    if requested_device == 'cpu':
        # CPU always available
        return {
            'actual_device': 'cpu',
            'device_name': 'CPU',
            'fallback_occurred': False,
            'fallback_reason': None
        }

    # Check if CUDA is actually available
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'actual_device': 'cuda',
                'device_name': f'GPU ({torch.cuda.get_device_name(0)})',
                'fallback_occurred': False,
                'fallback_reason': None
            }
        else:
            return {
                'actual_device': 'cpu',
                'device_name': 'CPU (fallback)',
                'fallback_occurred': True,
                'fallback_reason': 'CUDA requested but not available'
            }
    except ImportError:
        return {
            'actual_device': 'cpu',
            'device_name': 'CPU (fallback)',
            'fallback_occurred': True,
            'fallback_reason': 'PyTorch not installed'
        }
    except Exception as e:
        return {
            'actual_device': 'cpu',
            'device_name': 'CPU (fallback)',
            'fallback_occurred': True,
            'fallback_reason': f'Error: {str(e)}'
        }


def validate_preset_platform(preset_name: str, preset_config: dict) -> bool:
    """
    Validate that the preset matches the current platform.

    Args:
        preset_name: Name of the preset (e.g., 'jetson-orin-nano')
        preset_config: Preset configuration dict

    Returns:
        True if valid, False otherwise (prints error message)
    """
    platform_info = detect_platform()

    expected_platform = preset_config['platform']
    expected_device = preset_config['device']

    print("Platform Validation:")
    print(f"  Current architecture: {platform_info['architecture']}")
    print(f"  Expected architecture: {expected_platform}")

    # Check architecture match
    if platform_info['architecture'] != expected_platform:
        print()
        print("=" * 80)
        print("ERROR: Platform Mismatch!")
        print("=" * 80)
        print()
        print(f"Preset '{preset_name}' is designed for {expected_platform} architecture,")
        print(f"but you are running on {platform_info['architecture']}.")
        print()
        print("This will produce incorrect calibration data!")
        print()
        print("Available presets for your platform:")
        for name, config in PRESETS.items():
            if config['platform'] == platform_info['architecture']:
                print(f"  - {name}")
        print()
        return False

    # Check CUDA availability for GPU presets
    if expected_device == 'cuda':
        print(f"  CUDA available: {platform_info['has_cuda']}")
        if platform_info['has_cuda']:
            print(f"  CUDA device: {platform_info['cuda_device_name']}")

        if not platform_info['has_cuda']:
            print()
            print("=" * 80)
            print("ERROR: CUDA Not Available!")
            print("=" * 80)
            print()
            print(f"Preset '{preset_name}' requires CUDA, but no CUDA device was detected.")
            print()
            print("Possible issues:")
            print("  - PyTorch not installed with CUDA support")
            print("  - CUDA drivers not installed")
            print("  - No NVIDIA GPU available")
            print()
            return False

    print(f"  ✓ Platform validation passed")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Hardware Performance Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Hardware specification (auto-detect, database ID, or legacy preset)
    hw_group = parser.add_mutually_exclusive_group(required=False)
    hw_group.add_argument("--id", type=str,
                         help="Hardware ID from database (e.g., i7_12700k, h100_sxm5)")
    hw_group.add_argument("--preset", choices=PRESETS.keys(),
                         help="[DEPRECATED] Legacy preset (use --id or auto-detection instead)")
    hw_group.add_argument("--load", type=str,
                         help="Load and display existing calibration file")

    # Calibration options
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file (default: profiles/<hardware>.json)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick calibration (fewer sizes/trials)")
    parser.add_argument("--operations", type=str, default=None,
                       help="Comma-separated operations to calibrate (default: blas,stream). "
                            "BLAS options: blas (all), blas1, blas2, blas3, dot, axpy, gemv, gemm. "
                            "STREAM options: stream (all 4), stream_copy, stream_scale, stream_add, stream_triad. "
                            "Legacy: matmul, memory")
    parser.add_argument("--min-gflops", type=float, default=1.0,
                       help="Minimum GFLOPS threshold for precision early termination (default: 1.0). "
                            "Precisions achieving less than this will be skipped for larger sizes.")
    parser.add_argument("--framework", type=str, choices=['numpy', 'pytorch'], default=None,
                       help="Override framework selection (default: numpy for CPU, pytorch for GPU)")
    parser.add_argument("--skip-platform-check", action="store_true",
                       help="Skip platform validation (USE WITH CAUTION)")

    args = parser.parse_args()

    # Handle load mode
    if args.load:
        print(f"Loading calibration from: {args.load}\n")
        calibration = load_calibration(Path(args.load))
        calibration.print_summary()
        return 0

    # Load hardware database
    db = get_database()
    db.load_all()

    # Determine which hardware to calibrate
    hardware_spec = None

    if args.id:
        # Use specific hardware from database
        hardware_spec = db.get(args.id)
        if not hardware_spec:
            print(f"✗ Hardware not found in database: {args.id}")
            print()
            print("Available hardware:")
            for hw_id in sorted(db._cache.keys()):
                spec = db._cache[hw_id]
                print(f"  {hw_id:<30} {spec.vendor} {spec.model}")
            print()
            return 1

        print()
        print("=" * 80)
        print(f"Using Hardware from Database: {hardware_spec.id}")
        print("=" * 80)
        print(f"  Vendor:   {hardware_spec.vendor}")
        print(f"  Model:    {hardware_spec.model}")
        print(f"  Type:     {hardware_spec.device_type}")
        print(f"  Platform: {hardware_spec.platform}")
        print()

    elif args.preset:
        # Legacy preset mode (deprecated)
        print()
        print("⚠ WARNING: --preset is deprecated")
        print("  Use --id <hardware_id> or auto-detection instead")
        print("  Falling back to legacy preset mode...")
        print()

        preset = PRESETS[args.preset]

        # Convert preset to HardwareSpec format
        from graphs.hardware.database import HardwareSpec
        from datetime import datetime, timezone

        # Map preset device type
        device_type = 'gpu' if preset['device'] == 'cuda' else preset['device']

        # Create consolidated blocks for preset
        system_info = {
            "vendor": "Unknown",
            "model": preset['name'],
            "architecture": "Unknown",
            "device_type": device_type,
            "platform": preset['platform'],
            "os_compatibility": ["linux", "windows", "macos"],
            "isa_extensions": [],
            "special_features": []
        }

        memory_subsystem = {
            "total_size_gb": 0,
            "peak_bandwidth_gbps": preset['peak_bandwidth'],
            "memory_channels": []
        }

        mapper_info = {
            "mapper_class": "GPUMapper" if device_type == 'gpu' else "CPUMapper",
            "mapper_config": {},
            "hints": {}
        }

        hardware_spec = HardwareSpec(
            id=args.preset.replace('-', '_'),
            system=system_info,
            memory_subsystem=memory_subsystem,
            theoretical_peaks=preset['theoretical_peaks'],
            mapper=mapper_info,
            data_source="preset",
            last_updated=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        )

    else:
        # Auto-detect hardware
        hardware_spec, confidence = auto_detect_hardware(db)

        if not hardware_spec:
            print("Auto-detection failed. Please use --id to specify hardware.")
            return 1

        if confidence < 0.5:
            print(f"⚠ WARNING: Low confidence match ({confidence*100:.0f}%)")
            print("  Consider using --id to explicitly specify hardware")
            print()
            response = input("Continue with this hardware? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("Calibration cancelled.")
                return 1

    # Extract calibration parameters from hardware spec
    hardware_name = hardware_spec.model
    device = 'cuda' if hardware_spec.device_type == 'gpu' else hardware_spec.device_type
    theoretical_peaks = hardware_spec.theoretical_peaks
    peak_bandwidth = hardware_spec.peak_bandwidth_gbps

    # Use FP32 as the default peak for backward compatibility
    peak_gflops = theoretical_peaks.get('fp32', max(theoretical_peaks.values()) if theoretical_peaks else 100.0)

    # Platform validation (only for preset mode)
    if args.preset and not args.skip_platform_check:
        if not validate_preset_platform(args.preset, PRESETS[args.preset]):
            return 1
    elif args.skip_platform_check:
        print("WARNING: Platform validation skipped. Results may be incorrect!")
        print()

    # Parse operations
    operations = None
    if args.operations:
        operations = [op.strip() for op in args.operations.split(',')]

    # Detect actual device that will be used (may differ from requested if fallback occurs)
    device_info = detect_actual_device(device)

    # Determine framework (need this for filename)
    from graphs.hardware.calibration.calibrator import select_framework
    try:
        selected_framework = select_framework(device, args.framework)
    except RuntimeError as e:
        print()
        print("=" * 80)
        print("ERROR: Framework Selection Failed")
        print("=" * 80)
        print(f"  {e}")
        print()
        return 1

    # Determine output path (include framework to avoid overwriting)
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: profiles/<hardware_id>_<framework>.json
        # This prevents overwriting when running with different frameworks
        profiles_dir = Path(__file__).parent.parent / "src" / "graphs" / "hardware" / "calibration" / "profiles"
        safe_id = hardware_spec.id.lower().replace(" ", "_").replace("-", "_")
        output_path = profiles_dir / f"{safe_id}_{selected_framework}.json"

    # Show device information prominently
    print()
    print("=" * 80)
    print("EXECUTION DEVICE")
    print("=" * 80)
    print(f"  Requested device: {device.upper()}")
    print(f"  Actual device:    {device_info['device_name']}")
    print(f"  Framework:        {selected_framework.upper()}")

    if device_info['fallback_occurred']:
        print()
        print("  ⚠ WARNING: Device Fallback Occurred!")
        print(f"  Reason: {device_info['fallback_reason']}")
        print()
        print("  This will produce INCORRECT calibration data for the requested hardware!")
        print("  The calibration will reflect CPU performance, not GPU performance.")
        print()

        # Ask user if they want to continue
        response = input("  Continue anyway? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("  Calibration cancelled.")
            return 1
    print()

    # Run calibration
    try:
        calibration = calibrate_hardware(
            hardware_name=hardware_name,
            theoretical_peak_gflops=peak_gflops,
            theoretical_bandwidth_gbps=peak_bandwidth,
            theoretical_peaks=theoretical_peaks,  # NEW: per-precision peaks
            device=device,  # NEW: specify device type
            actual_device_info=device_info,  # NEW: actual device being used
            framework=args.framework,  # NEW: framework override
            output_path=output_path,
            operations=operations,
            quick=args.quick,
            min_useful_gflops=args.min_gflops  # NEW: configurable early termination threshold
        )

        print()
        print("=" * 80)
        print("Calibration Complete!")
        print("=" * 80)
        print()
        print(f"Calibration file: {output_path}")
        print()
        print("Next steps:")
        print("  1. Review the calibration results above")
        print("  2. Use this calibration in your analysis:")
        print(f"     ./cli/analyze_comprehensive.py --model resnet18 \\")
        print(f"         --hardware {hardware_spec.id} \\")
        print(f"         --calibration {output_path}")
        print()
        print("  3. Or export calibration to database:")
        print(f"     python scripts/hardware_db/update_hardware.py --id {hardware_spec.id} \\")
        print(f"         --field calibration_file --value {output_path}")
        print()

        return 0

    except Exception as e:
        print(f"\nError during calibration: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
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
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.calibration.calibrator import calibrate_hardware, load_calibration
from graphs.hardware.database import HardwareDatabase, HardwareDetector, get_database


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

    # Show board detection if available
    board_match = results.get('board_match')
    if results.get('board'):
        board = results['board']
        print(f"Board:    {board.model}")
        if board.soc:
            print(f"SoC:      {board.soc}")

    print()

    # Check for CPU match
    if results['cpu_matches']:
        best_match = results['cpu_matches'][0]
        conf_pct = best_match.confidence * 100
        spec = best_match.matched_spec

        via_board = ""
        if board_match and "via board:" in best_match.detected_string:
            via_board = f" (via board: {board_match.board_id})"

        print(f"✓ Matched to database: {spec.id}{via_board}")
        print(f"  Confidence: {conf_pct:.0f}%")
        print(f"  Device: {spec.device_type.upper()}")
        print()

        return spec, best_match.confidence

    # Check for GPU match
    if results['gpu_matches'] and results['gpu_matches'][0]:
        best_match = results['gpu_matches'][0][0]
        conf_pct = best_match.confidence * 100
        spec = best_match.matched_spec

        via_board = ""
        if board_match and "via board:" in best_match.detected_string:
            via_board = f" (via board: {board_match.board_id})"

        print(f"✓ Matched to database: {spec.id}{via_board}")
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


def detect_platform() -> dict:
    """
    Detect comprehensive platform and software stack information.

    Returns:
        dict with keys:
            - os: Operating system name (Linux, Windows, Darwin/macOS)
            - os_version: OS version string
            - kernel: Kernel version (Linux) or build (Windows/macOS)
            - architecture: CPU architecture (x86_64, aarch64, arm64)
            - hostname: Machine hostname
            - python_version: Python version
            - python_implementation: CPython, PyPy, etc.
            - numpy_version: NumPy version if installed
            - pytorch_version: PyTorch version if installed
            - cuda_version: CUDA version if available
            - cudnn_version: cuDNN version if available
            - rocm_version: ROCm version if available (AMD)
            - oneapi_version: oneAPI version if available (Intel)
            - openblas_version: OpenBLAS version if detected
            - mkl_version: Intel MKL version if detected
    """
    import os as os_module
    import platform

    info = {}

    # Operating system
    info['os'] = platform.system()
    info['os_version'] = platform.version()
    info['os_release'] = platform.release()
    info['architecture'] = platform.machine()
    info['hostname'] = platform.node()

    # For Linux, get distribution info
    if info['os'] == 'Linux':
        try:
            # Try to read /etc/os-release for distro info
            if Path('/etc/os-release').exists():
                with open('/etc/os-release') as f:
                    os_release = {}
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            os_release[key] = value.strip('"')
                    info['distro_name'] = os_release.get('NAME', 'Unknown')
                    info['distro_version'] = os_release.get('VERSION_ID', 'Unknown')
                    info['distro_id'] = os_release.get('ID', 'unknown')
        except Exception:
            pass

        # Get kernel version
        try:
            info['kernel'] = platform.release()
        except Exception:
            pass

    # Python information
    info['python_version'] = platform.python_version()
    info['python_implementation'] = platform.python_implementation()
    info['python_compiler'] = platform.python_compiler()

    # NumPy
    try:
        import numpy as np
        info['numpy_version'] = np.__version__

        # Try to detect BLAS backend
        try:
            blas_info = np.show_config(mode='dicts')
            if isinstance(blas_info, dict):
                blas_libs = blas_info.get('Build Dependencies', {}).get('blas', {})
                if blas_libs:
                    info['numpy_blas'] = blas_libs.get('name', 'unknown')
        except Exception:
            # Fallback: try to detect from numpy config string
            try:
                import io
                old_stdout = sys.stdout
                sys.stdout = mystdout = io.StringIO()
                np.show_config()
                config_str = mystdout.getvalue()
                sys.stdout = old_stdout

                if 'openblas' in config_str.lower():
                    info['numpy_blas'] = 'openblas'
                elif 'mkl' in config_str.lower():
                    info['numpy_blas'] = 'mkl'
                elif 'blis' in config_str.lower():
                    info['numpy_blas'] = 'blis'
            except Exception:
                pass
    except ImportError:
        info['numpy_version'] = None

    # PyTorch and CUDA
    try:
        import torch
        info['pytorch_version'] = torch.__version__

        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
            info['cuda_device_count'] = torch.cuda.device_count()
        else:
            info['cuda_available'] = False

        # Check for ROCm (AMD)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            info['rocm_version'] = torch.version.hip
            info['rocm_available'] = True
        else:
            info['rocm_available'] = False

        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['mps_available'] = True
        else:
            info['mps_available'] = False

    except ImportError:
        info['pytorch_version'] = None
        info['cuda_available'] = False

    # Intel oneAPI / MKL detection
    try:
        mkl_path = os_module.environ.get('MKLROOT')
        if mkl_path:
            info['mkl_root'] = mkl_path
        oneapi_root = os_module.environ.get('ONEAPI_ROOT')
        if oneapi_root:
            info['oneapi_root'] = oneapi_root
    except Exception:
        pass

    # OpenBLAS detection (check environment)
    try:
        openblas_num_threads = os_module.environ.get('OPENBLAS_NUM_THREADS')
        if openblas_num_threads:
            info['openblas_num_threads'] = openblas_num_threads
    except Exception:
        pass

    # Environment variables that affect performance
    info['omp_num_threads'] = os_module.environ.get('OMP_NUM_THREADS')
    info['mkl_num_threads'] = os_module.environ.get('MKL_NUM_THREADS')

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Hardware Performance Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Hardware specification (auto-detect or database ID)
    hw_group = parser.add_mutually_exclusive_group(required=False)
    hw_group.add_argument("--id", type=str,
                         help="Hardware ID from database (e.g., i7_12700k, h100_sxm5)")
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

#!/usr/bin/env python3
"""
Hardware Calibration CLI Tool

Runs calibration benchmarks and generates performance profiles for hardware mappers.

Usage:
    # Calibrate i7-12700K
    ./cli/calibrate_hardware.py --preset i7-12700k

    # Calibrate Jetson Orin Nano
    ./cli/calibrate_hardware.py --preset jetson-orin-nano

    # Quick calibration (fewer tests)
    ./cli/calibrate_hardware.py --preset i7-12700k --quick

    # Specific operations only
    ./cli/calibrate_hardware.py --preset i7-12700k --operations matmul

    # Load and view existing calibration
    ./cli/calibrate_hardware.py --load profiles/jetson_orin_nano.json
"""

import argparse
import sys
import platform
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.calibration.calibrator import calibrate_hardware, load_calibration


# Hardware presets with multi-precision theoretical peaks
# Format: precision_name -> theoretical GFLOPS/GOPS
PRESETS = {
    'i7-12700k': {
        'name': 'Intel-i7-12700K',
        'device': 'cpu',
        'platform': 'x86_64',
        'peak_bandwidth': 75.0,  # DDR5 dual-channel
        'theoretical_peaks': {
            'fp64': 360.0,      # 10 cores × 2 AVX2 lanes × 2 FMA × 2.0 GHz (P-cores)
            'fp32': 720.0,      # Double FP64 (2× throughput)
            'fp16': 720.0,      # Emulated, same as FP32
            'int32': 360.0,     # Same as FP64
            'int16': 720.0,     # VNNI 2× throughput
            'int8': 1440.0,     # VNNI 4× throughput
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
            'fp32': 5300.0,     # 64 Tensor Cores × 2 SM × 2.0 GHz (Ampere)
            'fp16': 10600.0,    # 2× FP32 (Tensor Cores)
            'int8': 21200.0,    # 4× FP32 (Tensor Cores)
        }
    },
    'jetson-orin-nano': {
        'name': 'NVIDIA-Jetson-Orin-Nano-8GB',
        'device': 'cuda',
        'platform': 'aarch64',
        'peak_bandwidth': 68.0,  # LPDDR5 (64-bit bus)
        'theoretical_peaks': {
            'fp32': 1000.0,     # 32 Tensor Cores (Ampere) @ 625 MHz
            'fp16': 2000.0,     # 2× FP32 (Tensor Cores)
            'int8': 4000.0,     # 4× FP32 (Tensor Cores)
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
            'int32': 1024.0,    # Same as FP64
            'int16': 2048.0,    # 2× INT32
            'int8': 4096.0,     # 4× INT32
        }
    },
}


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

    # Hardware specification (presets only)
    hw_group = parser.add_mutually_exclusive_group(required=True)
    hw_group.add_argument("--preset", choices=PRESETS.keys(),
                         help="Hardware preset to calibrate")
    hw_group.add_argument("--load", type=str,
                         help="Load and display existing calibration file")

    # Calibration options
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file (default: profiles/<hardware>.json)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick calibration (fewer sizes/trials)")
    parser.add_argument("--operations", type=str, default=None,
                       help="Comma-separated operations to calibrate (default: all)")
    parser.add_argument("--skip-platform-check", action="store_true",
                       help="Skip platform validation (USE WITH CAUTION)")

    args = parser.parse_args()

    # Handle load mode
    if args.load:
        print(f"Loading calibration from: {args.load}\n")
        calibration = load_calibration(Path(args.load))
        calibration.print_summary()
        return 0

    # Get preset configuration
    preset = PRESETS[args.preset]
    hardware_name = preset['name']
    device = preset['device']
    theoretical_peaks = preset['theoretical_peaks']
    peak_bandwidth = preset['peak_bandwidth']

    # Use FP32 as the default peak for backward compatibility
    peak_gflops = theoretical_peaks.get('fp32', max(theoretical_peaks.values()))

    # Platform validation
    if not args.skip_platform_check:
        if not validate_preset_platform(args.preset, preset):
            return 1
    else:
        print("WARNING: Platform validation skipped. Results may be incorrect!")
        print()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: profiles/<hardware_name>.json
        profiles_dir = Path(__file__).parent.parent / "src" / "graphs" / "hardware" / "calibration" / "profiles"
        safe_name = hardware_name.lower().replace(" ", "_").replace("-", "_")
        output_path = profiles_dir / f"{safe_name}.json"

    # Parse operations
    operations = None
    if args.operations:
        operations = [op.strip() for op in args.operations.split(',')]

    # Run calibration
    try:
        calibration = calibrate_hardware(
            hardware_name=hardware_name,
            theoretical_peak_gflops=peak_gflops,
            theoretical_bandwidth_gbps=peak_bandwidth,
            theoretical_peaks=theoretical_peaks,  # NEW: per-precision peaks
            device=device,  # NEW: specify device type
            output_path=output_path,
            operations=operations,
            quick=args.quick
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
        print(f"         --hardware {args.preset} \\")
        print(f"         --calibration {output_path}")
        print()

        return 0

    except Exception as e:
        print(f"\nError during calibration: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

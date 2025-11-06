#!/usr/bin/env python3
"""
Hardware Calibration CLI Tool

Runs calibration benchmarks and generates performance profiles for hardware mappers.

Usage:
    # Calibrate i7-12700K
    ./cli/calibrate_hardware.py --preset i7-12700k

    # Custom hardware
    ./cli/calibrate_hardware.py --name "My CPU" --peak-gflops 500 --peak-bandwidth 50

    # Quick calibration (fewer tests)
    ./cli/calibrate_hardware.py --preset i7-12700k --quick

    # Specific operations only
    ./cli/calibrate_hardware.py --preset i7-12700k --operations matmul

    # Load and view existing calibration
    ./cli/calibrate_hardware.py --load profiles/i7_12700k.json
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.calibration.calibrator import calibrate_hardware, load_calibration


# Hardware presets
PRESETS = {
    'i7-12700k': {
        'name': 'Intel-i7-12700K',
        'peak_gflops': 1000.0,  # All cores + HT realistic peak
        'peak_bandwidth': 75.0,  # DDR5 dual-channel
    },
    'h100': {
        'name': 'NVIDIA-H100-80GB',
        'peak_gflops': 60000.0,  # FP32 with sparsity
        'peak_bandwidth': 3352.0,  # HBM3
    },
    'jetson-orin': {
        'name': 'NVIDIA-Jetson-AGX-Orin',
        'peak_gflops': 5300.0,  # FP32
        'peak_bandwidth': 204.8,  # LPDDR5
    },
    'ampere-altra': {
        'name': 'Ampere-Altra-Max-128',
        'peak_gflops': 2048.0,  # 128 cores × 2 NEON × 4 floats × 2 GHz
        'peak_bandwidth': 204.8,  # DDR4-3200 8-channel
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Hardware Performance Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Hardware specification
    hw_group = parser.add_mutually_exclusive_group(required=True)
    hw_group.add_argument("--preset", choices=PRESETS.keys(),
                         help="Use predefined hardware preset")
    hw_group.add_argument("--name", type=str,
                         help="Custom hardware name")
    hw_group.add_argument("--load", type=str,
                         help="Load and display existing calibration file")

    # Custom hardware parameters (required if --name is used)
    parser.add_argument("--peak-gflops", type=float,
                       help="Theoretical peak GFLOPS (required with --name)")
    parser.add_argument("--peak-bandwidth", type=float,
                       help="Theoretical peak bandwidth GB/s (required with --name)")

    # Calibration options
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file (default: profiles/<hardware>.json)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick calibration (fewer sizes/trials)")
    parser.add_argument("--operations", type=str, default=None,
                       help="Comma-separated operations to calibrate (default: all)")

    args = parser.parse_args()

    # Handle load mode
    if args.load:
        print(f"Loading calibration from: {args.load}\n")
        calibration = load_calibration(Path(args.load))
        calibration.print_summary()
        return 0

    # Determine hardware parameters
    if args.preset:
        preset = PRESETS[args.preset]
        hardware_name = preset['name']
        peak_gflops = preset['peak_gflops']
        peak_bandwidth = preset['peak_bandwidth']
    else:
        # Custom hardware
        if not args.peak_gflops or not args.peak_bandwidth:
            parser.error("--name requires --peak-gflops and --peak-bandwidth")

        hardware_name = args.name
        peak_gflops = args.peak_gflops
        peak_bandwidth = args.peak_bandwidth

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
        print(f"         --hardware {args.preset or 'custom'} \\")
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

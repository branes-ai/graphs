#!/usr/bin/env python3
"""
DLA Calibration CLI

Benchmarks NVIDIA Deep Learning Accelerator (DLA) cores via TensorRT.
Runs synthetic single-layer benchmarks and/or reference model benchmarks,
saving timestamped JSON results to the calibration profiles directory.

Requires: TensorRT, PyCUDA, PyTorch (for ONNX export), torchvision

Usage:
  ./cli/calibrate_dla.py                    # All benchmarks, all cores, FP16+INT8
  ./cli/calibrate_dla.py --core 0           # Benchmark DLA core 0 only
  ./cli/calibrate_dla.py --precision fp16   # FP16 only (default: both)
  ./cli/calibrate_dla.py --synthetic-only   # Skip reference models
  ./cli/calibrate_dla.py --models-only      # Skip synthetic layers
  ./cli/calibrate_dla.py --no-fallback      # Disable GPU fallback (strict DLA)
  ./cli/calibrate_dla.py --iterations 200   # More measurement iterations
"""

import argparse
import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark NVIDIA DLA cores via TensorRT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Full calibration (all cores, FP16+INT8)
  %(prog)s --core 0 --precision fp16  # Core 0, FP16 only
  %(prog)s --synthetic-only           # Only synthetic layers
  %(prog)s --models-only              # Only reference models
  %(prog)s --no-fallback              # Strict DLA (no GPU fallback)
        """,
    )

    parser.add_argument(
        '--core', type=int, default=None,
        help='DLA core to benchmark (0 or 1). Default: all cores.',
    )
    parser.add_argument(
        '--precision', choices=['fp16', 'int8', 'both'], default='both',
        help='Precision to benchmark. Default: both.',
    )
    parser.add_argument(
        '--synthetic-only', action='store_true',
        help='Only run synthetic single-layer benchmarks.',
    )
    parser.add_argument(
        '--models-only', action='store_true',
        help='Only run reference model benchmarks.',
    )
    parser.add_argument(
        '--no-fallback', action='store_true',
        help='Disable GPU fallback (strict DLA mode).',
    )
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Inference batch size. Default: 1.',
    )
    parser.add_argument(
        '--warmup', type=int, default=10,
        help='Warmup iterations. Default: 10.',
    )
    parser.add_argument(
        '--iterations', type=int, default=100,
        help='Measurement iterations. Default: 100.',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results (auto-detected if not set).',
    )

    args = parser.parse_args()

    # Validate mutual exclusivity
    if args.synthetic_only and args.models_only:
        parser.error("--synthetic-only and --models-only are mutually exclusive")

    # Check dependencies
    try:
        from graphs.benchmarks.tensorrt_benchmarks.trt_utils import (
            check_trt_available, get_dla_core_count,
        )
        check_trt_available()
    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        print("Required: tensorrt, pycuda, torch, torchvision", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    from graphs.calibration.dla_calibrator import calibrate_dla, calibrate_all_dla_cores
    from pathlib import Path

    num_cores = get_dla_core_count()
    if num_cores == 0:
        print("Error: No DLA cores found on this platform.", file=sys.stderr)
        sys.exit(1)

    print(f"Detected {num_cores} DLA core(s)")

    run_synthetic = not args.models_only
    run_models = not args.synthetic_only
    gpu_fallback = not args.no_fallback
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Determine precisions to test
    if args.precision == 'both':
        precisions = ['fp16', 'int8']
    else:
        precisions = [args.precision]

    # Determine cores to test
    if args.core is not None:
        if args.core >= num_cores:
            print(f"Error: DLA core {args.core} not available (have {num_cores})",
                  file=sys.stderr)
            sys.exit(1)
        cores = [args.core]
    else:
        cores = list(range(num_cores))

    # Run calibration
    all_results = []
    for precision in precisions:
        for core in cores:
            print(f"\n{'#' * 70}")
            print(f"# DLA Core {core}, Precision: {precision.upper()}")
            print(f"{'#' * 70}")

            try:
                result = calibrate_dla(
                    dla_core=core,
                    precision=precision,
                    gpu_fallback=gpu_fallback,
                    run_synthetic=run_synthetic,
                    run_models=run_models,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    output_dir=output_dir,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\nError calibrating DLA core {core} ({precision}): {e}",
                      file=sys.stderr)

    # Print final summary
    if all_results:
        print(f"\n{'=' * 70}")
        print("  DLA Calibration Summary")
        print(f"{'=' * 70}")
        for result in all_results:
            result.print_summary()
    else:
        print("\nNo successful calibrations.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

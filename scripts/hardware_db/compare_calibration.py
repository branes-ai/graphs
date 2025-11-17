#!/usr/bin/env python3
"""
Compare Calibration Results vs Theoretical Performance

Compare measured calibration results against theoretical hardware specs from database.

Usage:
    python scripts/hardware_db/compare_calibration.py --calibration profiles/i7_12700k_numpy.json
    python scripts/hardware_db/compare_calibration.py --calibration profiles/h100_sxm5_pytorch.json --verbose
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareDatabase, get_database
from graphs.hardware.calibration.calibrator import load_calibration


def compare_theoretical_vs_calibrated(hw_spec, calibration, verbose=False):
    """
    Compare theoretical specs vs calibrated results.

    Args:
        hw_spec: HardwareSpec from database
        calibration: HardwareCalibration from file
        verbose: Show detailed comparison
    """
    print()
    print("=" * 80)
    print("Theoretical vs Calibrated Performance Comparison")
    print("=" * 80)
    print()

    print(f"Hardware: {hw_spec.model} ({hw_spec.id})")
    print(f"Vendor:   {hw_spec.vendor}")
    print(f"Type:     {hw_spec.device_type.upper()}")
    print()

    # Compare memory bandwidth
    print("Memory Bandwidth")
    print("-" * 80)
    theoretical_bw = hw_spec.peak_bandwidth_gbps
    measured_bw = calibration.measured_bandwidth_gbps if hasattr(calibration, 'measured_bandwidth_gbps') else None

    print(f"Theoretical: {theoretical_bw:.1f} GB/s")
    if measured_bw:
        efficiency = (measured_bw / theoretical_bw) * 100 if theoretical_bw > 0 else 0.0
        print(f"Measured:    {measured_bw:.1f} GB/s ({efficiency:.1f}% efficiency)")
    else:
        print(f"Measured:    Not available (run STREAM benchmark)")
    print()

    # Compare theoretical peaks by precision
    print("Compute Performance by Precision")
    print("-" * 80)
    print(f"{'Precision':<12} {'Theoretical':>15} {'Measured':>15} {'Efficiency':>12}")
    print("-" * 80)

    # Collect all precisions from both theoretical and measured
    all_precisions = set()
    if hw_spec.theoretical_peaks:
        all_precisions.update(hw_spec.theoretical_peaks.keys())

    # Get measured peaks from operation profiles
    measured_peaks = {}
    if hasattr(calibration, 'operation_profiles'):
        for op_name, op_profile in calibration.operation_profiles.items():
            if hasattr(op_profile, 'precision_results'):
                for prec_name, prec_result in op_profile.precision_results.items():
                    if prec_result.supported and prec_result.measured_gops:
                        # Keep the best performance across all operations
                        current_best = measured_peaks.get(prec_name, 0.0)
                        measured_peaks[prec_name] = max(current_best, prec_result.measured_gops)
                        all_precisions.add(prec_name)

    # Canonical precision order
    precision_order = ['fp64', 'fp32', 'fp16', 'fp8', 'fp4', 'bf16', 'int64', 'int32', 'int16', 'int8', 'int4']

    for prec in precision_order:
        if prec not in all_precisions:
            continue

        theoretical = hw_spec.theoretical_peaks.get(prec, 0.0) if hw_spec.theoretical_peaks else 0.0
        measured = measured_peaks.get(prec, 0.0)

        unit = "GIOPS" if prec.startswith('int') else "GFLOPS"

        theo_str = f"{theoretical:.1f} {unit}" if theoretical > 0 else "N/A"
        meas_str = f"{measured:.1f} {unit}" if measured > 0 else "N/A"

        if theoretical > 0 and measured > 0:
            efficiency = (measured / theoretical) * 100
            eff_str = f"{efficiency:.1f}%"
        else:
            eff_str = "N/A"

        print(f"{prec:<12} {theo_str:>15} {meas_str:>15} {eff_str:>12}")

    print()

    # Detailed operation breakdown (if verbose)
    if verbose and hasattr(calibration, 'operation_profiles'):
        print("Detailed Operation Breakdown")
        print("-" * 80)

        for op_name, op_profile in calibration.operation_profiles.items():
            if not hasattr(op_profile, 'precision_results'):
                continue

            print(f"\n{op_name.upper()}:")

            for prec_name in precision_order:
                if prec_name not in op_profile.precision_results:
                    continue

                prec_result = op_profile.precision_results[prec_name]
                if not prec_result.supported:
                    continue

                theoretical = hw_spec.theoretical_peaks.get(prec_name, 0.0) if hw_spec.theoretical_peaks else 0.0
                measured = prec_result.measured_gops

                unit = "GIOPS" if prec_name.startswith('int') else "GFLOPS"

                if theoretical > 0 and measured > 0:
                    efficiency = (measured / theoretical) * 100
                    print(f"  {prec_name:<10} {measured:>8.1f} {unit:<6} ({efficiency:>5.1f}% of theoretical)")
                else:
                    print(f"  {prec_name:<10} {measured:>8.1f} {unit}")


def identify_hardware_from_calibration(db: HardwareDatabase, calibration):
    """
    Try to identify which hardware spec matches the calibration.

    Args:
        db: HardwareDatabase
        calibration: HardwareCalibration

    Returns:
        HardwareSpec or None
    """
    # Try to match by name
    calibration_name = calibration.metadata.hardware_name

    # Search database for matching name
    for hw_id, spec in db._cache.items():
        if spec.model.lower() in calibration_name.lower() or calibration_name.lower() in spec.model.lower():
            return spec

        # Try matching by ID
        if hw_id.lower().replace('_', '-') in calibration_name.lower().replace('_', '-'):
            return spec

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare calibration results vs theoretical performance"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Path to calibration JSON file"
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Hardware ID from database (auto-detect if not specified)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed operation breakdown"
    )

    args = parser.parse_args()

    # Load calibration
    print(f"Loading calibration from: {args.calibration}")
    try:
        calibration = load_calibration(args.calibration)
    except Exception as e:
        print(f"✗ Error loading calibration: {e}")
        return 1

    # Load database
    db = HardwareDatabase(args.db)
    db.load_all()

    # Get hardware spec
    if args.id:
        hw_spec = db.get(args.id)
        if not hw_spec:
            print(f"✗ Hardware not found: {args.id}")
            return 1
    else:
        # Try to auto-detect from calibration name
        hw_spec = identify_hardware_from_calibration(db, calibration)

        if not hw_spec:
            print()
            print("⚠ Could not auto-identify hardware from calibration")
            print(f"  Calibration name: {calibration.metadata.hardware_name}")
            print()
            print("Please specify hardware with --id:")
            for hw_id in sorted(db._cache.keys()):
                spec = db._cache[hw_id]
                print(f"  {hw_id:<30} {spec.vendor} {spec.model}")
            return 1

        print(f"Auto-identified hardware: {hw_spec.id}")

    # Compare
    compare_theoretical_vs_calibrated(hw_spec, calibration, verbose=args.verbose)

    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    # Calculate overall efficiency from best precision results
    if hw_spec.theoretical_peaks and hasattr(calibration, 'operation_profiles'):
        measured_peaks = {}
        for op_profile in calibration.operation_profiles.values():
            if hasattr(op_profile, 'precision_results'):
                for prec_name, prec_result in op_profile.precision_results.items():
                    if prec_result.supported and prec_result.measured_gops:
                        current_best = measured_peaks.get(prec_name, 0.0)
                        measured_peaks[prec_name] = max(current_best, prec_result.measured_gops)

        # Get FP32 efficiency as reference
        theo_fp32 = hw_spec.theoretical_peaks.get('fp32', 0.0)
        meas_fp32 = measured_peaks.get('fp32', 0.0)

        if theo_fp32 > 0 and meas_fp32 > 0:
            efficiency = (meas_fp32 / theo_fp32) * 100
            print(f"FP32 Efficiency: {efficiency:.1f}%")

            if efficiency >= 80:
                print("  ✓ Excellent performance (≥80% of theoretical)")
            elif efficiency >= 50:
                print("  ✓ Good performance (≥50% of theoretical)")
            elif efficiency >= 20:
                print("  ⚠ Moderate performance (≥20% of theoretical)")
            else:
                print("  ⚠ Low performance (<20% of theoretical)")
                print("    Consider:")
                print("    - Using optimized BLAS library (MKL, OpenBLAS)")
                print("    - Enabling compiler optimizations")
                print("    - Checking thermal throttling")

    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Show Calibration Efficiency

Display calibration measurements as percentage of theoretical peak performance.

Usage:
    ./cli/show_calibration_efficiency.py --id jetson_orin_nano_gpu
    ./cli/show_calibration_efficiency.py --id jetson_orin_nano_gpu --power-mode 25W
    ./cli/show_calibration_efficiency.py --all
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.registry import get_registry


def format_gflops(val: float, is_int: bool = False) -> str:
    """Format GFLOPS/GOPS value with appropriate precision."""
    suffix = "GOPS" if is_int else "GFLOPS"
    tera_suffix = "TOPS" if is_int else "TFLOPS"

    if val >= 1000:
        return f"{val/1000:.2f} {tera_suffix}"
    elif val >= 100:
        return f"{val:.0f} {suffix}"
    elif val >= 10:
        return f"{val:.1f} {suffix}"
    else:
        return f"{val:.2f} {suffix}"


def format_efficiency(measured: float, theoretical: float) -> str:
    """Format efficiency as percentage with color indicator."""
    if theoretical <= 0:
        return "N/A"

    pct = (measured / theoretical) * 100

    if pct > 110:
        return f"{pct:6.1f}% ⚠"  # Above theoretical - need to check specs
    elif pct >= 80:
        return f"{pct:6.1f}% ✓"  # Excellent
    elif pct >= 50:
        return f"{pct:6.1f}%"    # Good
    elif pct >= 20:
        return f"{pct:6.1f}% ↓"  # Below optimal
    else:
        return f"{pct:6.1f}% ↓↓" # Very low


def show_calibration_efficiency(profile, calibration, power_mode: str = None):
    """Display efficiency report for a calibration."""
    print(f"\n{'='*80}")
    print(f"Hardware: {profile.model}")
    print(f"Profile:  {profile.id}")
    if power_mode:
        print(f"Power Mode: {power_mode}")
    if calibration.metadata.gpu_clock:
        gc = calibration.metadata.gpu_clock
        print(f"GPU Clock: {gc.sm_clock_mhz} MHz")
        if gc.power_mode_name:
            print(f"Power Mode: {gc.power_mode_name}")
    print(f"{'='*80}\n")

    # Get theoretical peaks from profile
    theoretical = profile.theoretical_peaks

    # Show precision matrix results
    if calibration.precision_matrix:
        pm = calibration.precision_matrix

        print("BLAS Performance (GEMM - Best Achieved):")
        print("-" * 80)
        print(f"{'Precision':<10} {'Measured':>15} {'Theoretical':>15} {'Efficiency':>12} {'Status'}")
        print("-" * 80)

        # Get per-precision peaks from calibration
        measured_peaks = pm.peak_gflops_by_precision

        for prec in ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'int8', 'int16', 'int32']:
            theo = theoretical.get(prec, 0)
            meas = measured_peaks.get(prec, 0)
            is_int = prec.startswith('int')

            if theo > 0 or meas > 0:
                theo_str = format_gflops(theo, is_int) if theo > 0 else "N/A"
                meas_str = format_gflops(meas, is_int) if meas > 0 else "N/A"
                eff_str = format_efficiency(meas, theo) if theo > 0 and meas > 0 else "N/A"

                # Determine status
                if meas == 0:
                    status = "Not measured"
                elif theo == 0:
                    status = "No spec"
                elif meas / theo > 1.1:
                    status = "Check spec!"
                elif meas / theo >= 0.8:
                    status = "Excellent"
                elif meas / theo >= 0.5:
                    status = "Good"
                elif meas / theo >= 0.2:
                    status = "Suboptimal"
                else:
                    status = "Very low"

                print(f"{prec:<10} {meas_str:>15} {theo_str:>15} {eff_str:>12} {status}")

        print()

    # Show memory bandwidth
    print("Memory Bandwidth (STREAM):")
    print("-" * 80)
    theo_bw = profile.peak_bandwidth_gbps
    meas_bw = calibration.measured_bandwidth_gbps
    bw_eff = calibration.bandwidth_efficiency

    print(f"{'Measured:':<20} {meas_bw:.1f} GB/s")
    print(f"{'Theoretical:':<20} {theo_bw:.1f} GB/s")
    print(f"{'Efficiency:':<20} {bw_eff*100:.1f}%")
    print()

    # Show summary
    print("Summary:")
    print("-" * 80)
    print(f"{'Best Compute Efficiency:':<30} {calibration.best_efficiency*100:.1f}%")
    print(f"{'Average Compute Efficiency:':<30} {calibration.avg_efficiency*100:.1f}%")
    print(f"{'Worst Compute Efficiency:':<30} {calibration.worst_efficiency*100:.1f}%")
    print(f"{'Memory Bandwidth Efficiency:':<30} {bw_eff*100:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Show calibration efficiency (percentage of theoretical peak)"
    )
    parser.add_argument(
        "--id",
        help="Hardware ID (e.g., jetson_orin_nano_gpu)"
    )
    parser.add_argument(
        "--power-mode", "-p",
        help="Filter by power mode (e.g., 25W, MAXN_SUPER)"
    )
    parser.add_argument(
        "--framework", "-f",
        help="Filter by framework (numpy, pytorch)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Show all calibrations"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available hardware profiles"
    )

    args = parser.parse_args()

    # Load registry
    registry = get_registry()
    count = registry.load_all()

    if args.list:
        print(f"Available hardware profiles ({count} loaded):\n")
        for hw_id in sorted(registry.list_all()):
            profile = registry.get(hw_id)
            cals = registry.list_calibrations(hw_id)
            cal_str = f"({len(cals)} calibrations)" if cals else "(no calibrations)"
            print(f"  {hw_id:<40} {cal_str}")
        return 0

    if not args.id and not args.all:
        parser.print_help()
        print("\nUse --list to see available hardware profiles")
        print("Use --id <id> to show efficiency for a specific profile")
        print("Use --all to show all calibrations")
        return 1

    # Get hardware IDs to process
    if args.all:
        hardware_ids = registry.list_all()
    else:
        hardware_ids = [args.id]

    found_any = False

    for hw_id in hardware_ids:
        profile = registry.get(hw_id)
        if not profile:
            if not args.all:
                print(f"Error: Hardware ID '{hw_id}' not found in registry")
                return 1
            continue

        # Get calibrations
        calibrations = registry.list_calibrations(hw_id)
        if not calibrations:
            if not args.all:
                print(f"No calibrations found for {hw_id}")
            continue

        for cal_info in calibrations:
            # Apply filters
            if args.power_mode and cal_info['power_mode'].upper() != args.power_mode.upper():
                continue
            if args.framework and cal_info['framework'].lower() != args.framework.lower():
                continue

            # Load full calibration
            cal_filter = {
                'power_mode': cal_info['power_mode'],
                'freq_mhz': cal_info['freq_mhz'],
                'framework': cal_info['framework'],
            }
            full_profile = registry.get(hw_id, calibration_filter=cal_filter)

            if full_profile and full_profile.calibration:
                found_any = True
                show_calibration_efficiency(
                    full_profile,
                    full_profile.calibration,
                    power_mode=cal_info['power_mode']
                )

    if not found_any:
        print("No matching calibrations found.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

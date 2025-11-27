#!/usr/bin/env python3
"""
List Calibrations

Display calibrations in the hardware registry with optional filtering.

Usage:
    python scripts/hardware_db/list_calibrations.py
    python scripts/hardware_db/list_calibrations.py --hardware jetson_orin_nano_cpu
    python scripts/hardware_db/list_calibrations.py --framework numpy
    python scripts/hardware_db/list_calibrations.py --power-mode MAXN
    python scripts/hardware_db/list_calibrations.py --detail
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.registry import get_registry
from graphs.hardware.registry.profile import HardwareProfile


def format_date(iso_date: str) -> str:
    """Format ISO date string to human-readable format."""
    try:
        dt = datetime.fromisoformat(iso_date)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso_date or "Unknown"


def main():
    parser = argparse.ArgumentParser(
        description="List calibrations in the hardware registry"
    )
    parser.add_argument(
        "--hardware", "-H",
        help="Filter by hardware ID (e.g., jetson_orin_nano_cpu)"
    )
    parser.add_argument(
        "--framework", "-f",
        help="Filter by framework (numpy, pytorch)"
    )
    parser.add_argument(
        "--power-mode", "-p",
        help="Filter by power mode (e.g., MAXN, 7W, performance)"
    )
    parser.add_argument(
        "--detail", "-d",
        action="store_true",
        help="Show detailed calibration information"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show summary statistics"
    )

    args = parser.parse_args()

    # Load registry
    registry = get_registry()
    count = registry.load_all()
    print(f"Loaded {count} hardware profiles from registry")
    print()

    # Collect all calibrations
    all_calibrations = []

    hardware_ids = [args.hardware] if args.hardware else registry.list_all()

    for hw_id in hardware_ids:
        profile = registry.get(hw_id)
        if not profile:
            if args.hardware:
                print(f"Error: Hardware '{hw_id}' not found in registry")
                return 1
            continue

        # Get calibrations for this profile
        calibrations = registry.list_calibrations(hw_id)

        for cal_info in calibrations:
            # Apply filters
            if args.framework and cal_info['framework'].lower() != args.framework.lower():
                continue
            if args.power_mode and cal_info['power_mode'].lower() != args.power_mode.lower():
                continue

            cal_info['hardware_id'] = hw_id
            cal_info['device_type'] = profile.device_type
            cal_info['model'] = profile.model

            # Load full calibration for detail view
            if args.detail:
                cal_filter = {
                    'power_mode': cal_info['power_mode'],
                    'freq_mhz': cal_info['freq_mhz'],
                    'framework': cal_info['framework'],
                }
                full_profile = registry.get(hw_id, calibration_filter=cal_filter)
                if full_profile and full_profile.calibration:
                    cal = full_profile.calibration
                    cal_info['date'] = cal.metadata.calibration_date
                    cal_info['best_gflops'] = cal.best_measured_gflops
                    cal_info['bandwidth_gbps'] = cal.measured_bandwidth_gbps
                    cal_info['governor'] = cal.metadata.cpu_clock.governor if cal.metadata.cpu_clock else None
                    cal_info['gpu_power_mode'] = cal.metadata.gpu_clock.power_mode_name if cal.metadata.gpu_clock else None

            all_calibrations.append(cal_info)

    if not all_calibrations:
        print("No calibrations found matching criteria.")
        return 0

    # Show summary if requested
    if args.summary:
        print("=" * 70)
        print("Calibration Summary")
        print("=" * 70)
        print(f"Total calibrations: {len(all_calibrations)}")
        print()

        # By hardware
        by_hardware = {}
        for cal in all_calibrations:
            hw_id = cal['hardware_id']
            by_hardware[hw_id] = by_hardware.get(hw_id, 0) + 1

        print("By Hardware:")
        for hw_id, cnt in sorted(by_hardware.items()):
            print(f"  {hw_id:<40} {cnt:>3}")
        print()

        # By framework
        by_framework = {}
        for cal in all_calibrations:
            fw = cal['framework']
            by_framework[fw] = by_framework.get(fw, 0) + 1

        print("By Framework:")
        for fw, cnt in sorted(by_framework.items()):
            print(f"  {fw:<15} {cnt:>3}")
        print()

        # By power mode
        by_power = {}
        for cal in all_calibrations:
            pm = cal['power_mode']
            by_power[pm] = by_power.get(pm, 0) + 1

        print("By Power Mode:")
        for pm, cnt in sorted(by_power.items()):
            print(f"  {pm:<15} {cnt:>3}")
        print()

        return 0

    # Display calibrations
    print("=" * 70)
    print(f"Found {len(all_calibrations)} calibration(s)")
    print("=" * 70)
    print()

    # Group by hardware
    current_hw = None
    for cal in sorted(all_calibrations, key=lambda c: (c['hardware_id'], c['power_mode'], c['freq_mhz'])):
        if cal['hardware_id'] != current_hw:
            current_hw = cal['hardware_id']
            print(f"{current_hw} ({cal['model']})")
            print("-" * 70)

        # Format calibration name
        cal_name = f"{cal['power_mode']}_{cal['freq_mhz']}MHz_{cal['framework']}"

        if args.detail:
            print(f"  {cal_name}")
            if cal.get('date'):
                print(f"    Date:      {format_date(cal['date'])}")
            if cal.get('best_gflops'):
                print(f"    Best:      {cal['best_gflops']:.1f} GFLOPS")
            if cal.get('bandwidth_gbps'):
                print(f"    Bandwidth: {cal['bandwidth_gbps']:.1f} GB/s")
            if cal.get('governor'):
                print(f"    Governor:  {cal['governor']}")
            if cal.get('gpu_power_mode'):
                print(f"    GPU Mode:  {cal['gpu_power_mode']}")
            print()
        else:
            # Compact view
            freq_str = f"{cal['freq_mhz']} MHz"
            print(f"  - {cal['power_mode']:<15} {freq_str:<12} {cal['framework']}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

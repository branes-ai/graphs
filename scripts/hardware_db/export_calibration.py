#!/usr/bin/env python3
"""
Export Calibration Results to Hardware Database

Extracts summary metrics from calibration profiles and updates the hardware_database
JSON files with calibration_summary and calibration_profiles fields.

This bridges the gap between:
- Theoretical specs in hardware_database/ (from datasheets)
- Empirical measurements in src/graphs/hardware/calibration/profiles/

Usage:
    # Export a single calibration profile to hardware database
    python scripts/hardware_db/export_calibration.py \
        --calibration src/graphs/hardware/calibration/profiles/nvidia/jetson_orin_nano/25W/nvidia_jetson_orin_nano_gpu_pytorch.json \
        --hardware-id jetson_orin_nano_gpu \
        --power-mode 25W

    # Auto-detect hardware ID from calibration filename
    python scripts/hardware_db/export_calibration.py \
        --calibration src/graphs/hardware/calibration/profiles/nvidia/jetson_orin_nano/25W/nvidia_jetson_orin_nano_gpu_pytorch.json \
        --power-mode 25W

    # Export all Jetson power modes
    python scripts/hardware_db/export_calibration.py \
        --calibration-dir src/graphs/hardware/calibration/profiles/nvidia/jetson_orin_nano \
        --hardware-id jetson_orin_nano_gpu

    # Dry run (show what would be updated without writing)
    python scripts/hardware_db/export_calibration.py \
        --calibration <path> --hardware-id <id> --dry-run
"""

import sys
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareDatabase
from graphs.hardware.database.schema import CalibrationSummary, CalibrationProfiles


def load_calibration_profile(profile_path: Path) -> Dict:
    """Load a calibration profile JSON file."""
    with open(profile_path) as f:
        return json.load(f)


def extract_gpu_clock_data(profile: Dict) -> Optional[Dict]:
    """
    Extract GPU clock data from a calibration profile.

    Args:
        profile: Loaded calibration profile dict

    Returns:
        Dict with GPU clock data or None if not available
    """
    metadata = profile.get('metadata', {})
    gpu_clock = metadata.get('gpu_clock')

    if not gpu_clock:
        return None

    # Extract relevant fields
    result = {}
    if gpu_clock.get('sm_clock_mhz'):
        result['sm_clock_mhz'] = gpu_clock['sm_clock_mhz']
    if gpu_clock.get('mem_clock_mhz'):
        result['mem_clock_mhz'] = gpu_clock['mem_clock_mhz']
    if gpu_clock.get('max_sm_clock_mhz'):
        result['max_sm_clock_mhz'] = gpu_clock['max_sm_clock_mhz']
    if gpu_clock.get('power_draw_watts'):
        result['power_draw_watts'] = gpu_clock['power_draw_watts']
    if gpu_clock.get('power_limit_watts'):
        result['power_limit_watts'] = gpu_clock['power_limit_watts']
    if gpu_clock.get('temperature_c'):
        result['temperature_c'] = gpu_clock['temperature_c']
    if gpu_clock.get('power_mode_name'):
        result['power_mode_name'] = gpu_clock['power_mode_name']
    if gpu_clock.get('nvpmodel_mode') is not None:
        result['nvpmodel_mode'] = gpu_clock['nvpmodel_mode']
    if gpu_clock.get('query_method'):
        result['query_method'] = gpu_clock['query_method']

    return result if result else None


def extract_calibration_summary(profile: Dict, power_mode: Optional[str] = None) -> CalibrationSummary:
    """
    Extract CalibrationSummary from a calibration profile.

    Args:
        profile: Loaded calibration profile dict
        power_mode: Optional power mode string (e.g., '25W')

    Returns:
        CalibrationSummary with aggregated metrics
    """
    metadata = profile.get('metadata', {})

    # Extract measured peaks from precision_matrix if available
    measured_peaks = {}
    efficiency = {}

    precision_matrix = profile.get('precision_matrix', {})
    if precision_matrix:
        peak_by_prec = precision_matrix.get('peak_gflops_by_precision', {})
        theoretical = precision_matrix.get('theoretical_peaks', {})

        for prec, measured in peak_by_prec.items():
            if measured and measured > 0:
                measured_peaks[prec] = measured
                if prec in theoretical and theoretical[prec] > 0:
                    efficiency[prec] = measured / theoretical[prec]

    # Fallback to top-level metrics if precision_matrix is sparse
    if not measured_peaks:
        if profile.get('best_measured_gflops'):
            measured_peaks['fp32'] = profile['best_measured_gflops']

    # Get bandwidth metrics
    measured_bandwidth = profile.get('measured_bandwidth_gbps')
    bandwidth_eff = profile.get('bandwidth_efficiency')

    # Get aggregate efficiency metrics
    best_eff = profile.get('best_efficiency')
    avg_eff = profile.get('avg_efficiency')
    worst_eff = profile.get('worst_efficiency')

    # Extract framework and date
    framework = metadata.get('framework', 'unknown')
    calibration_date = metadata.get('calibration_date')

    # Extract GPU clock data
    gpu_clock = extract_gpu_clock_data(profile)

    # Extract measured clock from GPU clock data
    measured_clock_mhz = None
    if gpu_clock and gpu_clock.get('sm_clock_mhz'):
        measured_clock_mhz = gpu_clock['sm_clock_mhz']

    return CalibrationSummary(
        measured_peaks=measured_peaks,
        efficiency=efficiency,
        measured_bandwidth_gbps=measured_bandwidth,
        bandwidth_efficiency=bandwidth_eff,
        best_efficiency=best_eff,
        avg_efficiency=avg_eff,
        worst_efficiency=worst_eff,
        calibration_date=calibration_date,
        power_mode=power_mode,
        framework=framework,
        profile_path=None,  # Will be set by caller
        measured_clock_mhz=measured_clock_mhz,
        gpu_clock=gpu_clock,
    )


def infer_hardware_id(profile_path: Path, profile: Dict) -> Optional[str]:
    """
    Try to infer hardware ID from calibration profile path or content.

    Args:
        profile_path: Path to calibration profile
        profile: Loaded profile dict

    Returns:
        Inferred hardware ID or None
    """
    # Common patterns to try
    path_str = str(profile_path).lower()

    # Jetson patterns
    if 'jetson_orin_nano' in path_str or 'jetson-orin-nano' in path_str:
        if 'gpu' in path_str:
            return 'jetson_orin_nano_gpu'
        elif 'cpu' in path_str:
            return 'jetson_orin_nano_cpu'
    elif 'jetson_orin_agx' in path_str or 'jetson-orin-agx' in path_str:
        if 'gpu' in path_str:
            return 'jetson_orin_agx_gpu'
        elif 'cpu' in path_str:
            return 'jetson_orin_agx_cpu'

    # Intel patterns
    if 'i7_12700k' in path_str or 'i7-12700k' in path_str:
        return 'intel_12th_gen_intelr_coretm_i7_12700k'

    # Try to extract from metadata
    metadata = profile.get('metadata', {})
    hw_name = metadata.get('hardware_name', '')

    if 'Jetson-Orin-Nano' in hw_name:
        if 'GPU' in hw_name:
            return 'jetson_orin_nano_gpu'
        elif 'CPU' in hw_name:
            return 'jetson_orin_nano_cpu'

    return None


def infer_power_mode(profile_path: Path) -> Optional[str]:
    """
    Try to infer power mode from calibration profile path.

    Args:
        profile_path: Path to calibration profile

    Returns:
        Inferred power mode (e.g., '25W') or None
    """
    path_str = str(profile_path)

    # Look for common power mode patterns
    # Pattern: .../25W/... or .../7W/... etc.
    match = re.search(r'/(\d+W)/', path_str)
    if match:
        return match.group(1)

    return None


def get_relative_profile_path(profile_path: Path, calibration_root: Path) -> str:
    """
    Get relative path from calibration root for storage in database.

    Args:
        profile_path: Absolute path to profile
        calibration_root: Root of calibration profiles directory

    Returns:
        Relative path string
    """
    try:
        return str(profile_path.relative_to(calibration_root))
    except ValueError:
        # If not under calibration_root, return full path
        return str(profile_path)


def find_calibration_profiles(
    calibration_dir: Path,
    hardware_id: str
) -> List[Tuple[Path, str]]:
    """
    Find all calibration profiles for a hardware ID in a directory.

    Args:
        calibration_dir: Directory to search
        hardware_id: Hardware ID to match

    Returns:
        List of (profile_path, power_mode) tuples
    """
    profiles = []

    for json_file in calibration_dir.rglob('*.json'):
        # Load and check if it matches
        try:
            profile = load_calibration_profile(json_file)
            inferred_id = infer_hardware_id(json_file, profile)

            if inferred_id == hardware_id:
                power_mode = infer_power_mode(json_file)
                profiles.append((json_file, power_mode))
        except (json.JSONDecodeError, KeyError):
            continue

    return profiles


def update_hardware_database(
    db: HardwareDatabase,
    hardware_id: str,
    calibration_summary: CalibrationSummary,
    profile_path: str,
    power_mode: Optional[str],
    dry_run: bool = False
) -> bool:
    """
    Update hardware database entry with calibration data.

    Args:
        db: HardwareDatabase instance
        hardware_id: Hardware ID to update
        calibration_summary: CalibrationSummary to add
        profile_path: Relative path to profile
        power_mode: Power mode string
        dry_run: If True, don't write changes

    Returns:
        True if successful
    """
    spec = db.get(hardware_id)
    if not spec:
        print(f"ERROR: Hardware ID '{hardware_id}' not found in database")
        return False

    # Update calibration_summary
    calibration_summary.profile_path = profile_path
    spec.calibration_summary = calibration_summary.to_dict()

    # Update calibration_profiles
    if not spec.calibration_profiles:
        spec.calibration_profiles = {}

    profiles = spec.calibration_profiles
    profiles['default'] = profile_path

    if power_mode:
        if 'by_power_mode' not in profiles:
            profiles['by_power_mode'] = {}
        profiles['by_power_mode'][power_mode] = profile_path

    # Extract framework from summary
    framework = calibration_summary.framework
    if framework and framework != 'unknown':
        if 'by_framework' not in profiles:
            profiles['by_framework'] = {}
        profiles['by_framework'][framework] = profile_path

    if dry_run:
        print("\n[DRY RUN] Would update hardware database with:")
        print(f"  Hardware ID: {hardware_id}")
        print(f"  calibration_summary: {json.dumps(spec.calibration_summary, indent=2)}")
        print(f"  calibration_profiles: {json.dumps(spec.calibration_profiles, indent=2)}")
        return True

    # Write to database
    try:
        db.add(spec, overwrite=True)
        print(f"Updated hardware database: {hardware_id}")
        return True
    except Exception as e:
        print(f"ERROR writing to database: {e}")
        return False


def print_calibration_summary(summary: CalibrationSummary, profile_path: str):
    """Print a human-readable summary of calibration data."""
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)

    print(f"\nProfile: {profile_path}")
    if summary.calibration_date:
        print(f"Date: {summary.calibration_date}")
    if summary.power_mode:
        print(f"Power Mode: {summary.power_mode}")
    if summary.framework != 'unknown':
        print(f"Framework: {summary.framework}")

    # Display GPU clock information (Phase 1 addition)
    if summary.gpu_clock:
        print("\nGPU Clock Info:")
        print("-" * 40)
        gc = summary.gpu_clock
        if gc.get('sm_clock_mhz'):
            clock_str = f"  SM Clock:     {gc['sm_clock_mhz']} MHz"
            if gc.get('max_sm_clock_mhz'):
                pct = gc['sm_clock_mhz'] / gc['max_sm_clock_mhz'] * 100
                clock_str += f" / {gc['max_sm_clock_mhz']} MHz ({pct:.0f}%)"
            print(clock_str)
        if gc.get('mem_clock_mhz'):
            print(f"  Memory Clock: {gc['mem_clock_mhz']} MHz")
        if gc.get('power_draw_watts'):
            power_str = f"  Power:        {gc['power_draw_watts']:.1f} W"
            if gc.get('power_limit_watts'):
                power_str += f" / {gc['power_limit_watts']:.1f} W limit"
            print(power_str)
        if gc.get('temperature_c'):
            print(f"  Temperature:  {gc['temperature_c']}°C")
        if gc.get('power_mode_name'):
            mode_str = f"  Power Mode:   {gc['power_mode_name']}"
            if gc.get('nvpmodel_mode') is not None:
                mode_str += f" (nvpmodel {gc['nvpmodel_mode']})"
            print(mode_str)
        if gc.get('query_method'):
            print(f"  Query Method: {gc['query_method']}")
    elif summary.measured_clock_mhz:
        print(f"\nGPU SM Clock: {summary.measured_clock_mhz} MHz")

    if summary.measured_peaks:
        print("\nMeasured Peak Performance:")
        print("-" * 40)
        for prec, gflops in sorted(summary.measured_peaks.items()):
            eff = summary.efficiency.get(prec)
            eff_str = f"({eff*100:.1f}% eff)" if eff else ""
            print(f"  {prec:<8} {gflops:>10.1f} GFLOPS {eff_str}")

    if summary.measured_bandwidth_gbps:
        bw_eff = summary.bandwidth_efficiency
        bw_eff_str = f"({bw_eff*100:.1f}% eff)" if bw_eff else ""
        print(f"\nMemory Bandwidth: {summary.measured_bandwidth_gbps:.1f} GB/s {bw_eff_str}")

    if summary.best_efficiency:
        print(f"\nAggregate Efficiency:")
        print(f"  Best:  {summary.best_efficiency*100:.1f}%")
        if summary.avg_efficiency:
            print(f"  Avg:   {summary.avg_efficiency*100:.1f}%")
        if summary.worst_efficiency:
            print(f"  Worst: {summary.worst_efficiency*100:.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Export calibration results to hardware database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--calibration",
        type=Path,
        help="Path to calibration profile JSON"
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="Directory containing calibration profiles (exports all matching profiles)"
    )
    parser.add_argument(
        "--hardware-id",
        help="Hardware ID in database (auto-detected if not specified)"
    )
    parser.add_argument(
        "--power-mode",
        help="Power mode (e.g., '25W', '7W') - auto-detected from path if not specified"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path(__file__).parent.parent.parent / "src" / "graphs" / "hardware" / "calibration" / "profiles",
        help="Root directory for calibration profiles (for relative paths)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.calibration and not args.calibration_dir:
        parser.error("Either --calibration or --calibration-dir is required")

    # Load database
    db = HardwareDatabase(args.db)
    print(f"Loading database from: {args.db}")
    db.load_all()
    print(f"Loaded {len(db._cache)} hardware specifications")

    # Process single calibration file
    if args.calibration:
        if not args.calibration.exists():
            print(f"ERROR: Calibration file not found: {args.calibration}")
            return 1

        # Load profile
        profile = load_calibration_profile(args.calibration)

        # Infer hardware ID if not specified
        hardware_id = args.hardware_id
        if not hardware_id:
            hardware_id = infer_hardware_id(args.calibration, profile)
            if not hardware_id:
                print("ERROR: Could not infer hardware ID. Please specify with --hardware-id")
                return 1
            print(f"Auto-detected hardware ID: {hardware_id}")

        # Infer power mode if not specified
        power_mode = args.power_mode
        if not power_mode:
            power_mode = infer_power_mode(args.calibration)
            if power_mode:
                print(f"Auto-detected power mode: {power_mode}")

        # Extract summary
        summary = extract_calibration_summary(profile, power_mode)

        # Get relative path
        profile_path = get_relative_profile_path(args.calibration, args.calibration_root)

        # Print summary
        if args.verbose:
            print_calibration_summary(summary, profile_path)

        # Update database
        success = update_hardware_database(
            db, hardware_id, summary, profile_path, power_mode, args.dry_run
        )

        return 0 if success else 1

    # Process directory of calibration files
    if args.calibration_dir:
        if not args.calibration_dir.exists():
            print(f"ERROR: Calibration directory not found: {args.calibration_dir}")
            return 1

        if not args.hardware_id:
            print("ERROR: --hardware-id is required when using --calibration-dir")
            return 1

        # Find all matching profiles
        profiles = find_calibration_profiles(args.calibration_dir, args.hardware_id)

        if not profiles:
            print(f"No calibration profiles found for hardware ID: {args.hardware_id}")
            return 1

        print(f"Found {len(profiles)} calibration profile(s)")

        # Process each profile
        all_profiles = {}
        default_summary = None
        default_profile_path = None

        for profile_path, power_mode in sorted(profiles, key=lambda x: x[1] or ''):
            profile = load_calibration_profile(profile_path)
            summary = extract_calibration_summary(profile, power_mode)
            rel_path = get_relative_profile_path(profile_path, args.calibration_root)

            if args.verbose:
                print_calibration_summary(summary, rel_path)

            # Track for calibration_profiles
            if power_mode:
                all_profiles[power_mode] = rel_path

            # Use highest power mode as default (or last one)
            default_summary = summary
            default_profile_path = rel_path

        # Update database with the default (highest power mode) summary
        if default_summary:
            # Build combined calibration_profiles
            spec = db.get(args.hardware_id)
            if spec:
                if not spec.calibration_profiles:
                    spec.calibration_profiles = {}

                spec.calibration_profiles['default'] = default_profile_path
                if all_profiles:
                    spec.calibration_profiles['by_power_mode'] = all_profiles

                default_summary.profile_path = default_profile_path
                spec.calibration_summary = default_summary.to_dict()

                if args.dry_run:
                    print("\n[DRY RUN] Would update hardware database with:")
                    print(f"  Hardware ID: {args.hardware_id}")
                    print(f"  calibration_summary: {json.dumps(spec.calibration_summary, indent=2)}")
                    print(f"  calibration_profiles: {json.dumps(spec.calibration_profiles, indent=2)}")
                else:
                    db.add(spec, overwrite=True)
                    print(f"\nUpdated hardware database: {args.hardware_id}")
                    print(f"  Profiles indexed: {len(all_profiles)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

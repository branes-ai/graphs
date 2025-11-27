#!/usr/bin/env python3
"""
Migrate hardware_database to new unified registry structure.

This script converts the old structure:
    hardware_database/
    ├── cpu/intel_i7_12700k.json
    ├── gpu/nvidia_h100.json
    └── boards/nvidia/jetson_orin_nano.json

To the new structure:
    hardware_registry/
    ├── cpu/i7_12700k/
    │   ├── spec.json
    │   └── calibration.json (if exists)
    ├── gpu/h100_sxm5/
    │   ├── spec.json
    │   └── calibration.json
    └── boards/jetson_orin_nano/
        ├── spec.json
        └── calibration.json

Usage:
    python scripts/hardware_db/migrate_to_registry.py
    python scripts/hardware_db/migrate_to_registry.py --dry-run
    python scripts/hardware_db/migrate_to_registry.py --output /path/to/registry
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def find_calibration_file(hardware_id: str, calibration_dir: Path) -> Path | None:
    """Find calibration file for a hardware ID."""
    # Try various naming patterns
    patterns = [
        f"{hardware_id}_*.json",
        f"{hardware_id}.json",
        f"*{hardware_id}*.json",
    ]

    for pattern in patterns:
        matches = list(calibration_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def convert_spec(old_spec: dict, device_type: str) -> dict:
    """Convert old hardware spec format to new profile format."""
    # Extract ID from old format
    old_id = old_spec.get('id', '')

    # Handle nested 'system' structure (new format from hardware_database)
    system = old_spec.get('system', {})
    core_info = old_spec.get('core_info', {})
    memory_subsystem = old_spec.get('memory_subsystem', {})

    # Extract vendor - try multiple locations
    vendor = (
        old_spec.get('vendor') or
        system.get('vendor') or
        'Unknown'
    )

    # Extract model - try multiple locations
    model = (
        old_spec.get('model') or
        system.get('model') or
        old_id
    )

    # Extract device type
    actual_device_type = (
        old_spec.get('device_type') or
        system.get('device_type') or
        device_type
    )

    # Extract architecture
    architecture = (
        old_spec.get('architecture') or
        system.get('architecture')
    )

    # Extract platform
    platform = (
        old_spec.get('platform') or
        system.get('platform')
    )

    # Extract compute units (cores for CPU, SMs for GPU)
    compute_units = (
        old_spec.get('compute_units') or
        old_spec.get('cores') or
        core_info.get('cores') or
        old_spec.get('sms')
    )

    # Extract memory
    memory_gb = (
        old_spec.get('memory_gb') or
        memory_subsystem.get('total_size_gb')
    )

    # Extract bandwidth
    peak_bandwidth = (
        old_spec.get('peak_bandwidth_gbps') or
        memory_subsystem.get('peak_bandwidth_gbps') or
        0.0
    )

    # Extract clock frequencies
    base_clock_mhz = None
    boost_clock_mhz = None
    if core_info.get('base_frequency_ghz'):
        base_clock_mhz = core_info['base_frequency_ghz'] * 1000
    if core_info.get('boost_frequency_ghz'):
        boost_clock_mhz = core_info['boost_frequency_ghz'] * 1000

    # Extract TDP
    tdp_watts = old_spec.get('tdp_watts') or old_spec.get('tdp')

    # Extract notes
    notes = old_spec.get('notes') or system.get('notes')

    # Build new spec
    new_spec = {
        'id': old_id,
        'vendor': vendor,
        'model': model,
        'device_type': actual_device_type,
        'theoretical_peaks': old_spec.get('theoretical_peaks', {}),
        'peak_bandwidth_gbps': peak_bandwidth,
    }

    # Add optional fields if present
    if architecture:
        new_spec['architecture'] = architecture
    if compute_units:
        new_spec['compute_units'] = compute_units
    if memory_gb:
        new_spec['memory_gb'] = memory_gb
    if tdp_watts:
        new_spec['tdp_watts'] = tdp_watts
    if base_clock_mhz:
        new_spec['base_clock_mhz'] = base_clock_mhz
    if boost_clock_mhz:
        new_spec['boost_clock_mhz'] = boost_clock_mhz
    if platform:
        new_spec['platform'] = platform
    if notes:
        new_spec['notes'] = notes

    # Copy tags if present
    if 'tags' in old_spec:
        new_spec['tags'] = old_spec['tags']

    return new_spec


def migrate_hardware_database(
    old_db_path: Path,
    new_registry_path: Path,
    calibration_path: Path | None,
    dry_run: bool = False
) -> tuple[int, int]:
    """
    Migrate hardware database to new registry structure.

    Returns:
        Tuple of (specs_migrated, calibrations_migrated)
    """
    specs_migrated = 0
    calibrations_migrated = 0

    # Process each device type directory
    device_types = ['cpu', 'gpu', 'boards']

    for device_type in device_types:
        old_device_dir = old_db_path / device_type
        if not old_device_dir.exists():
            continue

        # Find all JSON files (may be nested for boards)
        json_files = list(old_device_dir.rglob('*.json'))

        for json_file in json_files:
            # Skip schema files and other non-hardware files
            if json_file.name in ['schema.json', 'index.json']:
                continue

            print(f"Processing: {json_file}")

            try:
                with open(json_file) as f:
                    old_spec = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  ERROR: Invalid JSON: {e}")
                continue

            # Skip if not a hardware spec (check for required fields)
            if 'id' not in old_spec and 'model' not in old_spec:
                print(f"  SKIP: Not a hardware spec (no id or model)")
                continue

            # Get hardware ID
            hw_id = old_spec.get('id', json_file.stem)

            # Determine actual device type from spec or path
            actual_device_type = old_spec.get('device_type', device_type)
            if actual_device_type == 'board':
                actual_device_type = 'boards'

            # Convert spec to new format
            new_spec = convert_spec(old_spec, actual_device_type.rstrip('s'))  # 'boards' -> 'board'

            # Create output directory
            output_dir = new_registry_path / actual_device_type / hw_id

            if dry_run:
                print(f"  Would create: {output_dir}")
                print(f"    spec.json: {new_spec['model']}")
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                spec_path = output_dir / 'spec.json'
                with open(spec_path, 'w') as f:
                    json.dump(new_spec, f, indent=2)
                print(f"  Created: {spec_path}")

            specs_migrated += 1

            # Look for calibration file
            if calibration_path:
                cal_file = find_calibration_file(hw_id, calibration_path)
                if cal_file:
                    if dry_run:
                        print(f"    Would copy calibration: {cal_file.name}")
                    else:
                        cal_dest = output_dir / 'calibration.json'
                        shutil.copy(cal_file, cal_dest)
                        print(f"    Copied calibration: {cal_file.name}")
                    calibrations_migrated += 1

    return specs_migrated, calibrations_migrated


def main():
    parser = argparse.ArgumentParser(
        description="Migrate hardware_database to unified registry structure"
    )
    parser.add_argument(
        '--old-db',
        type=Path,
        default=None,
        help="Path to old hardware_database (default: auto-detect)"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help="Output path for new registry (default: hardware_registry/)"
    )
    parser.add_argument(
        '--calibration-dir',
        type=Path,
        default=None,
        help="Path to existing calibration profiles directory"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Set default paths
    old_db_path = args.old_db or project_root / 'hardware_database'
    new_registry_path = args.output or project_root / 'hardware_registry'
    calibration_path = args.calibration_dir or (
        project_root / 'src' / 'graphs' / 'hardware' / 'calibration' / 'profiles'
    )

    # Validate paths
    if not old_db_path.exists():
        print(f"ERROR: Old database not found: {old_db_path}")
        return 1

    print("=" * 60)
    print("Hardware Database Migration")
    print("=" * 60)
    print(f"Source:      {old_db_path}")
    print(f"Destination: {new_registry_path}")
    print(f"Calibration: {calibration_path}")
    print(f"Dry run:     {args.dry_run}")
    print()

    if not args.dry_run:
        new_registry_path.mkdir(parents=True, exist_ok=True)

    specs, cals = migrate_hardware_database(
        old_db_path,
        new_registry_path,
        calibration_path if calibration_path.exists() else None,
        dry_run=args.dry_run
    )

    print()
    print("=" * 60)
    print(f"Migration {'would migrate' if args.dry_run else 'complete'}:")
    print(f"  Hardware specs: {specs}")
    print(f"  Calibrations:   {cals}")
    print("=" * 60)

    if args.dry_run:
        print("\nRun without --dry-run to perform migration")

    return 0


if __name__ == '__main__':
    sys.exit(main())

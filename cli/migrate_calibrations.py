#!/usr/bin/env python
"""
Migrate old calibration.json files to new calibrations/ subdirectory structure.

This script migrates hardware_registry profiles from the old format:
    profile_dir/
        spec.json
        calibration.json

To the new format:
    profile_dir/
        spec.json
        calibrations/
            {power_mode}_{freq}MHz_{framework}.json

Usage:
    python cli/migrate_calibrations.py [--dry-run]
"""

import argparse
import json
import re
import shutil
from pathlib import Path


def make_calibration_filename(calibration_data: dict) -> str:
    """
    Generate calibration filename from calibration metadata.

    Format: {power_mode}_{frequency_mhz}MHz_{framework}.json
    """
    metadata = calibration_data.get('metadata', {})

    # Get power mode
    gpu_clock = metadata.get('gpu_clock', {})
    cpu_clock = metadata.get('cpu_clock', {})

    if metadata.get('device_type') == 'cuda' and gpu_clock.get('power_mode_name'):
        power_mode = gpu_clock['power_mode_name']
    elif gpu_clock.get('power_mode_name'):
        power_mode = gpu_clock['power_mode_name']
    elif cpu_clock.get('governor'):
        power_mode = cpu_clock['governor']
    else:
        power_mode = "unknown"

    # Get frequency (GPU SM clock or CPU frequency)
    if metadata.get('device_type') == 'cuda' and gpu_clock.get('sm_clock_mhz'):
        freq_mhz = gpu_clock['sm_clock_mhz']
    elif cpu_clock.get('current_freq_mhz'):
        freq_mhz = int(cpu_clock['current_freq_mhz'])
    else:
        freq_mhz = 0

    # Get framework
    framework = metadata.get('framework', 'unknown')

    # Sanitize power mode (replace spaces, special chars)
    power_mode = re.sub(r'[^a-zA-Z0-9]', '', power_mode)

    return f"{power_mode}_{freq_mhz}MHz_{framework}.json"


def migrate_profile(profile_dir: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single profile's calibration to new structure.

    Returns True if migration was performed, False otherwise.
    """
    old_calibration = profile_dir / 'calibration.json'
    calibrations_dir = profile_dir / 'calibrations'

    if not old_calibration.exists():
        return False

    # Load old calibration
    with open(old_calibration) as f:
        calibration_data = json.load(f)

    # Generate new filename
    new_filename = make_calibration_filename(calibration_data)
    new_path = calibrations_dir / new_filename

    print(f"  {profile_dir.name}:")
    print(f"    Old: calibration.json")
    print(f"    New: calibrations/{new_filename}")

    if dry_run:
        print(f"    [DRY RUN] Would migrate")
        return True

    # Create calibrations directory
    calibrations_dir.mkdir(exist_ok=True)

    # Move file (rename)
    shutil.move(str(old_calibration), str(new_path))
    print(f"    Migrated successfully")

    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate calibration.json to new structure")
    parser.add_argument('--dry-run', action='store_true',
                       help="Show what would be done without making changes")
    parser.add_argument('--registry-path', type=str, default=None,
                       help="Path to hardware registry (default: hardware_registry/)")
    args = parser.parse_args()

    # Find registry path
    if args.registry_path:
        registry_path = Path(args.registry_path)
    else:
        # Default: look relative to script
        script_dir = Path(__file__).parent
        registry_path = script_dir.parent / 'hardware_registry'

    if not registry_path.exists():
        print(f"Error: Registry path not found: {registry_path}")
        return 1

    print(f"Hardware Registry: {registry_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    # Find all calibration.json files
    migrated = 0
    for device_type_dir in registry_path.iterdir():
        if not device_type_dir.is_dir():
            continue
        if device_type_dir.name.startswith('.'):
            continue

        print(f"{device_type_dir.name}/")
        for profile_dir in device_type_dir.iterdir():
            if not profile_dir.is_dir():
                continue

            if migrate_profile(profile_dir, args.dry_run):
                migrated += 1

    print()
    if migrated > 0:
        print(f"{'Would migrate' if args.dry_run else 'Migrated'} {migrated} calibration(s)")
    else:
        print("No calibrations found to migrate")

    return 0


if __name__ == '__main__':
    exit(main())

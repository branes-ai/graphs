#!/usr/bin/env python3
"""
Migrate Hardware Presets to Database

Converts the existing PRESETS dictionary in cli/calibrate_hardware.py
to the new hardware database format (JSON files).

Usage:
    python scripts/hardware_db/migrate_presets.py
    python scripts/hardware_db/migrate_presets.py --dry-run
    python scripts/hardware_db/migrate_presets.py --output /custom/path
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareSpec, HardwareDatabase


# Import PRESETS from CLI (we'll read the file directly to avoid circular imports)
def load_presets_from_cli() -> dict:
    """Load PRESETS dictionary from cli/calibrate_hardware.py"""
    import re

    cli_path = Path(__file__).parent.parent.parent / "cli" / "calibrate_hardware.py"

    with open(cli_path) as f:
        content = f.read()

    # Extract PRESETS dictionary using regex
    # Look for PRESETS = { ... } spanning multiple lines
    match = re.search(r'PRESETS\s*=\s*\{(.*?)\n\}', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find PRESETS dictionary in calibrate_hardware.py")

    # Evaluate the dictionary safely
    presets_code = "PRESETS = {" + match.group(1) + "\n}"
    namespace = {}
    exec(presets_code, namespace)

    return namespace.get('PRESETS', {})


def migrate_preset(preset_id: str, preset_data: dict) -> HardwareSpec:
    """
    Convert a preset dictionary to HardwareSpec.

    Args:
        preset_id: Preset identifier (e.g., 'i7-12700k')
        preset_data: Preset configuration dict

    Returns:
        HardwareSpec instance
    """
    # Parse preset_id to extract vendor
    preset_lower = preset_id.lower()

    # Determine vendor from ID
    if 'intel' in preset_lower or 'i7' in preset_lower or 'i9' in preset_lower or 'xeon' in preset_lower:
        vendor = "Intel"
    elif 'amd' in preset_lower or 'ryzen' in preset_lower or 'epyc' in preset_lower:
        vendor = "AMD"
    elif 'ampere' in preset_lower or 'altra' in preset_lower:
        vendor = "Ampere Computing"
    elif 'nvidia' in preset_lower or 'h100' in preset_lower or 'a100' in preset_lower or 'jetson' in preset_lower:
        vendor = "NVIDIA"
    elif 'google' in preset_lower or 'tpu' in preset_lower:
        vendor = "Google"
    else:
        vendor = "Unknown"

    # Determine device type
    device = preset_data.get('device', 'cpu')

    # Convert device to device_type
    if device == 'cuda':
        device_type = 'gpu'
    else:
        device_type = device

    # Normalize ID (replace hyphens with underscores)
    normalized_id = preset_id.replace('-', '_')

    # Extract cores/threads if available from name
    cores = None
    threads = None
    if 'i7-12700k' in preset_lower:
        cores = 12  # 8P + 4E
        threads = 20
    elif 'altra-max' in preset_lower:
        cores = 128
        threads = 128
    elif 'jetson-orin-agx' in preset_lower:
        cores = 12
        threads = 12
    elif 'jetson-orin-nano' in preset_lower:
        cores = 6
        threads = 6

    # Create HardwareSpec
    spec = HardwareSpec(
        id=normalized_id,
        vendor=vendor,
        model=preset_data.get('name', preset_id),
        architecture="Unknown",  # Not in preset data, needs manual entry
        device_type=device_type,
        platform=preset_data.get('platform', 'x86_64'),

        # Detection (will need refinement)
        detection_patterns=[preset_data.get('name', preset_id)],
        os_compatibility=["linux", "windows", "macos"],

        # Core specs
        cores=cores,
        threads=threads,

        # Memory
        memory_type="DDR5" if device_type == 'cpu' else "Unknown",
        peak_bandwidth_gbps=preset_data.get('peak_bandwidth', 0.0),

        # Performance
        theoretical_peaks=preset_data.get('theoretical_peaks', {}),

        # Mapper
        mapper_class="GPUMapper" if device_type == 'gpu' else "CPUMapper",
        mapper_config={},

        # Metadata
        data_source="migrated",
        notes=f"Migrated from cli/calibrate_hardware.py preset '{preset_id}'"
    )

    return spec


def main():
    parser = argparse.ArgumentParser(
        description="Migrate hardware presets to database format"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Output directory for hardware database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without writing files"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Hardware Preset Migration")
    print("=" * 80)
    print()

    # Load presets
    print("Loading presets from cli/calibrate_hardware.py...")
    try:
        presets = load_presets_from_cli()
        print(f"Found {len(presets)} presets")
    except Exception as e:
        print(f"Error loading presets: {e}")
        return 1

    print()

    # Initialize database
    db = HardwareDatabase(args.output)
    if not args.dry_run:
        db.load_all()

    # Migrate each preset
    migrated_count = 0
    skipped_count = 0
    error_count = 0

    for preset_id, preset_data in presets.items():
        print(f"Migrating: {preset_id}")
        print(f"  Name: {preset_data.get('name', 'N/A')}")
        print(f"  Device: {preset_data.get('device', 'cpu')}")
        print(f"  Platform: {preset_data.get('platform', 'x86_64')}")

        try:
            # Convert to HardwareSpec
            spec = migrate_preset(preset_id, preset_data)

            # Validate
            errors = spec.validate()
            if errors:
                print(f"  ⚠ Validation warnings:")
                for error in errors:
                    print(f"    - {error}")

            if args.dry_run:
                print(f"  ✓ Would create: {spec.id}.json")
                migrated_count += 1
            else:
                # Add to database
                success = db.add(spec, overwrite=args.overwrite)
                if success:
                    migrated_count += 1
                else:
                    skipped_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_count += 1

        print()

    # Summary
    print("=" * 80)
    print("Migration Summary")
    print("=" * 80)
    print(f"Migrated: {migrated_count}")
    print(f"Skipped:  {skipped_count}")
    print(f"Errors:   {error_count}")
    print()

    if args.dry_run:
        print("This was a dry run. Use without --dry-run to actually migrate.")
    else:
        print(f"Hardware specs written to: {args.output}")
        print()
        print("Next steps:")
        print("  1. Review generated JSON files")
        print("  2. Fill in missing fields (architecture, detection_patterns, etc.)")
        print("  3. Run validation: python scripts/hardware_db/validate_database.py")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

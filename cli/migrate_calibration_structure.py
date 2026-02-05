#!/usr/bin/env python3
"""
Migrate Calibration Data to New Directory Structure

Moves calibration data from the old flat structure to the new precision-based structure.

Old structure:
    calibration_data/<hardware_id>/measurements/
    calibration_data/<hardware_id>/efficiency_curves.json

New structure:
    calibration_data/<hardware_id>/<precision>/measurements/
    calibration_data/<hardware_id>/<precision>/efficiency_curves.json

Usage:
    ./cli/migrate_calibration_structure.py --dry-run   # Preview changes
    ./cli/migrate_calibration_structure.py             # Apply migration
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
calibration_dir = repo_root / "calibration_data"

PRECISIONS = ['fp32', 'fp16', 'bf16', 'tf32', 'int8']


def infer_precision(hw_dir: Path) -> str:
    """Infer precision from efficiency_curves.json or measurements."""
    # Try efficiency_curves.json first
    curves_file = hw_dir / "efficiency_curves.json"
    if curves_file.exists():
        try:
            with open(curves_file) as f:
                curves = json.load(f)
                precision = curves.get("precision", "FP32")
                return precision.lower()
        except:
            pass

    # Try first measurement file
    measurements_dir = hw_dir / "measurements"
    if measurements_dir.exists():
        for json_file in measurements_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    precision = data.get("precision", "FP32")
                    return precision.lower()
            except:
                pass

    return "fp32"  # default


def needs_migration(hw_dir: Path) -> bool:
    """Check if a hardware directory needs migration.

    Returns True if it has the old flat structure (measurements/ at top level).
    Returns False if it already has precision subdirectories.
    """
    # Check if it has precision subdirectories
    for precision in PRECISIONS:
        prec_dir = hw_dir / precision
        if prec_dir.is_dir():
            if (prec_dir / "measurements").exists() or (prec_dir / "efficiency_curves.json").exists():
                return False  # Already migrated

    # Check if it has old structure
    if (hw_dir / "measurements").exists() or (hw_dir / "efficiency_curves.json").exists():
        return True

    return False


def migrate_hardware(hw_dir: Path, dry_run: bool = False) -> bool:
    """Migrate a single hardware directory to new structure.

    Returns True if migration was performed/would be performed.
    """
    hw_id = hw_dir.name
    precision = infer_precision(hw_dir)

    print(f"\n  Migrating {hw_id} -> {hw_id}/{precision}/")

    # Create target directory
    target_dir = hw_dir / precision

    if not dry_run:
        target_dir.mkdir(exist_ok=True)

    # Move measurements/
    measurements_dir = hw_dir / "measurements"
    if measurements_dir.exists():
        target_measurements = target_dir / "measurements"
        print(f"    Moving measurements/ -> {precision}/measurements/")
        if not dry_run:
            shutil.move(str(measurements_dir), str(target_measurements))

    # Move efficiency_curves.json
    curves_file = hw_dir / "efficiency_curves.json"
    if curves_file.exists():
        target_curves = target_dir / "efficiency_curves.json"
        print(f"    Moving efficiency_curves.json -> {precision}/efficiency_curves.json")
        if not dry_run:
            shutil.move(str(curves_file), str(target_curves))

    # Move calibration_report.md
    report_file = hw_dir / "calibration_report.md"
    if report_file.exists():
        target_report = target_dir / "calibration_report.md"
        print(f"    Moving calibration_report.md -> {precision}/calibration_report.md")
        if not dry_run:
            shutil.move(str(report_file), str(target_report))

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Migrate calibration data to new precision-based structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run   # Preview changes
  %(prog)s             # Apply migration
""")
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without applying them')

    args = parser.parse_args()

    if not calibration_dir.exists():
        print(f"No calibration_data directory found at {calibration_dir}")
        return 0

    print(f"Scanning {calibration_dir}...")
    if args.dry_run:
        print("DRY RUN - no changes will be made")

    # Find directories that need migration
    dirs_to_migrate = []
    dirs_already_migrated = []
    dirs_empty = []

    for hw_dir in sorted(calibration_dir.iterdir()):
        if not hw_dir.is_dir():
            continue

        if needs_migration(hw_dir):
            dirs_to_migrate.append(hw_dir)
        else:
            # Check if it's already migrated or empty
            has_precision_dirs = any(
                (hw_dir / p).is_dir() for p in PRECISIONS
            )
            if has_precision_dirs:
                dirs_already_migrated.append(hw_dir)
            else:
                dirs_empty.append(hw_dir)

    print(f"\nFound {len(dirs_to_migrate)} directories to migrate")
    print(f"      {len(dirs_already_migrated)} already migrated")
    print(f"      {len(dirs_empty)} empty/other")

    if not dirs_to_migrate:
        print("\nNo migration needed.")
        return 0

    # Perform migration
    for hw_dir in dirs_to_migrate:
        migrate_hardware(hw_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[DRY RUN] No changes made. Run without --dry-run to apply.")
    else:
        print("\nMigration complete.")

    return 0


if __name__ == '__main__':
    sys.exit(main())

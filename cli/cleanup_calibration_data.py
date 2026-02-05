#!/usr/bin/env python3
"""
Cleanup script to standardize calibration_data directory naming.

Renames directories from dashes to underscores and fixes hardware_id inside JSON files.

Usage:
    ./cli/cleanup_calibration_data.py --dry-run   # Preview changes
    ./cli/cleanup_calibration_data.py             # Apply changes
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
calibration_dir = repo_root / "calibration_data"


def dash_to_underscore(name: str) -> str:
    """Convert dashes to underscores, preserving case for model numbers."""
    # Special cases for CPU model numbers (preserve case)
    if name.startswith("i7-") or name.startswith("i9-") or name.startswith("i5-"):
        return name.replace("-", "_")
    if name.startswith("ryzen"):
        return name.replace("-", "_")
    # General case: replace dashes with underscores, lowercase for Jetson etc.
    return name.replace("-", "_").lower()


def fix_json_hardware_id(json_path: Path, old_id: str, new_id: str, dry_run: bool) -> bool:
    """Fix hardware_id in a JSON file."""
    try:
        with open(json_path, 'r') as f:
            content = f.read()

        # Check if this file has the old ID
        if f'"hardware_id": "{old_id}"' not in content:
            return False

        new_content = content.replace(
            f'"hardware_id": "{old_id}"',
            f'"hardware_id": "{new_id}"'
        )

        if not dry_run:
            with open(json_path, 'w') as f:
                f.write(new_content)

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Cleanup calibration_data naming')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without applying them')
    args = parser.parse_args()

    if not calibration_dir.exists():
        print(f"No calibration_data directory found at {calibration_dir}")
        return 0

    print(f"Scanning {calibration_dir}...")
    if args.dry_run:
        print("DRY RUN - no changes will be made\n")

    # Find directories that need renaming
    dirs_to_rename = []
    dirs_to_delete = []

    for d in sorted(calibration_dir.iterdir()):
        if not d.is_dir():
            continue

        old_name = d.name
        new_name = dash_to_underscore(old_name)

        if old_name != new_name:
            new_path = calibration_dir / new_name

            # Check if target already exists (duplicate)
            if new_path.exists():
                # Check which one has correct internal hardware_id
                old_json = d / "measurements" / "resnet18.json"
                new_json = new_path / "measurements" / "resnet18.json"

                old_hwid = None
                new_hwid = None

                if old_json.exists():
                    with open(old_json) as f:
                        old_hwid = json.load(f).get("hardware_id")
                if new_json.exists():
                    with open(new_json) as f:
                        new_hwid = json.load(f).get("hardware_id")

                # If new one has correct ID, delete old one
                if new_hwid == new_name:
                    print(f"DELETE: {old_name}")
                    print(f"  Reason: Duplicate of {new_name}, which has correct hardware_id")
                    print(f"  Old hardware_id: {old_hwid}")
                    print(f"  New hardware_id: {new_hwid}")
                    dirs_to_delete.append(d)
                else:
                    print(f"CONFLICT: {old_name} -> {new_name} (target exists)")
                    print(f"  Old hardware_id: {old_hwid}")
                    print(f"  New hardware_id: {new_hwid}")
                    print(f"  Manual resolution needed")
            else:
                dirs_to_rename.append((d, new_path, old_name, new_name))

    # Process deletions
    for d in dirs_to_delete:
        if not args.dry_run:
            shutil.rmtree(d)
            print(f"  Deleted {d.name}")

    # Process renames
    for old_path, new_path, old_name, new_name in dirs_to_rename:
        print(f"\nRENAME: {old_name} -> {new_name}")

        if not args.dry_run:
            old_path.rename(new_path)

        # Fix hardware_id in JSON files
        # Common old IDs that might be in the files
        old_ids = [
            old_name,
            old_name.replace("-", "_"),
            "Jetson-Orin-AGX",
            "Jetson-Orin-Nano",
            "Jetson-Orin-NX",
            "i7-12700K",
        ]

        json_files = list((new_path if not args.dry_run else old_path).glob("**/*.json"))
        for json_path in json_files:
            for old_id in old_ids:
                if fix_json_hardware_id(json_path, old_id, new_name, args.dry_run):
                    print(f"  Fixed hardware_id in {json_path.name}: {old_id} -> {new_name}")
                    break

    # Check remaining directories for internal ID mismatches
    print("\n--- Checking remaining directories ---")
    for d in sorted(calibration_dir.iterdir()):
        if not d.is_dir():
            continue

        expected_id = d.name
        json_files = list(d.glob("measurements/*.json"))[:1]
        if json_files:
            with open(json_files[0]) as f:
                actual_id = json.load(f).get("hardware_id")

            if actual_id != expected_id:
                print(f"MISMATCH: {d.name}")
                print(f"  Directory name: {expected_id}")
                print(f"  Internal ID:    {actual_id}")
                print(f"  -> Measurements need to be re-run with --id {expected_id}")
            else:
                print(f"OK: {d.name}")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

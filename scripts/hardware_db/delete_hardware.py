#!/usr/bin/env python3
"""
Delete Hardware from Database

Remove hardware specifications from the database.

Usage:
    python scripts/hardware_db/delete_hardware.py --id intel_i7_12700k
    python scripts/hardware_db/delete_hardware.py --id h100_sxm5 --yes
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareDatabase


def main():
    parser = argparse.ArgumentParser(
        description="Delete hardware from database"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--id",
        required=True,
        help="Hardware ID to delete"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Load database
    db = HardwareDatabase(args.db)
    print(f"Loading database from: {args.db}")
    db.load_all()
    print()

    # Check if hardware exists
    spec = db.get(args.id)
    if not spec:
        print(f"✗ Hardware not found: {args.id}")
        return 1

    # Show hardware info
    print("Hardware to delete:")
    print(f"  ID:       {spec.id}")
    print(f"  Vendor:   {spec.vendor}")
    print(f"  Model:    {spec.model}")
    print(f"  Type:     {spec.device_type}")
    print(f"  Platform: {spec.platform}")
    spec_file = db._find_spec_file(spec.id)
    if spec_file:
        print(f"  File:     {spec_file}")
    print()

    # Confirm deletion
    if not args.yes:
        confirm = input("Delete this hardware? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled")
            return 1

    # Delete
    try:
        success = db.delete(args.id)

        if success:
            print(f"✓ Deleted hardware: {args.id}")
            return 0
        else:
            print(f"✗ Failed to delete hardware: {args.id}")
            return 1

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

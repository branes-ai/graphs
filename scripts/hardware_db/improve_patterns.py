#!/usr/bin/env python3
"""
Improve Detection Patterns

Automatically improve detection patterns for existing hardware in the database.

Usage:
    python scripts/hardware_db/improve_patterns.py
    python scripts/hardware_db/improve_patterns.py --dry-run
    python scripts/hardware_db/improve_patterns.py --id intel_i7_12700k
"""

import sys
import re
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareDatabase


def generate_cpu_patterns(spec) -> list:
    """Generate detection patterns for CPU"""
    patterns = []
    model = spec.model
    vendor = spec.vendor

    # Extract key parts from model name
    # Examples: "Intel-i7-12700K", "Ampere-Altra-Max-128"

    # Pattern 1: Full model with vendor prefix
    patterns.append(f"{vendor}.*{re.escape(model)}")

    # Pattern 2: Model without vendor
    patterns.append(re.escape(model))

    # For Intel CPUs: handle generation variations
    if vendor == "Intel":
        # Extract generation and model (e.g., "12700K" from "i7-12700K")
        match = re.search(r'i(\d)-(\d+)', model.lower())
        if match:
            series = match.group(1)  # 7, 9, etc.
            gen_model = match.group(2)  # 12700, 13900, etc.
            gen = gen_model[:2]  # 12, 13, etc.

            # "12th Gen Intel.*Core.*i7-12700K"
            patterns.append(f"{gen}th Gen Intel.*Core.*i{series}-{gen_model}")

            # "Intel.*Core.*i7-12700K"
            patterns.append(f"Intel.*Core.*i{series}-{gen_model}")

    # For AMD CPUs: handle Ryzen/EPYC variations
    elif vendor == "AMD":
        # "Ryzen 9 7950X" -> "Ryzen.*9.*7950X"
        if "ryzen" in model.lower():
            patterns.append(f"Ryzen.*{re.escape(model.split('-')[-1])}")

    # For Ampere: handle Altra variations
    elif vendor == "Ampere Computing":
        if "altra" in model.lower():
            patterns.append("Ampere.*Altra.*")

    return patterns


def generate_gpu_patterns(spec) -> list:
    """Generate detection patterns for GPU"""
    patterns = []
    model = spec.model
    vendor = spec.vendor

    # Pattern 1: Full model with vendor
    patterns.append(f"{vendor}.*{re.escape(model)}")

    # For NVIDIA GPUs: handle variations
    if vendor == "NVIDIA":
        # Extract key parts: "H100", "A100", "Jetson Orin"
        # "NVIDIA-H100-SXM5-80GB" -> ["H100", "SXM5", "80GB"]
        parts = model.replace("NVIDIA-", "").split("-")

        if len(parts) >= 1:
            gpu_model = parts[0]  # H100, A100, Jetson

            # "NVIDIA H100.*80GB"
            if "gb" in model.lower():
                memory = [p for p in parts if "gb" in p.lower()]
                if memory:
                    patterns.append(f"NVIDIA {gpu_model}.*{memory[0]}")

            # "H100 SXM5"
            if len(parts) >= 2:
                form_factor = parts[1]  # SXM5, PCIe, etc.
                patterns.append(f"{gpu_model}.*{form_factor}")

            # Just "H100"
            patterns.append(gpu_model)

        # For Jetson: special handling
        if "jetson" in model.lower():
            # "Jetson Orin AGX" -> "Jetson.*Orin.*AGX"
            jetson_parts = [p for p in parts if p.lower() not in ['nvidia']]
            if jetson_parts:
                patterns.append(".*".join(jetson_parts))

    return patterns


def improve_hardware_patterns(db: HardwareDatabase, hw_id: str = None, dry_run: bool = False) -> int:
    """Improve detection patterns for hardware"""
    updated_count = 0

    # Get hardware to process
    if hw_id:
        spec = db.get(hw_id)
        if not spec:
            print(f"✗ Hardware not found: {hw_id}")
            return 0
        specs_to_process = [(hw_id, spec)]
    else:
        specs_to_process = list(db._cache.items())

    print(f"Processing {len(specs_to_process)} hardware specs...")
    print()

    for hw_id, spec in specs_to_process:
        # Check if patterns need improvement
        current_patterns = spec.detection_patterns or []

        # Skip if already has good patterns (more than 1 pattern with regex)
        has_regex = any('.*' in p or '[' in p or '(' in p for p in current_patterns)
        if len(current_patterns) > 1 and has_regex:
            continue

        # Generate new patterns
        if spec.device_type == 'cpu':
            new_patterns = generate_cpu_patterns(spec)
        elif spec.device_type == 'gpu':
            new_patterns = generate_gpu_patterns(spec)
        else:
            new_patterns = [spec.model]

        # Remove duplicates and keep order
        new_patterns = list(dict.fromkeys(new_patterns))

        # Show changes
        print(f"{hw_id}:")
        print(f"  Vendor: {spec.vendor}")
        print(f"  Model:  {spec.model}")
        print(f"  Old patterns ({len(current_patterns)}):")
        for p in current_patterns:
            print(f"    - {p}")
        print(f"  New patterns ({len(new_patterns)}):")
        for p in new_patterns:
            print(f"    + {p}")

        if dry_run:
            print("  (dry run - not updating)")
        else:
            # Update spec
            spec.detection_patterns = new_patterns
            spec.last_updated = datetime.utcnow().isoformat() + "Z"

            # Save (use add with overwrite to replace file)
            success = db.add(spec, overwrite=True)
            if success:
                print("  ✓ Updated")
                updated_count += 1
            else:
                print("  ✗ Failed to update")

        print()

    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Improve detection patterns in hardware database"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--id",
        help="Improve patterns for specific hardware ID"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without updating"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Improve Detection Patterns")
    print("=" * 80)
    print()

    # Load database
    db = HardwareDatabase(args.db)
    print(f"Loading database from: {args.db}")
    db.load_all()
    print(f"Loaded {len(db._cache)} hardware specs")
    print()

    # Process
    updated_count = improve_hardware_patterns(db, hw_id=args.id, dry_run=args.dry_run)

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Updated: {updated_count} hardware specs")

    if args.dry_run:
        print()
        print("This was a dry run. Use without --dry-run to apply changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

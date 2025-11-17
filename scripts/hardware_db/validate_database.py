#!/usr/bin/env python3
"""
Validate Hardware Database

Validate all hardware specifications in the database against the schema.

Usage:
    python scripts/hardware_db/validate_database.py
    python scripts/hardware_db/validate_database.py --strict
    python scripts/hardware_db/validate_database.py --fix-issues
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareDatabase, HardwareSpec


def validate_all_specs(db: HardwareDatabase, strict: bool = False) -> Tuple[int, int, Dict[str, List[str]]]:
    """
    Validate all hardware specs in database.

    Args:
        db: HardwareDatabase instance
        strict: If True, warnings are treated as errors

    Returns:
        (valid_count, invalid_count, issues_by_id)
    """
    valid_count = 0
    invalid_count = 0
    issues_by_id = {}

    for spec_id, spec in db._cache.items():
        errors = spec.validate()

        if errors:
            invalid_count += 1
            issues_by_id[spec_id] = errors
        else:
            valid_count += 1

    return valid_count, invalid_count, issues_by_id


def check_detection_patterns(db: HardwareDatabase) -> Dict[str, List[str]]:
    """
    Check for missing or weak detection patterns.

    Args:
        db: HardwareDatabase instance

    Returns:
        Dictionary mapping spec_id to warning messages
    """
    warnings = {}

    for spec_id, spec in db._cache.items():
        spec_warnings = []

        if not spec.detection_patterns:
            spec_warnings.append("No detection patterns defined")
        elif len(spec.detection_patterns) == 1 and spec.detection_patterns[0] == spec.model:
            spec_warnings.append("Only exact model name match - consider adding regex patterns")

        if spec_warnings:
            warnings[spec_id] = spec_warnings

    return warnings


def check_missing_fields(db: HardwareDatabase) -> Dict[str, List[str]]:
    """
    Check for important missing fields.

    Args:
        db: HardwareDatabase instance

    Returns:
        Dictionary mapping spec_id to warning messages
    """
    warnings = {}

    for spec_id, spec in db._cache.items():
        spec_warnings = []

        if spec.architecture == "Unknown":
            spec_warnings.append("Architecture is 'Unknown'")

        if spec.device_type == 'cpu':
            if not spec.cores:
                spec_warnings.append("CPU missing core count")
            if not spec.isa_extensions:
                spec_warnings.append("CPU missing ISA extensions")

        if spec.device_type == 'gpu':
            if spec.vendor == 'NVIDIA':
                if not spec.cuda_capability:
                    spec_warnings.append("NVIDIA GPU missing CUDA capability")
                if not spec.cuda_cores and not spec.sms:
                    spec_warnings.append("GPU missing CUDA cores or SM count")

        if not spec.theoretical_peaks or 'fp32' not in spec.theoretical_peaks:
            spec_warnings.append("Missing fp32 theoretical peak")

        if spec.data_source == 'migrated':
            spec_warnings.append("Data source is 'migrated' - needs manual review")

        if spec_warnings:
            warnings[spec_id] = spec_warnings

    return warnings


def main():
    parser = argparse.ArgumentParser(
        description="Validate hardware database"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    parser.add_argument(
        "--fix-issues",
        action="store_true",
        help="Attempt to auto-fix common issues"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Hardware Database Validation")
    print("=" * 80)
    print()

    # Load database
    db = HardwareDatabase(args.db)
    print(f"Loading database from: {args.db}")
    db.load_all()
    print(f"Loaded {len(db._cache)} hardware specs")
    print()

    # Validate all specs
    print("Validating Hardware Specifications...")
    print("-" * 80)
    valid_count, invalid_count, schema_issues = validate_all_specs(db, strict=args.strict)

    if schema_issues:
        print(f"✗ Schema Validation Errors: {invalid_count}")
        print()
        for spec_id, errors in schema_issues.items():
            print(f"{spec_id}:")
            for error in errors:
                print(f"  - {error}")
            print()
    else:
        print(f"✓ All {valid_count} specs pass schema validation")
        print()

    # Check detection patterns
    print("Checking Detection Patterns...")
    print("-" * 80)
    pattern_warnings = check_detection_patterns(db)

    if pattern_warnings:
        print(f"⚠ {len(pattern_warnings)} specs have detection pattern issues:")
        print()
        for spec_id, warnings in pattern_warnings.items():
            print(f"{spec_id}:")
            for warning in warnings:
                print(f"  - {warning}")
            print()
    else:
        print("✓ All specs have detection patterns")
        print()

    # Check missing fields
    print("Checking for Missing/Incomplete Fields...")
    print("-" * 80)
    field_warnings = check_missing_fields(db)

    if field_warnings:
        print(f"⚠ {len(field_warnings)} specs have missing/incomplete fields:")
        print()
        for spec_id, warnings in field_warnings.items():
            print(f"{spec_id}:")
            for warning in warnings:
                print(f"  - {warning}")
            print()
    else:
        print("✓ All specs have complete fields")
        print()

    # Summary
    print("=" * 80)
    print("Validation Summary")
    print("=" * 80)
    total_specs = len(db._cache)
    total_issues = len(schema_issues)
    total_warnings = len(pattern_warnings) + len(field_warnings)

    print(f"Total specs:      {total_specs}")
    print(f"Valid:            {valid_count}")
    print(f"Schema errors:    {total_issues}")
    print(f"Warnings:         {total_warnings}")
    print()

    if total_issues > 0:
        print("✗ Database has schema validation errors")
        print("  Fix these errors before using the database")
        return 1
    elif total_warnings > 0 and args.strict:
        print("✗ Database has warnings (strict mode enabled)")
        return 1
    elif total_warnings > 0:
        print("⚠ Database has warnings but passes validation")
        print("  Consider addressing warnings for better hardware detection")
        return 0
    else:
        print("✓ Database is valid")
        return 0


if __name__ == "__main__":
    sys.exit(main())

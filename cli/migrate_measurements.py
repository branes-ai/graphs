#!/usr/bin/env python3
"""
Migrate Measurement Data to v2.0 Schema and Canonical Layout.

Consolidates measurement files from both `measurements/` and
`calibration_data/` into the canonical layout:

    calibration_data/
      <hardware_id>/
        manifest.json
        measurements/
          <precision>/
            <model>_b<batch>.json

For each measurement file:
1. Reads v1.0 JSON
2. Infers batch_size from filename if not in JSON
3. Computes model_summary from subgraphs
4. Writes v2.0 JSON to canonical location
5. Generates manifest.json per hardware_id

Usage:
    # Preview what would happen
    python cli/migrate_measurements.py --dry-run

    # Execute migration
    python cli/migrate_measurements.py

    # Migrate from custom source
    python cli/migrate_measurements.py --source measurements/ --source calibration_data/

    # Rebuild manifests only (no file migration)
    python cli/migrate_measurements.py --manifests-only
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from graphs.calibration.ground_truth import (
    MeasurementRecord,
    GroundTruthLoader,
    _normalize_precision,
    _is_valid_model_name,
)


def find_measurement_files(source_dirs: List[Path]) -> List[Dict]:
    """Scan source directories for measurement JSON files.

    Returns list of dicts with keys:
        filepath, hardware_id, precision, model, batch_size, source_dir
    """
    results = []

    for source_dir in source_dirs:
        if not source_dir.exists():
            continue

        for hw_dir in sorted(source_dir.iterdir()):
            if not hw_dir.is_dir():
                continue
            hardware_id = hw_dir.name

            # Pattern 1: <hw>/<precision>/<model>_b<batch>.json (measurements/ dir)
            for prec_dir in sorted(hw_dir.iterdir()):
                if not prec_dir.is_dir():
                    continue
                precision = prec_dir.name

                # Direct JSON files in precision dir
                for f in sorted(prec_dir.glob("*.json")):
                    if f.name == 'efficiency_curves.json':
                        continue
                    model, batch_size = _parse_filename(f.name)
                    results.append({
                        'filepath': f,
                        'hardware_id': hardware_id,
                        'precision': precision,
                        'model': model,
                        'batch_size': batch_size,
                        'source_dir': source_dir.name,
                    })

                # Pattern 2: <hw>/<precision>/measurements/<model>.json
                meas_subdir = prec_dir / "measurements"
                if meas_subdir.exists():
                    for f in sorted(meas_subdir.glob("*.json")):
                        model, batch_size = _parse_filename(f.name)
                        results.append({
                            'filepath': f,
                            'hardware_id': hardware_id,
                            'precision': precision,
                            'model': model,
                            'batch_size': batch_size,
                            'source_dir': source_dir.name,
                        })

    return results


def _parse_filename(filename: str) -> Tuple[str, int]:
    """Parse model and batch size from filename.

    Uses _is_valid_model_name to distinguish batch suffixes from
    model names containing '_b<N>' (e.g., 'efficientnet_b1').
    """
    name = filename.replace('.json', '')
    match = re.match(r'^(.+?)_b(\d+)$', name)
    if match:
        candidate_model = match.group(1)
        batch = int(match.group(2))
        if batch >= 1 and _is_valid_model_name(candidate_model):
            return candidate_model, batch
    return name, 1


def migrate_file(
    info: Dict,
    output_base: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> Optional[Path]:
    """Migrate a single measurement file to v2.0 canonical layout.

    Returns the output path on success, None on failure.
    """
    filepath = info['filepath']
    hardware_id = info['hardware_id']
    precision = _normalize_precision(info['precision'])
    model = info['model']
    batch_size = info['batch_size']

    # Canonical output path
    output_path = (output_base / hardware_id / "measurements" /
                   precision / f"{model}_b{batch_size}.json")

    if dry_run:
        return output_path

    try:
        # Load using MeasurementRecord which handles v1.0 compat
        record = MeasurementRecord.load(filepath)

        # Ensure v2.0 fields are populated
        record.schema_version = "2.0"
        record.batch_size = batch_size

        # Ensure model_summary is computed
        if record.model_summary is None and record.subgraphs:
            from graphs.calibration.ground_truth import ModelSummary
            record.model_summary = ModelSummary.from_subgraphs(
                record.subgraphs, record.batch_size
            )

        # Bump tool version to indicate migration
        if 'v1.0' in record.tool_version:
            record.tool_version = record.tool_version.replace('v1.0', 'v2.0 (migrated)')

        # Save to canonical location
        record.save(output_path)

        if verbose:
            print(f"  OK: {filepath.name} -> {output_path.relative_to(output_base)}")

        return output_path

    except Exception as e:
        print(f"  FAIL: {filepath}: {e}")
        return None


def run_migration(
    source_dirs: List[Path],
    output_base: Path,
    dry_run: bool = False,
    verbose: bool = False,
    manifests_only: bool = False,
) -> Dict:
    """Run the full migration.

    Returns summary dict with migration statistics.
    """
    stats = {
        'total_found': 0,
        'migrated': 0,
        'skipped': 0,
        'failed': 0,
        'hardware_ids': set(),
        'duplicates': 0,
    }

    if manifests_only:
        print("Rebuilding manifests only (no file migration)...")
        loader = GroundTruthLoader(output_base)
        for hw_id in loader.list_hardware():
            manifest = loader.rebuild_manifest(hw_id)
            count = len(manifest.measurements)
            print(f"  {hw_id}: {count} entries")
            stats['hardware_ids'].add(hw_id)
        stats['hardware_ids'] = sorted(stats['hardware_ids'])
        return stats

    # Find all measurement files
    print("Scanning for measurement files...")
    files = find_measurement_files(source_dirs)
    stats['total_found'] = len(files)
    print(f"  Found {len(files)} measurement files")
    print()

    if not files:
        print("No measurement files found.")
        return stats

    # Group by hardware_id for organized output
    by_hardware: Dict[str, List[Dict]] = {}
    for info in files:
        hw = info['hardware_id']
        by_hardware.setdefault(hw, []).append(info)

    # Track unique keys to detect duplicates
    seen_keys: Set[Tuple[str, str, str, int]] = set()

    for hardware_id in sorted(by_hardware.keys()):
        hw_files = by_hardware[hardware_id]
        stats['hardware_ids'].add(hardware_id)

        print(f"{hardware_id}: {len(hw_files)} files")

        for info in hw_files:
            key = (info['hardware_id'], info['model'],
                   _normalize_precision(info['precision']), info['batch_size'])

            if key in seen_keys:
                stats['duplicates'] += 1
                if verbose:
                    print(f"  DUPLICATE (skipped): {info['filepath'].name}")
                stats['skipped'] += 1
                continue

            seen_keys.add(key)

            output_path = migrate_file(info, output_base, dry_run=dry_run, verbose=verbose)
            if output_path is not None:
                stats['migrated'] += 1
            else:
                stats['failed'] += 1

        if not dry_run:
            # Build manifest for this hardware
            print(f"  Building manifest...")
            loader = GroundTruthLoader(output_base)
            manifest = loader.rebuild_manifest(hardware_id)
            print(f"  Manifest: {len(manifest.measurements)} entries")

        print()

    stats['hardware_ids'] = sorted(stats['hardware_ids'])
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate measurement data to v2.0 schema and canonical layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview
    %(prog)s --dry-run

    # Execute
    %(prog)s

    # Custom source directories
    %(prog)s --source measurements/ --source calibration_data/

    # Rebuild manifests only
    %(prog)s --manifests-only
"""
    )

    parser.add_argument(
        '--source', action='append', type=Path,
        help='Source directories to scan (default: measurements/ and calibration_data/)'
    )
    parser.add_argument(
        '--output', type=Path, default=REPO_ROOT / "calibration_data",
        help='Output base directory (default: calibration_data/)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview without modifying files'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show per-file details'
    )
    parser.add_argument(
        '--manifests-only', action='store_true',
        help='Only rebuild manifests (no file migration)'
    )

    args = parser.parse_args()

    # Default source directories
    if args.source is None:
        args.source = [
            REPO_ROOT / "measurements",
            REPO_ROOT / "calibration_data",
        ]

    print()
    print("=" * 70)
    print("MEASUREMENT DATA MIGRATION")
    print("=" * 70)
    if args.dry_run:
        print("[DRY RUN - no files will be modified]")
    print(f"Sources: {', '.join(str(s) for s in args.source)}")
    print(f"Output:  {args.output}")
    print()

    stats = run_migration(
        source_dirs=args.source,
        output_base=args.output,
        dry_run=args.dry_run,
        verbose=args.verbose,
        manifests_only=args.manifests_only,
    )

    # Print summary
    print("=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    if not args.manifests_only:
        print(f"Files found:     {stats['total_found']}")
        print(f"Migrated:        {stats['migrated']}")
        print(f"Skipped (dupes): {stats['skipped']}")
        print(f"Failed:          {stats['failed']}")
        print(f"Duplicates:      {stats['duplicates']}")
    print(f"Hardware IDs:    {len(stats['hardware_ids'])}")
    for hw in stats['hardware_ids']:
        print(f"  - {hw}")

    if args.dry_run:
        print()
        print("This was a dry run. Run without --dry-run to execute migration.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

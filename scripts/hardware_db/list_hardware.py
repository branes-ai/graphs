#!/usr/bin/env python3
"""
List Hardware Database

Display all hardware specs in the database with optional filtering.

Usage:
    python scripts/hardware_db/list_hardware.py
    python scripts/hardware_db/list_hardware.py --vendor Intel
    python scripts/hardware_db/list_hardware.py --device-type gpu
    python scripts/hardware_db/list_hardware.py --platform aarch64
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareDatabase


def main():
    parser = argparse.ArgumentParser(
        description="List hardware specs in database"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--vendor",
        help="Filter by vendor"
    )
    parser.add_argument(
        "--device-type",
        help="Filter by device type (cpu, gpu, tpu, etc.)"
    )
    parser.add_argument(
        "--platform",
        help="Filter by platform (x86_64, aarch64, arm64)"
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )

    args = parser.parse_args()

    # Load database
    db = HardwareDatabase(args.db)
    print(f"Loading database from: {args.db}")
    db.load_all()
    print()

    # Show statistics if requested
    if args.stats:
        stats = db.get_statistics()
        print("=" * 80)
        print("Database Statistics")
        print("=" * 80)
        print(f"Total hardware specs: {stats['total_count']}")
        print()

        print("By Device Type:")
        for device_type, count in sorted(stats['by_device_type'].items()):
            print(f"  {device_type:<10} {count:>3}")
        print()

        print("By Vendor:")
        for vendor, count in sorted(stats['by_vendor'].items()):
            print(f"  {vendor:<20} {count:>3}")
        print()

        print("By Platform:")
        for platform, count in sorted(stats['by_platform'].items()):
            print(f"  {platform:<10} {count:>3}")
        print()

        print("By OS Compatibility:")
        for os_type, count in sorted(stats['by_os'].items()):
            print(f"  {os_type:<10} {count:>3}")
        print()
        return 0

    # Build filters
    filters = {}
    if args.vendor:
        filters['vendor'] = args.vendor
    if args.device_type:
        filters['device_type'] = args.device_type
    if args.platform:
        filters['platform'] = args.platform

    # Search
    if filters:
        results = db.search(**filters)
        print(f"Found {len(results)} matching hardware specs")
    else:
        results = list(db._cache.values())
        print(f"Listing all {len(results)} hardware specs")

    print("=" * 80)
    print()

    if not results:
        print("No hardware found matching criteria.")
        return 0

    # Display results
    for spec in sorted(results, key=lambda s: (s.device_type, s.vendor, s.id)):
        print(f"{spec.id}")
        print(f"  Vendor:   {spec.vendor}")
        print(f"  Model:    {spec.model}")
        print(f"  Type:     {spec.device_type}")
        print(f"  Platform: {spec.platform}")

        if args.detail:
            print(f"  Architecture: {spec.architecture}")

            if spec.cores:
                core_str = f"{spec.cores} cores"
                if spec.threads and spec.threads != spec.cores:
                    core_str += f", {spec.threads} threads"
                if spec.e_cores:
                    core_str += f" ({spec.cores - spec.e_cores}P + {spec.e_cores}E)"
                print(f"  Cores:    {core_str}")

            if spec.base_frequency_ghz:
                freq_str = f"{spec.base_frequency_ghz:.1f} GHz"
                if spec.boost_frequency_ghz:
                    freq_str += f" - {spec.boost_frequency_ghz:.1f} GHz boost"
                print(f"  Frequency: {freq_str}")

            if spec.memory_type:
                mem_str = spec.memory_type
                if spec.peak_bandwidth_gbps:
                    mem_str += f", {spec.peak_bandwidth_gbps:.1f} GB/s"
                print(f"  Memory:   {mem_str}")

            if spec.isa_extensions:
                print(f"  ISA:      {', '.join(spec.isa_extensions[:5])}")

            if spec.theoretical_peaks:
                peaks_str = ", ".join([
                    f"{prec}={gflops:.0f}"
                    for prec, gflops in list(spec.theoretical_peaks.items())[:3]
                ])
                print(f"  Peaks:    {peaks_str} (GFLOPS/GIOPS)")

            print(f"  Mapper:   {spec.mapper_class}")

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

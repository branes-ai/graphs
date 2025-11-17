#!/usr/bin/env python3
"""
Query Hardware Database

Query and display hardware specifications from the database.

Usage:
    python scripts/hardware_db/query_hardware.py --id intel_i7_12700k
    python scripts/hardware_db/query_hardware.py --vendor Intel
    python scripts/hardware_db/query_hardware.py --device-type gpu --platform x86_64
    python scripts/hardware_db/query_hardware.py --id h100_sxm5 --export h100.json
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareDatabase


def display_spec(spec, detail=False):
    """Display hardware spec in human-readable format"""
    print(f"\n{spec.id}")
    print("=" * 80)
    print(f"Vendor:       {spec.vendor}")
    print(f"Model:        {spec.model}")
    print(f"Architecture: {spec.architecture}")
    print(f"Device Type:  {spec.device_type}")
    print(f"Platform:     {spec.platform}")
    print()

    if detail:
        # Core specifications
        if spec.cores:
            core_str = f"{spec.cores} cores"
            if spec.threads and spec.threads != spec.cores:
                core_str += f", {spec.threads} threads"
            if spec.e_cores:
                p_cores = spec.cores - spec.e_cores
                core_str += f" ({p_cores}P + {spec.e_cores}E)"
            print(f"Cores:        {core_str}")

        if spec.base_frequency_ghz:
            freq_str = f"{spec.base_frequency_ghz:.2f} GHz"
            if spec.boost_frequency_ghz:
                freq_str += f" - {spec.boost_frequency_ghz:.2f} GHz boost"
            print(f"Frequency:    {freq_str}")

        # GPU-specific
        if spec.cuda_cores:
            print(f"CUDA Cores:   {spec.cuda_cores}")
        if spec.tensor_cores:
            print(f"Tensor Cores: {spec.tensor_cores}")
        if spec.sms:
            print(f"SMs:          {spec.sms}")
        if spec.cuda_capability:
            print(f"CUDA Cap:     {spec.cuda_capability}")

        print()

        # Memory
        if spec.memory_type:
            mem_str = spec.memory_type
            if spec.memory_channels:
                mem_str += f", {spec.memory_channels} channels"
            if spec.memory_bus_width:
                mem_str += f", {spec.memory_bus_width}-bit bus"
            print(f"Memory:       {mem_str}")

        if spec.peak_bandwidth_gbps:
            print(f"Bandwidth:    {spec.peak_bandwidth_gbps:.1f} GB/s")

        print()

        # ISA extensions
        if spec.isa_extensions:
            ext_str = ", ".join(spec.isa_extensions[:10])
            if len(spec.isa_extensions) > 10:
                ext_str += f" ... ({len(spec.isa_extensions)} total)"
            print(f"ISA:          {ext_str}")

        # Special features
        if spec.special_features:
            feat_str = ", ".join(spec.special_features)
            print(f"Features:     {feat_str}")

        print()

        # Theoretical peaks
        if spec.theoretical_peaks:
            print("Theoretical Peaks:")
            for prec, gops in spec.theoretical_peaks.items():
                unit = "GIOPS" if prec.startswith('int') else "GFLOPS"
                print(f"  {prec:<8} {gops:>12.1f} {unit}")
            print()

        # Cache
        if any([spec.l1_cache_kb, spec.l2_cache_kb, spec.l3_cache_kb]):
            print("Cache:")
            if spec.l1_cache_kb:
                print(f"  L1: {spec.l1_cache_kb} KB")
            if spec.l2_cache_kb:
                print(f"  L2: {spec.l2_cache_kb} KB")
            if spec.l3_cache_kb:
                print(f"  L3: {spec.l3_cache_kb} KB")
            print()

        # Power
        if spec.tdp_watts:
            pwr_str = f"{spec.tdp_watts} W TDP"
            if spec.max_power_watts:
                pwr_str += f", {spec.max_power_watts} W max"
            print(f"Power:        {pwr_str}")
            print()

        # Mapper
        print(f"Mapper:       {spec.mapper_class}")
        if spec.mapper_config:
            print(f"Config:       {spec.mapper_config}")
        print()

        # Metadata
        if spec.release_date:
            print(f"Released:     {spec.release_date}")
        if spec.manufacturer_url:
            print(f"URL:          {spec.manufacturer_url}")
        if spec.data_source:
            print(f"Data Source:  {spec.data_source}")
        if spec.last_updated:
            print(f"Updated:      {spec.last_updated}")

        # Detection
        if spec.detection_patterns:
            print()
            print("Detection Patterns:")
            for pattern in spec.detection_patterns:
                print(f"  - {pattern}")

        # OS compatibility
        if spec.os_compatibility:
            print(f"OS Support:   {', '.join(spec.os_compatibility)}")

        # Notes
        if spec.notes:
            print()
            print(f"Notes:        {spec.notes}")

    else:
        # Compact view
        if spec.theoretical_peaks:
            # Show first 3 peaks
            peaks_list = list(spec.theoretical_peaks.items())[:3]
            peaks_str = ", ".join([
                f"{prec}={gops:.0f}"
                for prec, gops in peaks_list
            ])
            print(f"Peaks:        {peaks_str} (GFLOPS/GIOPS)")

        if spec.peak_bandwidth_gbps:
            print(f"Bandwidth:    {spec.peak_bandwidth_gbps:.1f} GB/s")

        print(f"Mapper:       {spec.mapper_class}")


def main():
    parser = argparse.ArgumentParser(
        description="Query hardware specifications from database"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--id",
        help="Query by hardware ID"
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
        "--export",
        type=Path,
        help="Export result to JSON file"
    )

    args = parser.parse_args()

    # Load database
    db = HardwareDatabase(args.db)
    print(f"Loading database from: {args.db}")
    db.load_all()
    print(f"Loaded {len(db._cache)} hardware specs")

    # Query by ID
    if args.id:
        spec = db.get(args.id)
        if spec:
            display_spec(spec, detail=args.detail)

            if args.export:
                with open(args.export, 'w') as f:
                    json.dump(spec.to_json(), f, indent=2)
                print()
                print(f"✓ Exported to: {args.export}")

            return 0
        else:
            print(f"\n✗ Hardware not found: {args.id}")
            return 1

    # Search by filters
    filters = {}
    if args.vendor:
        filters['vendor'] = args.vendor
    if args.device_type:
        filters['device_type'] = args.device_type
    if args.platform:
        filters['platform'] = args.platform

    if filters:
        results = db.search(**filters)
        print(f"\nFound {len(results)} matching hardware specs")
        print("=" * 80)

        if not results:
            print("\nNo hardware found matching criteria.")
            return 0

        for spec in sorted(results, key=lambda s: (s.device_type, s.vendor, s.id)):
            display_spec(spec, detail=args.detail)

        if args.export:
            export_data = [spec.to_json() for spec in results]
            with open(args.export, 'w') as f:
                json.dump(export_data, f, indent=2)
            print()
            print(f"✓ Exported {len(results)} specs to: {args.export}")

        return 0

    # No query specified
    print()
    print("No query specified. Use --id, --vendor, --device-type, or --platform")
    print("Examples:")
    print("  --id intel_i7_12700k")
    print("  --vendor Intel")
    print("  --device-type gpu")
    print("  --platform x86_64")
    return 1


if __name__ == "__main__":
    sys.exit(main())

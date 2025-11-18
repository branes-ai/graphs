#!/usr/bin/env python3
"""
Automated Hardware Detection and Database Registration

Automatically detects current hardware using py-cpuinfo/psutil,
optionally runs calibration benchmarks for peak performance,
and adds the complete specification to the database.

Usage:
    python scripts/hardware_db/auto_detect_and_add.py
    python scripts/hardware_db/auto_detect_and_add.py --with-calibration
    python scripts/hardware_db/auto_detect_and_add.py --bandwidth 50.0 --fp32-gflops 1000.0
    python scripts/hardware_db/auto_detect_and_add.py --overwrite
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareDatabase, HardwareSpec, REQUIRED_PRECISIONS
from graphs.hardware.database.detector import HardwareDetector

# Optional calibration support
try:
    from graphs.hardware.calibration import HardwareCalibrator
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False


def detect_and_create_spec(
    detector: HardwareDetector,
    bandwidth_override: Optional[float] = None,
    fp32_override: Optional[float] = None,
    with_calibration: bool = False
) -> Optional[HardwareSpec]:
    """
    Detect hardware and create complete HardwareSpec.

    Args:
        detector: HardwareDetector instance
        bandwidth_override: Manual peak bandwidth in GB/s (if known)
        fp32_override: Manual fp32 peak in GFLOPS (if known)
        with_calibration: Run calibration benchmarks for actual performance

    Returns:
        HardwareSpec instance or None if detection fails
    """
    print("=" * 80)
    print("Hardware Detection")
    print("=" * 80)
    print()

    # Detect CPU
    cpu = detector.detect_cpu()
    if not cpu:
        print("✗ Failed to detect CPU")
        return None

    print(f"✓ Detected CPU:")
    print(f"  Model:        {cpu.model_name}")
    print(f"  Vendor:       {cpu.vendor}")
    print(f"  Architecture: {cpu.architecture}")
    print(f"  Cores:        {cpu.cores}")
    print(f"  Threads:      {cpu.threads}")
    if cpu.e_cores:
        print(f"  E-cores:      {cpu.e_cores} (P-cores: {cpu.cores - cpu.e_cores})")
    if cpu.base_frequency_ghz:
        print(f"  Base Freq:    {cpu.base_frequency_ghz:.2f} GHz")
    if cpu.boost_frequency_ghz:
        print(f"  Boost Freq:   {cpu.boost_frequency_ghz:.2f} GHz")
    print()

    # Cache information
    if cpu.l1_dcache_kb or cpu.l2_cache_kb or cpu.l3_cache_kb:
        print(f"  Cache Information:")

        # L1 Data Cache
        if cpu.l1_dcache_kb:
            l1d_parts = [f"{cpu.l1_dcache_kb} KB"]
            if cpu.l1_dcache_associativity:
                l1d_parts.append(f"{cpu.l1_dcache_associativity}-way")
            if cpu.l1_cache_line_size_bytes:
                l1d_parts.append(f"{cpu.l1_cache_line_size_bytes}-byte line")
            print(f"    L1 Data:       {', '.join(l1d_parts)}")

        # L1 Instruction Cache
        if cpu.l1_icache_kb:
            l1i_parts = [f"{cpu.l1_icache_kb} KB"]
            if cpu.l1_icache_associativity:
                l1i_parts.append(f"{cpu.l1_icache_associativity}-way")
            if cpu.l1_cache_line_size_bytes:
                l1i_parts.append(f"{cpu.l1_cache_line_size_bytes}-byte line")
            print(f"    L1 Instr:      {', '.join(l1i_parts)}")

        # L2 Cache
        if cpu.l2_cache_kb:
            l2_parts = [f"{cpu.l2_cache_kb} KB"]
            if cpu.l2_cache_associativity:
                l2_parts.append(f"{cpu.l2_cache_associativity}-way")
            if cpu.l2_cache_line_size_bytes:
                l2_parts.append(f"{cpu.l2_cache_line_size_bytes}-byte line")
            print(f"    L2:            {', '.join(l2_parts)}")

        # L3 Cache
        if cpu.l3_cache_kb:
            l3_parts = [f"{cpu.l3_cache_kb} KB"]
            if cpu.l3_cache_associativity:
                l3_parts.append(f"{cpu.l3_cache_associativity}-way")
            if cpu.l3_cache_line_size_bytes:
                l3_parts.append(f"{cpu.l3_cache_line_size_bytes}-byte line")
            print(f"    L3:            {', '.join(l3_parts)}")

        print()

    # ISA extensions
    if cpu.isa_extensions:
        print(f"  ISA Extensions: {', '.join(cpu.isa_extensions)}")
        print()

    # Generate hardware ID
    hw_id = f"{cpu.vendor}_{cpu.model_name}".lower()
    hw_id = hw_id.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    hw_id = ''.join(c for c in hw_id if c.isalnum() or c == '_')

    print(f"Generated ID: {hw_id}")
    print()

    # Detect or prompt for bandwidth
    peak_bandwidth_gbps = bandwidth_override
    if peak_bandwidth_gbps is None:
        print("=" * 80)
        print("Memory Bandwidth")
        print("=" * 80)
        print()
        print("Memory bandwidth is required but cannot be auto-detected.")
        print("Please look up the specification for your CPU/memory:")
        print()
        print("Example values:")
        print("  - DDR4-3200 dual-channel:  ~51 GB/s")
        print("  - DDR4-3600 dual-channel:  ~58 GB/s")
        print("  - DDR5-4800 dual-channel:  ~77 GB/s")
        print("  - DDR5-5600 dual-channel:  ~90 GB/s")
        print("  - DDR5-6400 dual-channel:  ~102 GB/s")
        print()

        while True:
            bw_input = input("Enter peak memory bandwidth (GB/s): ").strip()
            try:
                peak_bandwidth_gbps = float(bw_input)
                if peak_bandwidth_gbps > 0:
                    break
                else:
                    print("  Bandwidth must be positive")
            except ValueError:
                print("  Please enter a valid number")
        print()

    # Theoretical performance or calibration
    theoretical_peaks = {}

    if with_calibration and CALIBRATION_AVAILABLE:
        print("=" * 80)
        print("Hardware Calibration (Running Benchmarks)")
        print("=" * 80)
        print()
        print("This will run microbenchmarks to measure actual peak performance.")
        print("This may take several minutes...")
        print()

        try:
            calibrator = HardwareCalibrator()
            calibration_result = calibrator.calibrate_all()

            print(f"✓ Calibration complete")
            print()
            print(f"Measured Performance:")
            for precision, gflops in calibration_result.items():
                print(f"  {precision}: {gflops:.1f} GFLOPS")
                theoretical_peaks[precision] = gflops
            print()

        except Exception as e:
            print(f"✗ Calibration failed: {e}")
            print("  Falling back to manual entry")
            with_calibration = False

    if not with_calibration:
        print("=" * 80)
        print("Theoretical Performance")
        print("=" * 80)
        print()
        print("Theoretical peak performance is required.")
        print()

        if fp32_override:
            theoretical_peaks['fp32'] = fp32_override
            print(f"Using provided fp32 peak: {fp32_override} GFLOPS")
        else:
            print("Please calculate or look up theoretical FP32 peak GFLOPS:")
            print()
            print("Formula: Cores × FMA_units × 2 ops/FMA × Frequency_GHz")
            print()
            print("Examples:")
            print(f"  - Intel i7-12700K (8P+4E cores, 3.6 GHz base):")
            print(f"    P-cores: 8 × 2 (AVX2) × 2 × 3.6 = 115 GFLOPS")
            print(f"    E-cores: 4 × 2 (AVX2) × 2 × 3.6 = 58 GFLOPS")
            print(f"    Total: ~173 GFLOPS")
            print()

            while True:
                fp32_input = input("Enter fp32 peak (GFLOPS): ").strip()
                try:
                    fp32_peak = float(fp32_input)
                    if fp32_peak > 0:
                        theoretical_peaks['fp32'] = fp32_peak
                        break
                    else:
                        print("  Peak must be positive")
                except ValueError:
                    print("  Please enter a valid number")
            print()

        # Optional: other precisions (only in interactive mode)
        if sys.stdin.isatty():
            print("Optional: Enter peaks for other precisions (press Enter to skip)")
            for prec in ['fp64', 'fp16', 'bf16', 'int64', 'int32', 'int16', 'int8']:
                value_input = input(f"  {prec} (GFLOPS/GIOPS): ").strip()
                if value_input:
                    try:
                        value = float(value_input)
                        if value >= 0:  # Allow 0 for unsupported precisions
                            theoretical_peaks[prec] = value
                    except ValueError:
                        pass
        print()

    # Ensure ALL required precisions are present (set unsupported to 0.0)
    print("Ensuring all required precisions are present...")
    for precision in REQUIRED_PRECISIONS:
        if precision not in theoretical_peaks:
            theoretical_peaks[precision] = 0.0
            print(f"  {precision}: 0.0 (not supported)")

    print()
    print("Final theoretical peaks:")
    for prec in REQUIRED_PRECISIONS:
        value = theoretical_peaks[prec]
        if value > 0:
            print(f"  {prec}: {value:.1f}")
        else:
            print(f"  {prec}: 0.0 (not supported)")
    print()

    # Detect memory configuration
    print("=" * 80)
    print("Memory Detection")
    print("=" * 80)
    print()

    memory = detector.detect_memory()
    memory_subsystem = None
    memory_type = "DDR4"  # default

    if memory and memory.channels:
        print(f"✓ Detected {len(memory.channels)} memory channel(s):")
        memory_type = memory.channels[0].type.upper()

        # Display detected memory
        for ch in memory.channels:
            print(f"  {ch.name}: {ch.size_gb:.0f} GB {ch.type.upper()}-{ch.speed_mts}")
        print(f"  Total: {memory.total_gb:.0f} GB")
        print()

        # Generate memory_subsystem from detected channels
        memory_channels = []
        channel_map = {}  # Map controller to channels

        for ch in memory.channels:
            # Extract channel number from locator (e.g., "Controller0-DIMM1" -> controller 0)
            controller_num = 0
            if ch.locator and 'Controller' in ch.locator:
                try:
                    controller_num = int(ch.locator.split('Controller')[1].split('-')[0])
                except (IndexError, ValueError):
                    pass

            # Group by controller to identify channels
            if controller_num not in channel_map:
                channel_map[controller_num] = []
            channel_map[controller_num].append(ch)

        # Create memory channel entries
        for controller_num, dimms in sorted(channel_map.items()):
            # Calculate per-channel bandwidth
            # Bandwidth (GB/s) = (MT/s * bus_width_bits) / 8 / 1000
            # DDR typically has 64-bit bus per channel
            speed_mts = dimms[0].speed_mts
            bus_width_bits = 64
            bandwidth_gbps = (speed_mts * bus_width_bits) / 8 / 1000

            # Sum memory size for this channel
            channel_size_gb = sum(d.size_gb for d in dimms)

            # Determine how many DIMM slots (assume 2 per channel is common)
            dimm_slots = 2
            dimms_populated = len(dimms)

            memory_channels.append({
                "name": f"Channel {controller_num}",
                "type": dimms[0].type,
                "size_gb": channel_size_gb,
                "frequency_mhz": speed_mts // 2,  # MT/s to MHz (DDR = double data rate)
                "data_rate_mts": speed_mts,
                "bus_width_bits": bus_width_bits,
                "bandwidth_gbps": bandwidth_gbps,
                "dimm_slots": dimm_slots,
                "dimms_populated": dimms_populated,
                "dimm_size_gb": dimms[0].size_gb,
                "ecc_enabled": dimms[0].ecc_enabled if dimms[0].ecc_enabled is not None else False,
                "rank_count": dimms[0].rank_count,
                "numa_node": 0,  # Default to NUMA node 0 for consumer CPUs
                "physical_position": controller_num
            })

        # Calculate totals
        total_memory_gb = sum(ch["size_gb"] for ch in memory_channels)
        total_bandwidth_gbps = sum(ch["bandwidth_gbps"] for ch in memory_channels)

        if bandwidth_override is not None:
            total_bandwidth_gbps = bandwidth_override

        # Create consolidated memory_subsystem structure
        memory_subsystem = {
            "total_size_gb": total_memory_gb,
            "peak_bandwidth_gbps": total_bandwidth_gbps,
            "memory_channels": memory_channels
        }

        print(f"Total memory: {total_memory_gb:.0f} GB")
        print(f"Peak bandwidth: {total_bandwidth_gbps:.1f} GB/s")
        print()

    elif memory:
        print(f"✓ Detected {memory.total_gb:.0f} GB total memory (no detailed channel info)")
        print()
    else:
        print("✗ Could not detect memory configuration")
        print()

    # Generate core_info with core_clusters for heterogeneous CPUs (P/E-cores)
    core_info = {
        "cores": cpu.cores,
        "threads": cpu.threads,
        "base_frequency_ghz": cpu.base_frequency_ghz,
        "boost_frequency_ghz": cpu.boost_frequency_ghz,
    }

    if cpu.e_cores and cpu.e_cores > 0:
        # This is a heterogeneous CPU (e.g., Intel 12th gen+ with P/E-cores)
        p_cores = cpu.cores - cpu.e_cores
        e_cores = cpu.e_cores

        print(f"✓ Detected heterogeneous CPU: {p_cores} P-cores + {e_cores} E-cores")
        print()

        # Determine specific core architectures based on generation
        p_core_arch = cpu.architecture
        e_core_arch = cpu.architecture

        # Intel-specific architecture mapping
        if cpu.vendor == "Intel":
            if "12th Gen" in cpu.model_name or "Alder Lake" in cpu.architecture:
                p_core_arch = "Golden Cove"
                e_core_arch = "Gracemont"
            elif "13th Gen" in cpu.model_name or "Raptor Lake" in cpu.architecture:
                p_core_arch = "Raptor Cove"
                e_core_arch = "Gracemont"
            elif "14th Gen" in cpu.model_name or "Meteor Lake" in cpu.architecture:
                p_core_arch = "Redwood Cove"
                e_core_arch = "Crestmont"

        # Detect SIMD width based on ISA extensions
        p_core_simd = 512 if 'AVX512' in cpu.isa_extensions else (256 if 'AVX2' in cpu.isa_extensions else 128)
        e_core_simd = 128  # E-cores typically have more limited SIMD

        # Create core clusters with detailed architecture info
        core_info["core_clusters"] = [
            {
                "name": "Performance Cores",
                "type": "performance",
                "count": p_cores,
                "architecture": p_core_arch,
                "base_frequency_ghz": cpu.base_frequency_ghz,
                "boost_frequency_ghz": cpu.boost_frequency_ghz,
                "has_hyperthreading": True,
                "simd_width_bits": p_core_simd
            },
            {
                "name": "Efficiency Cores",
                "type": "efficiency",
                "count": e_cores,
                "architecture": e_core_arch,
                "base_frequency_ghz": cpu.base_frequency_ghz,
                "boost_frequency_ghz": cpu.boost_frequency_ghz,
                "has_hyperthreading": False,
                "simd_width_bits": e_core_simd
            }
        ]

    # Generate auto-notes about detection
    notes = []
    notes.append(f"Auto-detected on {detector.os_type} {detector.platform_arch}")

    # Warn about suspicious values
    if cpu.cache_levels:
        for cache in cpu.cache_levels:
            if cache.get('line_size_bytes', 64) > 256:
                notes.append(f"WARNING: {cache['name']} line size ({cache['line_size_bytes']} bytes) seems unusually large - please verify")

    if cpu.base_frequency_ghz and cpu.base_frequency_ghz < 0.5:
        notes.append(f"WARNING: Base frequency ({cpu.base_frequency_ghz:.2f} GHz) seems low - CPU may be throttled")

    if bandwidth_override is None:
        notes.append("Memory bandwidth should be verified with STREAM benchmark")

    if fp32_override is None:
        notes.append("Theoretical peak FP32 performance should be verified with calibration")

    # Create consolidated system info
    system_info = {
        "vendor": cpu.vendor,
        "model": cpu.model_name,
        "architecture": cpu.architecture,
        "device_type": "cpu",
        "platform": detector.platform_arch,
        "os_compatibility": [detector.os_type],
        "isa_extensions": cpu.isa_extensions,
        "special_features": [],
        "notes": " | ".join(notes) if notes else None
    }

    # Generate smart mapper hints based on hardware characteristics
    mapper_hints = {}

    # Suggest tile sizes based on L1/L2 cache
    if cpu.cache_levels:
        l1_size = None
        l2_size = None
        for cache in cpu.cache_levels:
            if cache['level'] == 1 and cache.get('cache_type') == 'data':
                l1_size = cache.get('size_per_unit_kb')
            elif cache['level'] == 2:
                l2_size = cache.get('size_per_unit_kb')

        # Suggest tile size that fits in L2 with some headroom
        if l2_size and l2_size >= 256:
            # For matrix operations, suggest square tiles
            # Aim for ~1/4 of L2 size to leave room for multiple tiles
            target_kb = l2_size // 4
            # For FP32, 128x128 matrix = 64KB, 256x256 = 256KB, 512x512 = 1024KB
            if target_kb >= 1024:
                mapper_hints["preferred_tile_size"] = [512, 512]
            elif target_kb >= 256:
                mapper_hints["preferred_tile_size"] = [256, 256]
            else:
                mapper_hints["preferred_tile_size"] = [128, 128]

    # Add SIMD hints based on ISA
    if 'AVX512' in cpu.isa_extensions:
        mapper_hints["simd_hint"] = "AVX512 available - use 512-bit SIMD"
    elif 'AVX2' in cpu.isa_extensions:
        mapper_hints["simd_hint"] = "AVX2 available - use 256-bit SIMD"

    # Add parallelism hints
    mapper_hints["parallelism_hint"] = f"Use up to {cpu.threads} threads (physical cores: {cpu.cores})"

    # Create consolidated mapper info
    mapper_info = {
        "mapper_class": "CPUMapper",
        "mapper_config": {},
        "hints": mapper_hints if mapper_hints else None
    }

    # Create HardwareSpec
    spec = HardwareSpec(
        id=hw_id,

        # Consolidated system/platform information
        system=system_info,

        detection_patterns=[cpu.model_name],

        # Consolidated core information
        core_info=core_info,

        # Consolidated memory subsystem
        memory_subsystem=memory_subsystem,

        # Performance
        theoretical_peaks=theoretical_peaks,

        # On-chip memory hierarchy (cache subsystem)
        # Only use cache_levels - the structured format
        onchip_memory_hierarchy={
            "cache_levels": cpu.cache_levels if cpu.cache_levels else [],
        },

        # Consolidated mapper configuration
        mapper=mapper_info,

        data_source="detected" + (" + calibrated" if with_calibration else ""),
        last_updated=datetime.utcnow().isoformat() + "Z"
    )

    return spec


def main():
    parser = argparse.ArgumentParser(
        description="Automatically detect hardware and add to database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for bandwidth and performance)
  python scripts/hardware_db/auto_detect_and_add.py

  # Write JSON to file for manual review/editing (recommended)
  python scripts/hardware_db/auto_detect_and_add.py --bandwidth 51.2 --fp32-gflops 115.2 -o my_cpu.json

  # Write to current directory with auto-generated filename
  python scripts/hardware_db/auto_detect_and_add.py --bandwidth 51.2 --fp32-gflops 115.2 -o .

  # With calibration benchmarks (automatic performance measurement)
  python scripts/hardware_db/auto_detect_and_add.py --with-calibration -o my_cpu.json

  # Direct database addition (skips manual review)
  python scripts/hardware_db/auto_detect_and_add.py --bandwidth 51.2 --fp32-gflops 115.2

  # Overwrite existing database entry
  python scripts/hardware_db/auto_detect_and_add.py --overwrite
        """
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        help="Peak memory bandwidth in GB/s (if known)"
    )
    parser.add_argument(
        "--fp32-gflops",
        type=float,
        help="FP32 peak performance in GFLOPS (if known)"
    )
    parser.add_argument(
        "--with-calibration",
        action="store_true",
        help="Run calibration benchmarks to measure actual performance"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing hardware entry"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and display spec but don't add to database"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write JSON to file instead of adding to database (e.g., my_cpu.json)"
    )

    args = parser.parse_args()

    if args.with_calibration and not CALIBRATION_AVAILABLE:
        print("✗ Error: Calibration module not available")
        print("  Please ensure calibration module is installed")
        return 1

    try:
        # Initialize detector
        detector = HardwareDetector()

        # Detect and create spec
        spec = detect_and_create_spec(
            detector,
            bandwidth_override=args.bandwidth,
            fp32_override=args.fp32_gflops,
            with_calibration=args.with_calibration
        )

        if not spec:
            return 1

        # Validate
        print("=" * 80)
        print("Validation")
        print("=" * 80)
        errors = spec.validate()

        if errors:
            print("⚠ Validation warnings:")
            for error in errors:
                print(f"  - {error}")
            print()
        else:
            print("✓ Specification is valid")
            print()

        # Review
        print("=" * 80)
        print("Review Hardware Specification")
        print("=" * 80)
        print(f"ID:       {spec.id}")

        # Get system info (consolidated fields)
        system_info = spec.get_system_info()
        if system_info:
            print(f"Vendor:   {system_info.vendor}")
            print(f"Model:    {system_info.model}")
            print(f"Type:     {system_info.device_type}")
            print(f"Platform: {system_info.platform}")

        # Get core info (consolidated fields)
        core_info = spec.get_core_info()
        if core_info:
            print(f"Cores:    {core_info.cores}")

        if spec.theoretical_peaks:
            peaks_str = ", ".join([f"{k}={v:.0f}" for k, v in list(spec.theoretical_peaks.items())[:3]])
            print(f"Peaks:    {peaks_str}")

        # Get memory info (consolidated fields)
        if spec.memory_subsystem and isinstance(spec.memory_subsystem, dict):
            peak_bw = spec.memory_subsystem.get('peak_bandwidth_gbps')
            if peak_bw:
                print(f"Bandwidth: {peak_bw:.1f} GB/s")
        print()

        # Dry run?
        if args.dry_run:
            print("Dry run - not adding to database")
            print()
            print("To add to database, run without --dry-run flag")
            return 0

        # Write JSON output?
        if args.output:
            output_path = args.output

            # Default filename if output is a directory
            if output_path.is_dir() or str(output_path).endswith('/'):
                output_path = output_path / f"{spec.id}.json"

            print()
            print(f"Writing JSON to: {output_path}")

            try:
                spec.to_json(output_path)
                print(f"✓ Successfully wrote hardware spec to {output_path}")
                print()
                print("Next steps:")
                print(f"  1. Review and edit the JSON file: {output_path}")
                print(f"  2. Fix any issues (e.g., cache line size)")
                print(f"  3. Move to database:")
                # Get system info for path construction
                system_info = spec.get_system_info()
                if system_info:
                    vendor_dir = system_info.vendor.lower().replace(' ', '_')
                    device_dir = system_info.device_type
                    print(f"     mkdir -p {args.db}/{device_dir}/{vendor_dir}")
                    print(f"     mv {output_path} {args.db}/{device_dir}/{vendor_dir}/{spec.id}.json")
                print(f"  4. Verify: python scripts/hardware_db/query_hardware.py --id {spec.id}")
                return 0
            except Exception as e:
                print(f"✗ Error writing JSON: {e}")
                return 1

        # Confirm
        confirm = input("Add this hardware to database? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled")
            return 1

        # Load database and add
        print()
        print("Adding to database...")
        db = HardwareDatabase(args.db)
        db.load_all()

        success = db.add(spec, overwrite=args.overwrite)

        if success:
            print()
            print(f"✓ Successfully added hardware: {spec.id}")
            spec_file = db._find_spec_file(spec.id)
            if spec_file:
                print(f"  File: {spec_file}")
            print()
            print("Next steps:")
            print(f"  1. Verify: python scripts/hardware_db/query_hardware.py --id {spec.id}")
            print("  2. Test detection: python scripts/hardware_db/detect_hardware.py")
            print("  3. Test mapping: python cli/analyze_comprehensive.py --model resnet18 --hardware", spec.id)
            return 0
        else:
            print()
            print(f"✗ Failed to add hardware")
            if not args.overwrite:
                print("  Hardware may already exist. Use --overwrite to replace.")
            return 1

    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

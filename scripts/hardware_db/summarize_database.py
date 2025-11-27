#!/usr/bin/env python3
"""
Summarize Hardware Database

Generate comprehensive reports and summaries of the hardware database content.
Provides multiple output formats and detail levels for reflecting on the data.

Usage:
    # Quick overview of all hardware
    python scripts/hardware_db/summarize_database.py

    # Detailed report for specific hardware
    python scripts/hardware_db/summarize_database.py --id jetson_orin_nano_gpu --detail

    # Compare theoretical peaks across all hardware
    python scripts/hardware_db/summarize_database.py --compare peaks

    # Compare memory bandwidth
    python scripts/hardware_db/summarize_database.py --compare bandwidth

    # Compare theoretical vs measured (calibration efficiency)
    python scripts/hardware_db/summarize_database.py --compare calibration

    # Export summary to markdown
    python scripts/hardware_db/summarize_database.py --output summary.md

    # Filter by device type
    python scripts/hardware_db/summarize_database.py --device-type gpu

    # Show only specific vendor
    python scripts/hardware_db/summarize_database.py --vendor NVIDIA
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareDatabase, HardwareSpec


def format_gflops(value: float) -> str:
    """Format GFLOPS/GIOPS value with appropriate unit"""
    if value >= 1000:
        return f"{value/1000:.1f} TFLOPS"
    elif value >= 1:
        return f"{value:.1f} GFLOPS"
    else:
        return f"{value*1000:.1f} MFLOPS"


def format_bandwidth(value: float) -> str:
    """Format bandwidth value"""
    if value >= 1000:
        return f"{value/1000:.2f} TB/s"
    else:
        return f"{value:.1f} GB/s"


def print_header(title: str, char: str = "=", width: int = 80):
    """Print a section header"""
    print()
    print(char * width)
    print(title)
    print(char * width)


def print_subheader(title: str, char: str = "-", width: int = 60):
    """Print a subsection header"""
    print()
    print(title)
    print(char * len(title))


def summarize_single_hardware(spec: HardwareSpec, detail: bool = False) -> str:
    """Generate summary for a single hardware spec"""
    lines = []

    # Header
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  {spec.model}")
    lines.append(f"  {spec.vendor} | {spec.device_type.upper()} | {spec.platform}")
    lines.append("=" * 80)
    lines.append("")

    # Basic Info
    lines.append("IDENTIFICATION")
    lines.append("-" * 40)
    lines.append(f"  ID:           {spec.id}")
    lines.append(f"  Vendor:       {spec.vendor}")
    lines.append(f"  Model:        {spec.model}")
    lines.append(f"  Architecture: {spec.architecture or 'N/A'}")
    lines.append(f"  Device Type:  {spec.device_type}")
    lines.append(f"  Platform:     {spec.platform}")
    lines.append("")

    # Core Configuration
    lines.append("COMPUTE RESOURCES")
    lines.append("-" * 40)

    if spec.device_type == 'gpu':
        if spec.cuda_cores:
            lines.append(f"  CUDA Cores:   {spec.cuda_cores:,}")
        if spec.tensor_cores:
            lines.append(f"  Tensor Cores: {spec.tensor_cores:,}")
        if spec.sms:
            lines.append(f"  SMs:          {spec.sms}")
        if spec.cuda_capability:
            lines.append(f"  CUDA Cap:     {spec.cuda_capability}")
    elif spec.device_type == 'cpu':
        if spec.cores:
            core_info = f"{spec.cores} cores"
            if spec.threads and spec.threads != spec.cores:
                core_info += f", {spec.threads} threads"
            if spec.e_cores:
                p_cores = spec.cores - spec.e_cores
                core_info += f" ({p_cores}P + {spec.e_cores}E)"
            lines.append(f"  Cores:        {core_info}")

    if spec.base_frequency_ghz:
        freq_str = f"{spec.base_frequency_ghz:.2f} GHz"
        if spec.boost_frequency_ghz:
            freq_str += f" (boost: {spec.boost_frequency_ghz:.2f} GHz)"
        lines.append(f"  Frequency:    {freq_str}")
    lines.append("")

    # Memory
    lines.append("MEMORY SUBSYSTEM")
    lines.append("-" * 40)
    if spec.memory_type:
        lines.append(f"  Type:         {spec.memory_type}")
    if spec.peak_bandwidth_gbps:
        lines.append(f"  Bandwidth:    {format_bandwidth(spec.peak_bandwidth_gbps)}")
    if spec.memory_channels:
        lines.append(f"  Channels:     {spec.memory_channels}")
    lines.append("")

    # Performance: Theoretical vs Measured
    has_calibration = spec.has_calibration_data()
    cal_summary = spec.get_calibration_summary() if has_calibration else None

    if spec.theoretical_peaks:
        if has_calibration:
            lines.append("PERFORMANCE: THEORETICAL vs MEASURED")
            lines.append("-" * 60)
            lines.append(f"  {'Precision':<10} {'Theoretical':>14} {'Measured':>14} {'Efficiency':>12}")
            lines.append("  " + "-" * 56)
        else:
            lines.append("THEORETICAL PEAKS (GFLOPS / GIOPS)")
            lines.append("-" * 40)

        # Sort by precision category
        float_precs = ['fp64', 'fp32', 'fp16', 'bf16', 'fp8', 'fp4']
        int_precs = ['int64', 'int32', 'int16', 'int8', 'int4']

        # Floating point
        has_float = False
        for prec in float_precs:
            if prec in spec.theoretical_peaks and spec.theoretical_peaks[prec] > 0:
                has_float = True
                theoretical = spec.theoretical_peaks[prec]

                if has_calibration:
                    measured = spec.get_measured_peak(prec)
                    efficiency = spec.get_efficiency(prec)
                    measured_str = f"{measured:.1f}" if measured else "-"
                    eff_str = f"{efficiency*100:.1f}%" if efficiency else "-"
                    lines.append(f"  {prec:<10} {theoretical:>14.1f} {measured_str:>14} {eff_str:>12}")
                else:
                    lines.append(f"  {prec:<8} {format_gflops(theoretical):>15}")

        if has_float:
            lines.append("")

        # Integer
        for prec in int_precs:
            if prec in spec.theoretical_peaks and spec.theoretical_peaks[prec] > 0:
                theoretical = spec.theoretical_peaks[prec]
                unit = "GIOPS" if theoretical < 1000 else "TOPS"

                if has_calibration:
                    measured = spec.get_measured_peak(prec)
                    efficiency = spec.get_efficiency(prec)
                    measured_str = f"{measured:.1f}" if measured else "-"
                    eff_str = f"{efficiency*100:.1f}%" if efficiency else "-"
                    lines.append(f"  {prec:<10} {theoretical:>14.1f} {measured_str:>14} {eff_str:>12}")
                else:
                    if theoretical >= 1000:
                        lines.append(f"  {prec:<8} {theoretical/1000:>12.1f} {unit}")
                    else:
                        lines.append(f"  {prec:<8} {theoretical:>12.1f} {unit}")
        lines.append("")

    # Calibration summary (if available and detailed)
    if has_calibration and cal_summary:
        lines.append("CALIBRATION METADATA")
        lines.append("-" * 40)
        if cal_summary.power_mode:
            lines.append(f"  Power Mode:   {cal_summary.power_mode}")
        if cal_summary.framework:
            lines.append(f"  Framework:    {cal_summary.framework}")
        if cal_summary.calibration_date:
            lines.append(f"  Date:         {cal_summary.calibration_date[:10]}")

        # Memory bandwidth
        if cal_summary.measured_bandwidth_gbps:
            theo_bw = spec.peak_bandwidth_gbps or 0
            bw_eff = cal_summary.bandwidth_efficiency
            if theo_bw > 0:
                lines.append(f"  Bandwidth:    {cal_summary.measured_bandwidth_gbps:.1f} / {theo_bw:.1f} GB/s ({bw_eff*100:.1f}% eff)")
            else:
                lines.append(f"  Bandwidth:    {cal_summary.measured_bandwidth_gbps:.1f} GB/s")

        # Aggregate efficiency
        if cal_summary.best_efficiency:
            lines.append(f"  Best Eff:     {cal_summary.best_efficiency*100:.1f}%")
        if cal_summary.avg_efficiency:
            lines.append(f"  Avg Eff:      {cal_summary.avg_efficiency*100:.1f}%")
        lines.append("")

    # Power
    if spec.tdp_watts or spec.max_power_watts:
        lines.append("POWER")
        lines.append("-" * 40)
        if spec.tdp_watts:
            lines.append(f"  TDP:          {spec.tdp_watts} W")
        if spec.max_power_watts:
            lines.append(f"  Max Power:    {spec.max_power_watts} W")
        lines.append("")

    # Cache (if detail mode)
    if detail and any([spec.l1_cache_kb, spec.l2_cache_kb, spec.l3_cache_kb]):
        lines.append("CACHE HIERARCHY")
        lines.append("-" * 40)
        if spec.l1_cache_kb:
            lines.append(f"  L1:           {spec.l1_cache_kb} KB")
        if spec.l2_cache_kb:
            lines.append(f"  L2:           {spec.l2_cache_kb} KB")
        if spec.l3_cache_kb:
            lines.append(f"  L3:           {spec.l3_cache_kb} KB")
        lines.append("")

    # ISA Extensions (if detail mode)
    if detail and spec.isa_extensions:
        lines.append("ISA EXTENSIONS")
        lines.append("-" * 40)
        # Format in columns
        exts = spec.isa_extensions
        for i in range(0, len(exts), 4):
            chunk = exts[i:i+4]
            lines.append("  " + ", ".join(chunk))
        lines.append("")

    # Special Features (if detail mode)
    if detail and spec.special_features:
        lines.append("SPECIAL FEATURES")
        lines.append("-" * 40)
        for feature in spec.special_features:
            lines.append(f"  - {feature}")
        lines.append("")

    # Mapper Configuration
    lines.append("MAPPER")
    lines.append("-" * 40)
    lines.append(f"  Class:        {spec.mapper_class}")
    if spec.mapper_config:
        lines.append(f"  Config:       {spec.mapper_config}")
    lines.append("")

    # Metadata
    if detail:
        lines.append("METADATA")
        lines.append("-" * 40)
        if spec.data_source:
            lines.append(f"  Data Source:  {spec.data_source}")
        if spec.last_updated:
            lines.append(f"  Last Updated: {spec.last_updated}")
        if spec.release_date:
            lines.append(f"  Released:     {spec.release_date}")
        if spec.manufacturer_url:
            lines.append(f"  URL:          {spec.manufacturer_url}")
        if spec.notes:
            lines.append(f"  Notes:        {spec.notes[:60]}...")
        lines.append("")

    return "\n".join(lines)


def generate_comparison_table(
    specs: List[HardwareSpec],
    compare_type: str = "peaks"
) -> str:
    """Generate a comparison table across multiple hardware specs"""
    lines = []

    if compare_type == "peaks":
        lines.append("")
        lines.append("THEORETICAL PEAK PERFORMANCE COMPARISON")
        lines.append("=" * 100)
        lines.append("")

        # Header row
        header = f"{'Hardware':<35} {'FP64':>10} {'FP32':>10} {'FP16':>10} {'INT8':>10} {'BW (GB/s)':>10}"
        lines.append(header)
        lines.append("-" * 100)

        # Sort by FP32 performance descending
        sorted_specs = sorted(
            specs,
            key=lambda s: s.theoretical_peaks.get('fp32', 0) if s.theoretical_peaks else 0,
            reverse=True
        )

        for spec in sorted_specs:
            name = f"{spec.vendor} {spec.model}"[:34]
            peaks = spec.theoretical_peaks or {}

            fp64 = peaks.get('fp64', 0)
            fp32 = peaks.get('fp32', 0)
            fp16 = peaks.get('fp16', 0)
            int8 = peaks.get('int8', 0)
            bw = spec.peak_bandwidth_gbps or 0

            # Format values
            fp64_str = f"{fp64:.0f}" if fp64 > 0 else "-"
            fp32_str = f"{fp32:.0f}" if fp32 > 0 else "-"
            fp16_str = f"{fp16:.0f}" if fp16 > 0 else "-"
            int8_str = f"{int8:.0f}" if int8 > 0 else "-"
            bw_str = f"{bw:.1f}" if bw > 0 else "-"

            row = f"{name:<35} {fp64_str:>10} {fp32_str:>10} {fp16_str:>10} {int8_str:>10} {bw_str:>10}"
            lines.append(row)

        lines.append("-" * 100)
        lines.append("Units: GFLOPS for FP*, GIOPS for INT*")
        lines.append("")

    elif compare_type == "bandwidth":
        lines.append("")
        lines.append("MEMORY BANDWIDTH COMPARISON")
        lines.append("=" * 80)
        lines.append("")

        header = f"{'Hardware':<40} {'Memory Type':<15} {'BW (GB/s)':>12} {'Arith Int':>10}"
        lines.append(header)
        lines.append("-" * 80)

        # Sort by bandwidth descending
        sorted_specs = sorted(
            specs,
            key=lambda s: s.peak_bandwidth_gbps or 0,
            reverse=True
        )

        for spec in sorted_specs:
            name = f"{spec.vendor} {spec.model}"[:39]
            mem_type = (spec.memory_type or "N/A")[:14]
            bw = spec.peak_bandwidth_gbps or 0

            # Compute arithmetic intensity ridge point (FP32)
            fp32 = spec.theoretical_peaks.get('fp32', 0) if spec.theoretical_peaks else 0
            ridge = fp32 / bw if bw > 0 else 0

            bw_str = f"{bw:.1f}" if bw > 0 else "-"
            ridge_str = f"{ridge:.1f}" if ridge > 0 else "-"

            row = f"{name:<40} {mem_type:<15} {bw_str:>12} {ridge_str:>10}"
            lines.append(row)

        lines.append("-" * 80)
        lines.append("Arith Int = Ridge point (FP32 GFLOPS / BW GB/s) - operations per byte")
        lines.append("")

    elif compare_type == "efficiency":
        lines.append("")
        lines.append("POWER EFFICIENCY COMPARISON")
        lines.append("=" * 90)
        lines.append("")

        header = f"{'Hardware':<35} {'FP32 (GFLOPS)':>14} {'TDP (W)':>10} {'GFLOPS/W':>12} {'Type':<8}"
        lines.append(header)
        lines.append("-" * 90)

        # Filter to specs with both FP32 and TDP
        valid_specs = [
            s for s in specs
            if s.tdp_watts and s.theoretical_peaks and s.theoretical_peaks.get('fp32', 0) > 0
        ]

        # Sort by efficiency descending
        sorted_specs = sorted(
            valid_specs,
            key=lambda s: s.theoretical_peaks.get('fp32', 0) / s.tdp_watts,
            reverse=True
        )

        for spec in sorted_specs:
            name = f"{spec.vendor} {spec.model}"[:34]
            fp32 = spec.theoretical_peaks.get('fp32', 0)
            tdp = spec.tdp_watts
            efficiency = fp32 / tdp if tdp > 0 else 0

            row = f"{name:<35} {fp32:>14.1f} {tdp:>10.1f} {efficiency:>12.2f} {spec.device_type:<8}"
            lines.append(row)

        if not sorted_specs:
            lines.append("  (No hardware with both FP32 peaks and TDP data)")

        lines.append("-" * 90)
        lines.append("")

    elif compare_type == "calibration":
        lines.append("")
        lines.append("THEORETICAL vs MEASURED PERFORMANCE COMPARISON")
        lines.append("=" * 110)
        lines.append("")

        header = f"{'Hardware':<30} {'Prec':>5} {'Theoretical':>12} {'Measured':>12} {'Efficiency':>10} {'BW Eff':>8} {'Calibrated':<12}"
        lines.append(header)
        lines.append("-" * 110)

        # Filter to specs with calibration data
        calibrated_specs = [s for s in specs if s.has_calibration_data()]
        uncalibrated_specs = [s for s in specs if not s.has_calibration_data()]

        # Sort calibrated by FP32 efficiency descending
        sorted_calibrated = sorted(
            calibrated_specs,
            key=lambda s: s.get_efficiency('fp32') or 0,
            reverse=True
        )

        for spec in sorted_calibrated:
            name = f"{spec.vendor} {spec.model}"[:29]
            cal_summary = spec.get_calibration_summary()

            # Show FP32 and FP16 rows
            for prec in ['fp32', 'fp16']:
                theoretical = spec.theoretical_peaks.get(prec, 0) if spec.theoretical_peaks else 0
                if theoretical == 0:
                    continue

                measured = spec.get_measured_peak(prec) or 0
                efficiency = spec.get_efficiency(prec) or 0
                bw_eff = cal_summary.bandwidth_efficiency if cal_summary else 0
                cal_date = cal_summary.calibration_date[:10] if cal_summary and cal_summary.calibration_date else "-"

                theo_str = f"{theoretical:.0f}"
                meas_str = f"{measured:.0f}" if measured > 0 else "-"
                eff_str = f"{efficiency*100:.1f}%" if efficiency > 0 else "-"
                bw_eff_str = f"{bw_eff*100:.0f}%" if bw_eff and prec == 'fp32' else ""

                row = f"{name:<30} {prec:>5} {theo_str:>12} {meas_str:>12} {eff_str:>10} {bw_eff_str:>8} {cal_date:<12}"
                lines.append(row)
                name = ""  # Only show name on first row

        if uncalibrated_specs:
            lines.append("")
            lines.append("NOT YET CALIBRATED:")
            for spec in uncalibrated_specs:
                name = f"{spec.vendor} {spec.model}"
                fp32 = spec.theoretical_peaks.get('fp32', 0) if spec.theoretical_peaks else 0
                lines.append(f"  {name:<40} FP32: {fp32:.0f} GFLOPS (theoretical)")

        lines.append("-" * 110)
        lines.append("")

    return "\n".join(lines)


def generate_overview_report(specs: List[HardwareSpec]) -> str:
    """Generate a high-level overview of all hardware in database"""
    lines = []

    lines.append("")
    lines.append("=" * 80)
    lines.append("HARDWARE DATABASE OVERVIEW")
    lines.append("=" * 80)
    lines.append("")

    # Statistics
    lines.append(f"Total hardware specifications: {len(specs)}")
    lines.append("")

    # By device type
    by_type = {}
    for spec in specs:
        by_type.setdefault(spec.device_type, []).append(spec)

    lines.append("BY DEVICE TYPE")
    lines.append("-" * 40)
    for dtype, type_specs in sorted(by_type.items()):
        lines.append(f"  {dtype.upper():<10} {len(type_specs):>3} specs")
    lines.append("")

    # By vendor
    by_vendor = {}
    for spec in specs:
        by_vendor.setdefault(spec.vendor, []).append(spec)

    lines.append("BY VENDOR")
    lines.append("-" * 40)
    for vendor, vendor_specs in sorted(by_vendor.items()):
        types = set(s.device_type for s in vendor_specs)
        types_str = ", ".join(sorted(types))
        lines.append(f"  {vendor:<20} {len(vendor_specs):>3} specs ({types_str})")
    lines.append("")

    # By platform
    by_platform = {}
    for spec in specs:
        by_platform.setdefault(spec.platform, []).append(spec)

    lines.append("BY PLATFORM")
    lines.append("-" * 40)
    for platform, platform_specs in sorted(by_platform.items()):
        lines.append(f"  {platform:<10} {len(platform_specs):>3} specs")
    lines.append("")

    # Quick list by type
    for dtype, type_specs in sorted(by_type.items()):
        lines.append(f"{dtype.upper()} HARDWARE")
        lines.append("-" * 60)

        # Sort by vendor, then model
        sorted_type_specs = sorted(type_specs, key=lambda s: (s.vendor, s.model))

        for spec in sorted_type_specs:
            fp32 = spec.theoretical_peaks.get('fp32', 0) if spec.theoretical_peaks else 0
            bw = spec.peak_bandwidth_gbps or 0

            name = f"{spec.vendor} {spec.model}"[:45]
            fp32_str = f"{fp32:.0f} GFLOPS" if fp32 > 0 else "N/A"
            bw_str = f"{bw:.1f} GB/s" if bw > 0 else "N/A"

            lines.append(f"  {name:<46} {fp32_str:>15} {bw_str:>12}")

        lines.append("")

    return "\n".join(lines)


def export_markdown(specs: List[HardwareSpec], output_path: Path):
    """Export database summary to Markdown format"""
    lines = []

    lines.append("# Hardware Database Summary")
    lines.append("")
    lines.append(f"Generated from hardware_database/ ({len(specs)} specifications)")
    lines.append("")

    # Statistics
    lines.append("## Overview")
    lines.append("")

    by_type = {}
    for spec in specs:
        by_type.setdefault(spec.device_type, []).append(spec)

    lines.append("| Device Type | Count |")
    lines.append("|-------------|-------|")
    for dtype, type_specs in sorted(by_type.items()):
        lines.append(f"| {dtype.upper()} | {len(type_specs)} |")
    lines.append("")

    # Performance comparison table
    lines.append("## Theoretical Peak Performance")
    lines.append("")
    lines.append("| Hardware | FP64 | FP32 | FP16 | INT8 | Bandwidth |")
    lines.append("|----------|------|------|------|------|-----------|")

    sorted_specs = sorted(
        specs,
        key=lambda s: s.theoretical_peaks.get('fp32', 0) if s.theoretical_peaks else 0,
        reverse=True
    )

    for spec in sorted_specs:
        name = f"{spec.vendor} {spec.model}"
        peaks = spec.theoretical_peaks or {}

        fp64 = peaks.get('fp64', 0)
        fp32 = peaks.get('fp32', 0)
        fp16 = peaks.get('fp16', 0)
        int8 = peaks.get('int8', 0)
        bw = spec.peak_bandwidth_gbps or 0

        fp64_str = format_gflops(fp64) if fp64 > 0 else "-"
        fp32_str = format_gflops(fp32) if fp32 > 0 else "-"
        fp16_str = format_gflops(fp16) if fp16 > 0 else "-"
        int8_str = f"{int8:.0f} GIOPS" if int8 > 0 else "-"
        bw_str = format_bandwidth(bw) if bw > 0 else "-"

        lines.append(f"| {name} | {fp64_str} | {fp32_str} | {fp16_str} | {int8_str} | {bw_str} |")

    lines.append("")

    # Individual hardware sections
    for dtype, type_specs in sorted(by_type.items()):
        lines.append(f"## {dtype.upper()} Hardware")
        lines.append("")

        for spec in sorted(type_specs, key=lambda s: s.model):
            lines.append(f"### {spec.vendor} {spec.model}")
            lines.append("")
            lines.append(f"- **ID:** `{spec.id}`")
            lines.append(f"- **Architecture:** {spec.architecture or 'N/A'}")
            lines.append(f"- **Platform:** {spec.platform}")

            if spec.peak_bandwidth_gbps:
                lines.append(f"- **Memory Bandwidth:** {format_bandwidth(spec.peak_bandwidth_gbps)}")

            if spec.theoretical_peaks:
                lines.append("- **Theoretical Peaks:**")
                for prec, value in spec.theoretical_peaks.items():
                    if value > 0:
                        unit = "GIOPS" if prec.startswith('int') else "GFLOPS"
                        lines.append(f"  - {prec}: {value:.1f} {unit}")

            lines.append("")

    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Exported summary to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and report on hardware database content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--id",
        help="Show detailed summary for specific hardware ID"
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
        help="Filter by platform (x86_64, aarch64)"
    )
    parser.add_argument(
        "--compare",
        choices=["peaks", "bandwidth", "efficiency", "calibration"],
        help="Generate comparison table (calibration shows theoretical vs measured)"
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Export to file (supports .md for Markdown)"
    )

    args = parser.parse_args()

    # Load database
    db = HardwareDatabase(args.db)
    print(f"Loading database from: {args.db}")
    db.load_all()
    print(f"Loaded {len(db._cache)} hardware specifications")

    # Get all specs or filter
    if args.id:
        spec = db.get(args.id)
        if spec:
            print(summarize_single_hardware(spec, detail=args.detail))
            return 0
        else:
            print(f"\nHardware not found: {args.id}")
            print("\nAvailable IDs:")
            for hw_id in sorted(db._cache.keys()):
                print(f"  {hw_id}")
            return 1

    # Build filters
    filters = {}
    if args.vendor:
        filters['vendor'] = args.vendor
    if args.device_type:
        filters['device_type'] = args.device_type
    if args.platform:
        filters['platform'] = args.platform

    # Get specs
    if filters:
        specs = db.search(**filters)
    else:
        specs = list(db._cache.values())

    if not specs:
        print("\nNo hardware found matching criteria.")
        return 0

    # Export if requested
    if args.output:
        if args.output.suffix == '.md':
            export_markdown(specs, args.output)
        else:
            # Default to text output
            with open(args.output, 'w') as f:
                if args.compare:
                    f.write(generate_comparison_table(specs, args.compare))
                else:
                    f.write(generate_overview_report(specs))
            print(f"Exported to: {args.output}")
        return 0

    # Generate and print report
    if args.compare:
        print(generate_comparison_table(specs, args.compare))
    elif args.detail:
        for spec in sorted(specs, key=lambda s: (s.device_type, s.vendor, s.model)):
            print(summarize_single_hardware(spec, detail=True))
    else:
        print(generate_overview_report(specs))

    return 0


if __name__ == "__main__":
    sys.exit(main())

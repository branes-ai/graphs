#!/usr/bin/env python
"""
Subgraph Inspection Tool - Diagnostic for Estimation Accuracy

Shows each subgraph with:
1. Workload characteristics (FLOPs, precision, memory bytes)
2. RAW roofline estimate (theoretical peak, no derates)
3. Each derating factor applied step-by-step
4. Final estimated latency
5. Measured latency (if calibration data available)

This tool helps identify which derating factors are causing estimation errors
and whether they are being applied correctly.

Usage:
    ./cli/inspect_subgraphs.py --model resnet18 --hardware Jetson-Orin-AGX

    # Show raw estimates without any derates
    ./cli/inspect_subgraphs.py --model resnet18 --hardware Jetson-Orin-AGX --raw

    # Compare against measured data
    ./cli/inspect_subgraphs.py --model resnet18 --hardware Jetson-Orin-AGX --measure

    # Output to file
    ./cli/inspect_subgraphs.py --model resnet18 --hardware Jetson-Orin-AGX --output report.txt
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch

from graphs.frontends.dynamo import trace_and_partition
from graphs.estimation.unified_analyzer import UnifiedAnalyzer
from graphs.hardware.resource_model import Precision
from graphs.core.structures import SubgraphDescriptor, BottleneckType


@dataclass
class SubgraphDiagnostic:
    """Diagnostic data for a single subgraph."""
    # Identity
    subgraph_id: str
    fusion_pattern: str
    num_ops: int
    node_names: List[str]

    # Workload characteristics
    total_flops: int
    total_macs: int
    input_bytes: int
    output_bytes: int
    weight_bytes: int
    total_bytes: int
    arithmetic_intensity: float

    # RAW estimates (theoretical peak, no derates)
    theoretical_peak_flops: float  # FLOPS
    theoretical_peak_bandwidth: float  # bytes/s
    raw_compute_time_us: float
    raw_memory_time_us: float
    raw_latency_us: float
    raw_bottleneck: str

    # Thermal profile derate
    thermal_efficiency: float
    thermal_derated_peak_flops: float
    thermal_derated_latency_us: float

    # Per-operation efficiency scale
    operation_efficiency_scale: float
    op_scaled_latency_us: float

    # Final estimate
    final_latency_us: float
    final_bottleneck: str

    # Measured (if available)
    measured_latency_us: Optional[float] = None
    error_pct: Optional[float] = None


def format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1e9:
        return f"{b/1e9:.2f} GB"
    elif b >= 1e6:
        return f"{b/1e6:.2f} MB"
    elif b >= 1e3:
        return f"{b/1e3:.2f} KB"
    return f"{b} B"


def format_flops(f: int) -> str:
    """Format FLOPs as human-readable string."""
    if f >= 1e12:
        return f"{f/1e12:.2f} TFLOPS"
    elif f >= 1e9:
        return f"{f/1e9:.2f} GFLOPS"
    elif f >= 1e6:
        return f"{f/1e6:.2f} MFLOPS"
    elif f >= 1e3:
        return f"{f/1e3:.2f} KFLOPS"
    return f"{f} FLOPS"


def format_time(us: float) -> str:
    """Format time in microseconds as human-readable string."""
    if us >= 1e6:
        return f"{us/1e6:.2f} s"
    elif us >= 1e3:
        return f"{us/1e3:.2f} ms"
    return f"{us:.2f} us"


def compute_operation_efficiency_scale(flops: int, hw_type: str) -> float:
    """
    Compute the per-operation efficiency scale factor.

    This is a copy of the logic from RooflineAnalyzer._get_operation_efficiency_scale
    to make the calculation explicit and inspectable.
    """
    if flops <= 0:
        return 1.0

    if hw_type == 'GPU':
        if flops < 10e6:
            return 0.4

        log_flops = math.log10(flops)

        if flops < 200e6:
            t = (log_flops - 7.0) / 1.3
            t = max(0.0, min(1.0, t))
            return 0.4 + 0.4 * t
        else:
            t = (log_flops - 8.3) / 0.48
            t = max(0.0, min(1.0, t))
            return 0.8 + 1.2 * t

    elif hw_type == 'CPU':
        if flops < 1e6:
            return 0.15

        log_flops = math.log10(flops)

        if flops < 10e6:
            t = (log_flops - 6.0) / 1.0
            t = max(0.0, min(1.0, t))
            return 0.15 + 0.10 * t
        elif flops < 100e6:
            t = (log_flops - 7.0) / 1.0
            t = max(0.0, min(1.0, t))
            return 0.25 + 0.45 * t
        else:
            t = (log_flops - 8.0) / 1.0
            t = max(0.0, min(1.0, t))
            return 0.70 + 0.30 * t

    return 1.0


def analyze_subgraph(
    sg: SubgraphDescriptor,
    theoretical_peak_flops: float,
    theoretical_bandwidth: float,
    thermal_efficiency: float,
    hw_type: str,
) -> SubgraphDiagnostic:
    """Analyze a single subgraph and compute all diagnostic values."""

    total_bytes = sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
    ai = sg.total_flops / total_bytes if total_bytes > 0 else 0.0

    # RAW estimates (no derates)
    raw_compute_time = sg.total_flops / theoretical_peak_flops if theoretical_peak_flops > 0 else 0
    raw_memory_time = total_bytes / theoretical_bandwidth if theoretical_bandwidth > 0 else 0
    raw_latency = max(raw_compute_time, raw_memory_time)
    raw_bottleneck = "COMPUTE" if raw_compute_time > raw_memory_time * 1.1 else (
        "MEMORY" if raw_memory_time > raw_compute_time * 1.1 else "BALANCED"
    )

    # Thermal derate
    thermal_derated_peak = theoretical_peak_flops * thermal_efficiency
    thermal_derated_compute = sg.total_flops / thermal_derated_peak if thermal_derated_peak > 0 else 0
    thermal_derated_memory = total_bytes / (theoretical_bandwidth * thermal_efficiency) if theoretical_bandwidth > 0 else 0
    thermal_derated_latency = max(thermal_derated_compute, thermal_derated_memory)

    # Per-operation efficiency scale
    op_scale = compute_operation_efficiency_scale(sg.total_flops, hw_type)
    op_scaled_peak = thermal_derated_peak * op_scale
    op_scaled_compute = sg.total_flops / op_scaled_peak if op_scaled_peak > 0 else 0
    op_scaled_memory = total_bytes / (theoretical_bandwidth * thermal_efficiency * op_scale) if theoretical_bandwidth > 0 else 0
    op_scaled_latency = max(op_scaled_compute, op_scaled_memory)

    final_bottleneck = "COMPUTE" if op_scaled_compute > op_scaled_memory * 1.1 else (
        "MEMORY" if op_scaled_memory > op_scaled_compute * 1.1 else "BALANCED"
    )

    return SubgraphDiagnostic(
        subgraph_id=sg.subgraph_id,
        fusion_pattern=sg.fusion_pattern,
        num_ops=sg.num_operators,
        node_names=sg.node_names[:3],  # First 3 for brevity

        total_flops=sg.total_flops,
        total_macs=sg.total_macs,
        input_bytes=sg.total_input_bytes,
        output_bytes=sg.total_output_bytes,
        weight_bytes=sg.total_weight_bytes,
        total_bytes=total_bytes,
        arithmetic_intensity=ai,

        theoretical_peak_flops=theoretical_peak_flops,
        theoretical_peak_bandwidth=theoretical_bandwidth,
        raw_compute_time_us=raw_compute_time * 1e6,
        raw_memory_time_us=raw_memory_time * 1e6,
        raw_latency_us=raw_latency * 1e6,
        raw_bottleneck=raw_bottleneck,

        thermal_efficiency=thermal_efficiency,
        thermal_derated_peak_flops=thermal_derated_peak,
        thermal_derated_latency_us=thermal_derated_latency * 1e6,

        operation_efficiency_scale=op_scale,
        op_scaled_latency_us=op_scaled_latency * 1e6,

        final_latency_us=op_scaled_latency * 1e6,
        final_bottleneck=final_bottleneck,
    )


def print_subgraph_diagnostic(diag: SubgraphDiagnostic, index: int, show_raw: bool = False) -> str:
    """Format a single subgraph diagnostic as a string."""
    lines = []

    # Header
    lines.append(f"{'='*80}")
    lines.append(f"SUBGRAPH {index}: {diag.fusion_pattern}")
    lines.append(f"{'='*80}")

    # Workload characteristics
    lines.append("")
    lines.append("WORKLOAD CHARACTERISTICS:")
    lines.append(f"  Operators:  {diag.num_ops}")
    lines.append(f"  FLOPs:      {diag.total_flops:>15,} ({format_flops(diag.total_flops)})")
    lines.append(f"  MACs:       {diag.total_macs:>15,}")
    lines.append(f"  Input:      {diag.input_bytes:>15,} bytes ({format_bytes(diag.input_bytes)})")
    lines.append(f"  Output:     {diag.output_bytes:>15,} bytes ({format_bytes(diag.output_bytes)})")
    lines.append(f"  Weights:    {diag.weight_bytes:>15,} bytes ({format_bytes(diag.weight_bytes)})")
    lines.append(f"  Total I/O:  {diag.total_bytes:>15,} bytes ({format_bytes(diag.total_bytes)})")
    lines.append(f"  Arith Int:  {diag.arithmetic_intensity:>15.2f} FLOPs/byte")

    # RAW estimates
    lines.append("")
    lines.append("RAW ROOFLINE (theoretical peak, NO derates):")
    lines.append(f"  Peak FLOPs:     {diag.theoretical_peak_flops/1e12:.2f} TFLOPS")
    lines.append(f"  Peak BW:        {diag.theoretical_peak_bandwidth/1e9:.2f} GB/s")
    lines.append(f"  Compute time:   {format_time(diag.raw_compute_time_us)}")
    lines.append(f"  Memory time:    {format_time(diag.raw_memory_time_us)}")
    lines.append(f"  RAW latency:    {format_time(diag.raw_latency_us)} [{diag.raw_bottleneck}]")

    if not show_raw:
        # Derating chain
        lines.append("")
        lines.append("DERATING CHAIN:")

        lines.append(f"  1. Thermal efficiency:        {diag.thermal_efficiency:.2f}x")
        lines.append(f"     -> Derated peak:           {diag.thermal_derated_peak_flops/1e12:.2f} TFLOPS")
        lines.append(f"     -> Derated latency:        {format_time(diag.thermal_derated_latency_us)}")

        lines.append(f"  2. Per-op efficiency scale:   {diag.operation_efficiency_scale:.2f}x")
        lines.append(f"     -> Final latency:          {format_time(diag.op_scaled_latency_us)} [{diag.final_bottleneck}]")

        # Total derate
        total_derate = diag.thermal_efficiency * diag.operation_efficiency_scale
        slowdown = diag.final_latency_us / diag.raw_latency_us if diag.raw_latency_us > 0 else 0
        lines.append("")
        lines.append(f"  TOTAL DERATE:                 {total_derate:.3f}x (effective peak)")
        lines.append(f"  SLOWDOWN vs RAW:              {slowdown:.2f}x")

    # Measured comparison
    if diag.measured_latency_us is not None:
        lines.append("")
        lines.append("MEASURED COMPARISON:")
        lines.append(f"  Measured:       {format_time(diag.measured_latency_us)}")
        lines.append(f"  Estimated:      {format_time(diag.final_latency_us)}")
        lines.append(f"  Error:          {diag.error_pct:+.1f}%")

    return "\n".join(lines)


def print_summary(diagnostics: List[SubgraphDiagnostic], model_name: str, hardware_name: str) -> str:
    """Print summary statistics."""
    lines = []

    lines.append("")
    lines.append("=" * 80)
    lines.append(f"SUMMARY: {model_name} on {hardware_name}")
    lines.append("=" * 80)

    total_flops = sum(d.total_flops for d in diagnostics)
    total_bytes = sum(d.total_bytes for d in diagnostics)
    total_raw_latency = sum(d.raw_latency_us for d in diagnostics)
    total_final_latency = sum(d.final_latency_us for d in diagnostics)

    lines.append("")
    lines.append("TOTALS:")
    lines.append(f"  Subgraphs:        {len(diagnostics)}")
    lines.append(f"  Total FLOPs:      {format_flops(total_flops)}")
    lines.append(f"  Total I/O:        {format_bytes(total_bytes)}")
    lines.append(f"  Avg AI:           {total_flops/total_bytes:.2f} FLOPs/byte" if total_bytes > 0 else "  Avg AI:           N/A")

    lines.append("")
    lines.append("LATENCY:")
    lines.append(f"  RAW (theoretical):    {format_time(total_raw_latency)}")
    lines.append(f"  Final (with derates): {format_time(total_final_latency)}")
    lines.append(f"  Slowdown factor:      {total_final_latency/total_raw_latency:.2f}x" if total_raw_latency > 0 else "")

    # Distribution of bottlenecks
    compute_bound = sum(1 for d in diagnostics if d.final_bottleneck == "COMPUTE")
    memory_bound = sum(1 for d in diagnostics if d.final_bottleneck == "MEMORY")
    balanced = sum(1 for d in diagnostics if d.final_bottleneck == "BALANCED")

    lines.append("")
    lines.append("BOTTLENECK DISTRIBUTION:")
    lines.append(f"  Compute-bound:  {compute_bound:>3} ({100*compute_bound/len(diagnostics):.1f}%)")
    lines.append(f"  Memory-bound:   {memory_bound:>3} ({100*memory_bound/len(diagnostics):.1f}%)")
    lines.append(f"  Balanced:       {balanced:>3} ({100*balanced/len(diagnostics):.1f}%)")

    # Per-op efficiency distribution
    scales = [d.operation_efficiency_scale for d in diagnostics]
    lines.append("")
    lines.append("PER-OPERATION EFFICIENCY SCALE DISTRIBUTION:")
    lines.append(f"  Min:    {min(scales):.2f}")
    lines.append(f"  Max:    {max(scales):.2f}")
    lines.append(f"  Avg:    {sum(scales)/len(scales):.2f}")

    # Top 5 by latency
    sorted_by_latency = sorted(diagnostics, key=lambda d: d.final_latency_us, reverse=True)
    lines.append("")
    lines.append("TOP 5 SUBGRAPHS BY LATENCY:")
    for i, d in enumerate(sorted_by_latency[:5], 1):
        pct = 100 * d.final_latency_us / total_final_latency if total_final_latency > 0 else 0
        lines.append(f"  {i}. {d.fusion_pattern[:40]:<40} {format_time(d.final_latency_us):>12} ({pct:.1f}%)")

    # Top 5 by op-scale (lowest = most penalized)
    sorted_by_scale = sorted(diagnostics, key=lambda d: d.operation_efficiency_scale)
    lines.append("")
    lines.append("TOP 5 MOST PENALIZED BY OP-SCALE (lowest scale):")
    for i, d in enumerate(sorted_by_scale[:5], 1):
        lines.append(f"  {i}. {d.fusion_pattern[:40]:<40} scale={d.operation_efficiency_scale:.2f} ({d.total_flops/1e6:.1f}M FLOPs)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Inspect subgraphs with diagnostic information for estimation accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model resnet18 --hardware Jetson-Orin-AGX
  %(prog)s --model vit_b_16 --hardware Jetson-Orin-AGX --raw
  %(prog)s --model mobilenet_v2 --hardware i7-12700K --top 10
""")
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--hardware', required=True, help='Hardware name')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32', help='Precision')
    parser.add_argument('--raw', action='store_true', help='Show only RAW estimates (no derates)')
    parser.add_argument('--top', type=int, default=None, help='Show only top N subgraphs by latency')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--quiet', action='store_true', help='Only show summary')

    args = parser.parse_args()

    # Map precision
    precision = Precision.FP32 if args.precision == 'fp32' else Precision.FP16

    # Create analyzer and get results
    print(f"Analyzing {args.model} on {args.hardware}...", file=sys.stderr)

    analyzer = UnifiedAnalyzer(verbose=False)

    try:
        result = analyzer.analyze_model(
            model_name=args.model,
            hardware_name=args.hardware,
            precision=precision,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Get resource model info
    partition_report = result.partition_report
    roofline_report = result.roofline_report

    # Get theoretical peaks from roofline report
    theoretical_peak_flops = roofline_report.peak_flops
    theoretical_bandwidth = roofline_report.peak_bandwidth

    # We need to extract the thermal efficiency from the hardware model
    # This requires looking at how the roofline analyzer was configured
    # For now, estimate it from the relationship between theoretical and effective
    hw_type = "GPU" if "Jetson" in args.hardware or "H100" in args.hardware or "A100" in args.hardware else "CPU"

    # Get thermal efficiency (this is approximate - we should expose it better)
    # The roofline_report.peak_flops already has thermal derate applied if thermal profile was used
    # We need the raw theoretical to compute the derate
    from graphs.hardware.mappers.gpu import (
        create_jetson_orin_agx_64gb_mapper,
        create_h100_sxm5_80gb_mapper,
    )
    from graphs.hardware.mappers.cpu import create_i7_12700k_mapper

    if "Jetson-Orin-AGX" in args.hardware:
        mapper = create_jetson_orin_agx_64gb_mapper("30W")
        rm = mapper.resource_model
        # Get theoretical from precision profile
        if precision in rm.precision_profiles:
            theoretical_peak_flops = rm.precision_profiles[precision].peak_ops_per_sec
        # Get thermal efficiency
        thermal_eff = 0.35  # Default for 30W FP32
        if rm.thermal_operating_points and "30W" in rm.thermal_operating_points:
            thermal_point = rm.thermal_operating_points["30W"]
            if precision in thermal_point.performance_specs:
                thermal_eff = thermal_point.performance_specs[precision].efficiency_factor
        theoretical_bandwidth = rm.peak_bandwidth
    elif "i7-12700K" in args.hardware:
        mapper = create_i7_12700k_mapper()
        rm = mapper.resource_model
        if precision in rm.precision_profiles:
            theoretical_peak_flops = rm.precision_profiles[precision].peak_ops_per_sec
        thermal_eff = 0.50  # From calibration
        if rm.thermal_operating_points:
            default_profile = rm.default_thermal_profile
            if default_profile and default_profile in rm.thermal_operating_points:
                thermal_point = rm.thermal_operating_points[default_profile]
                if precision in thermal_point.performance_specs:
                    thermal_eff = thermal_point.performance_specs[precision].efficiency_factor
        theoretical_bandwidth = rm.peak_bandwidth
    else:
        # Generic - use roofline values
        thermal_eff = 1.0

    # Analyze each subgraph
    diagnostics = []
    for sg in partition_report.subgraphs:
        diag = analyze_subgraph(
            sg,
            theoretical_peak_flops=theoretical_peak_flops,
            theoretical_bandwidth=theoretical_bandwidth,
            thermal_efficiency=thermal_eff,
            hw_type=hw_type,
        )
        diagnostics.append(diag)

    # Sort by latency if --top specified
    if args.top:
        diagnostics_to_show = sorted(diagnostics, key=lambda d: d.final_latency_us, reverse=True)[:args.top]
    else:
        diagnostics_to_show = diagnostics

    # Generate output
    output_lines = []

    output_lines.append("=" * 80)
    output_lines.append(f"SUBGRAPH INSPECTION: {args.model} on {args.hardware}")
    output_lines.append(f"Batch size: {args.batch_size}, Precision: {args.precision.upper()}")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append(f"Theoretical peak FLOPs: {theoretical_peak_flops/1e12:.2f} TFLOPS")
    output_lines.append(f"Theoretical peak BW:    {theoretical_bandwidth/1e9:.2f} GB/s")
    output_lines.append(f"Thermal efficiency:     {thermal_eff:.2f}")
    output_lines.append(f"Hardware type:          {hw_type}")

    if not args.quiet:
        for i, diag in enumerate(diagnostics_to_show, 1):
            output_lines.append("")
            output_lines.append(print_subgraph_diagnostic(diag, i, show_raw=args.raw))

    # Summary
    output_lines.append(print_summary(diagnostics, args.model, args.hardware))

    # Compare with actual result
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("COMPARISON WITH UNIFIED ANALYZER RESULT")
    output_lines.append("=" * 80)
    output_lines.append(f"  Unified analyzer total latency: {result.total_latency_ms:.3f} ms")
    our_total = sum(d.final_latency_us for d in diagnostics) / 1000
    output_lines.append(f"  Our diagnostic total latency:   {our_total:.3f} ms")
    if result.total_latency_ms > 0:
        diff_pct = (our_total - result.total_latency_ms) / result.total_latency_ms * 100
        output_lines.append(f"  Difference:                     {diff_pct:+.1f}%")

    output = "\n".join(output_lines)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == '__main__':
    sys.exit(main())

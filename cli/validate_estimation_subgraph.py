#!/usr/bin/env python
"""
Per-Subgraph Estimation Validation Tool

Uses torch.fx.Interpreter to measure actual per-node latency, then maps
node times to subgraphs from the fusion partitioner. Compares measured
per-subgraph latency against roofline estimates to identify which fusion
patterns contribute most to the total estimation error.

Usage:
    # Identify which subgraphs cause ViT estimation errors
    ./cli/validate_estimation_subgraph.py --model vit_b_16 --hardware i7-12700K

    # Validate ResNet-18 estimation on Jetson GPU
    ./cli/validate_estimation_subgraph.py --model resnet18 --hardware Jetson-Orin-AGX --device cuda

    # CSV output for further analysis
    ./cli/validate_estimation_subgraph.py --model vit_b_16 --hardware i7-12700K --output results.csv
"""

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
from torch.fx import Interpreter, symbolic_trace

from graphs.estimation.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner
from graphs.hardware.resource_model import Precision
from torch.fx.passes.shape_prop import ShapeProp


# ============================================================================
# TimingInterpreter with CUDA Events
# ============================================================================

class CUDAEventTimingInterpreter(Interpreter):
    """FX Interpreter that times each node using CUDA events (no sync overhead)."""

    def __init__(self, module, device='cpu'):
        super().__init__(module)
        self.device = device
        self.node_times = {}  # node_name -> time_ms
        # Pre-create events for CUDA timing
        if device == 'cuda':
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)

    def run_node(self, n):
        if self.device == 'cuda':
            self._start_event.record()
            result = super().run_node(n)
            self._end_event.record()
            self._end_event.synchronize()  # Wait only for this node
            self.node_times[n.name] = self._start_event.elapsed_time(self._end_event)
        else:
            start = time.perf_counter()
            result = super().run_node(n)
            self.node_times[n.name] = (time.perf_counter() - start) * 1000
        return result


class LegacyTimingInterpreter(Interpreter):
    """FX Interpreter that times each node (legacy with sync overhead)."""

    def __init__(self, module, device='cpu'):
        super().__init__(module)
        self.device = device
        self.node_times = {}  # node_name -> time_ms

    def run_node(self, n):
        if self.device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = super().run_node(n)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.node_times[n.name] = (time.perf_counter() - start) * 1000
        return result


def measure_full_model_inference(model, input_tensor, device='cpu',
                                  warmup_runs=50, timing_runs=100):
    """Measure full model inference time with CUDA events (gold standard).

    This measures the actual end-to-end inference time with all kernel fusion
    and pipelining benefits. This is the "ground truth" for validation.

    Args:
        model: PyTorch model (nn.Module)
        input_tensor: Input tensor
        device: 'cpu' or 'cuda'
        warmup_runs: Warmup iterations to stabilize performance
        timing_runs: Measurement iterations

    Returns:
        Tuple of (median_time_ms, min_time_ms, max_time_ms)
    """
    if device == 'cuda':
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Measure with CUDA events for GPU, time.perf_counter for CPU
    times = []
    with torch.no_grad():
        for _ in range(timing_runs):
            if device == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(input_tensor)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            else:
                start = time.perf_counter()
                _ = model(input_tensor)
                times.append((time.perf_counter() - start) * 1000)

    times.sort()
    median_time = statistics.median(times)
    return median_time, min(times), max(times)


def measure_node_times(fx_graph, input_tensor, device='cpu',
                       warmup_runs=5, timing_runs=20, use_cuda_events=True):
    """Run TimingInterpreter multiple times and return median per-node times.

    NOTE: FX Interpreter runs each node as a SEPARATE kernel without fusion.
    The sum of node times will be MUCH higher than actual inference time.
    Use measure_full_model_inference() for the actual inference time.

    Args:
        fx_graph: Traced FX GraphModule
        input_tensor: Input tensor (on correct device)
        device: 'cpu' or 'cuda'
        warmup_runs: Discarded warmup iterations
        timing_runs: Measurement iterations
        use_cuda_events: Use CUDA events for accurate GPU timing (default True)

    Returns:
        Dict[node_name, median_time_ms]
    """
    if device == 'cuda':
        fx_graph = fx_graph.cuda()
        input_tensor = input_tensor.cuda()

    # Select interpreter class
    if device == 'cuda' and use_cuda_events:
        InterpreterClass = CUDAEventTimingInterpreter
    else:
        InterpreterClass = LegacyTimingInterpreter

    # Collect per-node times across runs
    all_times = defaultdict(list)  # node_name -> [time_ms, ...]

    total_runs = warmup_runs + timing_runs
    for i in range(total_runs):
        interp = InterpreterClass(fx_graph, device=device)
        with torch.no_grad():
            interp.run(input_tensor)

        # Skip warmup runs
        if i >= warmup_runs:
            for name, t in interp.node_times.items():
                all_times[name].append(t)

    # Take median per node
    median_times = {}
    for name, times in all_times.items():
        median_times[name] = statistics.median(times)

    return median_times


# ============================================================================
# Subgraph aggregation and comparison
# ============================================================================

def aggregate_to_subgraphs(node_times, partition_report, roofline_report):
    """Map per-node measured times to subgraphs and compare with estimates.

    Args:
        node_times: Dict[node_name, time_ms] from TimingInterpreter
        partition_report: PartitionReport with SubgraphDescriptors
        roofline_report: RooflineReport with LatencyDescriptors

    Returns:
        List of dicts with per-subgraph comparison data
    """
    latencies = roofline_report.latencies
    subgraphs = partition_report.subgraphs

    # Subgraphs and latencies are produced in the same order by the analyzer,
    # so we iterate them in parallel. (SubgraphDescriptor.subgraph_id is int
    # while LatencyDescriptor.subgraph_id is str, so matching by ID doesn't work.)
    if len(subgraphs) != len(latencies):
        print(f"  WARNING: subgraph count ({len(subgraphs)}) != latency count "
              f"({len(latencies)}), using min")

    rows = []
    for sg, lat in zip(subgraphs, latencies):

        # Sum measured times for all nodes in this subgraph
        measured_ms = 0.0
        matched_nodes = 0
        for name in sg.node_names:
            if name in node_times:
                measured_ms += node_times[name]
                matched_nodes += 1

        estimated_ms = lat.actual_latency * 1000  # seconds -> ms

        if measured_ms > 0:
            error_pct = (estimated_ms - measured_ms) / measured_ms * 100
        else:
            error_pct = float('inf') if estimated_ms > 0 else 0.0

        gap_ms = estimated_ms - measured_ms

        rows.append({
            'subgraph_id': sg.subgraph_id,
            'pattern': sg.fusion_pattern,
            'num_ops': sg.num_operators,
            'flops': sg.total_flops,
            'total_bytes': sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes,
            'arithmetic_intensity': sg.arithmetic_intensity,
            'estimated_ms': estimated_ms,
            'measured_ms': measured_ms,
            'error_pct': error_pct,
            'gap_ms': gap_ms,
            'abs_gap_ms': abs(gap_ms),
            'bottleneck': lat.bottleneck.name if hasattr(lat.bottleneck, 'name') else str(lat.bottleneck),
            'compute_time_ms': lat.compute_time * 1000,
            'memory_time_ms': lat.memory_time * 1000,
            'overhead_ms': lat.overhead * 1000,
            'matched_nodes': matched_nodes,
            'total_nodes': len(sg.node_names),
        })

    return rows


def aggregate_by_pattern(rows):
    """Aggregate subgraph rows by fusion pattern.

    Returns:
        List of dicts sorted by absolute gap (descending)
    """
    patterns = defaultdict(lambda: {
        'count': 0, 'est_ms': 0.0, 'meas_ms': 0.0, 'flops': 0,
    })

    for r in rows:
        p = patterns[r['pattern']]
        p['count'] += 1
        p['est_ms'] += r['estimated_ms']
        p['meas_ms'] += r['measured_ms']
        p['flops'] += r['flops']

    result = []
    for pattern, p in patterns.items():
        gap = p['est_ms'] - p['meas_ms']
        error = (gap / p['meas_ms'] * 100) if p['meas_ms'] > 0 else float('inf')
        result.append({
            'pattern': pattern,
            'count': p['count'],
            'est_ms': p['est_ms'],
            'meas_ms': p['meas_ms'],
            'gap_ms': gap,
            'abs_gap_ms': abs(gap),
            'error_pct': error,
            'flops': p['flops'],
        })

    result.sort(key=lambda x: x['abs_gap_ms'], reverse=True)
    return result


# ============================================================================
# Display
# ============================================================================

def format_flops(flops):
    """Format FLOPs with appropriate unit."""
    if flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.1f}M"
    elif flops >= 1e3:
        return f"{flops/1e3:.0f}K"
    else:
        return f"{flops:.0f}"


def format_bytes(nbytes):
    """Format bytes as human readable string."""
    if nbytes >= 1e9:
        return f"{nbytes/1e9:.1f}GB"
    elif nbytes >= 1e6:
        return f"{nbytes/1e6:.1f}MB"
    elif nbytes >= 1e3:
        return f"{nbytes/1e3:.0f}KB"
    else:
        return f"{nbytes:.0f}B"


def print_subgraph_table(rows, top_n=None):
    """Print per-subgraph comparison table with memory sizes."""
    # Sort by measured time descending (large kernels first - more reliable measurements)
    sorted_rows = sorted(rows, key=lambda r: r['measured_ms'], reverse=True)
    if top_n:
        sorted_rows = sorted_rows[:top_n]

    print()
    header = (f"  {'Subgraph':<10} {'Pattern':<24} {'FLOPs':>8} {'Memory':>8} "
              f"{'Est (ms)':>9} {'Meas (ms)':>9} {'Error':>8} {'Bottleneck':<14}")
    print(header)
    print("  " + "-" * 106)

    for r in sorted_rows:
        flops_str = format_flops(r['flops'])
        mem_str = format_bytes(r['total_bytes'])
        err_str = f"{r['error_pct']:+.1f}%" if abs(r['error_pct']) < 10000 else "N/A"
        print(f"  sg_{r['subgraph_id']:<7} {r['pattern']:<24} {flops_str:>8} {mem_str:>8} "
              f"{r['estimated_ms']:>9.3f} {r['measured_ms']:>9.3f} {err_str:>8} "
              f"{r['bottleneck']:<14}")


def print_pattern_table(pattern_rows, total_gap_ms):
    """Print error aggregated by fusion pattern."""
    print()
    print("Error by Fusion Pattern (aggregated):")
    print(f"  {'Pattern':<24} {'Count':>5} {'Est (ms)':>10} {'Meas (ms)':>10} "
          f"{'Gap (ms)':>10} {'Error':>8}")
    print("  " + "-" * 75)

    for p in pattern_rows:
        err_str = f"{p['error_pct']:+.1f}%" if abs(p['error_pct']) < 10000 else "N/A"
        print(f"  {p['pattern']:<24} {p['count']:>5} {p['est_ms']:>10.3f} "
              f"{p['meas_ms']:>10.3f} {p['gap_ms']:>+10.3f} {err_str:>8}")


def print_top_contributors(pattern_rows, total_gap_ms):
    """Print top error contributors by absolute gap."""
    print()
    print("Top Error Contributors (by absolute ms gap):")
    abs_total = sum(p['abs_gap_ms'] for p in pattern_rows)

    for i, p in enumerate(pattern_rows[:10]):
        pct_of_total = p['abs_gap_ms'] / abs_total * 100 if abs_total > 0 else 0
        direction = "underestimate" if p['gap_ms'] < 0 else "overestimate"
        print(f"  {i+1:>2}. {p['pattern']:<20} {p['gap_ms']:>+8.2f} ms  "
              f"({p['count']} instances, {pct_of_total:.1f}% of total error, {direction})")


def print_unaccounted_time(node_times, partition_report, total_meas_ms):
    """Print time spent in nodes not mapped to any subgraph."""
    # Collect all node names that are in subgraphs
    mapped_nodes = set()
    for sg in partition_report.subgraphs:
        mapped_nodes.update(sg.node_names)

    # Find unmapped node time
    unmapped_time = 0.0
    unmapped_nodes = []
    for name, t in node_times.items():
        if name not in mapped_nodes and t > 0.001:  # > 1 microsecond
            unmapped_time += t
            unmapped_nodes.append((name, t))

    if unmapped_time > 0.01:  # > 10 microseconds
        print()
        print(f"Unmapped node time: {unmapped_time:.3f} ms "
              f"({unmapped_time/total_meas_ms*100:.1f}% of measured total)")
        unmapped_nodes.sort(key=lambda x: -x[1])
        for name, t in unmapped_nodes[:5]:
            print(f"    {name:<30} {t:.3f} ms")


# ============================================================================
# CSV output
# ============================================================================

def write_csv_output(rows, pattern_rows, output_path):
    """Write per-subgraph and per-pattern results to CSV."""
    # Per-subgraph CSV
    sg_path = output_path
    with open(sg_path, 'w', newline='') as f:
        fieldnames = ['subgraph_id', 'pattern', 'num_ops', 'flops', 'total_bytes',
                      'arithmetic_intensity', 'estimated_ms', 'measured_ms',
                      'error_pct', 'gap_ms', 'bottleneck',
                      'compute_time_ms', 'memory_time_ms', 'overhead_ms']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-subgraph results: {sg_path}")

    # Per-pattern CSV
    pat_path = Path(output_path).with_suffix('.patterns.csv')
    with open(pat_path, 'w', newline='') as f:
        fieldnames = ['pattern', 'count', 'est_ms', 'meas_ms', 'gap_ms', 'error_pct', 'flops']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(pattern_rows)
    print(f"Per-pattern results:  {pat_path}")


# ============================================================================
# Main
# ============================================================================

HARDWARE_CHOICES = [
    'H100', 'A100', 'V100', 'Jetson-Orin-AGX', 'Jetson-Orin-Nano',
    'TPU-v4', 'Coral-Edge-TPU', 'KPU-T768', 'KPU-T256', 'KPU-T64',
    'EPYC', 'Xeon', 'Ampere-One', 'i7-12700K', 'Ryzen',
    'QRB5165', 'TI-TDA4VM', 'DPU', 'CGRA',
]

GPU_HARDWARE = {
    'h100', 'a100', 'v100',
    'jetson-orin-agx', 'jetson-orin-nano',
}


def main():
    parser = argparse.ArgumentParser(
        description='Per-subgraph estimation validation using FX Interpreter timing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model vit_b_16 --hardware i7-12700K
  %(prog)s --model resnet18 --hardware Jetson-Orin-AGX --device cuda
  %(prog)s --model vit_b_16 --hardware i7-12700K --output results.csv
""")
    parser.add_argument('--model', required=True,
                        help='Model name (e.g., resnet18, vit_b_16)')
    parser.add_argument('--hardware', required=True, choices=HARDWARE_CHOICES,
                        help='Target hardware for estimation')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='Device for measurement (default: auto from hardware)')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32',
                        help='Precision (default: fp32)')
    parser.add_argument('--warmup-runs', type=int, default=5,
                        help='TimingInterpreter warmup runs (default: 5)')
    parser.add_argument('--timing-runs', type=int, default=20,
                        help='TimingInterpreter measurement runs (default: 20)')
    parser.add_argument('--top', type=int, default=None,
                        help='Show top N subgraphs by error (default: all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path')
    parser.add_argument('--thermal-profile', type=str, default=None,
                        help='Thermal/power profile (e.g., 15W, 30W, 50W, MAXN for Jetson)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose estimation output')
    parser.add_argument('--legacy-timing', action='store_true',
                        help='Use legacy timing with sync overhead (for comparison)')
    parser.add_argument('--subgraph', action='store_true',
                        help='Show per-subgraph breakdown (FX Interpreter, useful for large compute kernels)')

    args = parser.parse_args()

    # Auto-detect device
    device = args.device
    if device is None:
        if args.hardware.lower() in GPU_HARDWARE:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = 'cpu'

    precision_map = {'fp32': Precision.FP32, 'fp16': Precision.FP16}
    precision = precision_map[args.precision]

    # Thermal profile
    thermal_profile = getattr(args, 'thermal_profile', None)

    print()
    print("=" * 80)
    print("  PER-SUBGRAPH ESTIMATION VALIDATION")
    print("=" * 80)
    print(f"  Model:     {args.model}")
    print(f"  Hardware:  {args.hardware}")
    print(f"  Thermal:   {thermal_profile or '(default)'}")
    print(f"  Device:    {device}")
    print(f"  Precision: {args.precision.upper()}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  Runs:      {args.warmup_runs} warmup + {args.timing_runs} timed")
    print("=" * 80)

    # Step 1: Create model
    print("\nStep 1: Creating model...", flush=True)
    analyzer = UnifiedAnalyzer(verbose=not args.quiet)
    model, input_tensor, display_name = analyzer._create_model(args.model, args.batch_size)
    model.eval()

    if args.precision == 'fp16':
        model = model.half()
        input_tensor = input_tensor.half()

    # Step 2: Trace with symbolic_trace (needed for Interpreter AND estimation)
    # We use the SAME traced graph for both estimation and measurement so
    # node names match exactly between SubgraphDescriptor and TimingInterpreter.
    print("Step 2: Tracing with symbolic_trace...", flush=True)
    fx_graph = symbolic_trace(model)

    # Shape propagation (needed for partitioner to compute FLOPs/memory)
    print("  Running shape propagation...", flush=True)
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    # Step 3: Partition and run roofline on the symbolic_trace graph
    print("Step 3: Partitioning and running roofline estimation...", flush=True)
    partitioner = FusionBasedPartitioner()
    partition_report = partitioner.partition(fx_graph)
    print(f"  Partitioned into {len(partition_report.subgraphs)} subgraphs, "
          f"{partition_report.total_flops/1e9:.2f} GFLOPs", flush=True)

    hardware_mapper = analyzer._create_hardware_mapper(args.hardware, thermal_profile=thermal_profile)
    hardware = hardware_mapper.resource_model
    roofline_analyzer = RooflineAnalyzer(hardware, precision=precision, thermal_profile=thermal_profile)
    roofline_report = roofline_analyzer.analyze(partition_report.subgraphs, partition_report)

    if roofline_report is None:
        print("ERROR: Roofline analysis failed")
        return 1

    # Step 4: Measure FULL MODEL inference time (gold standard)
    print("Step 4: Measuring full model inference time (gold standard)...", flush=True)
    full_model_median, full_model_min, full_model_max = measure_full_model_inference(
        model, input_tensor, device=device,
        warmup_runs=50, timing_runs=100,
    )
    estimated_total = roofline_report.total_latency * 1000  # convert from seconds to ms
    full_model_error = (estimated_total - full_model_median) / full_model_median * 100
    print(f"  Full model inference: {full_model_median:.3f} ms "
          f"(min={full_model_min:.3f}, max={full_model_max:.3f})", flush=True)
    print(f"  Estimated total:      {estimated_total:.3f} ms", flush=True)
    print(f"  Error:                {full_model_error:+.1f}%", flush=True)

    # Step 5: Measure per-node times with TimingInterpreter (only if --subgraph or --output)
    node_times = {}
    total_node_time = 0.0
    rows = []
    pattern_rows = []
    total_gap = 0.0

    if args.subgraph or args.output:
        use_cuda_events = device == 'cuda' and not args.legacy_timing
        timing_method = "CUDA events" if use_cuda_events else "wall clock (legacy)"
        print(f"Step 5: Measuring per-node latency for breakdown ({args.warmup_runs} warmup + "
              f"{args.timing_runs} timed runs on {device}, {timing_method})...", flush=True)
        node_times = measure_node_times(
            fx_graph, input_tensor, device=device,
            warmup_runs=args.warmup_runs, timing_runs=args.timing_runs,
            use_cuda_events=use_cuda_events,
        )

        total_node_time = sum(node_times.values())
        print(f"  Total measured node time: {total_node_time:.2f} ms "
              f"({len(node_times)} nodes)", flush=True)

        # Step 6: Map to subgraphs and compare
        print("Step 6: Mapping nodes to subgraphs and comparing...", flush=True)
        rows = aggregate_to_subgraphs(
            node_times, partition_report, roofline_report,
        )

        if not rows:
            print("ERROR: No subgraphs matched between estimation and measurement")
            return 1

        total_gap = sum(r['gap_ms'] for r in rows)
        pattern_rows = aggregate_by_pattern(rows)

    # Print FULL MODEL validation summary first (most important)
    print()
    print("=" * 80)
    print("  FULL MODEL VALIDATION (Gold Standard)")
    print("=" * 80)
    print(f"  Model:              {display_name}")
    print(f"  Hardware:           {args.hardware} ({thermal_profile or 'default'})")
    print(f"  Measured inference: {full_model_median:.3f} ms (CUDA events, 100 runs)")
    print(f"  Estimated latency:  {estimated_total:.3f} ms (roofline model)")
    print(f"  Error:              {full_model_error:+.1f}%")
    if abs(full_model_error) < 15:
        rating = "EXCELLENT"
    elif abs(full_model_error) < 30:
        rating = "GOOD"
    elif abs(full_model_error) < 50:
        rating = "FAIR"
    else:
        rating = "POOR"
    print(f"  Rating:             {rating}")
    print("=" * 80)

    # Per-subgraph breakdown (only with --subgraph flag)
    if args.subgraph:
        print()
        print("=" * 106)
        print("  PER-SUBGRAPH BREAKDOWN (FX Interpreter - no kernel fusion)")
        print("=" * 106)
        print()
        print("  IMPORTANT: FX Interpreter runs each node as a SEPARATE kernel without fusion.")
        print(f"  Total node time ({total_node_time:.1f} ms) is {total_node_time/full_model_median:.1f}x higher than")
        print(f"  actual inference ({full_model_median:.1f} ms) due to per-kernel dispatch overhead (~0.2-0.3ms).")
        print()
        print("  This table is useful for:")
        print("  - Identifying operation types and memory footprints in the model")
        print("  - Comparing LARGE compute kernels (>100M FLOPs) where dispatch overhead is <30%")
        print("  - Understanding which fusion patterns dominate the workload")
        print()
        print("  NOT useful for: validating estimation accuracy (use Full Model Validation above)")
        print()

        print_subgraph_table(rows, top_n=args.top)
        print()
        print_pattern_table(pattern_rows, total_gap)
        print_top_contributors(pattern_rows, total_gap)
        print_unaccounted_time(node_times, partition_report, total_node_time)
    else:
        print()
        print("  Use --subgraph to see per-subgraph breakdown (FX Interpreter, for debugging)")

    # CSV output
    if args.output:
        print()
        write_csv_output(rows, pattern_rows, args.output)

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())

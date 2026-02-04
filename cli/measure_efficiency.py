#!/usr/bin/env python3
"""
Per-Subgraph Efficiency Measurement Tool

Measures actual efficiency for each subgraph in a model by comparing
achieved FLOPS against theoretical peak. Outputs statistical characterization
for use in efficiency curve calibration.

Usage:
    # Measure efficiency on CPU
    ./cli/measure_efficiency.py --model resnet18 --hardware i7-12700K --device cpu

    # Measure on GPU with specific power mode
    ./cli/measure_efficiency.py --model resnet18 --hardware Jetson-Orin-AGX --device cuda --thermal-profile 50W

    # Save measurements to JSON
    ./cli/measure_efficiency.py --model resnet18 --hardware i7-12700K --output measurements/resnet18_i7.json

    # Multiple models for calibration
    for model in resnet18 resnet50 vgg16 mobilenet_v2 vit_b_16; do
        ./cli/measure_efficiency.py --model $model --hardware i7-12700K --output measurements/${model}_i7.json
    done
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
from torch.fx import Interpreter, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.estimation.unified_analyzer import UnifiedAnalyzer
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner
from graphs.hardware.resource_model import Precision


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class LatencyStats:
    """Statistical summary of latency measurements."""
    mean: float
    std: float
    min: float
    max: float
    ci_lower: float  # 95% CI
    ci_upper: float
    samples: int

    def to_dict(self):
        return asdict(self)


@dataclass
class EfficiencyStats:
    """Statistical summary of efficiency measurements."""
    mean: float
    std: float
    min: float
    max: float
    ci_lower: float  # 95% CI
    ci_upper: float
    samples: int

    def to_dict(self):
        return asdict(self)


@dataclass
class SubgraphMeasurement:
    """Complete measurement data for a single subgraph."""
    subgraph_id: int
    fusion_pattern: str
    operation_type: str
    flops: int
    total_bytes: int
    arithmetic_intensity: float
    theoretical_peak_flops: float
    measured_latency: LatencyStats
    achieved_flops: float
    efficiency: EfficiencyStats
    node_names: List[str]
    source_model: str

    def to_dict(self):
        d = asdict(self)
        d['measured_latency'] = self.measured_latency.to_dict()
        d['efficiency'] = self.efficiency.to_dict()
        return d


# ============================================================================
# TimingInterpreter (adapted from validate_estimation_subgraph.py)
# ============================================================================

class CUDAEventTimingInterpreter(Interpreter):
    """FX Interpreter that times each node using CUDA events."""

    def __init__(self, module, device='cpu'):
        super().__init__(module)
        self.device = device
        self.node_times = {}
        if device == 'cuda':
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)

    def run_node(self, n):
        if self.device == 'cuda':
            self._start_event.record()
            result = super().run_node(n)
            self._end_event.record()
            self._end_event.synchronize()
            self.node_times[n.name] = self._start_event.elapsed_time(self._end_event)
        else:
            start = time.perf_counter()
            result = super().run_node(n)
            self.node_times[n.name] = (time.perf_counter() - start) * 1000
        return result


class CPUTimingInterpreter(Interpreter):
    """FX Interpreter that times each node using perf_counter."""

    def __init__(self, module, device='cpu'):
        super().__init__(module)
        self.device = device
        self.node_times = {}

    def run_node(self, n):
        start = time.perf_counter()
        result = super().run_node(n)
        self.node_times[n.name] = (time.perf_counter() - start) * 1000
        return result


# ============================================================================
# Operation type mapping
# ============================================================================

def map_fusion_pattern_to_operation_type(fusion_pattern: str) -> str:
    """Map a fusion pattern string to a calibration operation type.

    This mapping determines which efficiency curve to use for estimation.

    Args:
        fusion_pattern: Fusion pattern from partitioner (e.g., "Conv2d_BatchNorm2d_ReLU")

    Returns:
        Operation type for calibration lookup
    """
    pattern_lower = fusion_pattern.lower()

    # Depthwise convolutions (very low efficiency)
    if 'depthwise' in pattern_lower or 'dw' in pattern_lower:
        return 'conv2d_depthwise'

    # MBConv blocks (EfficientNet/MobileNet style)
    if 'mbconv' in pattern_lower or 'inverted' in pattern_lower:
        return 'mbconv'

    # Standard convolutions
    if 'conv2d' in pattern_lower or 'conv' in pattern_lower:
        if 'batchnorm' in pattern_lower or 'bn' in pattern_lower:
            return 'conv2d_batchnorm'
        else:
            return 'conv2d'

    # Matrix operations
    if 'matmul' in pattern_lower or 'linear' in pattern_lower or 'attention' in pattern_lower:
        return 'matmul'

    # Pooling operations
    if 'pool' in pattern_lower:
        return 'pooling'

    # Normalization
    if 'layernorm' in pattern_lower or 'layer_norm' in pattern_lower:
        return 'layernorm'
    if 'batchnorm' in pattern_lower:
        return 'batchnorm'

    # Activations
    if any(act in pattern_lower for act in ['relu', 'gelu', 'silu', 'swish', 'sigmoid']):
        return 'activation'

    # Element-wise operations
    if 'add' in pattern_lower or 'mul' in pattern_lower or 'elementwise' in pattern_lower:
        return 'elementwise'

    # Reshape/view operations
    if any(op in pattern_lower for op in ['reshape', 'view', 'flatten', 'squeeze', 'unsqueeze']):
        return 'reshape'

    # Unfused or unknown
    if 'unfused' in pattern_lower:
        return 'unfused'

    return 'generic'


# ============================================================================
# Measurement functions
# ============================================================================

def collect_node_times_multiple_runs(
    fx_graph,
    input_tensor,
    device: str = 'cpu',
    warmup_runs: int = 10,
    timing_runs: int = 50
) -> Dict[str, List[float]]:
    """Run TimingInterpreter multiple times and collect all per-node times.

    Args:
        fx_graph: Traced FX GraphModule
        input_tensor: Input tensor
        device: 'cpu' or 'cuda'
        warmup_runs: Discarded warmup iterations
        timing_runs: Measurement iterations

    Returns:
        Dict mapping node_name -> list of time measurements (ms)
    """
    if device == 'cuda':
        fx_graph = fx_graph.cuda()
        input_tensor = input_tensor.cuda()

    # Select interpreter
    if device == 'cuda':
        InterpreterClass = CUDAEventTimingInterpreter
    else:
        InterpreterClass = CPUTimingInterpreter

    all_times = defaultdict(list)

    total_runs = warmup_runs + timing_runs
    for i in range(total_runs):
        interp = InterpreterClass(fx_graph, device=device)
        with torch.no_grad():
            interp.run(input_tensor)

        # Skip warmup runs
        if i >= warmup_runs:
            for name, t in interp.node_times.items():
                all_times[name].append(t)

    return dict(all_times)


def compute_latency_stats(times: List[float]) -> LatencyStats:
    """Compute statistical summary of latency measurements.

    Args:
        times: List of latency measurements in ms

    Returns:
        LatencyStats with mean, std, min, max, CI
    """
    n = len(times)
    if n == 0:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0)

    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0
    min_val = min(times)
    max_val = max(times)

    # 95% confidence interval (t-distribution approximation for small n)
    se = std / math.sqrt(n) if n > 0 else 0
    t_value = 1.96 if n >= 30 else 2.0  # Simplified
    ci_lower = mean - t_value * se
    ci_upper = mean + t_value * se

    return LatencyStats(
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        ci_lower=max(0, ci_lower),
        ci_upper=ci_upper,
        samples=n
    )


def compute_efficiency_stats(
    latency_times: List[float],
    flops: int,
    theoretical_peak_flops: float
) -> Tuple[EfficiencyStats, float]:
    """Compute efficiency statistics from latency measurements.

    Efficiency = achieved_flops / theoretical_peak
    Achieved_flops = flops / (latency_ms / 1000)

    Args:
        latency_times: List of latency measurements in ms
        flops: FLOPs for this operation
        theoretical_peak_flops: Hardware peak FLOPS

    Returns:
        Tuple of (EfficiencyStats, mean_achieved_flops)
    """
    if not latency_times or flops == 0 or theoretical_peak_flops == 0:
        return EfficiencyStats(0, 0, 0, 0, 0, 0, 0), 0

    # Compute efficiency for each measurement
    efficiencies = []
    achieved_flops_list = []
    for t_ms in latency_times:
        if t_ms > 0:
            t_s = t_ms / 1000.0
            achieved = flops / t_s
            eff = achieved / theoretical_peak_flops
            efficiencies.append(eff)
            achieved_flops_list.append(achieved)

    if not efficiencies:
        return EfficiencyStats(0, 0, 0, 0, 0, 0, 0), 0

    n = len(efficiencies)
    mean = statistics.mean(efficiencies)
    std = statistics.stdev(efficiencies) if n > 1 else mean * 0.1  # 10% assumed
    min_val = min(efficiencies)
    max_val = max(efficiencies)

    se = std / math.sqrt(n) if n > 0 else 0
    t_value = 1.96 if n >= 30 else 2.0
    ci_lower = mean - t_value * se
    ci_upper = mean + t_value * se

    mean_achieved = statistics.mean(achieved_flops_list) if achieved_flops_list else 0

    return EfficiencyStats(
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        ci_lower=max(0, ci_lower),
        ci_upper=ci_upper,
        samples=n
    ), mean_achieved


def measure_subgraph_efficiency(
    subgraph,
    node_times: Dict[str, List[float]],
    theoretical_peak_flops: float,
    model_name: str
) -> SubgraphMeasurement:
    """Compute efficiency measurement for a single subgraph.

    Args:
        subgraph: SubgraphDescriptor from partitioner
        node_times: Dict mapping node_name -> list of times (ms)
        theoretical_peak_flops: Hardware peak FLOPS
        model_name: Source model name

    Returns:
        SubgraphMeasurement with full statistics
    """
    # Aggregate times across all nodes in subgraph
    # For each run, sum the node times to get subgraph time
    num_runs = 0
    for name in subgraph.node_names:
        if name in node_times:
            num_runs = max(num_runs, len(node_times[name]))

    if num_runs == 0:
        # No timing data for this subgraph
        return None

    # Sum node times per run to get subgraph latencies
    subgraph_times = []
    for run_idx in range(num_runs):
        run_total = 0.0
        for name in subgraph.node_names:
            if name in node_times and run_idx < len(node_times[name]):
                run_total += node_times[name][run_idx]
        subgraph_times.append(run_total)

    # Compute statistics
    latency_stats = compute_latency_stats(subgraph_times)
    efficiency_stats, achieved_flops = compute_efficiency_stats(
        subgraph_times, subgraph.total_flops, theoretical_peak_flops
    )

    # Map to operation type
    op_type = map_fusion_pattern_to_operation_type(subgraph.fusion_pattern)

    total_bytes = (subgraph.total_input_bytes +
                   subgraph.total_output_bytes +
                   subgraph.total_weight_bytes)

    return SubgraphMeasurement(
        subgraph_id=subgraph.subgraph_id,
        fusion_pattern=subgraph.fusion_pattern,
        operation_type=op_type,
        flops=subgraph.total_flops,
        total_bytes=total_bytes,
        arithmetic_intensity=subgraph.arithmetic_intensity,
        theoretical_peak_flops=theoretical_peak_flops,
        measured_latency=latency_stats,
        achieved_flops=achieved_flops,
        efficiency=efficiency_stats,
        node_names=subgraph.node_names,
        source_model=model_name
    )


# ============================================================================
# Output
# ============================================================================

def format_flops(flops: float) -> str:
    """Format FLOPs with appropriate unit."""
    if flops >= 1e12:
        return f"{flops/1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.1f}M"
    elif flops >= 1e3:
        return f"{flops/1e3:.0f}K"
    else:
        return f"{flops:.0f}"


def print_measurements_table(measurements: List[SubgraphMeasurement]):
    """Print measurement results in table format."""
    print()
    print(f"  {'SG':<4} {'Pattern':<30} {'OpType':<18} {'FLOPs':>8} "
          f"{'Lat(ms)':>8} {'Eff':>7} {'Eff Std':>7}")
    print("  " + "-" * 105)

    for m in measurements:
        if m is None:
            continue
        flops_str = format_flops(m.flops)
        eff_str = f"{m.efficiency.mean:.3f}" if m.efficiency.mean < 10 else f"{m.efficiency.mean:.1f}"
        std_str = f"{m.efficiency.std:.3f}" if m.efficiency.std < 10 else f"{m.efficiency.std:.1f}"
        print(f"  {m.subgraph_id:<4} {m.fusion_pattern:<30} {m.operation_type:<18} {flops_str:>8} "
              f"{m.measured_latency.mean:>8.3f} {eff_str:>7} {std_str:>7}")


def print_operation_type_summary(measurements: List[SubgraphMeasurement]):
    """Print summary by operation type."""
    by_type = defaultdict(list)
    for m in measurements:
        if m is not None:
            by_type[m.operation_type].append(m)

    print()
    print("Summary by Operation Type:")
    print(f"  {'Operation Type':<20} {'Count':>5} {'Total FLOPs':>12} "
          f"{'Avg Eff':>8} {'Eff Range':>15}")
    print("  " + "-" * 70)

    for op_type, items in sorted(by_type.items()):
        count = len(items)
        total_flops = sum(m.flops for m in items)
        effs = [m.efficiency.mean for m in items]
        avg_eff = statistics.mean(effs) if effs else 0
        min_eff = min(effs) if effs else 0
        max_eff = max(effs) if effs else 0
        range_str = f"{min_eff:.3f}-{max_eff:.3f}"
        print(f"  {op_type:<20} {count:>5} {format_flops(total_flops):>12} "
              f"{avg_eff:>8.3f} {range_str:>15}")


def save_measurements_json(
    measurements: List[SubgraphMeasurement],
    model_name: str,
    hardware_id: str,
    device: str,
    precision: str,
    thermal_profile: Optional[str],
    theoretical_peak: float,
    output_path: Path
):
    """Save measurements to JSON file."""
    output = {
        "schema_version": "1.0",
        "measurement_type": "efficiency",
        "model": model_name,
        "hardware_id": hardware_id,
        "device": device,
        "precision": precision,
        "thermal_profile": thermal_profile,
        "theoretical_peak_flops": theoretical_peak,
        "measurement_date": datetime.now().isoformat(),
        "tool_version": "measure_efficiency.py v1.0",
        "subgraphs": [m.to_dict() for m in measurements if m is not None]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nMeasurements saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

HARDWARE_CHOICES = [
    'H100', 'A100', 'V100', 'Jetson-Orin-AGX', 'Jetson-Orin-Nano',
    'i7-12700K', 'Ryzen', 'EPYC', 'Xeon', 'Ampere-One',
]

GPU_HARDWARE = {'h100', 'a100', 'v100', 'jetson-orin-agx', 'jetson-orin-nano'}


def main():
    parser = argparse.ArgumentParser(
        description='Measure per-subgraph efficiency for calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model resnet18 --hardware i7-12700K
  %(prog)s --model vit_b_16 --hardware Jetson-Orin-AGX --device cuda --thermal-profile 50W
  %(prog)s --model resnet18 --hardware i7-12700K --output measurements/resnet18_i7.json
""")
    parser.add_argument('--model', required=True,
                        help='Model name (e.g., resnet18, vit_b_16)')
    parser.add_argument('--hardware', required=True, choices=HARDWARE_CHOICES,
                        help='Target hardware')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='Device for measurement (default: auto)')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32',
                        help='Precision (default: fp32)')
    parser.add_argument('--thermal-profile', type=str, default=None,
                        help='Thermal/power profile (e.g., 15W, 30W, 50W)')
    parser.add_argument('--warmup-runs', type=int, default=10,
                        help='Warmup runs (default: 10)')
    parser.add_argument('--timing-runs', type=int, default=50,
                        help='Timing runs (default: 50)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Auto-detect device
    device = args.device
    if device is None:
        if args.hardware.lower() in GPU_HARDWARE:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = 'cpu'

    print()
    print("=" * 80)
    print("  EFFICIENCY MEASUREMENT")
    print("=" * 80)
    print(f"  Model:     {args.model}")
    print(f"  Hardware:  {args.hardware}")
    print(f"  Thermal:   {args.thermal_profile or '(default)'}")
    print(f"  Device:    {device}")
    print(f"  Precision: {args.precision.upper()}")
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

    # Step 2: Trace
    print("Step 2: Tracing with symbolic_trace...", flush=True)
    fx_graph = symbolic_trace(model)
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    # Step 3: Partition
    print("Step 3: Partitioning...", flush=True)
    partitioner = FusionBasedPartitioner()
    partition_report = partitioner.partition(fx_graph)
    print(f"  {len(partition_report.subgraphs)} subgraphs, "
          f"{partition_report.total_flops/1e9:.2f} GFLOPs")

    # Step 4: Get hardware specs
    print("Step 4: Loading hardware specifications...", flush=True)
    hardware_mapper = analyzer._create_hardware_mapper(
        args.hardware, thermal_profile=args.thermal_profile
    )
    hardware = hardware_mapper.resource_model

    # Get theoretical peak based on precision
    precision_map = {'fp32': Precision.FP32, 'fp16': Precision.FP16}
    precision = precision_map[args.precision]
    theoretical_peak = hardware.get_peak_ops(precision)
    print(f"  Theoretical peak: {format_flops(theoretical_peak)}FLOPS")

    # Step 5: Measure node times
    print(f"Step 5: Measuring node times ({args.warmup_runs} warmup + "
          f"{args.timing_runs} timed)...", flush=True)
    node_times = collect_node_times_multiple_runs(
        fx_graph, input_tensor, device=device,
        warmup_runs=args.warmup_runs, timing_runs=args.timing_runs
    )
    print(f"  Collected times for {len(node_times)} nodes")

    # Step 6: Compute efficiency per subgraph
    print("Step 6: Computing efficiency statistics...", flush=True)
    measurements = []
    for sg in partition_report.subgraphs:
        m = measure_subgraph_efficiency(
            sg, node_times, theoretical_peak, args.model
        )
        measurements.append(m)

    valid_measurements = [m for m in measurements if m is not None]
    print(f"  {len(valid_measurements)} subgraphs measured")

    # Print results
    if not args.quiet:
        print_measurements_table(valid_measurements)
        print_operation_type_summary(valid_measurements)

    # Save to JSON
    if args.output:
        save_measurements_json(
            measurements=valid_measurements,
            model_name=args.model,
            hardware_id=args.hardware,
            device=device,
            precision=args.precision.upper(),
            thermal_profile=args.thermal_profile,
            theoretical_peak=theoretical_peak,
            output_path=Path(args.output)
        )

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())

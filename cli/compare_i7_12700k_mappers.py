#!/usr/bin/env python
"""
Compare i7-12700K Mapper Variants

Demonstrates the performance difference between:
- create_i7_12700k_mapper() - Tuned for tiny models (batch 1-32)
- create_i7_12700k_large_mapper() - Tuned for large models (batch≥32)

This script characterizes the same model using both mappers to show
how calibration target affects performance estimates.
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # cli/ → graphs/

from src.graphs.transform.partitioning import FusionBasedPartitioner, FusionReport
from src.graphs.hardware.mappers.cpu import create_i7_12700k_mapper, create_i7_12700k_large_mapper
from src.graphs.hardware.resource_model import Precision


def extract_execution_stages(fusion_report: FusionReport):
    """
    Extract execution stages from fusion report.

    Simplified version: each fused subgraph is its own stage.
    For more accurate modeling, would analyze dependencies and create stages.
    """
    stages = [[i] for i in range(len(fusion_report.fused_subgraphs))]
    return stages


def create_test_mlp(input_dim=1024, hidden_dim=2048, output_dim=512):
    """Create a simple MLP for testing"""
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x

    return MLP()


def analyze_with_mapper(model, input_tensor, mapper_name, mapper):
    """Run full characterization pipeline with a specific mapper"""
    print(f"\n{'='*80}")
    print(f"Analyzing with: {mapper_name}")
    print(f"{'='*80}")

    # FX trace and shape propagation
    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(input_tensor)

    # Fusion-based partitioning
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(traced)

    # Calculate arithmetic intensity
    ai = 0.0
    if fusion_report.total_memory_traffic_fused > 0:
        ai = fusion_report.total_flops / fusion_report.total_memory_traffic_fused

    print(f"\nFusion Report:")
    print(f"  Total FLOPs: {fusion_report.total_flops / 1e9:.6f} GFLOPs ({fusion_report.total_flops:,} ops)")
    print(f"  Total Memory Traffic (fused): {fusion_report.total_memory_traffic_fused / 1e6:.2f} MB")
    print(f"  Arithmetic Intensity: {ai:.2f} FLOPs/Byte")
    print(f"  Fused Subgraphs: {len(fusion_report.fused_subgraphs)}")

    # Extract execution stages
    execution_stages = extract_execution_stages(fusion_report)

    # DEBUG: Check if mapper has thermal profile
    print(f"\nDEBUG - Mapper Configuration:")
    print(f"  Thermal Profile: {mapper.thermal_profile}")
    print(f"  Has Thermal Operating Points: {bool(mapper.resource_model.thermal_operating_points)}")

    if mapper.thermal_profile and mapper.resource_model.thermal_operating_points:
        thermal_point = mapper.resource_model.thermal_operating_points[mapper.thermal_profile]
        if Precision.FP32 in thermal_point.performance_specs:
            perf = thermal_point.performance_specs[Precision.FP32]
            print(f"  FP32 Effective Ops/sec: {perf.effective_ops_per_sec / 1e9:.2f} GFLOPS")
            print(f"  FP32 efficiency_factor: {perf.efficiency_factor:.3f}")

    # Hardware mapping
    batch_size = input_tensor.shape[0]
    hw_report = mapper.map_graph(
        fusion_report,
        execution_stages,
        batch_size=batch_size,
        precision='fp32'
    )

    # Calculate throughput from FLOPs and latency
    effective_throughput = 0.0
    if hw_report.total_latency > 0:
        effective_throughput = fusion_report.total_flops / hw_report.total_latency

    print(f"\nHardware Report ({mapper_name}):")
    print(f"  Estimated Latency: {hw_report.total_latency * 1000:.3f} ms")
    print(f"  Effective Throughput: {effective_throughput / 1e9:.2f} GFLOPS")
    print(f"  Average Utilization: {hw_report.average_utilization * 100:.1f}%")
    print(f"  Peak Utilization: {hw_report.peak_utilization * 100:.1f}%")
    print(f"  Compute Units Used: {hw_report.average_compute_units_used:.1f} avg, {hw_report.peak_compute_units_used} peak")

    return hw_report


def compare_mappers():
    """Compare both mapper variants on different model sizes"""

    test_cases = [
        {
            "name": "Tiny MLP (batch=1)",
            "input_dim": 256,
            "hidden_dim": 512,
            "output_dim": 128,
            "batch_size": 1,
            "expected_better": "tiny",  # Which mapper should be more accurate
        },
        {
            "name": "Small MLP (batch=16)",
            "input_dim": 512,
            "hidden_dim": 1024,
            "output_dim": 256,
            "batch_size": 16,
            "expected_better": "tiny/large",  # Border case
        },
        {
            "name": "Medium MLP (batch=64)",
            "input_dim": 1024,
            "hidden_dim": 2048,
            "output_dim": 512,
            "batch_size": 64,
            "expected_better": "large",
        },
        {
            "name": "Large MLP (batch=128)",
            "input_dim": 2048,
            "hidden_dim": 4096,
            "output_dim": 1024,
            "batch_size": 128,
            "expected_better": "large",
        },
    ]

    print("="*80)
    print("i7-12700K Mapper Comparison")
    print("="*80)
    print("\nThis comparison shows how calibration target affects estimates.")
    print("The SAME hardware is modeled with DIFFERENT efficiency factors.\n")

    for test_case in test_cases:
        print("\n" + "="*80)
        print(f"TEST CASE: {test_case['name']}")
        print(f"Expected Better Mapper: {test_case['expected_better']}")
        print("="*80)

        # Create model and input
        model = create_test_mlp(
            test_case['input_dim'],
            test_case['hidden_dim'],
            test_case['output_dim']
        )
        input_tensor = torch.randn(
            test_case['batch_size'],
            test_case['input_dim']
        )

        # Create mappers
        mapper_tiny = create_i7_12700k_mapper()
        mapper_large = create_i7_12700k_large_mapper()

        # Analyze with both
        report_tiny = analyze_with_mapper(model, input_tensor, "Tiny Model Mapper", mapper_tiny)
        report_large = analyze_with_mapper(model, input_tensor, "Large Model Mapper", mapper_large)

        # Comparison
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")

        latency_tiny = report_tiny.total_latency * 1000
        latency_large = report_large.total_latency * 1000
        speedup = latency_tiny / latency_large if latency_large > 0 else 0

        print(f"\nLatency Estimates:")
        print(f"  Tiny Mapper:  {latency_tiny:.3f} ms")
        print(f"  Large Mapper: {latency_large:.3f} ms")
        print(f"  Ratio (tiny/large): {speedup:.2f}×")

        if speedup > 1.5:
            print(f"  → Large mapper predicts {speedup:.1f}× FASTER execution")
            print(f"     (Higher efficiency_factor: 0.60 vs 0.20)")
        elif speedup < 0.67:
            print(f"  → Tiny mapper predicts {1/speedup:.1f}× FASTER execution")
            print(f"     (More pessimistic about efficiency)")
        else:
            print(f"  → Similar predictions (within 50%)")

        print(f"\nUtilization:")
        print(f"  Tiny Mapper:  {report_tiny.average_utilization * 100:.1f}% avg")
        print(f"  Large Mapper: {report_large.average_utilization * 100:.1f}% avg")

        print(f"\nKey Coefficient Differences:")
        print(f"  efficiency_factor:         0.20 vs 0.60  (3.0× difference)")
        print(f"  memory_bottleneck_factor:  0.25 vs 0.65  (2.6× difference)")
        print(f"  tile_utilization:          0.50 vs 0.80  (1.6× difference)")
        print(f"  instruction_efficiency:    0.65 vs 0.80  (1.2× difference)")

        print(f"\n⚠ CALIBRATION RECOMMENDATION:")
        if test_case['expected_better'] == "tiny":
            print(f"  Use create_i7_12700k_mapper() - calibrated on tiny models")
        elif test_case['expected_better'] == "large":
            print(f"  Use create_i7_12700k_large_mapper() - better for large models")
        else:
            print(f"  Border case - either mapper acceptable, expect 15-25% MAPE")


if __name__ == "__main__":
    compare_mappers()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The two mappers model the SAME hardware but with DIFFERENT efficiency factors:

1. create_i7_12700k_mapper():
   - Calibrated on tiny MLPs (batch 1-32)
   - efficiency_factor=0.20 (pessimistic, accounts for overhead)
   - Best for: Real-time inference, edge AI, batch=1

2. create_i7_12700k_large_mapper():
   - Estimated for large models (batch≥32)
   - efficiency_factor=0.60 (optimistic, amortizes overhead)
   - Best for: Training, batch inference, transformers

Choose the mapper that matches your workload characteristics!

To calibrate the large mapper, run:
    python validation/empirical/sweep_mlp.py --batch-sizes 32,64,128 \\
        --hidden-dims "[[2048,2048,2048]]" --device cpu
""")

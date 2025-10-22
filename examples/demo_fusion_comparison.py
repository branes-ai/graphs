#!/usr/bin/env python
"""
Test Fusion-Based Partitioner

Compare unfused (one subgraph per operator) vs fused (aggregated subgraphs)
to see the reduction in data movement and kernel launches.
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.insert(0, 'src')

from graphs.characterize.fusion_partitioner import FusionBasedPartitioner
from graphs.characterize.graph_partitioner import GraphPartitioner


def test_fusion(model_name='resnet18', input_shape=(1, 3, 224, 224)):
    """Test fusion partitioner on a model"""

    print("=" * 80)
    print(f"Testing Fusion-Based Partitioning: {model_name}")
    print("=" * 80)

    # Load model
    print(f"\n[1/4] Loading {model_name}...")
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
    else:
        print(f"Unknown model: {model_name}")
        return

    model.eval()

    # Trace
    print("[2/4] Tracing with PyTorch FX...")
    input_tensor = torch.randn(*input_shape)
    fx_graph = symbolic_trace(model)

    # Shape propagation
    print("[3/4] Propagating shapes...")
    ShapeProp(fx_graph).propagate(input_tensor)

    # Partition with both methods
    print("[4/4] Partitioning graph...")

    print("\n  a) Unfused partitioning (old method)...")
    unfused_partitioner = GraphPartitioner()
    unfused_report = unfused_partitioner.partition(fx_graph)

    print("  b) Fusion-based partitioning (new method)...")
    fusion_partitioner = FusionBasedPartitioner()
    fusion_report = fusion_partitioner.partition(fx_graph)

    # Display results
    print("\n" + "=" * 80)
    print("COMPARISON: UNFUSED vs FUSED")
    print("=" * 80)

    print(f"\nExecution Units:")
    print(f"  Unfused (old):  {unfused_report.total_subgraphs} subgraphs (1 per operator)")
    print(f"  Fused (new):    {fusion_report.total_subgraphs} subgraphs (aggregated)")
    print(f"  Reduction:      {unfused_report.total_subgraphs / max(1, fusion_report.total_subgraphs):.1f}× fewer kernel launches")

    print(f"\nCompute:")
    print(f"  Unfused FLOPs:  {unfused_report.total_flops / 1e9:.2f} G")
    print(f"  Fused FLOPs:    {fusion_report.total_flops / 1e9:.2f} G")
    print(f"  Difference:     {abs(unfused_report.total_flops - fusion_report.total_flops) / 1e9:.3f} G")

    print(f"\nMemory Traffic:")
    unfused_memory = unfused_report.total_memory_traffic / 1e6
    fused_memory = fusion_report.total_memory_traffic_fused / 1e6
    unfused_estimate = fusion_report.total_memory_traffic_unfused / 1e6

    print(f"  Unfused traffic: {unfused_estimate:.2f} MB (estimated from fusion analysis)")
    print(f"  Fused traffic:   {fused_memory:.2f} MB")
    print(f"  Reduction:       {fusion_report.data_movement_reduction * 100:.1f}%")
    print(f"  Savings:         {unfused_estimate - fused_memory:.2f} MB")

    print(f"\nFusion Statistics:")
    print(f"  Average ops per subgraph: {fusion_report.avg_fusion_size:.1f}")
    print(f"  Largest fused subgraph:   {fusion_report.max_fusion_size} operators")

    # Detailed fusion report
    print("\n" + "=" * 80)
    print("FUSION REPORT")
    print("=" * 80)
    print(fusion_report.summary_stats())

    # Show top fused subgraphs
    print("\n" + "=" * 80)
    print("TOP 10 FUSED SUBGRAPHS BY FLOPS")
    print("=" * 80)

    sorted_subgraphs = sorted(fusion_report.fused_subgraphs,
                             key=lambda sg: sg.total_flops, reverse=True)[:10]

    for i, sg in enumerate(sorted_subgraphs, 1):
        flop_pct = sg.total_flops / max(1, fusion_report.total_flops) * 100

        print(f"\n{i}. Subgraph {sg.subgraph_id}: {sg.fusion_pattern}")
        print(f"   Operators: {sg.num_operators} ({', '.join(op.value for op in sg.operation_types[:3])}...)")
        print(f"   FLOPs: {sg.total_flops / 1e9:.3f} G ({flop_pct:.1f}% of total)")
        print(f"   Memory (external): {(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes) / 1e6:.2f} MB")
        print(f"   Memory (internal): {sg.internal_bytes / 1e6:.2f} MB (stays in cache)")
        print(f"   Data movement reduction: {sg.data_movement_reduction() * 100:.1f}%")
        print(f"   Arithmetic Intensity: {sg.arithmetic_intensity:.2f} FLOPs/byte")
        print(f"   Bottleneck: {sg.recommended_bottleneck.value}")

    # Fusion pattern breakdown
    print("\n" + "=" * 80)
    print("FUSION PATTERN ANALYSIS")
    print("=" * 80)

    patterns_by_size = {}
    for sg in fusion_report.fused_subgraphs:
        size = sg.num_operators
        if size not in patterns_by_size:
            patterns_by_size[size] = []
        patterns_by_size[size].append(sg)

    for size in sorted(patterns_by_size.keys(), reverse=True):
        subgraphs = patterns_by_size[size]
        print(f"\n{size}-operator fusions: {len(subgraphs)} subgraphs")

        # Show unique patterns
        pattern_counts = {}
        for sg in subgraphs:
            pattern = sg.fusion_pattern
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pattern}: {count}")

    # Summary
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    reduction_ratio = unfused_report.total_subgraphs / max(1, fusion_report.total_subgraphs)

    print(f"\n1. Execution Efficiency:")
    print(f"   - {reduction_ratio:.1f}× fewer kernel launches ({unfused_report.total_subgraphs} → {fusion_report.total_subgraphs})")
    print(f"   - Each subgraph averages {fusion_report.avg_fusion_size:.1f} fused operators")

    print(f"\n2. Memory Efficiency:")
    print(f"   - {fusion_report.data_movement_reduction * 100:.1f}% reduction in global memory traffic")
    print(f"   - {(unfused_estimate - fused_memory):.1f} MB of data stays in cache/registers")

    print(f"\n3. Hardware Mapping:")
    print(f"   - Unfused: {unfused_report.total_subgraphs} tiny kernels → poor GPU utilization")
    print(f"   - Fused: {fusion_report.total_subgraphs} coarse kernels → better SM occupancy")

    print(f"\n4. Most Common Fusion Pattern:")
    top_pattern = max(fusion_report.fusion_patterns.items(), key=lambda x: x[1])
    print(f"   - {top_pattern[0]}: {top_pattern[1]} occurrences")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test fusion-based partitioning')
    parser.add_argument('--model', default='resnet18',
                       choices=['resnet18', 'mobilenet_v2', 'efficientnet_b0'],
                       help='Model to test')
    args = parser.parse_args()

    test_fusion(args.model)


if __name__ == "__main__":
    main()

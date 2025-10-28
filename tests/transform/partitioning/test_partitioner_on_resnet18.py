#!/usr/bin/env python
"""
Test GraphPartitioner and ConcurrencyAnalyzer on ResNet-18

This validates Phase 1 of the realistic performance modeling:
- Graph partitioning works correctly
- Concurrency analysis produces reasonable results
- Statistics are accurate
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.transform.partitioning import GraphPartitioner
from graphs.analysis.concurrency import ConcurrencyAnalyzer


def test_resnet18():
    """Test partitioner on ResNet-18"""

    print("=" * 80)
    print("Testing GraphPartitioner and ConcurrencyAnalyzer on ResNet-18")
    print("=" * 80)

    # Load model
    print("\n[1/5] Loading ResNet-18...")
    model = models.resnet18(weights=None)
    model.eval()

    # FX trace
    print("[2/5] FX tracing...")
    input_tensor = torch.randn(1, 3, 224, 224)

    try:
        fx_graph = symbolic_trace(model)
    except Exception as e:
        print(f"Error during FX trace: {e}")
        print("This is expected for some models. Trying with custom tracer...")
        assert False, f"FX trace failed: {e}"

    # Shape propagation
    print("[3/5] Shape propagation...")
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    # Partition graph
    print("[4/5] Partitioning graph...")
    partitioner = GraphPartitioner()
    partition_report = partitioner.partition(fx_graph)

    # Analyze concurrency
    print("[5/5] Analyzing concurrency...")
    analyzer = ConcurrencyAnalyzer()
    concurrency = analyzer.analyze(partition_report)

    # Add concurrency to report
    partition_report.concurrency = concurrency

    # Print results
    print("\n" + "=" * 80)
    print("PARTITION REPORT")
    print("=" * 80)
    print(partition_report.summary_stats())

    print("\n" + "=" * 80)
    print("CONCURRENCY ANALYSIS")
    print("=" * 80)
    print(concurrency.explanation)

    # Detailed subgraph analysis
    print("\n" + "=" * 80)
    print("TOP 10 SUBGRAPHS BY FLOPS")
    print("=" * 80)

    sorted_subgraphs = sorted(partition_report.subgraphs,
                             key=lambda sg: sg.flops, reverse=True)[:10]

    for i, sg in enumerate(sorted_subgraphs, 1):
        parallelism_str = ""
        if sg.parallelism:
            parallelism_str = (f"{sg.parallelism.total_threads:,} threads "
                             f"(B={sg.parallelism.batch}, "
                             f"C={sg.parallelism.channels}, "
                             f"S={sg.parallelism.spatial})")

        print(f"\n{i}. {sg.node_name}")
        print(f"   Type: {sg.operation_type.value}")
        print(f"   FLOPs: {sg.flops / 1e9:.3f} G ({sg.flops / partition_report.total_flops * 100:.1f}%)")
        print(f"   Memory: {(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes) / 1e6:.2f} MB")
        print(f"   Arithmetic Intensity: {sg.arithmetic_intensity:.2f} FLOPs/byte")
        print(f"   Parallelism: {parallelism_str}")
        print(f"   Bottleneck: {sg.recommended_bottleneck.value}")

        # Concurrency info
        if sg.node_id in concurrency.subgraph_concurrency:
            sc = concurrency.subgraph_concurrency[sg.node_id]
            print(f"   Concurrency: {sc.independent_threads:,} independent threads, "
                 f"optimal {sc.optimal_hardware_units} hardware units")

    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    checks_passed = 0
    checks_total = 0

    # Check 1: Total FLOPs reasonable for ResNet-18
    checks_total += 1
    expected_flops = 3.79e9  # 3.79 GFLOPs for ResNet-18
    actual_flops = partition_report.total_flops
    error = abs(actual_flops - expected_flops) / expected_flops
    if error < 0.2:  # within 20%
        print(f"✓ Total FLOPs: {actual_flops / 1e9:.2f} G (expected ~3.79 G, error {error * 100:.1f}%)")
        checks_passed += 1
    else:
        print(f"✗ Total FLOPs: {actual_flops / 1e9:.2f} G (expected ~3.79 G, error {error * 100:.1f}%)")

    # Check 2: Reasonable number of subgraphs
    checks_total += 1
    if 50 < partition_report.total_subgraphs < 200:
        print(f"✓ Subgraph count: {partition_report.total_subgraphs} (reasonable for ResNet-18)")
        checks_passed += 1
    else:
        print(f"✗ Subgraph count: {partition_report.total_subgraphs} (expected 50-200)")

    # Check 3: Concurrency analysis present
    checks_total += 1
    if concurrency.total_subgraphs > 0:
        print(f"✓ Concurrency analysis: {concurrency.num_stages} stages, "
             f"max {concurrency.max_parallel_ops_per_stage} parallel ops")
        checks_passed += 1
    else:
        print(f"✗ Concurrency analysis failed")

    # Check 4: Critical path exists
    checks_total += 1
    if concurrency.critical_path_length > 0:
        print(f"✓ Critical path: {concurrency.critical_path_length} operations, "
             f"{concurrency.critical_path_flops / 1e9:.2f} GFLOPs")
        checks_passed += 1
    else:
        print(f"✗ Critical path analysis failed")

    # Check 5: Parallelism detected
    checks_total += 1
    avg_threads = sum(sg.parallelism.total_threads for sg in partition_report.subgraphs
                     if sg.parallelism) / max(1, partition_report.total_subgraphs)
    if avg_threads > 1000:
        print(f"✓ Thread-level parallelism: {avg_threads:,.0f} threads average")
        checks_passed += 1
    else:
        print(f"✗ Thread-level parallelism too low: {avg_threads:,.0f} threads")

    print(f"\n{checks_passed}/{checks_total} validation checks passed")

    assert checks_passed == checks_total, f"Only {checks_passed}/{checks_total} validation checks passed"


if __name__ == "__main__":
    try:
        test_resnet18()
        exit(0)
    except AssertionError:
        exit(1)

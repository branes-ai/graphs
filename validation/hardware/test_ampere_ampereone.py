#!/usr/bin/env python
"""
Test Ampere AmpereOne 192-core ARM Server CPU Mapper

This script validates the Ampere AmpereOne mapper implementation by running
ResNet-50 inference and verifying performance characteristics.

Expected characteristics:
- High core count (192 cores)
- Good parallelism utilization
- Strong INT8/FP16 performance (native ARM SIMD)
- High memory bandwidth (332.8 GB/s DDR5)
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graphs.characterize.fusion_partitioner import FusionBasedPartitioner
from src.graphs.characterize.cpu_mapper import create_ampere_ampereone_192_mapper
from src.graphs.characterize.hardware_mapper import Precision


def test_ampere_ampereone_resnet50():
    """Test Ampere AmpereOne with ResNet-50"""
    print("=" * 80)
    print("Testing Ampere AmpereOne 192-core ARM Server CPU")
    print("=" * 80)

    # Create ResNet-50 model
    print("\n1. Creating ResNet-50 model...")
    model = models.resnet50(weights=None)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)

    # Trace and propagate shapes
    print("2. Tracing model with PyTorch FX...")
    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(input_tensor)

    # Partition into fused subgraphs
    print("3. Partitioning into fused subgraphs...")
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(traced)
    print(f"   → {len(fusion_report.fused_subgraphs)} fused subgraphs")

    # Create execution stages (sequential for simplicity)
    execution_stages = [[i] for i in range(len(fusion_report.fused_subgraphs))]

    # Test with different precisions
    precisions = [Precision.FP32, Precision.FP16, Precision.INT8]

    for precision in precisions:
        print(f"\n{'='*80}")
        print(f"Testing {precision.name}")
        print(f"{'='*80}")

        # Create mapper
        mapper = create_ampere_ampereone_192_mapper()

        # Map graph to hardware
        print(f"4. Mapping to Ampere AmpereOne (192 cores)...")
        hw_report = mapper.map_graph(
            fusion_report,
            execution_stages,
            batch_size=1,
            precision=precision
        )

        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS: {precision.name}")
        print(f"{'='*80}")
        print(f"Hardware: {hw_report.hardware_name}")
        print(f"Total Subgraphs: {hw_report.total_subgraphs}")
        print(f"Execution Stages: {hw_report.total_execution_stages}")
        print(f"\nCore Utilization:")
        print(f"  Peak Cores Used: {hw_report.peak_compute_units_used:.1f} / 192")
        print(f"  Average Cores Used: {hw_report.average_compute_units_used:.1f}")
        print(f"  Peak Utilization: {hw_report.peak_utilization*100:.1f}%")
        print(f"  Average Utilization: {hw_report.average_utilization*100:.1f}%")
        print(f"\nPerformance:")
        print(f"  Total Latency: {hw_report.total_latency*1000:.2f} ms")
        print(f"  Throughput: {1.0/hw_report.total_latency:.2f} FPS")
        print(f"  Total Energy: {hw_report.total_energy*1000:.2f} mJ")
        print(f"\nBottleneck Analysis:")
        print(f"  Compute Bound: {hw_report.compute_bound_count}")
        print(f"  Memory Bound: {hw_report.memory_bound_count}")
        print(f"  Bandwidth Bound: {hw_report.bandwidth_bound_count}")
        print(f"  Balanced: {hw_report.balanced_count}")
        print(f"\nEfficiency:")
        print(f"  Naive Latency: {hw_report.naive_latency*1000:.2f} ms")
        print(f"  Correction Factor: {hw_report.latency_correction_factor:.2f}×")

        # Verify expected characteristics
        print(f"\n{'='*80}")
        print("VALIDATION CHECKS")
        print(f"{'='*80}")

        checks = []

        # Check 1: High core count available
        checks.append(("192 cores available", hw_report.peak_compute_units_used <= 192))

        # Check 2: Reasonable utilization (should use many cores for ResNet-50)
        checks.append((
            "Reasonable utilization (>10%)",
            hw_report.average_utilization > 0.1
        ))

        # Check 3: Latency should be finite and positive
        checks.append((
            "Valid latency",
            hw_report.total_latency > 0 and hw_report.total_latency < 100
        ))

        # Check 4: Energy should be finite and positive
        checks.append((
            "Valid energy",
            hw_report.total_energy > 0 and hw_report.total_energy < 1000
        ))

        # Check 5: Should have bottleneck classifications
        checks.append((
            "Bottleneck analysis present",
            (hw_report.compute_bound_count + hw_report.memory_bound_count +
             hw_report.bandwidth_bound_count + hw_report.balanced_count) > 0
        ))

        # Print validation results
        all_passed = True
        for check_name, passed in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {check_name}")
            if not passed:
                all_passed = False

        if all_passed:
            print(f"\n✓ All validation checks passed for {precision.name}!")
        else:
            print(f"\n✗ Some validation checks failed for {precision.name}")
            return False

    print("\n" + "=" * 80)
    print("✓ Ampere AmpereOne mapper test completed successfully!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_ampere_ampereone_resnet50()
    sys.exit(0 if success else 1)

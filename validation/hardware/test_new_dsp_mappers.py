#!/usr/bin/env python
"""
Test script for newly added DSP/NPU mappers:
- CEVA NeuPro-M NPM11
- Cadence Tensilica Vision Q8
- Synopsys ARC EV7x

Tests basic functionality with ResNet-50 model.
"""

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.graphs.characterize.fusion_partitioner import FusionBasedPartitioner
from src.graphs.characterize.dsp_mapper import (
    create_ceva_neupro_npm11_mapper,
    create_cadence_vision_q8_mapper,
    create_synopsys_arc_ev7x_mapper,
)
from src.graphs.characterize.hardware_mapper import Precision


def extract_execution_stages(fusion_report):
    """Extract execution stages from fusion report"""
    stages = [[i] for i in range(len(fusion_report.fused_subgraphs))]
    return stages


def test_mapper(mapper_name, mapper_factory, precision='int8'):
    """Test a single mapper with ResNet-50"""
    print(f"\n{'='*80}")
    print(f"Testing: {mapper_name}")
    print(f"{'='*80}")

    # Create ResNet-50 model
    print("Creating ResNet-50 model...")
    model = models.resnet50(weights=None)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)

    # FX trace
    print("Tracing model with FX...")
    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(input_tensor)

    # Fusion partitioning
    print("Running fusion-based partitioning...")
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(traced)

    print(f"\nFusion Report:")
    print(f"  Total FLOPs: {fusion_report.total_flops / 1e9:.3f} GFLOPs")
    print(f"  Total Memory Traffic: {fusion_report.total_memory_traffic_fused / 1e6:.2f} MB")
    print(f"  Fused Subgraphs: {len(fusion_report.fused_subgraphs)}")

    # Extract execution stages
    execution_stages = extract_execution_stages(fusion_report)

    # Create mapper
    print(f"\nCreating {mapper_name} mapper...")
    mapper = mapper_factory()

    # Map to hardware
    print(f"Mapping to hardware...")
    if precision == 'int8':
        prec = Precision.INT8
    elif precision == 'fp32':
        prec = Precision.FP32
    else:
        prec = Precision.FP16

    hw_report = mapper.map_graph(
        fusion_report,
        execution_stages,
        batch_size=1,
        precision=prec
    )

    # Print results
    print(f"\n{mapper_name} Results:")
    print(f"  Latency: {hw_report.total_latency * 1000:.3f} ms")
    print(f"  Throughput: {1000.0 / (hw_report.total_latency * 1000):.1f} FPS")
    print(f"  Energy: {hw_report.total_energy * 1000:.3f} mJ")
    print(f"  Average Utilization: {hw_report.average_utilization * 100:.1f}%")
    print(f"  Peak Utilization: {hw_report.peak_utilization * 100:.1f}%")
    print(f"  Compute Units: {hw_report.average_compute_units_used:.1f} avg, {hw_report.peak_compute_units_used} peak")

    # Bottleneck analysis
    print(f"\nBottleneck Analysis:")
    print(f"  Compute Bound: {hw_report.compute_bound_count}")
    print(f"  Memory Bound: {hw_report.memory_bound_count}")
    print(f"  Bandwidth Bound: {hw_report.bandwidth_bound_count}")
    print(f"  Balanced: {hw_report.balanced_count}")

    return hw_report


def main():
    """Run tests for all new DSP/NPU mappers"""
    print("="*80)
    print("Testing New DSP/NPU Mappers")
    print("="*80)
    print("\nTesting CEVA NeuPro, Cadence Tensilica, and Synopsys ARC mappers")
    print("with ResNet-50 model @ INT8 precision")

    # Test CEVA NeuPro-M NPM11
    try:
        ceva_report = test_mapper(
            "CEVA NeuPro-M NPM11",
            create_ceva_neupro_npm11_mapper,
            precision='int8'
        )
        print(f"\n✓ CEVA NeuPro-M NPM11 mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ CEVA NeuPro-M NPM11 mapper: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Test Cadence Tensilica Vision Q8
    try:
        cadence_report = test_mapper(
            "Cadence Tensilica Vision Q8",
            create_cadence_vision_q8_mapper,
            precision='int8'
        )
        print(f"\n✓ Cadence Tensilica Vision Q8 mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ Cadence Tensilica Vision Q8 mapper: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Test Synopsys ARC EV7x
    try:
        synopsys_report = test_mapper(
            "Synopsys ARC EV7x",
            create_synopsys_arc_ev7x_mapper,
            precision='int8'
        )
        print(f"\n✓ Synopsys ARC EV7x mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ Synopsys ARC EV7x mapper: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON (ResNet-50 @ INT8)")
    print("="*80)
    print(f"\n{'Hardware':<30} {'Latency (ms)':<15} {'FPS':<10} {'Energy (mJ)':<12} {'Util %':<10}")
    print("-"*80)

    try:
        print(f"{'CEVA NeuPro-M NPM11':<30} "
              f"{ceva_report.total_latency * 1000:<15.3f} "
              f"{1000.0 / (ceva_report.total_latency * 1000):<10.1f} "
              f"{ceva_report.total_energy * 1000:<12.3f} "
              f"{ceva_report.average_utilization * 100:<10.1f}")
    except:
        pass

    try:
        print(f"{'Cadence Tensilica Vision Q8':<30} "
              f"{cadence_report.total_latency * 1000:<15.3f} "
              f"{1000.0 / (cadence_report.total_latency * 1000):<10.1f} "
              f"{cadence_report.total_energy * 1000:<12.3f} "
              f"{cadence_report.average_utilization * 100:<10.1f}")
    except:
        pass

    try:
        print(f"{'Synopsys ARC EV7x':<30} "
              f"{synopsys_report.total_latency * 1000:<15.3f} "
              f"{1000.0 / (synopsys_report.total_latency * 1000):<10.1f} "
              f"{synopsys_report.total_energy * 1000:<12.3f} "
              f"{synopsys_report.average_utilization * 100:<10.1f}")
    except:
        pass

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Test script for IP core mappers:
- CEVA NeuPro-M NPM11
- Cadence Tensilica Vision Q8
- Synopsys ARC EV7x

These are licensable IP cores for SoC integration, targeting
semiconductor vendors who integrate them into custom chips.

Tests basic functionality with ResNet-50 model @ INT8 precision.
"""

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.transform.partitioning import FusionBasedPartitioner
from graphs.hardware.mappers.dsp import (
    create_ceva_neupro_npm11_mapper,
    create_cadence_vision_q8_mapper,
    create_synopsys_arc_ev7x_mapper,
)
from graphs.hardware.resource_model import Precision


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

    # BOM cost if available
    if mapper.resource_model.bom_cost_profile:
        bom = mapper.resource_model.bom_cost_profile
        print(f"\nBOM Cost Profile:")
        print(f"  Total BOM Cost: ${bom.total_bom_cost:.2f}")
        print(f"  Silicon Die: ${bom.silicon_die_cost:.2f}")
        print(f"  Package: ${bom.package_cost:.2f}")
        print(f"  Process Node: {bom.process_node}")
        print(f"  Volume Tier: {bom.volume_tier}")

    return hw_report, mapper


def main():
    """Run tests for all IP core mappers"""
    print("="*80)
    print("Testing IP Core Mappers")
    print("="*80)
    print("\nThese are licensable IP cores for SoC integration.")
    print("Target customers: Semiconductor vendors, chip designers")
    print("Testing with ResNet-50 model @ INT8 precision\n")

    reports = {}
    mappers = {}

    # Test CEVA NeuPro-M NPM11
    try:
        ceva_report, ceva_mapper = test_mapper(
            "CEVA NeuPro-M NPM11",
            create_ceva_neupro_npm11_mapper,
            precision='int8'
        )
        reports['CEVA NeuPro-M NPM11'] = ceva_report
        mappers['CEVA NeuPro-M NPM11'] = ceva_mapper
        print(f"\n✓ CEVA NeuPro-M NPM11 mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ CEVA NeuPro-M NPM11 mapper: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Test Cadence Tensilica Vision Q8
    try:
        cadence_report, cadence_mapper = test_mapper(
            "Cadence Tensilica Vision Q8",
            create_cadence_vision_q8_mapper,
            precision='int8'
        )
        reports['Cadence Tensilica Vision Q8'] = cadence_report
        mappers['Cadence Tensilica Vision Q8'] = cadence_mapper
        print(f"\n✓ Cadence Tensilica Vision Q8 mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ Cadence Tensilica Vision Q8 mapper: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Test Synopsys ARC EV7x
    try:
        synopsys_report, synopsys_mapper = test_mapper(
            "Synopsys ARC EV7x",
            create_synopsys_arc_ev7x_mapper,
            precision='int8'
        )
        reports['Synopsys ARC EV7x'] = synopsys_report
        mappers['Synopsys ARC EV7x'] = synopsys_mapper
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
    print(f"\n{'IP Core':<30} {'Latency (ms)':<15} {'FPS':<10} {'Energy (mJ)':<12} {'Util %':<10}")
    print("-"*80)

    for name, report in reports.items():
        try:
            print(f"{name:<30} "
                  f"{report.total_latency * 1000:<15.3f} "
                  f"{1000.0 / (report.total_latency * 1000):<10.1f} "
                  f"{report.total_energy * 1000:<12.3f} "
                  f"{report.average_utilization * 100:<10.1f}")
        except:
            pass

    # BOM comparison if available
    print("\n" + "="*80)
    print("BOM COST COMPARISON")
    print("="*80)
    print(f"\n{'IP Core':<30} {'Total BOM':<15} {'Die Cost':<12} {'Process':<10}")
    print("-"*80)

    for name, mapper in mappers.items():
        try:
            if mapper.resource_model.bom_cost_profile:
                bom = mapper.resource_model.bom_cost_profile
                print(f"{name:<30} "
                      f"${bom.total_bom_cost:<14.2f} "
                      f"${bom.silicon_die_cost:<11.2f} "
                      f"{bom.process_node:<10}")
        except:
            pass

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

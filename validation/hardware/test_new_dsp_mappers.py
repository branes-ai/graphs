#!/usr/bin/env python
"""
Test script for DSP-based AI accelerators:
- Qualcomm Hexagon DSPs (QRB5165, QCS6490, SA8775P, Snapdragon Ride)
- TI C7x DSPs (TDA4VM, TDA4VL, TDA4AL, TDA4VH)

These are integrated DSP processors in complete SoC platforms,
NOT licensable IP cores.

Tests basic functionality with ResNet-50 model.
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
    create_qrb5165_mapper,
    create_qualcomm_sa8775p_mapper,
    create_qualcomm_snapdragon_ride_mapper,
    create_ti_tda4vm_mapper,
    create_ti_tda4vl_mapper,
    create_ti_tda4al_mapper,
    create_ti_tda4vh_mapper,
)
from graphs.hardware.resource_model import Precision


def extract_execution_stages(fusion_report):
    """Extract execution stages from fusion report"""
    stages = [[i] for i in range(len(fusion_report.fused_subgraphs))]
    return stages


def _test_mapper_helper(mapper_name, mapper_factory, precision='int8'):
    """Helper function to test a single mapper with ResNet-50"""
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

    return hw_report, mapper


def main():
    """Run tests for all DSP-based AI accelerator mappers"""
    print("="*80)
    print("Testing DSP-Based AI Accelerators")
    print("="*80)
    print("\nThese are integrated DSP processors in complete SoC platforms.")
    print("Target customers: OEMs building edge/automotive AI systems")
    print("Testing with ResNet-50 model @ INT8 precision\n")

    reports = {}
    mappers = {}

    # Test Qualcomm Hexagon DSPs
    print("\n" + "="*80)
    print("QUALCOMM HEXAGON DSPs")
    print("="*80)

    try:
        qrb5165_report, qrb5165_mapper = _test_mapper_helper(
            "Qualcomm QRB5165 (Hexagon 698)",
            create_qrb5165_mapper,
            precision='int8'
        )
        reports['QRB5165'] = qrb5165_report
        mappers['QRB5165'] = qrb5165_mapper
        print(f"\n✓ QRB5165 mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ QRB5165 mapper: FAILED - {e}")
        import traceback
        traceback.print_exc()


    try:
        sa8775p_report, sa8775p_mapper = _test_mapper_helper(
            "Qualcomm SA8775P (Automotive)",
            create_qualcomm_sa8775p_mapper,
            precision='int8'
        )
        reports['SA8775P'] = sa8775p_report
        mappers['SA8775P'] = sa8775p_mapper
        print(f"\n✓ SA8775P mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ SA8775P mapper: FAILED - {e}")
        import traceback
        traceback.print_exc()

    try:
        ride_report, ride_mapper = _test_mapper_helper(
            "Qualcomm Snapdragon Ride (L4/L5)",
            create_qualcomm_snapdragon_ride_mapper,
            precision='int8'
        )
        reports['Snapdragon Ride'] = ride_report
        mappers['Snapdragon Ride'] = ride_mapper
        print(f"\n✓ Snapdragon Ride mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ Snapdragon Ride mapper: FAILED - {e}")
        import traceback
        traceback.print_exc()

    # Test TI C7x DSPs
    print("\n" + "="*80)
    print("TEXAS INSTRUMENTS C7x DSPs")
    print("="*80)

    try:
        tda4vm_report, tda4vm_mapper = _test_mapper_helper(
            "TI TDA4VM (8 TOPS)",
            create_ti_tda4vm_mapper,
            precision='int8'
        )
        reports['TDA4VM'] = tda4vm_report
        mappers['TDA4VM'] = tda4vm_mapper
        print(f"\n✓ TDA4VM mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ TDA4VM mapper: FAILED - {e}")
        import traceback
        traceback.print_exc()

    try:
        tda4vl_report, tda4vl_mapper = _test_mapper_helper(
            "TI TDA4VL (4 TOPS)",
            create_ti_tda4vl_mapper,
            precision='int8'
        )
        reports['TDA4VL'] = tda4vl_report
        mappers['TDA4VL'] = tda4vl_mapper
        print(f"\n✓ TDA4VL mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ TDA4VL mapper: FAILED - {e}")
        import traceback
        traceback.print_exc()

    try:
        tda4al_report, tda4al_mapper = _test_mapper_helper(
            "TI TDA4AL (2 TOPS)",
            create_ti_tda4al_mapper,
            precision='int8'
        )
        reports['TDA4AL'] = tda4al_report
        mappers['TDA4AL'] = tda4al_mapper
        print(f"\n✓ TDA4AL mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ TDA4AL mapper: FAILED - {e}")
        import traceback
        traceback.print_exc()

    try:
        tda4vh_report, tda4vh_mapper = _test_mapper_helper(
            "TI TDA4VH (4 TOPS)",
            create_ti_tda4vh_mapper,
            precision='int8'
        )
        reports['TDA4VH'] = tda4vh_report
        mappers['TDA4VH'] = tda4vh_mapper
        print(f"\n✓ TDA4VH mapper: SUCCESS")
    except Exception as e:
        print(f"\n✗ TDA4VH mapper: FAILED - {e}")
        import traceback
        traceback.print_exc()

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON (ResNet-50 @ INT8)")
    print("="*80)
    print(f"\n{'Platform':<30} {'Latency (ms)':<15} {'FPS':<10} {'Energy (mJ)':<12} {'Util %':<10}")
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
    print(f"\n{'Platform':<30} {'Total BOM':<15} {'Process':<12} {'Market':<15}")
    print("-"*80)

    for name, mapper in mappers.items():
        try:
            if mapper.resource_model.bom_cost_profile:
                bom = mapper.resource_model.bom_cost_profile
                market = "Automotive" if "TDA4" in name or "SA8775P" in name or "Ride" in name else "Robotics/Edge"
                print(f"{name:<30} "
                      f"${bom.total_bom_cost:<14.2f} "
                      f"{bom.process_node:<12} "
                      f"{market:<15}")
        except:
            pass

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

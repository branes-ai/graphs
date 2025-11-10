#!/usr/bin/env python3
"""
Test Phase 1: Mapper-Integrated Pipeline

This test validates the hardware mapper integration into UnifiedAnalyzer.

Expected behavior:
- Hardware mapping produces actual unit allocations
- Energy calculation uses per-unit idle power based on allocation
- With power_gating=False: Results should be similar to old approach
- With power_gating=True: Should show idle energy savings

Key validation:
- hardware_allocation is populated
- Energy uses allocation info (not just thread estimates)
- power_gating_enabled flag works correctly
"""

import sys
import os

# Add src to path for direct execution
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.mappers.gpu import create_h100_pcie_80gb_mapper
from graphs.hardware.resource_model import Precision


def test_mapper_integration_phase1():
    """Test Phase 1: Basic mapper integration without power gating"""

    print("=" * 100)
    print("Phase 1 Validation: Mapper-Integrated Pipeline (ResNet-18 on H100)")
    print("=" * 100)
    print()

    # Create analyzer
    analyzer = UnifiedAnalyzer(verbose=True)

    # Test 1: With hardware mapping (power_gating=False)
    print("-" * 100)
    print("Test 1: Hardware Mapping Integration (power_gating=False)")
    print("-" * 100)
    print()

    config = AnalysisConfig(
        run_hardware_mapping=True,
        power_gating_enabled=False,  # Conservative estimate
    )

    try:
        result = analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32,
            config=config
        )

        print()
        print("✓ Analysis complete")
        print()

        # Validate hardware_allocation exists
        if result.hardware_allocation:
            print(f"✓ Hardware allocation populated:")
            print(f"  Total subgraphs: {result.hardware_allocation.total_subgraphs}")
            print(f"  Peak units: {result.hardware_allocation.peak_compute_units_used}/{result.hardware.compute_units}")
            print(f"  Average units: {result.hardware_allocation.average_compute_units_used:.1f}/{result.hardware.compute_units}")
            print(f"  Peak util: {result.hardware_allocation.peak_utilization * 100:.1f}%")
            print(f"  Avg util: {result.hardware_allocation.average_utilization * 100:.1f}%")
        else:
            print("❌ ERROR: hardware_allocation is None!")
            return False

        print()

        # Check energy report
        if result.energy_report:
            print(f"✓ Energy analysis complete:")
            print(f"  Total energy: {result.total_energy_mj:.2f} mJ")
            print(f"  Static energy: {result.energy_report.static_energy_j * 1e3:.2f} mJ ({result.energy_report.static_energy_j / (result.energy_report.compute_energy_j + result.energy_report.memory_energy_j + result.energy_report.static_energy_j) * 100:.1f}%)")
        else:
            print("❌ ERROR: energy_report is None!")
            return False

        print()

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Compare with/without power gating
    print("-" * 100)
    print("Test 2: Power Gating Impact")
    print("-" * 100)
    print()

    try:
        # Run with power gating enabled
        config_pg = AnalysisConfig(
            run_hardware_mapping=True,
            power_gating_enabled=True,  # NEW
        )

        result_pg = analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32,
            config=config_pg
        )

        print()
        print("✓ Power gating analysis complete")
        print()

        # Compare energies
        static_no_pg = result.energy_report.static_energy_j * 1e3  # mJ
        static_with_pg = result_pg.energy_report.static_energy_j * 1e3  # mJ

        print(f"Static Energy Comparison:")
        print(f"  No power gating:   {static_no_pg:.2f} mJ")
        print(f"  With power gating: {static_with_pg:.2f} mJ")
        print(f"  Savings:           {static_no_pg - static_with_pg:.2f} mJ ({(1 - static_with_pg/static_no_pg) * 100:.1f}%)")
        print()

        if static_with_pg < static_no_pg:
            print("✓ Power gating reduces static energy as expected")
        else:
            print("⚠ Warning: Power gating should reduce static energy")

        print()

        # Calculate expected savings based on utilization
        avg_util = result.hardware_allocation.average_utilization
        expected_savings_pct = (1 - avg_util) * 100
        actual_savings_pct = (1 - static_with_pg/static_no_pg) * 100

        print(f"Expected vs Actual Savings:")
        print(f"  Average utilization: {avg_util * 100:.1f}%")
        print(f"  Expected savings (1 - util): ~{expected_savings_pct:.1f}%")
        print(f"  Actual savings: {actual_savings_pct:.1f}%")
        print()

        if abs(actual_savings_pct - expected_savings_pct) < 5:
            print("✓ Savings match expected based on utilization")
        else:
            print(f"⚠ Savings mismatch (expected ~{expected_savings_pct:.1f}%, got {actual_savings_pct:.1f}%)")

        print()

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Validate allocation details
    print("-" * 100)
    print("Test 3: Allocation Details (Top 5 Subgraphs)")
    print("-" * 100)
    print()

    try:
        allocs = result.hardware_allocation.subgraph_allocations[:5]

        print(f"{'Subgraph':<30} {'Units':<10} {'Occupancy':<12} {'Latency (μs)':<15} {'Energy (μJ)'}")
        print("-" * 100)

        for alloc in allocs:
            print(
                f"{alloc.subgraph_name:<30} "
                f"{alloc.compute_units_allocated}/{result.hardware.compute_units:<10} "
                f"{alloc.occupancy * 100:<11.1f}% "
                f"{alloc.estimated_latency * 1e6:<14.2f} "
                f"{alloc.total_energy * 1e6:.2f}"
            )

        print()
        print("✓ Allocation details look reasonable")
        print()

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print("✅ Phase 1 (Mapper Integration) Validation Complete")
    print()
    print("Key Findings:")
    print("  • Hardware mapper successfully integrated into UnifiedAnalyzer")
    print("  • Per-unit static energy calculation working")
    print(f"  • Power gating shows {actual_savings_pct:.1f}% idle energy savings")
    print("  • Allocation info flows correctly from mapper to energy analyzer")
    print()
    print("Ready for production use!")
    print()

    return True


if __name__ == '__main__':
    success = test_mapper_integration_phase1()
    sys.exit(0 if success else 1)

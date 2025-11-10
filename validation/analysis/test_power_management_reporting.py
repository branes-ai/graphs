#!/usr/bin/env python3
"""
Test Enhanced Power Management Reporting

This test validates the enhanced energy reporting with power management fields.

Expected behavior:
- EnergyReport shows power management section
- Allocated vs unallocated unit energy breakdown
- Power gating savings clearly visible
- Per-subgraph power management details available
"""

import sys
import os

# Add src to path for direct execution
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.mappers.gpu import create_h100_pcie_80gb_mapper
from graphs.hardware.resource_model import Precision


def test_power_management_reporting():
    """Test enhanced power management reporting"""

    print("=" * 100)
    print("Enhanced Power Management Reporting Test (ResNet-18 on H100)")
    print("=" * 100)
    print()

    analyzer = UnifiedAnalyzer(verbose=False)

    # Test 1: Report WITHOUT power gating
    print("-" * 100)
    print("Test 1: Energy Report WITHOUT Power Gating")
    print("-" * 100)
    print()

    config_no_pg = AnalysisConfig(
        run_hardware_mapping=True,
        power_gating_enabled=False,
    )

    result_no_pg = analyzer.analyze_model(
        model_name='resnet18',
        hardware_name='H100',
        batch_size=1,
        precision=Precision.FP32,
        config=config_no_pg
    )

    # Print full energy report
    print(result_no_pg.energy_report.format_report())
    print()

    # Test 2: Report WITH power gating
    print("-" * 100)
    print("Test 2: Energy Report WITH Power Gating")
    print("-" * 100)
    print()

    config_with_pg = AnalysisConfig(
        run_hardware_mapping=True,
        power_gating_enabled=True,
    )

    result_with_pg = analyzer.analyze_model(
        model_name='resnet18',
        hardware_name='H100',
        batch_size=1,
        precision=Precision.FP32,
        config=config_with_pg
    )

    # Print full energy report
    print(result_with_pg.energy_report.format_report())
    print()

    # Test 3: Per-subgraph power management details
    print("-" * 100)
    print("Test 3: Per-Subgraph Power Management Details (Top 5)")
    print("-" * 100)
    print()

    top_subgraphs = sorted(
        result_with_pg.energy_report.energy_descriptors,
        key=lambda d: d.total_energy_j,
        reverse=True
    )[:5]

    for i, desc in enumerate(top_subgraphs, 1):
        print(f"{i}. {desc.subgraph_name}")
        print(desc.format_summary())
        print()

    # Test 4: Verify power management fields are populated
    print("-" * 100)
    print("Test 4: Validation Checks")
    print("-" * 100)
    print()

    checks_passed = []
    checks_failed = []

    # Check 1: Power management fields populated
    if result_with_pg.energy_report.average_allocated_units > 0:
        checks_passed.append("✓ average_allocated_units populated")
    else:
        checks_failed.append("✗ average_allocated_units is 0")

    # Check 2: Power gating enabled flag
    if result_with_pg.energy_report.power_gating_enabled:
        checks_passed.append("✓ power_gating_enabled = True")
    else:
        checks_failed.append("✗ power_gating_enabled = False (should be True)")

    if not result_no_pg.energy_report.power_gating_enabled:
        checks_passed.append("✓ power_gating_enabled = False (no PG config)")
    else:
        checks_failed.append("✗ power_gating_enabled = True (should be False)")

    # Check 3: Power gating savings exist
    if result_with_pg.energy_report.total_power_gating_savings_j > 0:
        checks_passed.append(f"✓ Power gating savings: {result_with_pg.energy_report.total_power_gating_savings_j * 1e3:.2f} mJ")
    else:
        checks_failed.append("✗ No power gating savings calculated")

    # Check 4: No savings when power gating disabled
    if result_no_pg.energy_report.total_power_gating_savings_j == 0:
        checks_passed.append("✓ No savings when power gating disabled")
    else:
        checks_failed.append(f"✗ Unexpected savings with PG disabled: {result_no_pg.energy_report.total_power_gating_savings_j * 1e3:.2f} mJ")

    # Check 5: Allocated vs unallocated breakdown
    total_idle_no_pg = (result_no_pg.energy_report.total_allocated_units_energy_j +
                        result_no_pg.energy_report.total_unallocated_units_energy_j)
    total_idle_with_pg = (result_with_pg.energy_report.total_allocated_units_energy_j +
                          result_with_pg.energy_report.total_unallocated_units_energy_j)

    if abs(total_idle_no_pg - result_no_pg.energy_report.static_energy_j) < 1e-6:
        checks_passed.append("✓ Idle energy breakdown sums to static energy (no PG)")
    else:
        checks_failed.append(f"✗ Idle energy breakdown mismatch (no PG): {total_idle_no_pg:.6f} vs {result_no_pg.energy_report.static_energy_j:.6f}")

    if abs(total_idle_with_pg - result_with_pg.energy_report.static_energy_j) < 1e-6:
        checks_passed.append("✓ Idle energy breakdown sums to static energy (with PG)")
    else:
        checks_failed.append(f"✗ Idle energy breakdown mismatch (with PG): {total_idle_with_pg:.6f} vs {result_with_pg.energy_report.static_energy_j:.6f}")

    # Check 6: Per-subgraph fields populated
    subgraph_with_pg = result_with_pg.energy_report.energy_descriptors[0]
    if subgraph_with_pg.allocated_units > 0:
        checks_passed.append(f"✓ Per-subgraph allocated_units: {subgraph_with_pg.allocated_units}")
    else:
        checks_failed.append("✗ Per-subgraph allocated_units is 0")

    if subgraph_with_pg.power_gating_savings_j > 0:
        checks_passed.append(f"✓ Per-subgraph power gating savings: {subgraph_with_pg.power_gating_savings_j * 1e6:.2f} μJ")
    else:
        checks_failed.append("✗ Per-subgraph power gating savings is 0")

    # Print results
    print("Passed Checks:")
    for check in checks_passed:
        print(f"  {check}")
    print()

    if checks_failed:
        print("Failed Checks:")
        for check in checks_failed:
            print(f"  {check}")
        print()
        return False

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print("✅ Enhanced Power Management Reporting Validation Complete")
    print()
    print("Key Findings:")
    print(f"  • Average units allocated: {result_with_pg.energy_report.average_allocated_units:.1f}/{result_with_pg.hardware.compute_units}")
    print(f"  • Allocated units idle energy: {result_with_pg.energy_report.total_allocated_units_energy_j * 1e3:.2f} mJ")
    print(f"  • Unallocated units idle energy (no PG): {result_no_pg.energy_report.total_unallocated_units_energy_j * 1e3:.2f} mJ")
    print(f"  • Power gating savings: {result_with_pg.energy_report.total_power_gating_savings_j * 1e3:.2f} mJ")
    print(f"  • Per-subgraph details available for {len(result_with_pg.energy_report.energy_descriptors)} subgraphs")
    print()
    print("Enhanced reporting provides complete power management visibility!")
    print()

    return True


if __name__ == '__main__':
    success = test_power_management_reporting()
    sys.exit(0 if success else 1)

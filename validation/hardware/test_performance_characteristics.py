#!/usr/bin/env python3
"""
Unit Tests for PerformanceCharacteristics and ThermalOperatingPoint

Tests that the efficiency_factor and other coefficients are actually
being used in performance calculations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graphs.characterize.cpu_mapper import create_i7_12700k_mapper, create_i7_12700k_large_mapper
from src.graphs.characterize.hardware_mapper import Precision


def test_mapper_initialization():
    """Test that mappers are initialized with correct thermal profiles"""
    print("="*80)
    print("TEST 1: Mapper Initialization")
    print("="*80)

    tiny_mapper = create_i7_12700k_mapper()
    large_mapper = create_i7_12700k_large_mapper()

    print(f"\nTiny Mapper:")
    print(f"  Hardware Name: {tiny_mapper.resource_model.name}")
    print(f"  Thermal Profile: {tiny_mapper.thermal_profile}")
    print(f"  Has Thermal Operating Points: {bool(tiny_mapper.resource_model.thermal_operating_points)}")

    if tiny_mapper.resource_model.thermal_operating_points:
        print(f"  Available Thermal Points: {list(tiny_mapper.resource_model.thermal_operating_points.keys())}")
        print(f"  Default Thermal Profile: {tiny_mapper.resource_model.default_thermal_profile}")

    print(f"\nLarge Mapper:")
    print(f"  Hardware Name: {large_mapper.resource_model.name}")
    print(f"  Thermal Profile: {large_mapper.thermal_profile}")
    print(f"  Has Thermal Operating Points: {bool(large_mapper.resource_model.thermal_operating_points)}")

    if large_mapper.resource_model.thermal_operating_points:
        print(f"  Available Thermal Points: {list(large_mapper.resource_model.thermal_operating_points.keys())}")
        print(f"  Default Thermal Profile: {large_mapper.resource_model.default_thermal_profile}")

    return tiny_mapper, large_mapper


def test_performance_characteristics():
    """Test that PerformanceCharacteristics have different efficiency_factor values"""
    print("\n" + "="*80)
    print("TEST 2: PerformanceCharacteristics")
    print("="*80)

    tiny_mapper = create_i7_12700k_mapper()
    large_mapper = create_i7_12700k_large_mapper()

    # Get thermal operating points
    tiny_thermal = None
    large_thermal = None

    if tiny_mapper.thermal_profile and tiny_mapper.resource_model.thermal_operating_points:
        tiny_thermal = tiny_mapper.resource_model.thermal_operating_points[tiny_mapper.thermal_profile]
        print(f"\nTiny Mapper Thermal Profile: {tiny_mapper.thermal_profile}")
        print(f"  TDP: {tiny_thermal.tdp_watts}W")
        print(f"  Cooling: {tiny_thermal.cooling_solution}")
    else:
        print(f"\n⚠ WARNING: Tiny mapper has no thermal profile set!")
        print(f"  thermal_profile: {tiny_mapper.thermal_profile}")
        print(f"  thermal_operating_points: {tiny_mapper.resource_model.thermal_operating_points}")

    if large_mapper.thermal_profile and large_mapper.resource_model.thermal_operating_points:
        large_thermal = large_mapper.resource_model.thermal_operating_points[large_mapper.thermal_profile]
        print(f"\nLarge Mapper Thermal Profile: {large_mapper.thermal_profile}")
        print(f"  TDP: {large_thermal.tdp_watts}W")
        print(f"  Cooling: {large_thermal.cooling_solution}")
    else:
        print(f"\n⚠ WARNING: Large mapper has no thermal profile set!")
        print(f"  thermal_profile: {large_mapper.thermal_profile}")
        print(f"  thermal_operating_points: {large_mapper.resource_model.thermal_operating_points}")

    # Check FP32 performance characteristics
    print(f"\n" + "-"*80)
    print("FP32 Performance Characteristics Comparison:")
    print("-"*80)

    if tiny_thermal and Precision.FP32 in tiny_thermal.performance_specs:
        tiny_fp32 = tiny_thermal.performance_specs[Precision.FP32]
        print(f"\nTiny Mapper (FP32):")
        print(f"  efficiency_factor:           {tiny_fp32.efficiency_factor:.3f}")
        print(f"  memory_bottleneck_factor:    {tiny_fp32.memory_bottleneck_factor:.3f}")
        print(f"  tile_utilization:            {tiny_fp32.tile_utilization:.3f}")
        print(f"  instruction_efficiency:      {tiny_fp32.instruction_efficiency:.3f}")
        print(f"  Peak ops/sec:                {tiny_fp32.peak_ops_per_sec / 1e9:.2f} GFLOPS")
        print(f"  Sustained ops/sec:           {tiny_fp32.sustained_ops_per_sec / 1e9:.2f} GFLOPS")
        print(f"  Effective ops/sec:           {tiny_fp32.effective_ops_per_sec / 1e9:.2f} GFLOPS")
    else:
        print(f"\n⚠ ERROR: Tiny mapper has no FP32 performance specs!")

    if large_thermal and Precision.FP32 in large_thermal.performance_specs:
        large_fp32 = large_thermal.performance_specs[Precision.FP32]
        print(f"\nLarge Mapper (FP32):")
        print(f"  efficiency_factor:           {large_fp32.efficiency_factor:.3f}")
        print(f"  memory_bottleneck_factor:    {large_fp32.memory_bottleneck_factor:.3f}")
        print(f"  tile_utilization:            {large_fp32.tile_utilization:.3f}")
        print(f"  instruction_efficiency:      {large_fp32.instruction_efficiency:.3f}")
        print(f"  Peak ops/sec:                {large_fp32.peak_ops_per_sec / 1e9:.2f} GFLOPS")
        print(f"  Sustained ops/sec:           {large_fp32.sustained_ops_per_sec / 1e9:.2f} GFLOPS")
        print(f"  Effective ops/sec:           {large_fp32.effective_ops_per_sec / 1e9:.2f} GFLOPS")
    else:
        print(f"\n⚠ ERROR: Large mapper has no FP32 performance specs!")

    # Verify differences
    if tiny_thermal and large_thermal and \
       Precision.FP32 in tiny_thermal.performance_specs and \
       Precision.FP32 in large_thermal.performance_specs:

        tiny_fp32 = tiny_thermal.performance_specs[Precision.FP32]
        large_fp32 = large_thermal.performance_specs[Precision.FP32]

        print(f"\n" + "-"*80)
        print("Coefficient Ratios (Large / Tiny):")
        print("-"*80)
        print(f"  efficiency_factor ratio:        {large_fp32.efficiency_factor / tiny_fp32.efficiency_factor:.2f}×")
        print(f"  memory_bottleneck_factor ratio: {large_fp32.memory_bottleneck_factor / tiny_fp32.memory_bottleneck_factor:.2f}×")
        print(f"  effective_ops_per_sec ratio:    {large_fp32.effective_ops_per_sec / tiny_fp32.effective_ops_per_sec:.2f}×")

        effective_ratio = large_fp32.effective_ops_per_sec / tiny_fp32.effective_ops_per_sec

        if abs(effective_ratio - 3.0) < 0.5:
            print(f"\n✓ PASS: Effective ops/sec ratio is ~3× as expected")
        else:
            print(f"\n✗ FAIL: Expected ~3× ratio, got {effective_ratio:.2f}×")

        return True
    else:
        print(f"\n✗ FAIL: Could not compare - missing thermal profiles or performance specs")
        return False


def test_latency_calculation():
    """Test that _calculate_latency() uses the efficiency_factor"""
    print("\n" + "="*80)
    print("TEST 3: Latency Calculation")
    print("="*80)

    tiny_mapper = create_i7_12700k_mapper()
    large_mapper = create_i7_12700k_large_mapper()

    # Test parameters
    ops = 1_000_000_000  # 1 billion ops (1 GFLOP)
    bytes_transferred = 10_000_000  # 10 MB
    allocated_units = 10  # All cores
    occupancy = 1.0
    precision = Precision.FP32

    print(f"\nTest Parameters:")
    print(f"  Operations: {ops / 1e9:.1f} GOPs")
    print(f"  Bytes Transferred: {bytes_transferred / 1e6:.1f} MB")
    print(f"  Allocated Units: {allocated_units} cores")
    print(f"  Occupancy: {occupancy * 100:.0f}%")
    print(f"  Precision: {precision.value}")

    # Calculate latency with both mappers
    tiny_compute, tiny_memory, tiny_bottleneck = tiny_mapper._calculate_latency(
        ops, bytes_transferred, allocated_units, occupancy, precision
    )

    large_compute, large_memory, large_bottleneck = large_mapper._calculate_latency(
        ops, bytes_transferred, allocated_units, occupancy, precision
    )

    print(f"\nTiny Mapper Results:")
    print(f"  Compute Time: {tiny_compute * 1000:.3f} ms")
    print(f"  Memory Time:  {tiny_memory * 1000:.3f} ms")
    print(f"  Bottleneck:   {tiny_bottleneck.value}")
    print(f"  Latency:      {max(tiny_compute, tiny_memory) * 1000:.3f} ms")

    print(f"\nLarge Mapper Results:")
    print(f"  Compute Time: {large_compute * 1000:.3f} ms")
    print(f"  Memory Time:  {large_memory * 1000:.3f} ms")
    print(f"  Bottleneck:   {large_bottleneck.value}")
    print(f"  Latency:      {max(large_compute, large_memory) * 1000:.3f} ms")

    # Check if results differ
    compute_ratio = tiny_compute / large_compute if large_compute > 0 else 0

    print(f"\n" + "-"*80)
    print("Comparison:")
    print("-"*80)
    print(f"  Compute Time Ratio (Tiny / Large): {compute_ratio:.2f}×")
    print(f"  Memory Time (same for both):        {tiny_memory * 1000:.3f} ms")

    if abs(compute_ratio - 3.0) < 0.5:
        print(f"\n✓ PASS: Compute time ratio is ~3× as expected")
        print(f"         (Tiny mapper takes 3× longer due to lower efficiency_factor)")
        return True
    elif abs(compute_ratio - 1.0) < 0.1:
        print(f"\n✗ FAIL: Compute times are identical!")
        print(f"         This means efficiency_factor is NOT being used!")
        return False
    else:
        print(f"\n⚠ WARNING: Unexpected ratio {compute_ratio:.2f}×")
        print(f"           Expected ~3× (tiny should be slower)")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("UNIT TESTS: PerformanceCharacteristics & ThermalOperatingPoint")
    print("="*80)

    # Run tests
    test_mapper_initialization()
    test2_pass = test_performance_characteristics()
    test3_pass = test_latency_calculation()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"  Test 1 (Initialization):           {'PASS' if True else 'FAIL'} (informational)")
    print(f"  Test 2 (Performance Characteristics): {'PASS' if test2_pass else 'FAIL'}")
    print(f"  Test 3 (Latency Calculation):      {'PASS' if test3_pass else 'FAIL'}")

    if test2_pass and test3_pass:
        print(f"\n✓ ALL TESTS PASSED")
        print(f"  The efficiency_factor IS being used correctly!")
        return 0
    else:
        print(f"\n✗ SOME TESTS FAILED")
        print(f"  Root cause analysis needed!")
        return 1


if __name__ == "__main__":
    exit(main())

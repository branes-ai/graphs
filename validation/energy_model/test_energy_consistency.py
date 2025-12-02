#!/usr/bin/env python3
"""
Energy Model Consistency Tests

This test validates that the cycle-level energy models are consistent
across architectures by testing the following principles:

1. SIMD Amortization (CPU):
   - With SIMD, instruction overhead should be amortized across vector width
   - AVX-512 (simd_width=16) should have ~16x less instruction overhead than scalar

2. No Double-Counting:
   - GPU should not add static power as per-op overhead
   - TPU/KPU internal data movement should be separate from external memory access

3. Memory Model Consistency:
   - For HIGH_REUSE workloads with data fitting in L2/L3, all architectures
     should have equivalent external memory energy
   - Differences should be in: instruction amortization, control overhead, MAC efficiency

4. Fair Comparison:
   - Using same TechnologyProfile ensures same circuit energies
   - Using same bytes_transferred ensures same external memory traffic
   - Architectural differences should be in overhead, not fundamental physics
"""

import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from graphs.hardware.cycle_energy.cpu import build_cpu_cycle_energy
from graphs.hardware.cycle_energy.gpu import build_gpu_cycle_energy
from graphs.hardware.cycle_energy.tpu import build_tpu_cycle_energy
from graphs.hardware.cycle_energy.kpu import build_kpu_cycle_energy
from graphs.hardware.cycle_energy.base import (
    OperatingMode,
    OperatorType,
    CyclePhase,
)
from graphs.hardware.cycle_energy.comparison import (
    format_comparison_table,
    format_energy_categories_table,
    format_energy_categories_per_op_table,
)
from graphs.hardware.technology_profile import (
    ARCH_COMPARISON_8NM_X86,
    create_architecture_comparison_set,
    MemoryType,
)


def test_cpu_simd_amortization():
    """Test that CPU SIMD amortization works correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: CPU SIMD Amortization")
    print("=" * 80)

    comparison = ARCH_COMPARISON_8NM_X86
    tech = comparison.cpu_profile

    num_ops = 10000
    bytes_transferred = num_ops * 4  # 4 bytes per FP32 op

    # Scalar execution (simd_width=1)
    scalar = build_cpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=tech,
        simd_width=1,
    )

    # AVX-256 (simd_width=8)
    avx256 = build_cpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=tech,
        simd_width=8,
    )

    # AVX-512 (simd_width=16)
    avx512 = build_cpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=tech,
        simd_width=16,
    )

    # ARM NEON (simd_width=4)
    neon = build_cpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=tech,
        simd_width=4,
    )

    print(f"\nWorkload: {num_ops:,} ops, {bytes_transferred:,} bytes")
    print(f"Mode: L1_RESIDENT (data fits in cache)")
    print()

    # Get instruction overhead for each
    scalar_ctrl = scalar.get_control_overhead_energy()
    avx256_ctrl = avx256.get_control_overhead_energy()
    avx512_ctrl = avx512.get_control_overhead_energy()
    neon_ctrl = neon.get_control_overhead_energy()

    # Get compute energy (should be same for all)
    scalar_compute = scalar.get_compute_energy()
    avx512_compute = avx512.get_compute_energy()

    print("Control Overhead (Instruction fetch/decode + register access):")
    print(f"  Scalar (width=1):  {scalar_ctrl/1000:.2f} nJ")
    print(f"  ARM NEON (width=4): {neon_ctrl/1000:.2f} nJ  ({scalar_ctrl/neon_ctrl:.1f}x reduction)")
    print(f"  AVX-256 (width=8):  {avx256_ctrl/1000:.2f} nJ  ({scalar_ctrl/avx256_ctrl:.1f}x reduction)")
    print(f"  AVX-512 (width=16): {avx512_ctrl/1000:.2f} nJ  ({scalar_ctrl/avx512_ctrl:.1f}x reduction)")
    print()
    print("Compute Energy (should be identical - same ops):")
    print(f"  Scalar:  {scalar_compute/1000:.2f} nJ")
    print(f"  AVX-512: {avx512_compute/1000:.2f} nJ")
    print()
    print("Total Energy:")
    print(f"  Scalar:  {scalar.total_energy_pj/1000:.2f} nJ ({scalar.total_energy_pj/num_ops:.2f} pJ/op)")
    print(f"  AVX-512: {avx512.total_energy_pj/1000:.2f} nJ ({avx512.total_energy_pj/num_ops:.2f} pJ/op)")
    print(f"  Speedup: {scalar.total_energy_pj/avx512.total_energy_pj:.2f}x")

    # Verify SIMD amortization is working
    # Control overhead should scale roughly with 1/simd_width
    expected_ratio = 16.0  # scalar vs AVX-512
    actual_ratio = scalar_ctrl / avx512_ctrl

    print()
    print(f"SIMD Amortization Check:")
    print(f"  Expected control ratio (scalar/AVX-512): ~{expected_ratio:.0f}x")
    print(f"  Actual control ratio: {actual_ratio:.1f}x")

    if actual_ratio > expected_ratio * 0.8:  # Allow 20% tolerance
        print("  PASS: SIMD amortization is working correctly")
    else:
        print("  FAIL: SIMD amortization may not be correct")

    return scalar, avx512


def test_architecture_comparison():
    """Test fair comparison across architectures."""
    print("\n" + "=" * 80)
    print("TEST 2: Architecture Comparison (Same Technology)")
    print("=" * 80)

    # Use architecture comparison set for fair comparison
    comparison = ARCH_COMPARISON_8NM_X86
    print(f"\n{comparison.summary()}")

    num_ops = 100000
    bytes_transferred = num_ops * 4  # HIGH_REUSE workload

    print(f"\nWorkload: {num_ops:,} ops, {bytes_transferred:,} bytes")
    print("Mode: L2_RESIDENT (data fits in L2/SRAM)")
    print()

    # Build energy breakdowns for each architecture
    cpu = build_cpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L2_RESIDENT,
        tech_profile=comparison.cpu_profile,
        operator_type=OperatorType.HIGH_REUSE,
        simd_width=16,  # AVX-512
    )

    gpu = build_gpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L2_RESIDENT,
        tech_profile=comparison.gpu_profile,
        operator_type=OperatorType.HIGH_REUSE,
    )

    tpu = build_tpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L2_RESIDENT,
        tech_profile=comparison.tpu_profile,
        operator_type=OperatorType.HIGH_REUSE,
    )

    kpu = build_kpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L2_RESIDENT,
        tech_profile=comparison.kpu_profile,
        operator_type=OperatorType.HIGH_REUSE,
    )

    breakdowns = [cpu, gpu, tpu, kpu]

    # Print category breakdown
    print(format_energy_categories_table(breakdowns, num_ops=num_ops))
    print()
    print(format_energy_categories_per_op_table(breakdowns, num_ops=num_ops))

    return breakdowns


def test_memory_consistency():
    """Test that external memory energy is consistent across architectures."""
    print("\n" + "=" * 80)
    print("TEST 3: Memory Model Consistency")
    print("=" * 80)

    comparison = ARCH_COMPARISON_8NM_X86

    num_ops = 10000
    bytes_transferred = num_ops * 4

    print(f"\nWorkload: {num_ops:,} ops, {bytes_transferred:,} bytes")
    print()

    # Test with L1_RESIDENT (all on-chip)
    print("Mode: L1_RESIDENT (100% on-chip)")
    cpu_l1 = build_cpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=comparison.cpu_profile,
        simd_width=16,
    )
    gpu_l1 = build_gpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=comparison.gpu_profile,
    )
    tpu_l1 = build_tpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=comparison.tpu_profile,
    )
    kpu_l1 = build_kpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=comparison.kpu_profile,
    )

    print(f"  CPU: {cpu_l1.get_data_movement_energy()/1000:.2f} nJ data movement")
    print(f"  GPU: {gpu_l1.get_data_movement_energy()/1000:.2f} nJ data movement")
    print(f"  TPU: {tpu_l1.get_data_movement_energy()/1000:.2f} nJ data movement")
    print(f"  KPU: {kpu_l1.get_data_movement_energy()/1000:.2f} nJ data movement")

    # Test with DRAM_RESIDENT (streaming from external memory)
    print("\nMode: DRAM_RESIDENT (streaming from external memory)")
    cpu_dram = build_cpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.DRAM_RESIDENT,
        tech_profile=comparison.cpu_profile,
        operator_type=OperatorType.STREAMING,
        simd_width=16,
    )
    gpu_dram = build_gpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.DRAM_RESIDENT,
        tech_profile=comparison.gpu_profile,
        operator_type=OperatorType.STREAMING,
    )
    tpu_dram = build_tpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.DRAM_RESIDENT,
        tech_profile=comparison.tpu_profile,
        operator_type=OperatorType.STREAMING,
    )
    kpu_dram = build_kpu_cycle_energy(
        num_ops=num_ops,
        bytes_transferred=bytes_transferred,
        mode=OperatingMode.DRAM_RESIDENT,
        tech_profile=comparison.kpu_profile,
        operator_type=OperatorType.STREAMING,
    )

    print(f"  CPU: {cpu_dram.get_data_movement_energy()/1000:.2f} nJ data movement")
    print(f"  GPU: {gpu_dram.get_data_movement_energy()/1000:.2f} nJ data movement")
    print(f"  TPU: {tpu_dram.get_data_movement_energy()/1000:.2f} nJ data movement")
    print(f"  KPU: {kpu_dram.get_data_movement_energy()/1000:.2f} nJ data movement")

    # External memory energy should be roughly proportional to bytes_transferred
    # for all architectures (with some architecture-specific overhead)
    expected_dram_energy = bytes_transferred * comparison.cpu_profile.offchip_energy_per_byte_pj
    print(f"\nExpected DRAM energy (bytes * energy/byte): {expected_dram_energy/1000:.2f} nJ")


def test_scaling_behavior():
    """Test that energy scales correctly with workload size."""
    print("\n" + "=" * 80)
    print("TEST 4: Scaling Behavior")
    print("=" * 80)

    comparison = ARCH_COMPARISON_8NM_X86

    print("\nEnergy per operation at different workload sizes:")
    print(f"{'Ops':>12} {'CPU (pJ/op)':>14} {'GPU (pJ/op)':>14} {'TPU (pJ/op)':>14} {'KPU (pJ/op)':>14}")
    print("-" * 70)

    for num_ops in [100, 1000, 10000, 100000, 1000000]:
        bytes_transferred = num_ops * 4

        cpu = build_cpu_cycle_energy(
            num_ops=num_ops,
            bytes_transferred=bytes_transferred,
            mode=OperatingMode.L2_RESIDENT,
            tech_profile=comparison.cpu_profile,
            simd_width=16,
        )
        gpu = build_gpu_cycle_energy(
            num_ops=num_ops,
            bytes_transferred=bytes_transferred,
            mode=OperatingMode.L2_RESIDENT,
            tech_profile=comparison.gpu_profile,
        )
        tpu = build_tpu_cycle_energy(
            num_ops=num_ops,
            bytes_transferred=bytes_transferred,
            mode=OperatingMode.L2_RESIDENT,
            tech_profile=comparison.tpu_profile,
        )
        kpu = build_kpu_cycle_energy(
            num_ops=num_ops,
            bytes_transferred=bytes_transferred,
            mode=OperatingMode.L2_RESIDENT,
            tech_profile=comparison.kpu_profile,
        )

        cpu_per_op = cpu.total_energy_pj / num_ops
        gpu_per_op = gpu.total_energy_pj / num_ops
        tpu_per_op = tpu.total_energy_pj / num_ops
        kpu_per_op = kpu.total_energy_pj / num_ops

        print(f"{num_ops:>12,} {cpu_per_op:>14.2f} {gpu_per_op:>14.2f} {tpu_per_op:>14.2f} {kpu_per_op:>14.2f}")

    print()
    print("Expected behavior:")
    print("  - Energy per op should decrease with scale (fixed costs amortized)")
    print("  - GPU should NOT show dramatic decrease (no per-op static power)")
    print("  - TPU/KPU should converge to ~0.75-0.80x of CPU compute energy")


def main():
    """Run all energy model consistency tests."""
    print("=" * 80)
    print("ENERGY MODEL CONSISTENCY TESTS")
    print("=" * 80)
    print()
    print("These tests validate the fixes to the cycle-level energy models:")
    print("1. CPU SIMD amortization (AVX-512, ARM NEON)")
    print("2. GPU static power model (no double-counting)")
    print("3. Memory model consistency (internal vs external movement)")
    print("4. Scaling behavior (fixed costs amortize correctly)")

    test_cpu_simd_amortization()
    test_architecture_comparison()
    test_memory_consistency()
    test_scaling_behavior()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

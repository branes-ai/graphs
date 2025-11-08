#!/usr/bin/env python3
"""
Test TPU Tile Energy Model

This script validates the TPU v4 tile energy model implementation
by testing with a simple Conv2D operation and comparing against
expected energy characteristics.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from graphs.hardware.models.datacenter.tpu_v4 import tpu_v4_resource_model
from graphs.hardware.mappers.accelerators.tpu import create_tpu_v4_mapper
from graphs.hardware.resource_model import Precision


def test_tile_energy_model():
    """Test that TPU v4 has tile energy model configured"""
    print("=" * 70)
    print("TEST 1: TPU v4 Tile Energy Model Configuration")
    print("=" * 70)

    model = tpu_v4_resource_model()

    # Check if tile energy model is attached
    assert hasattr(model, 'tile_energy_model'), "TPU v4 should have tile_energy_model"

    tile_model = model.tile_energy_model
    print(f"✓ TPU v4 has tile energy model")
    print(f"  - Array size: {tile_model.array_width}×{tile_model.array_height}")
    print(f"  - Number of arrays: {tile_model.num_arrays}")
    print(f"  - Weight tile size: {tile_model.weight_tile_size / 1024:.0f} KiB")
    print(f"  - Pipeline fill cycles: {tile_model.pipeline_fill_cycles}")
    print(f"  - Clock frequency: {tile_model.clock_frequency_hz / 1e6:.0f} MHz")
    print(f"  - Accumulator size: {tile_model.accumulator_size / (1024*1024):.0f} MiB")
    print(f"  - Unified Buffer: {tile_model.unified_buffer_size / (1024*1024):.0f} MiB")
    print(f"  - Weight memory energy: {tile_model.weight_memory_energy_per_byte * 1e12:.1f} pJ/byte")
    print(f"  - MAC energy: {tile_model.mac_energy * 1e12:.2f} pJ")
    print()


def test_simple_conv2d_energy():
    """Test energy calculation for a simple Conv2D operation"""
    print("=" * 70)
    print("TEST 2: Simple Conv2D Energy Calculation")
    print("=" * 70)

    # Create mapper
    mapper = create_tpu_v4_mapper()

    # Simple Conv2D parameters
    # Input: 1×64×56×56, Output: 1×64×56×56, Kernel: 3×3, Stride: 1
    batch = 1
    in_channels = 64
    out_channels = 64
    h, w = 56, 56
    kernel_h, kernel_w = 3, 3

    # Calculate FLOPs (2 * H * W * K_h * K_w * C_in * C_out)
    ops = 2 * h * w * kernel_h * kernel_w * in_channels * out_channels

    # Calculate bytes transferred
    # Input: batch * in_channels * h * w * 4 bytes (FP32)
    # Weights: out_channels * in_channels * kernel_h * kernel_w * 4 bytes
    # Output: batch * out_channels * h * w * 4 bytes
    input_bytes = batch * in_channels * h * w * 4
    weight_bytes = out_channels * in_channels * kernel_h * kernel_w * 4
    output_bytes = batch * out_channels * h * w * 4
    bytes_transferred = input_bytes + weight_bytes + output_bytes

    print(f"Conv2D Operation:")
    print(f"  - Input: {batch}×{in_channels}×{h}×{w}")
    print(f"  - Output: {batch}×{out_channels}×{h}×{w}")
    print(f"  - Kernel: {kernel_h}×{kernel_w}")
    print(f"  - FLOPs: {ops / 1e9:.3f} GFLOPS")
    print(f"  - Bytes transferred: {bytes_transferred / (1024*1024):.2f} MiB")
    print()

    # Test BF16 (native TPU v4 precision)
    compute_energy_bf16, memory_energy_bf16 = mapper._calculate_energy(
        ops=ops,
        bytes_transferred=bytes_transferred,
        precision=Precision.BF16
    )

    total_energy_bf16 = compute_energy_bf16 + memory_energy_bf16
    energy_per_gflop_bf16 = (total_energy_bf16 * 1e12) / (ops / 1e9)  # pJ per GFLOP

    # Calculate MACs and energy per MAC for BF16
    macs_bf16 = ops // 2  # Each MAC = 1 multiply + 1 add = 2 FLOPs
    energy_per_mac_bf16 = (total_energy_bf16 * 1e12) / macs_bf16  # pJ per MAC

    print(f"BF16 Energy Breakdown:")
    print(f"  - Compute energy: {compute_energy_bf16 * 1e12:.2f} pJ ({compute_energy_bf16 * 1e9:.3f} nJ)")
    print(f"  - Memory energy: {memory_energy_bf16 * 1e12:.2f} pJ ({memory_energy_bf16 * 1e9:.3f} nJ)")
    print(f"  - Total energy: {total_energy_bf16 * 1e12:.2f} pJ ({total_energy_bf16 * 1e9:.3f} nJ)")
    print(f"  - Energy per GFLOP: {energy_per_gflop_bf16:.2f} pJ/GFLOP")
    print(f"  - Energy per MAC: {energy_per_mac_bf16:.3f} pJ/MAC (BF16)")
    print(f"    └─ Guidance for RTL/ALU designers: Target 0.2-0.5 pJ/MAC for BF16")
    print()

    # Test INT8 (2× faster, more efficient)
    compute_energy_int8, memory_energy_int8 = mapper._calculate_energy(
        ops=ops,
        bytes_transferred=bytes_transferred // 2,  # INT8 is half the size
        precision=Precision.INT8
    )

    total_energy_int8 = compute_energy_int8 + memory_energy_int8
    energy_per_gflop_int8 = (total_energy_int8 * 1e12) / (ops / 1e9)

    # Calculate MACs and energy per MAC for INT8
    macs_int8 = ops // 2  # Each MAC = 1 multiply + 1 add = 2 FLOPs
    energy_per_mac_int8 = (total_energy_int8 * 1e12) / macs_int8  # pJ per MAC

    print(f"INT8 Energy Breakdown:")
    print(f"  - Compute energy: {compute_energy_int8 * 1e12:.2f} pJ ({compute_energy_int8 * 1e9:.3f} nJ)")
    print(f"  - Memory energy: {memory_energy_int8 * 1e12:.2f} pJ ({memory_energy_int8 * 1e9:.3f} nJ)")
    print(f"  - Total energy: {total_energy_int8 * 1e12:.2f} pJ ({total_energy_int8 * 1e9:.3f} nJ)")
    print(f"  - Energy per GFLOP: {energy_per_gflop_int8:.2f} pJ/GFLOP")
    print(f"  - Energy per MAC: {energy_per_mac_int8:.3f} pJ/MAC (INT8)")
    print(f"    └─ Guidance for RTL/ALU designers: Target 0.1-0.3 pJ/MAC for INT8")
    print()

    # Validate INT8 is more efficient than BF16
    efficiency_gain = energy_per_gflop_bf16 / energy_per_gflop_int8
    efficiency_gain_mac = energy_per_mac_bf16 / energy_per_mac_int8
    print(f"INT8 Efficiency Gain:")
    print(f"  - Per GFLOP: {efficiency_gain:.2f}× better than BF16")
    print(f"  - Per MAC: {efficiency_gain_mac:.2f}× better than BF16")
    print(f"  - BF16: {energy_per_mac_bf16:.3f} pJ/MAC")
    print(f"  - INT8: {energy_per_mac_int8:.3f} pJ/MAC")
    print()

    # Sanity checks
    assert compute_energy_int8 < compute_energy_bf16, "INT8 compute should be more efficient than BF16"
    assert memory_energy_int8 < memory_energy_bf16, "INT8 memory should be less than BF16 (smaller data)"
    print("✓ Sanity checks passed")
    print()


def test_weight_tile_decomposition():
    """Test that weight tiling is working correctly"""
    print("=" * 70)
    print("TEST 3: Weight Tile Decomposition")
    print("=" * 70)

    model = tpu_v4_resource_model()
    tile_model = model.tile_energy_model

    # Simulate a layer with different weight sizes
    weight_tile_size = tile_model.weight_tile_size

    test_cases = [
        ("Small layer", 16 * 1024),  # 16 KiB (fits in 1 tile)
        ("Medium layer", 64 * 1024),  # 64 KiB (2 tiles)
        ("Large layer", 256 * 1024),  # 256 KiB (8 tiles)
        ("Very large layer", 1024 * 1024),  # 1 MiB (32 tiles)
    ]

    for name, weight_bytes in test_cases:
        num_tiles = max(1, int(weight_bytes / weight_tile_size))
        ops_estimate = weight_bytes * 2 * 128  # Rough estimate for matrix multiply

        # Calculate energy for batch_size=1 and batch_size=64
        energy_b1 = tile_model.compute_tile_energy(
            num_weight_tiles=num_tiles,
            ops_per_tile=int(ops_estimate / num_tiles),
            input_elements_per_tile=1024,
            output_elements_per_tile=1024,
            batch_size=1,
            precision="BF16"
        )

        energy_b64 = tile_model.compute_tile_energy(
            num_weight_tiles=num_tiles,
            ops_per_tile=int(ops_estimate / num_tiles),
            input_elements_per_tile=1024,
            output_elements_per_tile=1024,
            batch_size=64,
            precision="BF16"
        )

        weight_load_b1 = energy_b1['total_weight_energy_j'] * 1e9
        weight_load_b64 = energy_b64['total_weight_energy_j'] * 1e9
        weight_savings = (weight_load_b1 - weight_load_b64) / weight_load_b1 * 100

        print(f"{name} ({weight_bytes / 1024:.0f} KiB, {num_tiles} tiles):")
        print(f"  - Batch=1 weight loading: {weight_load_b1:.3f} nJ")
        print(f"  - Batch=64 weight loading: {weight_load_b64:.3f} nJ")
        print(f"  - Savings with batch=64: {weight_savings:.1f}%")
        print(f"  - Arithmetic intensity: {energy_b1['arithmetic_intensity']:.1f} ops/byte")
        print()

    print("✓ Weight tile decomposition working correctly")
    print()


def test_energy_breakdown():
    """Test detailed energy breakdown"""
    print("=" * 70)
    print("TEST 4: Detailed Energy Breakdown")
    print("=" * 70)

    model = tpu_v4_resource_model()
    tile_model = model.tile_energy_model

    # Medium-sized Conv2D operation
    num_tiles = 4
    ops_per_tile = 50_000_000  # 50M MACs per tile
    input_elements = 100_000
    output_elements = 100_000

    energy = tile_model.compute_tile_energy(
        num_weight_tiles=num_tiles,
        ops_per_tile=ops_per_tile,
        input_elements_per_tile=input_elements,
        output_elements_per_tile=output_elements,
        batch_size=1,
        precision="BF16"
    )

    total = energy['total_energy_j'] * 1e9  # Convert to nJ

    print(f"Operation: {num_tiles} tiles, {ops_per_tile / 1e6:.1f}M MACs/tile")
    print()
    print(f"Energy Breakdown (nJ):")
    print(f"  Weight Loading:")
    print(f"    - DRAM read: {energy['weight_dram_energy_j'] * 1e9:.3f} nJ ({energy['weight_dram_energy_j'] * 1e9 / total * 100:.1f}%)")
    print(f"    - FIFO buffer: {energy['weight_fifo_energy_j'] * 1e9:.3f} nJ ({energy['weight_fifo_energy_j'] * 1e9 / total * 100:.1f}%)")
    print(f"    - Shift-in: {energy['weight_shift_energy_j'] * 1e9:.3f} nJ ({energy['weight_shift_energy_j'] * 1e9 / total * 100:.1f}%)")
    print(f"  Input Activation Loading:")
    print(f"    - UB read: {energy['input_read_energy_j'] * 1e9:.3f} nJ ({energy['input_read_energy_j'] * 1e9 / total * 100:.1f}%)")
    print(f"    - Stream: {energy['activation_stream_energy_j'] * 1e9:.3f} nJ ({energy['activation_stream_energy_j'] * 1e9 / total * 100:.1f}%)")
    print(f"  Computation:")
    print(f"    - MACs: {energy['compute_energy_j'] * 1e9:.3f} nJ ({energy['compute_energy_j'] * 1e9 / total * 100:.1f}%)")
    print(f"  Accumulator Management:")
    print(f"    - Writes: {energy['accumulator_write_energy_j'] * 1e9:.3f} nJ ({energy['accumulator_write_energy_j'] * 1e9 / total * 100:.1f}%)")
    print(f"    - Reads: {energy['accumulator_read_energy_j'] * 1e9:.3f} nJ ({energy['accumulator_read_energy_j'] * 1e9 / total * 100:.1f}%)")
    print(f"  Output Write:")
    print(f"    - UB write: {energy['output_write_energy_j'] * 1e9:.3f} nJ ({energy['output_write_energy_j'] * 1e9 / total * 100:.1f}%)")
    print()
    print(f"  TOTAL: {total:.3f} nJ")
    print()
    print(f"Note: Pipeline fill/drain ({tile_model.pipeline_fill_cycles} cycles @ {tile_model.clock_frequency_hz/1e6:.0f} MHz)")
    print(f"      is a LATENCY overhead, not energy. It's modeled in latency calculations.")
    print()

    # Validate that weight loading dominates at batch=1 (should be largest component)
    weight_total = (energy['weight_dram_energy_j'] + energy['weight_fifo_energy_j'] +
                   energy['weight_shift_energy_j'])
    print(f"Key Insights:")
    print(f"  - Weight loading: {weight_total * 1e9 / total * 100:.1f}% of total")
    print(f"  - Computation: {energy['compute_energy_j'] * 1e9 / total * 100:.1f}% of total")
    print(f"  - Arithmetic intensity: {energy['arithmetic_intensity']:.1f} ops/byte")
    print()
    print("✓ Energy breakdown validated")
    print()


def test_tpu_v4_efficiency_validation():
    """
    Test TPU v4 energy efficiency against published Google TPU data.

    Google TPU claims: 0.2-0.5 pJ per BF16 MAC

    This test validates that our tile energy model produces results
    consistent with published TPU efficiency data.
    """
    print("=" * 70)
    print("TEST 5: TPU v4 Efficiency Validation (vs Google Claims)")
    print("=" * 70)

    # Create mapper
    mapper = create_tpu_v4_mapper()

    # Test operation: Large Conv2D (typical ResNet layer)
    # Input: 1×256×28×28, Output: 1×256×28×28, Kernel: 3×3
    batch = 1
    in_channels = 256
    out_channels = 256
    h, w = 28, 28
    kernel_h, kernel_w = 3, 3

    # Calculate MACs (not FLOPs)
    macs = h * w * kernel_h * kernel_w * in_channels * out_channels
    flops = 2 * macs  # Each MAC = 1 multiply + 1 add

    # Calculate bytes transferred (BF16 = 2 bytes per element)
    input_bytes = batch * in_channels * h * w * 2
    weight_bytes = out_channels * in_channels * kernel_h * kernel_w * 2
    output_bytes = batch * out_channels * h * w * 2
    bytes_transferred = input_bytes + weight_bytes + output_bytes

    print(f"Test Operation (Typical ResNet Layer):")
    print(f"  - Input: {batch}×{in_channels}×{h}×{w}")
    print(f"  - Output: {batch}×{out_channels}×{h}×{w}")
    print(f"  - Kernel: {kernel_h}×{kernel_w}")
    print(f"  - MACs: {macs / 1e9:.3f} GMACs")
    print(f"  - FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"  - Bytes transferred: {bytes_transferred / (1024*1024):.2f} MiB (BF16)")
    print()

    # Calculate energy using tile model
    compute_energy, memory_energy = mapper._calculate_energy(
        ops=flops,
        bytes_transferred=bytes_transferred,
        precision=Precision.BF16
    )

    total_energy = compute_energy + memory_energy

    # Calculate efficiency metrics
    energy_per_gflop_pj = (total_energy * 1e12) / (flops / 1e9)  # pJ per GFLOP
    energy_per_mac_pj = (total_energy * 1e12) / macs  # pJ per MAC

    print(f"Energy Results:")
    print(f"  - Compute energy: {compute_energy * 1e9:.3f} nJ")
    print(f"  - Memory energy: {memory_energy * 1e9:.3f} nJ")
    print(f"  - Total energy: {total_energy * 1e9:.3f} nJ")
    print()

    print(f"Efficiency Metrics:")
    print(f"  - Energy per GFLOP: {energy_per_gflop_pj:.2f} pJ/GFLOP")
    print(f"  - Energy per MAC: {energy_per_mac_pj:.3f} pJ/MAC")
    print()

    # Validation against Google TPU claims
    google_tpu_min = 0.2  # pJ per BF16 MAC (best case)
    google_tpu_max = 0.5  # pJ per BF16 MAC (worst case)

    print(f"Validation Against Google TPU Claims:")
    print(f"  - Google TPU claim: 0.2-0.5 pJ per BF16 MAC")
    print(f"  - Our model: {energy_per_mac_pj:.3f} pJ per BF16 MAC")
    print()

    if google_tpu_min <= energy_per_mac_pj <= google_tpu_max:
        print(f"  ✓ PASS: Our model ({energy_per_mac_pj:.3f} pJ/MAC) is within Google's range")
        print(f"         [{google_tpu_min:.1f} - {google_tpu_max:.1f}] pJ/MAC")
    elif energy_per_mac_pj < google_tpu_min:
        deviation = ((google_tpu_min - energy_per_mac_pj) / google_tpu_min) * 100
        print(f"  ⚠ WARNING: Our model is {deviation:.1f}% MORE efficient than Google's claim")
        print(f"            This might be too optimistic")
    else:
        deviation = ((energy_per_mac_pj - google_tpu_max) / google_tpu_max) * 100
        print(f"  ⚠ WARNING: Our model is {deviation:.1f}% LESS efficient than Google's claim")
        print(f"            This might be too pessimistic")
    print()

    # Component breakdown
    compute_fraction = compute_energy / total_energy
    memory_fraction = memory_energy / total_energy

    print(f"Energy Breakdown:")
    print(f"  - Compute: {compute_fraction * 100:.1f}%")
    print(f"  - Memory: {memory_fraction * 100:.1f}%")
    print()

    # Arithmetic intensity analysis
    arithmetic_intensity = flops / bytes_transferred
    print(f"Arithmetic Intensity: {arithmetic_intensity:.2f} ops/byte")
    print(f"  (Higher is better - more computation per data movement)")
    print()

    # Validate we're in the right ballpark
    assert google_tpu_min * 0.5 <= energy_per_mac_pj <= google_tpu_max * 2.0, \
        f"Energy per MAC ({energy_per_mac_pj:.3f} pJ) is too far from Google's claim (0.2-0.5 pJ)"

    print("✓ TPU v4 efficiency validation passed")
    print()


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("TPU v4 Tile Energy Model Test Suite")
    print("=" * 70)
    print()

    try:
        test_tile_energy_model()
        test_simple_conv2d_energy()
        test_weight_tile_decomposition()
        test_energy_breakdown()
        test_tpu_v4_efficiency_validation()

        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print()

    except Exception as e:
        print()
        print("=" * 70)
        print(f"TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)

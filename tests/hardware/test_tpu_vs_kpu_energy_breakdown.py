#!/usr/bin/env python3
"""
TPU vs KPU Energy Event Breakdown Comparison

This test performs a detailed apples-to-apples comparison of TPU and KPU energy models
by explicitly tracking ALL energy events for the SAME workload.

The goal is to ensure fair comparison by accounting for ALL data movement energy,
including TPU's Unified Buffer and Weight FIFO paths that may be implicit.

Workload: Standard MatMul (1024×1024 @ 1024, batch=1, BF16)
- Input:  [1, 1024, 1024] = 2 MiB
- Weight: [1024, 1024] = 2 MiB
- Output: [1, 1024, 1024] = 2 MiB
- Ops:    2.15 GOps (2 * 1024^3)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.models.datacenter.tpu_v4 import tpu_v4_resource_model
from graphs.hardware.models.automotive.kpu_t768 import kpu_t768_resource_model


def analyze_tpu_energy_events():
    """
    Detailed TPU energy event breakdown.

    TPU Memory Hierarchy (2-stage):
    1. Off-chip: HBM2e (DRAM equivalent)
    2. On-chip:  Unified Buffer (24-32 MiB, like L3)
                 Weight FIFO (staging buffer)
                 Accumulators (partial sum storage)

    Data Paths:
    - Weights:     HBM → Weight FIFO → Matrix Unit
    - Activations: HBM → Unified Buffer → Matrix Unit
    - Outputs:     Matrix Unit → Accumulators → Unified Buffer → HBM
    """
    print("="*80)
    print("TPU v4 Energy Event Breakdown")
    print("="*80)

    model = tpu_v4_resource_model()
    energy_model = model.tile_energy_model

    # Standard MatMul: 1024×1024 @ 1024
    M, N, K = 1024, 1024, 1024
    batch = 1

    # Calculate tile parameters
    # TPU v4: 128×128 array, so tiles are 128×128
    tile_size = 128

    # Proper 3D tiling: tile across M, N, K dimensions
    M_tiles = (M + tile_size - 1) // tile_size  # = 8
    N_tiles = (N + tile_size - 1) // tile_size  # = 8
    K_tiles = (K + tile_size - 1) // tile_size  # = 8

    # Total number of tiles = M_tiles × N_tiles × K_tiles
    num_weight_tiles = M_tiles * N_tiles * K_tiles  # = 512

    # Ops and bytes per tile
    ops_per_tile = 2 * tile_size * tile_size * tile_size  # 2 * 128^3
    input_elements_per_tile = tile_size * tile_size  # 128^2
    output_elements_per_tile = tile_size * tile_size  # 128^2

    # Run energy model
    result = energy_model.compute_tile_energy(
        num_weight_tiles=num_weight_tiles,
        ops_per_tile=ops_per_tile,
        input_elements_per_tile=input_elements_per_tile,
        output_elements_per_tile=output_elements_per_tile,
        batch_size=batch,
        precision="BF16"
    )

    print(f"\nWorkload: {M}×{N} @ {K} MatMul (batch={batch}, BF16)")
    print(f"  Total Ops:   {result['total_ops']/1e9:.3f} GOps")
    print(f"  Total Bytes: {result['total_bytes']/1e6:.3f} MB")
    print(f"  Num Tiles:   {result['num_tiles']}")

    print(f"\n{'Energy Event':<40} {'Energy (µJ)':<15} {'% of Total':<12}")
    print("-"*70)

    total_energy_uj = result['total_energy_j'] * 1e6

    # Weight loading (OFF-CHIP → ON-CHIP)
    print(f"\n{'WEIGHT LOADING (HBM → Weight FIFO → MXU)':=<70}")
    weight_dram_uj = result['weight_dram_energy_j'] * 1e6
    weight_dram_pct = (result['weight_dram_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'1. HBM read (off-chip DRAM)':<38} {weight_dram_uj:>13.2f}   {weight_dram_pct:>10.1f}%")

    weight_fifo_uj = result['weight_fifo_energy_j'] * 1e6
    weight_fifo_pct = (result['weight_fifo_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'2. Weight FIFO buffering (on-chip)':<38} {weight_fifo_uj:>13.2f}   {weight_fifo_pct:>10.1f}%")

    weight_shift_uj = result['weight_shift_energy_j'] * 1e6
    weight_shift_pct = (result['weight_shift_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'3. Shift into Matrix Unit':<38} {weight_shift_uj:>13.2f}   {weight_shift_pct:>10.1f}%")

    total_weight_uj = result['total_weight_energy_j'] * 1e6
    total_weight_pct = (result['total_weight_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'-'*38} {'-------------':>13}   {'----------':>10}")
    print(f"  {'SUBTOTAL: Weight Path':<38} {total_weight_uj:>13.2f}   {total_weight_pct:>10.1f}%")

    # Input activation loading (OFF-CHIP → ON-CHIP → MXU)
    print(f"\n{'ACTIVATION LOADING (HBM → Unified Buffer → MXU)':=<70}")
    input_read_uj = result['input_read_energy_j'] * 1e6
    input_read_pct = (result['input_read_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'4. Unified Buffer read (on-chip)':<38} {input_read_uj:>13.2f}   {input_read_pct:>10.1f}%")

    activation_stream_uj = result['activation_stream_energy_j'] * 1e6
    activation_stream_pct = (result['activation_stream_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'5. Stream into Matrix Unit':<38} {activation_stream_uj:>13.2f}   {activation_stream_pct:>10.1f}%")

    total_input_uj = result['total_input_energy_j'] * 1e6
    total_input_pct = (result['total_input_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'-'*38} {'-------------':>13}   {'----------':>10}")
    print(f"  {'SUBTOTAL: Activation Path':<38} {total_input_uj:>13.2f}   {total_input_pct:>10.1f}%")

    # Computation
    print(f"\n{'COMPUTATION (Matrix Unit MACs)':=<70}")
    compute_uj = result['compute_energy_j'] * 1e6
    compute_pct = (result['compute_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'6. Systolic array MACs':<38} {compute_uj:>13.2f}   {compute_pct:>10.1f}%")

    # Accumulator management
    print(f"\n{'ACCUMULATOR MANAGEMENT (Partial Sums)':=<70}")
    accum_write_uj = result['accumulator_write_energy_j'] * 1e6
    accum_write_pct = (result['accumulator_write_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'7. Write partial sums (during compute)':<38} {accum_write_uj:>13.2f}   {accum_write_pct:>10.1f}%")

    accum_read_uj = result['accumulator_read_energy_j'] * 1e6
    accum_read_pct = (result['accumulator_read_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'8. Read completed results (to UB)':<38} {accum_read_uj:>13.2f}   {accum_read_pct:>10.1f}%")

    total_accum_uj = result['total_accumulator_energy_j'] * 1e6
    total_accum_pct = (result['total_accumulator_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'-'*38} {'-------------':>13}   {'----------':>10}")
    print(f"  {'SUBTOTAL: Accumulator':<38} {total_accum_uj:>13.2f}   {total_accum_pct:>10.1f}%")

    # Output write (ON-CHIP → OFF-CHIP)
    print(f"\n{'OUTPUT WRITE (Unified Buffer → HBM)':=<70}")
    output_write_uj = result['output_write_energy_j'] * 1e6
    output_write_pct = (result['output_write_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'9. Unified Buffer write':<38} {output_write_uj:>13.2f}   {output_write_pct:>10.1f}%")

    # Total
    print(f"\n{'='*70}")
    print(f"  {'TOTAL ENERGY':<38} {total_energy_uj:>13.2f}   {'100.0%':>10}")

    # Calculate energy per MAC
    energy_per_mac_pj = (result['total_energy_j'] / (result['total_ops'] / 2)) * 1e12
    print(f"  {'Energy per MAC':<38} {energy_per_mac_pj:>13.3f} pJ")
    print(f"  {'Arithmetic Intensity':<38} {result['arithmetic_intensity']:>13.1f} ops/byte")

    # Categorize by hierarchy level
    print(f"\n{'ENERGY BY HIERARCHY LEVEL':=<70}")

    # Off-chip (HBM)
    off_chip_uj = weight_dram_uj
    off_chip_pct = (off_chip_uj / total_energy_uj) * 100
    print(f"  {'Off-chip (HBM read only)':<38} {off_chip_uj:>13.2f}   {off_chip_pct:>10.1f}%")

    # On-chip buffers (Weight FIFO, Unified Buffer, Accumulators)
    on_chip_buffer_uj = (weight_fifo_uj + input_read_uj +
                         total_accum_uj + output_write_uj)
    on_chip_buffer_pct = (on_chip_buffer_uj / total_energy_uj) * 100
    print(f"  {'On-chip buffers (FIFO/UB/Accum)':<38} {on_chip_buffer_uj:>13.2f}   {on_chip_buffer_pct:>10.1f}%")

    # Data movement (shift/stream)
    data_movement_uj = weight_shift_uj + activation_stream_uj
    data_movement_pct = (data_movement_uj / total_energy_uj) * 100
    print(f"  {'Data movement (shift/stream)':<38} {data_movement_uj:>13.2f}   {data_movement_pct:>10.1f}%")

    # Compute
    print(f"  {'Compute (MACs)':<38} {compute_uj:>13.2f}   {compute_pct:>10.1f}%")

    return result


def analyze_kpu_energy_events():
    """
    Detailed KPU energy event breakdown.

    KPU Memory Hierarchy (4-stage):
    1. Off-chip: HBM2 (DRAM equivalent)
    2. L3:       Distributed scratchpad (256 KiB/tile, like Unified Buffer)
    3. L2:       Tile-local cache (32 KiB/tile)
    4. L1:       PE-local cache (4 KiB/PE)

    Data Movement Engines:
    - DMA:        DRAM ↔ L3
    - BlockMover: L3 ↔ L2 (inter-tile)
    - Streamer:   L2 ↔ L1 (intra-tile)

    Unique KPU Events:
    - Token signature matching (distributed CAM-like)
    - SURE program loading
    - Token routing through mesh
    """
    print("\n\n")
    print("="*80)
    print("KPU-T768 Energy Event Breakdown")
    print("="*80)

    model = kpu_t768_resource_model()
    energy_model = model.tile_energy_model

    # Same workload: 1024×1024 @ 1024
    M, N, K = 1024, 1024, 1024
    batch = 1

    result = energy_model.compute_gemm_energy(
        M=M, N=N, K=K,
        batch_size=batch,
        precision="BF16",
        enable_fusion=False,
        num_fused_ops=1
    )

    print(f"\nWorkload: {M}×{N} @ {K} MatMul (batch={batch}, BF16)")
    print(f"  Total Ops:   {result['total_ops']/1e9:.3f} GOps")
    print(f"  Total Bytes: {result['total_bytes']/1e6:.3f} MB")

    print(f"\n{'Energy Event':<40} {'Energy (µJ)':<15} {'% of Total':<12}")
    print("-"*70)

    total_energy_uj = result['total_energy_j'] * 1e6

    # 4-Stage Memory Hierarchy
    print(f"\n{'4-STAGE MEMORY HIERARCHY':=<70}")

    dram_read_uj = result['dram_read_energy_j'] * 1e6
    dram_read_pct = (result['dram_read_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'1. DRAM read (off-chip HBM2)':<38} {dram_read_uj:>13.2f}   {dram_read_pct:>10.1f}%")

    dram_write_uj = result['dram_write_energy_j'] * 1e6
    dram_write_pct = (result['dram_write_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'2. DRAM write (off-chip HBM2)':<38} {dram_write_uj:>13.2f}   {dram_write_pct:>10.1f}%")

    l3_read_uj = result['l3_read_energy_j'] * 1e6
    l3_read_pct = (result['l3_read_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'3. L3 read (distributed scratchpad)':<38} {l3_read_uj:>13.2f}   {l3_read_pct:>10.1f}%")

    l3_write_uj = result['l3_write_energy_j'] * 1e6
    l3_write_pct = (result['l3_write_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'4. L3 write':<38} {l3_write_uj:>13.2f}   {l3_write_pct:>10.1f}%")

    l2_read_uj = result['l2_read_energy_j'] * 1e6
    l2_read_pct = (result['l2_read_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'5. L2 read (tile-local)':<38} {l2_read_uj:>13.2f}   {l2_read_pct:>10.1f}%")

    l2_write_uj = result['l2_write_energy_j'] * 1e6
    l2_write_pct = (result['l2_write_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'6. L2 write':<38} {l2_write_uj:>13.2f}   {l2_write_pct:>10.1f}%")

    l1_read_uj = result['l1_read_energy_j'] * 1e6
    l1_read_pct = (result['l1_read_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'7. L1 read (PE-local)':<38} {l1_read_uj:>13.2f}   {l1_read_pct:>10.1f}%")

    l1_write_uj = result['l1_write_energy_j'] * 1e6
    l1_write_pct = (result['l1_write_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'8. L1 write':<38} {l1_write_uj:>13.2f}   {l1_write_pct:>10.1f}%")

    total_mem_hier_uj = result['total_memory_hierarchy_energy_j'] * 1e6
    total_mem_hier_pct = (result['total_memory_hierarchy_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'-'*38} {'-------------':>13}   {'----------':>10}")
    print(f"  {'SUBTOTAL: Memory Hierarchy':<38} {total_mem_hier_uj:>13.2f}   {total_mem_hier_pct:>10.1f}%")

    # Data Movement Engines
    print(f"\n{'DATA MOVEMENT ENGINES':=<70}")

    dma_uj = result['dma_energy_j'] * 1e6
    dma_pct = (result['dma_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'9. DMA (DRAM ↔ L3)':<38} {dma_uj:>13.2f}   {dma_pct:>10.1f}%")

    blockmover_uj = result['blockmover_energy_j'] * 1e6
    blockmover_pct = (result['blockmover_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'10. BlockMover (L3 ↔ L2 inter-tile)':<38} {blockmover_uj:>13.2f}   {blockmover_pct:>10.1f}%")

    streamer_uj = result['streamer_energy_j'] * 1e6
    streamer_pct = (result['streamer_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'11. Streamer (L2 ↔ L1 intra-tile)':<38} {streamer_uj:>13.2f}   {streamer_pct:>10.1f}%")

    total_dme_uj = result['total_dme_energy_j'] * 1e6
    total_dme_pct = (result['total_dme_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'-'*38} {'-------------':>13}   {'----------':>10}")
    print(f"  {'SUBTOTAL: Data Movement Engines':<38} {total_dme_uj:>13.2f}   {total_dme_pct:>10.1f}%")

    # Token-based Execution (UNIQUE TO KPU)
    print(f"\n{'TOKEN-BASED EXECUTION (UNIQUE TO KPU)':=<70}")

    sig_match_uj = result['signature_matching_energy_j'] * 1e6
    sig_match_pct = (result['signature_matching_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'12. Token signature matching (CAM-like)':<38} {sig_match_uj:>13.2f}   {sig_match_pct:>10.1f}%")

    handshake_uj = result['handshake_energy_j'] * 1e6
    handshake_pct = (result['handshake_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'13. Token handshake':<38} {handshake_uj:>13.2f}   {handshake_pct:>10.1f}%")

    token_routing_uj = result['token_routing_energy_j'] * 1e6
    token_routing_pct = (result['token_routing_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'14. Token routing (mesh hops)':<38} {token_routing_uj:>13.2f}   {token_routing_pct:>10.1f}%")

    total_token_uj = (result['total_token_matching_energy_j'] +
                      result['token_routing_energy_j']) * 1e6
    total_token_pct = (total_token_uj / total_energy_uj) * 100
    print(f"  {'-'*38} {'-------------':>13}   {'----------':>10}")
    print(f"  {'SUBTOTAL: Token Execution':<38} {total_token_uj:>13.2f}   {total_token_pct:>10.1f}%")

    # SURE Program Management (UNIQUE TO KPU)
    print(f"\n{'SURE PROGRAM MANAGEMENT (UNIQUE TO KPU)':=<70}")

    prog_load_uj = result['program_load_energy_j'] * 1e6
    prog_load_pct = (result['program_load_energy_j'] / result['total_energy_j']) * 100
    cache_hit_rate = result['cache_hit_rate']
    print(f"  {'15. SURE program loading':<38} {prog_load_uj:>13.2f}   {prog_load_pct:>10.1f}%")
    print(f"      {'(cache hit rate: ' + f'{cache_hit_rate*100:.0f}%)':<38}")

    # Computation
    print(f"\n{'COMPUTATION (PE BLAS Operators)':=<70}")
    compute_uj = result['compute_energy_j'] * 1e6
    compute_pct = (result['compute_energy_j'] / result['total_energy_j']) * 100
    print(f"  {'16. PE MACs':<38} {compute_uj:>13.2f}   {compute_pct:>10.1f}%")

    # Total
    print(f"\n{'='*70}")
    print(f"  {'TOTAL ENERGY':<38} {total_energy_uj:>13.2f}   {'100.0%':>10}")
    print(f"  {'Energy per MAC':<38} {result['energy_per_mac_pj']:>13.3f} pJ")
    print(f"  {'Arithmetic Intensity':<38} {result['arithmetic_intensity']:>13.1f} ops/byte")

    # Categorize by hierarchy level
    print(f"\n{'ENERGY BY HIERARCHY LEVEL':=<70}")

    # Off-chip (DRAM)
    off_chip_uj = dram_read_uj + dram_write_uj
    off_chip_pct = (off_chip_uj / total_energy_uj) * 100
    print(f"  {'Off-chip (HBM2 read+write)':<38} {off_chip_uj:>13.2f}   {off_chip_pct:>10.1f}%")

    # On-chip buffers (L3, L2, L1)
    on_chip_buffer_uj = l3_read_uj + l3_write_uj + l2_read_uj + l2_write_uj + l1_read_uj + l1_write_uj
    on_chip_buffer_pct = (on_chip_buffer_uj / total_energy_uj) * 100
    print(f"  {'On-chip buffers (L3/L2/L1)':<38} {on_chip_buffer_uj:>13.2f}   {on_chip_buffer_pct:>10.1f}%")

    # Data movement engines
    print(f"  {'Data movement engines':<38} {total_dme_uj:>13.2f}   {total_dme_pct:>10.1f}%")

    # KPU-specific overhead
    kpu_overhead_uj = total_token_uj + prog_load_uj
    kpu_overhead_pct = (kpu_overhead_uj / total_energy_uj) * 100
    print(f"  {'KPU-specific (tokens + programs)':<38} {kpu_overhead_uj:>13.2f}   {kpu_overhead_pct:>10.1f}%")

    # Compute
    print(f"  {'Compute (MACs)':<38} {compute_uj:>13.2f}   {compute_pct:>10.1f}%")

    return result


def compare_tpu_vs_kpu():
    """
    Side-by-side comparison of TPU and KPU for the same workload.
    """
    print("\n\n")
    print("="*80)
    print("TPU v4 vs KPU-T768 Comparison")
    print("="*80)
    print(f"\nWorkload: 1024×1024 @ 1024 MatMul (batch=1, BF16)")
    print(f"  Total Ops:   2.147 GOps")
    print(f"  Input:       2.00 MB")
    print(f"  Weight:      2.00 MB")
    print(f"  Output:      2.00 MB")

    # Get results (already computed)
    tpu_model = tpu_v4_resource_model()
    kpu_model = kpu_t768_resource_model()

    # TPU
    M, N, K = 1024, 1024, 1024
    tile_size = 128  # TPU v4 array is 128×128

    # Proper 3D tiling: tile across M, N, K dimensions
    M_tiles = (M + tile_size - 1) // tile_size  # = 8
    N_tiles = (N + tile_size - 1) // tile_size  # = 8
    K_tiles = (K + tile_size - 1) // tile_size  # = 8

    # Total number of tiles = M_tiles × N_tiles × K_tiles
    num_weight_tiles = M_tiles * N_tiles * K_tiles  # = 512

    # Each tile processes tile_size × tile_size @ tile_size
    ops_per_tile = 2 * tile_size * tile_size * tile_size  # 2 × 128³
    input_elements_per_tile = tile_size * tile_size  # 128²
    output_elements_per_tile = tile_size * tile_size  # 128²

    tpu_result = tpu_model.tile_energy_model.compute_tile_energy(
        num_weight_tiles=num_weight_tiles,
        ops_per_tile=ops_per_tile,
        input_elements_per_tile=input_elements_per_tile,
        output_elements_per_tile=output_elements_per_tile,
        batch_size=1,
        precision="BF16"
    )

    # KPU
    kpu_result = kpu_model.tile_energy_model.compute_gemm_energy(
        M=M, N=N, K=K,
        batch_size=1,
        precision="BF16",
        enable_fusion=False,
        num_fused_ops=1
    )

    # Compare
    print(f"\n{'Metric':<40} {'TPU v4':<20} {'KPU-T768':<20} {'Ratio (KPU/TPU)':<15}")
    print("-"*95)

    tpu_total_mj = tpu_result['total_energy_j'] * 1000
    kpu_total_mj = kpu_result['total_energy_j'] * 1000
    ratio = kpu_total_mj / tpu_total_mj
    print(f"{'Total Energy (mJ)':<40} {tpu_total_mj:<20.3f} {kpu_total_mj:<20.3f} {ratio:<15.2f}x")

    # Energy per MAC is already in result for KPU, need to calculate for TPU
    tpu_energy_per_mac_pj = (tpu_result['total_energy_j'] / (tpu_result['total_ops'] / 2)) * 1e12
    kpu_energy_per_mac_pj = kpu_result['energy_per_mac_pj']
    ratio = kpu_energy_per_mac_pj / tpu_energy_per_mac_pj
    print(f"{'Energy per MAC (pJ)':<40} {tpu_energy_per_mac_pj:<20.3f} {kpu_energy_per_mac_pj:<20.3f} {ratio:<15.2f}x")

    # Compute percentage
    tpu_compute_pct = (tpu_result['compute_energy_j'] / tpu_result['total_energy_j']) * 100
    kpu_compute_pct = kpu_result['compute_percentage']
    print(f"{'Compute % (higher = better)':<40} {tpu_compute_pct:<20.1f} {kpu_compute_pct:<20.1f}")

    print(f"\n{'Component Breakdown':<40} {'TPU v4 (µJ)':<20} {'KPU-T768 (µJ)':<20}")
    print("-"*80)

    # Off-chip memory
    tpu_offchip = tpu_result['weight_dram_energy_j'] * 1e6
    kpu_offchip = (kpu_result['dram_read_energy_j'] + kpu_result['dram_write_energy_j']) * 1e6
    print(f"{'Off-chip memory (DRAM/HBM)':<40} {tpu_offchip:<20.2f} {kpu_offchip:<20.2f}")

    # On-chip buffers
    tpu_onchip = (tpu_result['weight_fifo_energy_j'] + tpu_result['input_read_energy_j'] +
                  tpu_result['total_accumulator_energy_j'] + tpu_result['output_write_energy_j']) * 1e6
    kpu_onchip = (kpu_result['l3_read_energy_j'] + kpu_result['l3_write_energy_j'] +
                  kpu_result['l2_read_energy_j'] + kpu_result['l2_write_energy_j'] +
                  kpu_result['l1_read_energy_j'] + kpu_result['l1_write_energy_j']) * 1e6
    print(f"{'On-chip buffers':<40} {tpu_onchip:<20.2f} {kpu_onchip:<20.2f}")

    # Data movement
    tpu_movement = (tpu_result['weight_shift_energy_j'] + tpu_result['activation_stream_energy_j']) * 1e6
    kpu_movement = kpu_result['total_dme_energy_j'] * 1e6
    print(f"{'Data movement':<40} {tpu_movement:<20.2f} {kpu_movement:<20.2f}")

    # Architecture-specific overhead
    tpu_overhead = 0.0  # TPU has no explicit overhead (implicit in control)
    kpu_overhead = ((kpu_result['total_token_matching_energy_j'] +
                     kpu_result['token_routing_energy_j'] +
                     kpu_result['program_load_energy_j']) * 1e6)
    print(f"{'Architecture-specific overhead':<40} {tpu_overhead:<20.2f} {kpu_overhead:<20.2f}")

    # Compute
    tpu_compute = tpu_result['compute_energy_j'] * 1e6
    kpu_compute = kpu_result['compute_energy_j'] * 1e6
    print(f"{'Compute (MACs)':<40} {tpu_compute:<20.2f} {kpu_compute:<20.2f}")

    print(f"\n{'='*80}")
    print(f"KEY INSIGHT:")
    print(f"  TPU v4:   {tpu_energy_per_mac_pj:.3f} pJ/MAC ({tpu_compute_pct:.1f}% compute)")
    print(f"  KPU-T768: {kpu_energy_per_mac_pj:.3f} pJ/MAC ({kpu_compute_pct:.1f}% compute)")
    print(f"  Ratio:    {ratio:.2f}x (KPU uses {ratio:.2f}x more energy than TPU)")
    print(f"\n  KPU overhead from programmability:")
    print(f"    - Token routing: {kpu_overhead:.2f} µJ ({(kpu_overhead/kpu_total_mj/10):.1f}% of total)")
    print(f"    - 4-stage hierarchy vs TPU's 2-stage")
    print(f"    - Advantage: Programmable (all BLAS ops vs TPU's GEMM only)")


if __name__ == "__main__":
    # Run detailed breakdowns
    tpu_result = analyze_tpu_energy_events()
    kpu_result = analyze_kpu_energy_events()

    # Compare side-by-side
    compare_tpu_vs_kpu()

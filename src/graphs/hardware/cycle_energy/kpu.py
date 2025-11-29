"""
KPU (Domain Flow Architecture) Cycle-Level Energy Model

Models energy consumption for Stillwater Supercomputing KPU, a Domain Flow
Architecture that implements direct execution of Systems of Uniform
Recurrence Equations (SURE).

The KPU is a PROGRAMMABLE dataflow machine - NOT a fixed-function accelerator.
It can execute ANY algorithm expressible as a system of uniform recurrence
equations, including:
- Signal processing (FFT, FIR, IIR filters, convolutions)
- Linear algebra (matrix multiply, decompositions, solvers)
- Constraint solvers and optimizers
- Neural network inference

KPU Architecture (Domain Flow):
  +-----------+     +------------------+     +------------------+
  |  DOMAIN   |     |  TILE ARRAY      |     |  OUTPUT          |
  |  PROGRAM  |---->|  (8x8 to 24x32)  |---->|  AGGREGATION     |
  |  LOAD     |     |  Heterogeneous:  |     |                  |
  +-----------+     |  INT8/BF16/Matrix|     +------------------+
       ^            +------------------+              |
       |                   ^                          v
  +----+----+        +-----+-----+          +--------+--------+
  |  DOMAIN |        |  STREAM   |          |  STREAM         |
  |  TRACKER|        |  INPUT    |          |  OUTPUT         |
  +---------+        +-----------+          +-----------------+
       ^                   ^                          |
       |                   |                          v
  +--------------------------------------------------------+
  |   DISTRIBUTED SCRATCHPADS (256KB/tile) + SHARED SRAM   |
  +--------------------------------------------------------+

Key characteristics (Domain Flow vs other architectures):
- PROGRAMMABLE: Executes ANY system of uniform recurrence equations
- NO instruction fetch per operation (domain program loaded once)
- Data-driven spatial execution based on SURE theory
- Uniform dependency vectors encode temporal AND spatial distance
- Near 100% utilization at batch=1 (unlike TPU's 10-20%)
- Streaming dataflow - no flat global memory
- Heterogeneous tiles (INT8/BF16/Matrix) in checkerboard pattern
- 2.5-4x more energy efficient than stored program machines

Key Innovation - Distributed CAM:
Traditional dataflow machines have a centralized Content-Addressable Memory
(CAM) that becomes a bottleneck as the machine scales, forcing reduced cycle
times. The KPU distributes the CAM across processing elements, enabling
scalability without cycle time reduction - solving the quintessential problem
that limited dataflow machines in general parallel computing.

Theoretical Foundation:
- Systems of Uniform Recurrence Equations (Karp-Miller-Winograd)
- Computational Space-Times (Omtzigt, 1994)
- Affine transformations decompose into time and spatial projections
- Uniform dependencies map to physical PE-to-PE distances
"""

from typing import Optional

from .base import (
    CyclePhase,
    OperatingMode,
    HitRatios,
    DEFAULT_HIT_RATIOS,
    CycleEnergyBreakdown,
)


# KPU Energy Parameters (based on Stillwater KPU-T64 / Domain Flow Architecture)
# Process: 16nm (edge), moving to 7nm for higher tiers
# Frequency: 900 MHz (T64), 1.4 GHz (T256)
# Voltage: ~0.8V

KPU_ENERGY_PARAMS = {
    # Tile array configuration (T64 baseline)
    'num_tiles': 64,                  # 8x8 heterogeneous tile array
    'int8_tiles': 44,                 # 70% INT8 tiles
    'bf16_tiles': 13,                 # 20% BF16 tiles
    'matrix_tiles': 7,                # 10% Matrix tiles

    # Domain Flow overhead (per inference / domain program execution)
    # These are one-time costs, NOT per-operation
    'domain_program_load_pj': 500.0,  # Load domain flow program (once per layer)
    'domain_tracker_pj': 20.0,        # Domain tracking per tile activation
    'network_overlay_pj': 100.0,      # Configure SURE network overlay

    # Stream processing (continuous data flow)
    'stream_setup_pj': 50.0,          # Stream buffer initialization
    'stream_sync_pj': 10.0,           # Inter-tile synchronization
    'stream_pj_per_byte': 0.15,       # Per-byte streaming (very efficient)

    # Domain Flow compute (extremely efficient - no instruction fetch!)
    'mac_int8_pj': 0.25,              # INT8 MAC in domain flow tile
    'mac_bf16_pj': 0.40,              # BF16 MAC
    'mac_fp32_pj': 0.80,              # FP32 MAC (less common)
    'accumulator_pj': 0.08,           # Per accumulation
    'activation_pj': 0.12,            # Per activation (ReLU/etc)

    # Workload mix (typical AI inference)
    'int8_fraction': 0.70,            # 70% INT8 ops
    'bf16_fraction': 0.20,            # 20% BF16 ops
    'fp32_fraction': 0.10,            # 10% FP32 ops

    # Distributed scratchpad memory (256KB per tile - no cache tags!)
    'scratchpad_read_pj_per_byte': 0.2,   # L1 scratchpad read
    'scratchpad_write_pj_per_byte': 0.3,  # L1 scratchpad write
    'l2_sram_pj_per_byte': 0.7,           # Tile-local L2 SRAM
    'l3_sram_pj_per_byte': 1.5,           # Shared L3 SRAM
    'dram_pj_per_byte': 12.0,             # LPDDR5 (edge devices)

    # Efficiency factors (Domain Flow advantage)
    'tile_utilization': 0.85,         # 85% typical (near 100% at batch=1!)
    'pipeline_efficiency': 0.90,      # Minimal pipeline bubbles
    'domain_flow_efficiency': 0.95,   # SURE execution efficiency

    # Tiling overhead (when data doesn't fit in 256KB scratchpad)
    'tiling_overhead_per_iteration': 0.10,  # 10% overhead per tiling iteration
}


def build_kpu_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    num_layers: int = 1,
    verbose: bool = False
) -> CycleEnergyBreakdown:
    """
    Build the KPU (Domain Flow Architecture) cycle energy breakdown.

    Args:
        num_ops: Number of MAC operations to execute
        bytes_transferred: Total bytes of data accessed
        mode: Operating mode (L1/SRAM or DRAM resident)
        hit_ratios: Custom hit ratios (uses defaults for mode if None)
        num_layers: Number of neural network layers (affects domain program loads)
        verbose: Enable verbose output

    Returns:
        CycleEnergyBreakdown with detailed energy breakdown

    KPU Energy Model (Domain Flow Architecture):
    - NO instruction fetch per operation (domain program loaded once per layer)
    - Data-driven spatial execution based on SURE theory
    - Domain tracking overhead (minimal, amortized across operations)
    - Stream processing through heterogeneous tile array
    - Distributed scratchpads (256KB/tile) - no cache coherence overhead
    - Near 100% utilization even at batch=1 (unlike TPU/GPU)

    Key advantages over stored program machines:
    - 2.5-4x more energy efficient than CPU
    - No instruction fetch/decode per operation
    - No cache coherence machinery (unlike GPU)
    - Predictable latency (no cache misses, no branch misprediction)

    Key advantages over TPU (systolic):
    - Near 100% utilization at batch=1 (TPU: 10-20%)
    - Overlapped compute/data movement (no fill/drain bubbles)
    - Programmable spatial schedule (not fixed function)
    """
    ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[mode]
    params = KPU_ENERGY_PARAMS

    breakdown = CycleEnergyBreakdown(
        architecture_name="KPU (Stillwater Domain Flow)",
        architecture_class="Domain Flow Architecture (SURE)"
    )

    num_tiles = params['num_tiles']

    # Calculate effective operations considering efficiency
    effective_ops = int(num_ops * params['tile_utilization'] *
                       params['pipeline_efficiency'] * params['domain_flow_efficiency'])

    # Calculate tile invocations
    ops_per_tile_cycle = 512  # Typical ops per tile per cycle
    total_tile_ops = num_tiles * ops_per_tile_cycle
    num_invocations = max(1, (num_ops + total_tile_ops - 1) // total_tile_ops)

    # Cycles based on domain flow execution
    breakdown.num_cycles = num_invocations
    breakdown.ops_per_cycle = min(total_tile_ops, num_ops)

    # ==========================================================================
    # DOMAIN FLOW CONFIGURATION (One-time per layer, NOT per operation!)
    # ==========================================================================
    # This is the key advantage: domain program is loaded once, then
    # data flows through the configured SURE network

    breakdown.add_event(
        CyclePhase.SPATIAL_CONFIG,
        f"Domain program load ({num_layers} layers)",
        params['domain_program_load_pj'],
        num_layers
    )

    # Domain tracker configures which tiles participate
    active_tiles = min(num_tiles, max(1, num_ops // 100))
    breakdown.add_event(
        CyclePhase.SPATIAL_CONFIG,
        f"Domain tracker ({active_tiles} active tiles)",
        params['domain_tracker_pj'],
        active_tiles
    )

    # SURE network overlay configuration
    breakdown.add_event(
        CyclePhase.SPATIAL_CONFIG,
        f"SURE network overlay ({num_layers} layers)",
        params['network_overlay_pj'],
        num_layers
    )

    # ==========================================================================
    # DOMAIN FLOW STREAMING (Data-driven execution)
    # ==========================================================================
    # Data flows through the tile network driven by dependencies,
    # not by instruction fetches

    breakdown.add_event(
        CyclePhase.SPATIAL_STREAM,
        "Stream buffer initialization",
        params['stream_setup_pj'],
        num_invocations
    )

    # Inter-tile synchronization (minimal in domain flow)
    sync_points = num_invocations * (active_tiles // 8)  # Sync per row of tiles
    breakdown.add_event(
        CyclePhase.SPATIAL_STREAM,
        "Inter-tile synchronization (SURE)",
        params['stream_sync_pj'],
        max(1, sync_points)
    )

    # Input data streaming
    input_bytes = bytes_transferred // 2
    breakdown.add_event(
        CyclePhase.SPATIAL_STREAM,
        "Input data streaming",
        params['stream_pj_per_byte'],
        input_bytes
    )

    # Output data streaming
    output_bytes = bytes_transferred // 2
    breakdown.add_event(
        CyclePhase.SPATIAL_STREAM,
        "Output data streaming",
        params['stream_pj_per_byte'],
        output_bytes
    )

    # ==========================================================================
    # DOMAIN FLOW COMPUTE (Heterogeneous tile array)
    # ==========================================================================
    # This is where Domain Flow Architecture excels:
    # - No instruction fetch per operation
    # - Data-driven execution based on SURE dependencies
    # - Near 100% utilization even at batch=1

    # Distribute operations across tile types
    int8_ops = int(num_ops * params['int8_fraction'])
    bf16_ops = int(num_ops * params['bf16_fraction'])
    fp32_ops = num_ops - int8_ops - bf16_ops

    breakdown.add_event(
        CyclePhase.SPATIAL_COMPUTE,
        f"INT8 MACs ({params['int8_tiles']} tiles, domain flow)",
        params['mac_int8_pj'],
        int8_ops
    )

    breakdown.add_event(
        CyclePhase.SPATIAL_COMPUTE,
        f"BF16 MACs ({params['bf16_tiles']} tiles, domain flow)",
        params['mac_bf16_pj'],
        bf16_ops
    )

    if fp32_ops > 0:
        breakdown.add_event(
            CyclePhase.SPATIAL_COMPUTE,
            f"FP32 MACs (matrix tiles, domain flow)",
            params['mac_fp32_pj'],
            fp32_ops
        )

    # Accumulation at tile boundaries
    num_accumulations = num_ops // ops_per_tile_cycle
    if num_accumulations > 0:
        breakdown.add_event(
            CyclePhase.SPATIAL_COMPUTE,
            "Tile boundary accumulation",
            params['accumulator_pj'],
            num_accumulations
        )

    # Activation functions (fused in tiles)
    num_activations = num_ops // (num_tiles * 64)  # ~1 activation per 64 ops per tile
    if num_activations > 0:
        breakdown.add_event(
            CyclePhase.SPATIAL_COMPUTE,
            "Activation function (ReLU/etc)",
            params['activation_pj'],
            num_activations
        )

    # ==========================================================================
    # DOMAIN FLOW INTERCONNECT (SURE dependency network)
    # ==========================================================================
    # Uniform dependency vectors encode physical PE-to-PE distances
    # Data flows between adjacent tiles based on SURE mappings

    # Inter-tile data movement follows SURE dependency patterns
    # Average distance is ~2 hops in 8x8 array with good mapping
    avg_hops = 2.0
    inter_tile_transfers = num_invocations * active_tiles

    # Energy for data movement through SURE network
    # Much more efficient than GPU coherence or CPU cache hierarchy
    interconnect_energy_per_transfer = params['stream_pj_per_byte'] * 4 * avg_hops
    breakdown.add_event(
        CyclePhase.SPATIAL_INTERCONNECT,
        f"SURE dependency network ({avg_hops:.0f} avg hops)",
        interconnect_energy_per_transfer,
        inter_tile_transfers
    )

    # ==========================================================================
    # MEMORY ACCESS (Distributed Scratchpads + Shared SRAM)
    # ==========================================================================
    # KPU uses distributed scratchpads (256KB/tile) - NO cache tags!
    # No coherence overhead like GPU
    # No cache miss penalties like CPU

    scratchpad_bytes = int(bytes_transferred * ratios.l1_hit)
    l2_bytes = int((bytes_transferred - scratchpad_bytes) * ratios.l2_hit)
    remaining = bytes_transferred - scratchpad_bytes - l2_bytes
    l3_bytes = int(remaining * ratios.l3_hit) if mode != OperatingMode.L1_RESIDENT else 0
    dram_bytes = remaining - l3_bytes

    is_scratchpad_resident = (mode == OperatingMode.L1_RESIDENT)

    if scratchpad_bytes > 0:
        if is_scratchpad_resident:
            desc = "Distributed scratchpads (256KB/tile, 100% resident)"
        else:
            desc = f"Distributed scratchpads ({int(ratios.l1_hit * 100)}% hit)"

        # Split between read and write
        read_bytes = scratchpad_bytes // 2
        write_bytes = scratchpad_bytes - read_bytes

        breakdown.add_event(
            CyclePhase.MEM_SRAM,
            desc,
            (params['scratchpad_read_pj_per_byte'] + params['scratchpad_write_pj_per_byte']) / 2,
            scratchpad_bytes
        )

    if l2_bytes > 0:
        breakdown.add_event(
            CyclePhase.MEM_L2,
            f"Tile-local L2 SRAM ({l2_bytes} bytes)",
            params['l2_sram_pj_per_byte'],
            l2_bytes
        )

    if l3_bytes > 0:
        breakdown.add_event(
            CyclePhase.MEM_L3,
            f"Shared L3 SRAM ({l3_bytes} bytes)",
            params['l3_sram_pj_per_byte'],
            l3_bytes
        )

    if dram_bytes > 0:
        breakdown.add_event(
            CyclePhase.MEM_DRAM,
            f"LPDDR5 ({dram_bytes} bytes)",
            params['dram_pj_per_byte'],
            dram_bytes
        )

    return breakdown

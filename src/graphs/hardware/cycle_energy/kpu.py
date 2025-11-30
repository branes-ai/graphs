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

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graphs.hardware.technology_profile import TechnologyProfile

from .base import (
    CyclePhase,
    OperatingMode,
    HitRatios,
    DEFAULT_HIT_RATIOS,
    CycleEnergyBreakdown,
)


# NOTE: Energy parameters are now REQUIRED via TechnologyProfile.
# The hardcoded KPU_ENERGY_PARAMS dict has been removed to eliminate
# dual sources of energy definitions. Use a TechnologyProfile instance
# (e.g., DEFAULT_PROFILE from technology_profile.py) for all energy values.


def build_kpu_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    num_layers: int = 1,
    tech_profile: 'TechnologyProfile' = None,
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
        tech_profile: REQUIRED TechnologyProfile for energy parameters
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

    Technology Profile (REQUIRED):
        A TechnologyProfile must be provided to specify energy parameters.
        Use DEFAULT_PROFILE for typical datacenter values.

        Example:
            from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
            breakdown = build_kpu_cycle_energy(
                num_ops=1000,
                tech_profile=EDGE_8NM_LPDDR5
            )

    Raises:
        ValueError: If tech_profile is not provided.
    """
    if tech_profile is None:
        raise ValueError(
            "tech_profile is required. Use DEFAULT_PROFILE from "
            "graphs.hardware.technology_profile or provide a specific profile."
        )
    ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[mode]

    # Derive all energy parameters from technology profile
    process_scale = tech_profile.process_node_nm / 16.0  # 16nm is baseline for KPU
    params = {
        # Tile array configuration (fixed)
        'num_tiles': 64,
        'int8_tiles': 44,
        'bf16_tiles': 13,
        'matrix_tiles': 7,

        # Domain Flow overhead (scales with process)
        'domain_program_load_pj': 500.0 * process_scale,
        'domain_tracker_pj': 20.0 * process_scale,
        'network_overlay_pj': 100.0 * process_scale,

        # Stream processing
        'stream_setup_pj': 50.0 * process_scale,
        'stream_sync_pj': 10.0 * process_scale,
        'stream_pj_per_byte': tech_profile.sram_energy_per_byte_pj * 0.5,

        # Domain Flow compute (uses domain flow MAC energy from profile)
        'mac_int8_pj': tech_profile.domain_flow_mac_energy_pj * 0.5,  # INT8 is cheaper
        'mac_bf16_pj': tech_profile.domain_flow_mac_energy_pj * 0.8,
        'mac_fp32_pj': tech_profile.domain_flow_mac_energy_pj * 1.6,
        'accumulator_pj': tech_profile.domain_flow_mac_energy_pj * 0.16,
        'activation_pj': tech_profile.domain_flow_mac_energy_pj * 0.24,

        # Workload mix (fixed)
        'int8_fraction': 0.70,
        'bf16_fraction': 0.20,
        'fp32_fraction': 0.10,

        # Memory hierarchy
        'scratchpad_read_pj_per_byte': tech_profile.sram_energy_per_byte_pj * 0.8,
        'scratchpad_write_pj_per_byte': tech_profile.sram_energy_per_byte_pj * 1.2,
        'l2_sram_pj_per_byte': tech_profile.l2_cache_energy_per_byte_pj,
        'l3_sram_pj_per_byte': tech_profile.l3_cache_energy_per_byte_pj,
        'dram_pj_per_byte': tech_profile.offchip_energy_per_byte_pj,

        # Efficiency factors (fixed)
        'tile_utilization': 0.85,
        'pipeline_efficiency': 0.90,
        'domain_flow_efficiency': 0.95,
        'tiling_overhead_per_iteration': 0.10,
    }

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
    # EDDO SCRATCHPAD HIERARCHY (Software-Managed, NOT caches!)
    # ==========================================================================
    # EDDO = Explicit Data Distribution and Orchestration
    #
    # KPU uses SOFTWARE-MANAGED scratchpads - NOT hardware-managed caches:
    # - Tile Scratchpad: 256KB per tile, directly addressed (no tags!)
    # - Global Scratchpad: Shared SRAM across tile groups
    # - Streaming Buffer: DMA staging for off-chip transfers
    #
    # Key differences from cache hierarchy:
    # - No tag lookup energy (scratchpads are directly addressed)
    # - No coherence protocol (explicit data distribution)
    # - No cache misses (compiler pre-stages all data)
    # - Deterministic timing (no variable miss latencies)
    #
    # The compiler determines data placement at compile time (EDDO).
    # Data is proactively staged before it's needed - no reactive fetching.

    # EDDO data distribution (compiler-determined, not hit-rate based)
    # For modeling purposes, we use similar ratios but interpret them differently:
    # - "l1_hit" -> fraction in tile scratchpad (pre-staged by compiler)
    # - "l2_hit" -> fraction in global scratchpad
    # - "l3_hit" -> fraction in streaming buffer
    # - remainder -> DRAM via DMA

    tile_scratchpad_bytes = int(bytes_transferred * ratios.l1_hit)
    global_scratchpad_bytes = int((bytes_transferred - tile_scratchpad_bytes) * ratios.l2_hit)
    remaining = bytes_transferred - tile_scratchpad_bytes - global_scratchpad_bytes
    streaming_buffer_bytes = int(remaining * ratios.l3_hit) if mode != OperatingMode.L1_RESIDENT else 0
    dram_bytes = remaining - streaming_buffer_bytes

    is_scratchpad_resident = (mode == OperatingMode.L1_RESIDENT)

    # Tile Scratchpad (per-tile local SRAM, 256KB/tile)
    # Directly addressed - NO tag lookup energy!
    if tile_scratchpad_bytes > 0:
        if is_scratchpad_resident:
            desc = "Tile scratchpad (256KB/tile, EDDO pre-staged)"
        else:
            desc = f"Tile scratchpad (EDDO, {tile_scratchpad_bytes/1024:.1f}KB)"

        # Scratchpad energy: ~60% of cache energy (no tags, no coherence)
        scratchpad_energy = (params['scratchpad_read_pj_per_byte'] +
                           params['scratchpad_write_pj_per_byte']) / 2

        breakdown.add_event(
            CyclePhase.EDDO_TILE_SCRATCHPAD,
            desc,
            scratchpad_energy,
            tile_scratchpad_bytes
        )

    # Global Scratchpad (shared SRAM across tile groups)
    # Software-managed, no coherence protocol
    if global_scratchpad_bytes > 0:
        breakdown.add_event(
            CyclePhase.EDDO_GLOBAL_SCRATCHPAD,
            f"Global scratchpad (EDDO, {global_scratchpad_bytes/1024:.1f}KB)",
            params['l2_sram_pj_per_byte'] * 0.5,  # 50% of cache energy (no tags)
            global_scratchpad_bytes
        )

    # Streaming Buffer (DMA staging area for off-chip transfers)
    # FIFO access pattern, no tag lookup needed
    if streaming_buffer_bytes > 0:
        breakdown.add_event(
            CyclePhase.EDDO_STREAMING_BUFFER,
            f"Streaming buffer (DMA staging, {streaming_buffer_bytes/1024:.1f}KB)",
            params['l3_sram_pj_per_byte'] * 0.4,  # 40% of cache energy (FIFO, no tags)
            streaming_buffer_bytes
        )

    # DRAM access via DMA (double-buffered, overlapped with compute)
    if dram_bytes > 0:
        # DMA descriptor setup (per 4KB block)
        dma_block_size = 4096
        num_dma_transfers = max(1, dram_bytes // dma_block_size)
        dma_setup_energy = 5.0 * (tech_profile.process_node_nm / 7.0)  # pJ per descriptor

        breakdown.add_event(
            CyclePhase.EDDO_DMA_SETUP,
            f"DMA setup ({num_dma_transfers} transfers)",
            dma_setup_energy,
            num_dma_transfers
        )

        breakdown.add_event(
            CyclePhase.MEM_DRAM,
            f"DRAM via DMA ({dram_bytes/1024:.1f}KB, double-buffered)",
            params['dram_pj_per_byte'],
            dram_bytes
        )

    return breakdown

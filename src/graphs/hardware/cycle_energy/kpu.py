"""
KPU (Domain Flow Architecture) Cycle-Level Energy Model

Models energy consumption for Stillwater Supercomputing KPU, a Domain Flow
Architecture that implements direct execution of Systems of Uniform
Recurrence Equations (SURE).

KPU vs GPU: The Fundamental Architectural Difference
=====================================================
The critical energy difference between KPU and GPU is NOT the compute units
themselves (both have MACs), but HOW operands reach those compute units.

GPU (Stored Program Machine - Request/Reply Cycle):
  For EACH warp instruction:
    1. Scoreboard lookup - Check RAW/WAW/WAR hazards
    2. Generate register addresses - Decode src1, src2, dst
    3. Operand collector - Gather operands from banked register file
    4. Bank arbitration - Resolve bank conflicts
    5. Operand routing - Crossbar to route operands to ALUs
    6. ALU execution - Perform computation
    7. Result routing - Route result back to register file
    8. Register write - Store result

  Each instruction must explicitly specify WHERE its operands come from.
  This "request/reply" cycle is UNAVOIDABLE in stored program machines.

KPU (Spatial Dataflow - Data Arrives at PE):
  For EACH operation:
    1. Operands ARRIVE from neighboring PE via SURE network (wire delay only)
    2. ALU execution - Perform computation
    3. Result written to LOCAL register (next PE's input)

  NO scoreboard, NO operand collector, NO register arbitration!
  The routing is determined at COMPILE TIME and baked into the spatial layout.
  Data flows through the network based on SURE dependency vectors.

This is why KPU has dramatically lower "data movement" energy:
  - GPU: Every operand requires address generation, arbitration, routing
  - KPU: Operands arrive via pre-configured wire connections (near-zero cost)

KPU Architecture (Domain Flow):
  +-----------+     +------------------+     +------------------+
  |  DOMAIN   |     |  TILE ARRAY      |     |  OUTPUT          |
  |  PROGRAM  |---->|  (checkerboard)  |---->|  AGGREGATION     |
  |  LOAD     |     |  64 tiles total  |     |                  |
  +-----------+     +------------------+     +------------------+
       ^                   ^                          |
       |                   |                          v
  +----+----+        +-----+-----+          +--------+--------+
  |  DOMAIN |        |  STREAM   |          |  STREAM         |
  |  TRACKER|        |  INPUT    |          |  OUTPUT         |
  +---------+        +-----------+          +-----------------+

Each tile is a 16x16 PROGRAMMABLE SYSTOLIC ARRAY:
  - 256 MACs per tile per cycle
  - Total: 64 tiles x 256 MACs = 16,384 MACs/cycle (same as TPU 128x128)
  - But organized as 64 smaller tiles for better workload mapping

Key characteristics (Domain Flow vs other architectures):
- PROGRAMMABLE: Executes ANY system of uniform recurrence equations
- NO instruction fetch per operation (domain program loaded once)
- NO register address generation (operands arrive via SURE network)
- NO operand collector (data already positioned by spatial layout)
- NO bank arbitration (each PE has local register, not shared regfile)
- Data-driven spatial execution based on SURE theory
- Near 100% utilization at batch=1 (unlike TPU's 10-20%)
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graphs.hardware.technology_profile import TechnologyProfile

from .base import (
    CyclePhase,
    OperatingMode,
    OperatorType,
    HitRatios,
    DEFAULT_HIT_RATIOS,
    KPU_L1_STREAMING_BUFFER_SIZES,
    KPU_L2_TILE_STAGING_SIZES,
    KPU_L3_GLOBAL_SCRATCHPAD_SIZES,
    CycleEnergyBreakdown,
    compute_cache_hit_ratio,
)


def build_kpu_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    num_layers: int = 1,
    tech_profile: 'TechnologyProfile' = None,
    verbose: bool = False,
    # New parameters for working-set-based memory modeling
    operator_type: OperatorType = OperatorType.HIGH_REUSE,
    working_set_bytes: Optional[int] = None,
    l3_scratchpad_bytes: int = KPU_L3_GLOBAL_SCRATCHPAD_SIZES['default'],
) -> CycleEnergyBreakdown:
    """
    Build the KPU (Domain Flow Architecture) cycle energy breakdown.

    KPU Execution Model (Spatial Dataflow):
        Unlike GPU (stored program with request/reply cycle), KPU uses
        spatial dataflow where operands ARRIVE at each PE:

        1. Domain program loaded ONCE per layer (NOT per operation!)
        2. For each operation:
           - Operands arrive from neighboring PE (via SURE network - wire only)
           - ALU executes
           - Result written to local register (next PE's input)

        NO scoreboard, NO operand collector, NO register arbitration!
        This is the fundamental source of KPU's energy efficiency.

    Internal Data Movement (SURE Network):
        Data flows between PEs via the SURE (Uniform Recurrence) network:
        - Routing determined at compile time (not runtime)
        - PE-to-PE transfer is just wire delay + local register write
        - Energy: ~0.05 pJ per transfer (vs GPU's ~3 pJ for operand collection)

    External Data Movement (EDDO Hierarchy):
        EDDO = Explicit Data Distribution and Orchestration
        - Software-managed scratchpads (not hardware caches)
        - Compiler pre-stages all data before compute begins
        - No tag lookup, no coherence protocol

    Args:
        num_ops: Number of MAC operations to execute
        bytes_transferred: Total bytes of EXTERNAL data (from DRAM/scratchpad)
        mode: Operating mode (affects external memory access)
        hit_ratios: Custom hit ratios for external memory
        num_layers: Number of neural network layers
        tech_profile: REQUIRED TechnologyProfile for energy parameters
        verbose: Enable verbose output
        operator_type: Type of operator (for memory placement)
        working_set_bytes: Size of working set
        l3_scratchpad_bytes: Size of global scratchpad

    Returns:
        CycleEnergyBreakdown with detailed energy breakdown
    """
    if tech_profile is None:
        raise ValueError(
            "tech_profile is required. Use DEFAULT_PROFILE from "
            "graphs.hardware.technology_profile or provide a specific profile."
        )

    # KPU uses EXPLICIT (software-managed) memory via EDDO
    ws_bytes = working_set_bytes if working_set_bytes is not None else bytes_transferred

    if hit_ratios is not None:
        ratios = hit_ratios
    else:
        # Compute placement using explicit memory model
        l1_size = KPU_L1_STREAMING_BUFFER_SIZES['default']
        l2_size = KPU_L2_TILE_STAGING_SIZES['default']
        l3_size = l3_scratchpad_bytes

        l1_hit = compute_cache_hit_ratio(
            operator_type=operator_type,
            working_set_bytes=ws_bytes,
            cache_size_bytes=l1_size,
            cache_is_cold=False,
            is_explicit_memory=True,
        )
        l2_hit = compute_cache_hit_ratio(
            operator_type=operator_type,
            working_set_bytes=ws_bytes,
            cache_size_bytes=l2_size,
            cache_is_cold=False,
            is_explicit_memory=True,
        )
        l3_hit = compute_cache_hit_ratio(
            operator_type=operator_type,
            working_set_bytes=ws_bytes,
            cache_size_bytes=l3_size,
            cache_is_cold=False,
            is_explicit_memory=True,
        )

        ratios = HitRatios(l1_hit=l1_hit, l2_hit=l2_hit, l3_hit=l3_hit)

        if verbose:
            print(f"  EDDO memory placement (explicit, ws={ws_bytes/1024/1024:.1f}MB):")
            print(f"    L1 (streaming buffer, {l1_size/1024:.0f}KB): {'fits' if l1_hit > 0 else 'spills'}")
            print(f"    L2 (tile staging, {l2_size/1024:.0f}KB): {'fits' if l2_hit > 0 else 'spills'}")
            print(f"    L3 (global scratchpad, {l3_size/1024/1024:.0f}MB): {'fits' if l3_hit > 0 else 'spills to DRAM'}")

    # ==========================================================================
    # Energy Parameters
    # ==========================================================================
    process_scale = tech_profile.process_node_nm / 16.0  # 16nm baseline for KPU

    params = {
        # Tile array configuration
        'num_tiles': 64,
        'tile_dim': 16,              # 16x16 systolic array per tile
        'macs_per_tile': 256,        # 16 x 16 = 256 MACs per tile per cycle
        'total_macs_per_cycle': 16384,  # 64 tiles x 256 MACs

        # === DOMAIN FLOW CONFIGURATION (One-time per layer) ===
        # These costs are amortized over ALL operations in a layer
        'domain_program_load_pj': 500.0 * process_scale,
        'domain_tracker_pj': 20.0 * process_scale,
        'network_overlay_pj': 100.0 * process_scale,

        # === INTERNAL DATA MOVEMENT (SURE Network) ===
        # This is the KEY difference from GPU!
        #
        # GPU request/reply cycle per instruction:
        #   scoreboard (0.3pJ) + addr_gen (0.6pJ) + operand_collector (0.8pJ)
        #   + bank_arb (0.3pJ) + routing (0.7pJ) = ~2.7 pJ
        #
        # KPU SURE network per operation:
        #   PE-to-PE wire delay + local register write only
        #   = ~0.05 pJ (just the wire and latch energy)
        #
        # This 50x reduction in internal data movement is the main source
        # of KPU's energy efficiency over stored program machines.
        #
        'pe_to_pe_transfer_pj': 0.05 * process_scale,  # Wire + local latch
        'local_register_write_pj': 0.02 * process_scale,  # PE-local register

        # === COMPUTE (Domain Flow MACs) ===
        # The MAC energy itself is similar to GPU TensorCore
        # (both are systolic-style MAC arrays)
        'mac_int8_pj': tech_profile.domain_flow_mac_energy_pj * 0.5,
        'mac_bf16_pj': tech_profile.domain_flow_mac_energy_pj * 0.8,
        'mac_fp32_pj': tech_profile.domain_flow_mac_energy_pj * 1.6,
        'accumulator_pj': tech_profile.domain_flow_mac_energy_pj * 0.16,

        # Workload mix
        'int8_fraction': 0.70,
        'bf16_fraction': 0.20,
        'fp32_fraction': 0.10,

        # === EXTERNAL MEMORY (EDDO Hierarchy) ===
        # Scratchpads are software-managed, NO tag lookup energy
        # ~50-60% of equivalent cache energy
        'tile_scratchpad_pj_per_byte': tech_profile.sram_energy_per_byte_pj * 0.5,
        'global_scratchpad_pj_per_byte': tech_profile.l2_cache_energy_per_byte_pj * 0.5,
        'streaming_buffer_pj_per_byte': tech_profile.l3_cache_energy_per_byte_pj * 0.4,
        'dram_pj_per_byte': tech_profile.offchip_energy_per_byte_pj,

        # DMA overhead
        'dma_setup_pj': 5.0 * process_scale,  # Per 4KB block
    }

    breakdown = CycleEnergyBreakdown(
        architecture_name="KPU (Stillwater Domain Flow)",
        architecture_class="Spatial Dataflow (SURE)"
    )

    num_tiles = params['num_tiles']
    total_macs_per_cycle = params['total_macs_per_cycle']

    # Calculate tile cycles needed
    num_tile_cycles = max(1, (num_ops + total_macs_per_cycle - 1) // total_macs_per_cycle)
    active_tiles = min(num_tiles, max(1, num_ops // 100))

    breakdown.num_cycles = num_tile_cycles
    breakdown.ops_per_cycle = min(total_macs_per_cycle, num_ops)

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

    breakdown.add_event(
        CyclePhase.SPATIAL_CONFIG,
        f"Domain tracker ({active_tiles} active tiles)",
        params['domain_tracker_pj'],
        active_tiles
    )

    breakdown.add_event(
        CyclePhase.SPATIAL_CONFIG,
        f"SURE network overlay ({num_layers} layers)",
        params['network_overlay_pj'],
        num_layers
    )

    # ==========================================================================
    # INTERNAL DATA MOVEMENT (SURE Network - Near Zero Cost!)
    # ==========================================================================
    # This is the FUNDAMENTAL difference from GPU!
    #
    # In KPU, operands ARRIVE at each PE via the SURE network:
    # - Routing is determined at compile time (not runtime)
    # - PE-to-PE transfer is just wire delay + local register latch
    # - NO address generation, NO operand collector, NO bank arbitration
    #
    # Each operation needs 2 input operands to arrive and 1 output to leave.
    # But these are just wire transfers, not memory accesses.

    # Number of PE-to-PE transfers: each op has 2 inputs + 1 output = 3 transfers
    # But many are local (same PE), so average ~2 transfers per op
    num_pe_transfers = num_ops * 2

    breakdown.add_event(
        CyclePhase.SPATIAL_STREAM,
        "PE-to-PE transfer (SURE network, wire only)",
        params['pe_to_pe_transfer_pj'],
        num_pe_transfers
    )

    # Local register writes (each PE has its own register, no arbitration)
    breakdown.add_event(
        CyclePhase.SPATIAL_STREAM,
        "Local register write (PE-local, no arbitration)",
        params['local_register_write_pj'],
        num_ops
    )

    # ==========================================================================
    # DOMAIN FLOW COMPUTE (Heterogeneous tile array)
    # ==========================================================================
    # The MAC energy itself is similar to GPU TensorCore
    # The savings come from the data movement, not the compute

    int8_ops = int(num_ops * params['int8_fraction'])
    bf16_ops = int(num_ops * params['bf16_fraction'])
    fp32_ops = num_ops - int8_ops - bf16_ops

    breakdown.add_event(
        CyclePhase.SPATIAL_COMPUTE,
        "INT8 MACs (domain flow, no fetch/decode)",
        params['mac_int8_pj'],
        int8_ops
    )

    breakdown.add_event(
        CyclePhase.SPATIAL_COMPUTE,
        "BF16 MACs (domain flow, no fetch/decode)",
        params['mac_bf16_pj'],
        bf16_ops
    )

    if fp32_ops > 0:
        breakdown.add_event(
            CyclePhase.SPATIAL_COMPUTE,
            "FP32 MACs (domain flow, no fetch/decode)",
            params['mac_fp32_pj'],
            fp32_ops
        )

    # Accumulation at tile boundaries
    num_accumulations = num_ops // total_macs_per_cycle
    if num_accumulations > 0:
        breakdown.add_event(
            CyclePhase.SPATIAL_COMPUTE,
            "Tile boundary accumulation",
            params['accumulator_pj'],
            num_accumulations
        )

    # ==========================================================================
    # EXTERNAL MEMORY ACCESS (EDDO Scratchpad Hierarchy)
    # ==========================================================================
    # This models the energy to GET data into/out of the KPU chip.
    # EDDO = Explicit Data Distribution and Orchestration
    #
    # Key differences from cache hierarchy:
    # - No tag lookup energy (scratchpads are directly addressed)
    # - No coherence protocol (explicit data distribution)
    # - Compiler pre-stages all data (no reactive fetching)

    if mode == OperatingMode.L1_RESIDENT:
        tile_scratchpad_bytes = bytes_transferred
        global_scratchpad_bytes = 0
        streaming_buffer_bytes = 0
        dram_bytes = 0
    else:
        tile_scratchpad_bytes = int(bytes_transferred * ratios.l1_hit)
        remaining_after_l1 = bytes_transferred - tile_scratchpad_bytes

        global_scratchpad_bytes = int(remaining_after_l1 * ratios.l2_hit)
        remaining_after_l2 = remaining_after_l1 - global_scratchpad_bytes

        streaming_buffer_bytes = int(remaining_after_l2 * ratios.l3_hit)
        dram_bytes = remaining_after_l2 - streaming_buffer_bytes

    # Tile Scratchpad (per-tile local SRAM, 256KB/tile)
    if tile_scratchpad_bytes > 0:
        breakdown.add_event(
            CyclePhase.EDDO_TILE_SCRATCHPAD,
            f"Tile scratchpad (EDDO, no tags)",
            params['tile_scratchpad_pj_per_byte'],
            tile_scratchpad_bytes
        )

    # Global Scratchpad
    if global_scratchpad_bytes > 0:
        breakdown.add_event(
            CyclePhase.EDDO_GLOBAL_SCRATCHPAD,
            f"Global scratchpad (EDDO, no coherence)",
            params['global_scratchpad_pj_per_byte'],
            global_scratchpad_bytes
        )

    # Streaming Buffer
    if streaming_buffer_bytes > 0:
        breakdown.add_event(
            CyclePhase.EDDO_STREAMING_BUFFER,
            f"Streaming buffer (DMA staging)",
            params['streaming_buffer_pj_per_byte'],
            streaming_buffer_bytes
        )

    # DRAM access via DMA
    if dram_bytes > 0:
        dma_block_size = 4096
        num_dma_transfers = max(1, dram_bytes // dma_block_size)

        breakdown.add_event(
            CyclePhase.EDDO_DMA_SETUP,
            f"DMA setup ({num_dma_transfers} transfers)",
            params['dma_setup_pj'],
            num_dma_transfers
        )

        breakdown.add_event(
            CyclePhase.MEM_DRAM,
            f"DRAM via DMA (double-buffered)",
            params['dram_pj_per_byte'],
            dram_bytes
        )

    return breakdown

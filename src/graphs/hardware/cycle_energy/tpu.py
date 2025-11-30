"""
TPU Cycle-Level Energy Model

Models energy consumption for systolic array architectures like
Google TPU v4 or Coral Edge TPU.

TPU Architecture (Systolic Array):
  +-----------+     +------------------+     +------------------+
  |  WEIGHT   |     |  SYSTOLIC ARRAY  |     |  ACCUMULATOR     |
  |  BUFFER   |---->|  (MxM multiply)  |---->|  + ACTIVATION    |
  |  (SRAM)   |     |                  |     |                  |
  +-----------+     +------------------+     +------------------+
        ^                   ^                        |
        |                   |                        v
  +-----+-----+       +-----+-----+          +------+------+
  |  WEIGHT   |       |  INPUT    |          |  OUTPUT     |
  |  LOADER   |       |  FEEDER   |          |  DRAIN      |
  +-----------+       +-----------+          +-------------+
        ^                   ^                        |
        |                   |                        v
  +-----------------------------------------------------+
  |               ON-CHIP SRAM / HBM                     |
  +-----------------------------------------------------+

Key characteristics:
- Weight-stationary dataflow (weights stay in place)
- Data flows through systolic array (no instruction fetch per op!)
- Very high MAC efficiency (0.1-0.2 pJ per MAC)
- But significant data movement overhead
- Fixed-function (limited to matrix operations)
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
# The hardcoded TPU_ENERGY_PARAMS dict has been removed to eliminate
# dual sources of energy definitions. Use a TechnologyProfile instance
# (e.g., DEFAULT_PROFILE from technology_profile.py) for all energy values.


def build_tpu_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    matrix_size: int = 128,
    tech_profile: 'TechnologyProfile' = None,
    verbose: bool = False
) -> CycleEnergyBreakdown:
    """
    Build the TPU (systolic array) cycle energy breakdown.

    Args:
        num_ops: Number of MAC operations to execute
        bytes_transferred: Total bytes of data accessed
        mode: Operating mode (L1/SRAM or DRAM/HBM resident)
        hit_ratios: Custom hit ratios (uses defaults for mode if None)
        matrix_size: Systolic array size (default 128x128)
        tech_profile: REQUIRED TechnologyProfile for energy parameters
        verbose: Enable verbose output

    Returns:
        CycleEnergyBreakdown with detailed energy breakdown

    TPU Energy Model:
    - No instruction fetch/decode overhead (fixed function)
    - Systolic MACs are extremely efficient (0.1 pJ each)
    - Main overhead is data movement:
      - Weight loading (amortized by weight-stationary)
      - Input feeding (one edge of array)
      - Output draining (one edge of array)
    - Very efficient for large matrix operations
    - Inefficient for small/irregular operations

    Technology Profile (REQUIRED):
        A TechnologyProfile must be provided to specify energy parameters.
        Use DEFAULT_PROFILE for typical datacenter values.

        Example:
            from graphs.hardware.technology_profile import DATACENTER_4NM_HBM3
            breakdown = build_tpu_cycle_energy(
                num_ops=1000,
                tech_profile=DATACENTER_4NM_HBM3
            )

    Raises:
        ValueError: If tech_profile is not provided.
    """
    if tech_profile is None:
        raise ValueError(
            "tech_profile is required. Use DEFAULT_PROFILE from "
            "graphs.hardware.technology_profile or provide a specific profile."
        )
    # TPU has no L3 cache - L3 mode should behave like DRAM mode
    # (SRAM misses go directly to HBM)
    effective_mode = mode
    if mode == OperatingMode.L3_RESIDENT:
        effective_mode = OperatingMode.DRAM_RESIDENT

    ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[effective_mode]

    # Derive all energy parameters from technology profile
    process_scale = tech_profile.process_node_nm / 7.0  # 7nm is baseline for TPU
    params = {
        # Systolic array dimensions
        'array_size': matrix_size,

        # Systolic compute (uses systolic MAC energy from profile)
        'mac_pj': tech_profile.systolic_mac_energy_pj,
        'accumulator_pj': tech_profile.systolic_mac_energy_pj * 0.5,
        'activation_pj': tech_profile.systolic_mac_energy_pj * 2.0,

        # Data loading overhead (scales with process)
        'weight_load_pj_per_byte': 0.5 * process_scale,
        'input_feed_pj_per_byte': 0.3 * process_scale,
        'output_drain_pj_per_byte': 0.3 * process_scale,

        # Control overhead (scales with process)
        'control_pj_per_tile': 50.0 * process_scale,
        'config_pj': 1000.0 * process_scale,

        # Memory hierarchy
        'sram_pj_per_byte': tech_profile.sram_energy_per_byte_pj,
        'hbm_pj_per_byte': tech_profile.offchip_energy_per_byte_pj,

        # Utilization factors (fixed)
        'weight_reuse': 64,
        'input_reuse': 128,
    }

    breakdown = CycleEnergyBreakdown(
        architecture_name="TPU (Google TPU v4 / Coral)",
        architecture_class="Systolic Array (Weight-Stationary)"
    )

    array_size = params['array_size']

    # Calculate number of systolic tiles needed
    # Each tile is array_size x array_size MACs
    macs_per_tile = array_size * array_size
    num_tiles = max(1, (num_ops + macs_per_tile - 1) // macs_per_tile)

    breakdown.num_cycles = num_tiles * array_size  # Cycles to drain results
    breakdown.ops_per_cycle = array_size  # One row completes per cycle

    # ==========================================================================
    # SYSTOLIC CONTROL (Minimal - no instruction fetch!)
    # ==========================================================================
    # TPU has NO instruction fetch/decode per operation
    # Just configuration and control signals

    breakdown.add_event(
        CyclePhase.SYSTOLIC_CONTROL,
        "Initial configuration",
        params['config_pj'],
        1
    )
    breakdown.add_event(
        CyclePhase.SYSTOLIC_CONTROL,
        f"Tile control ({num_tiles} tiles)",
        params['control_pj_per_tile'],
        num_tiles
    )

    # ==========================================================================
    # SYSTOLIC WEIGHT LOADING
    # ==========================================================================
    # Weight-stationary: load weights once, reuse for many inputs
    # Amortized across weight_reuse factor

    weight_bytes_per_tile = array_size * array_size * 2  # INT16 weights
    total_weight_bytes = weight_bytes_per_tile * num_tiles
    amortized_weight_bytes = total_weight_bytes // params['weight_reuse']

    breakdown.add_event(
        CyclePhase.SYSTOLIC_WEIGHT_LOAD,
        f"Weight load (amortized {params['weight_reuse']}x reuse)",
        params['weight_load_pj_per_byte'],
        amortized_weight_bytes
    )

    # ==========================================================================
    # SYSTOLIC DATA LOADING (Input feeding)
    # ==========================================================================
    # Inputs flow along one edge of systolic array
    # Each tile needs array_size inputs per cycle for array_size cycles

    input_bytes_per_tile = array_size * array_size * 2  # INT16 inputs
    total_input_bytes = input_bytes_per_tile * num_tiles

    breakdown.add_event(
        CyclePhase.SYSTOLIC_DATA_LOAD,
        "Input data feeding (one edge)",
        params['input_feed_pj_per_byte'],
        total_input_bytes
    )

    # ==========================================================================
    # SYSTOLIC COMPUTE
    # ==========================================================================
    # This is where systolic arrays shine - extremely efficient MACs

    breakdown.add_event(
        CyclePhase.SYSTOLIC_COMPUTE,
        f"Systolic MACs ({array_size}x{array_size} array)",
        params['mac_pj'],
        num_ops
    )

    # Accumulation at edge of array
    num_accumulations = num_ops // array_size  # One accumulation per output
    breakdown.add_event(
        CyclePhase.SYSTOLIC_COMPUTE,
        "Partial sum accumulation",
        params['accumulator_pj'],
        num_accumulations
    )

    # Activation function (optional, depends on workload)
    num_activations = num_ops // (array_size * array_size)  # Per output element
    if num_activations > 0:
        breakdown.add_event(
            CyclePhase.SYSTOLIC_COMPUTE,
            "Activation function (ReLU/etc)",
            params['activation_pj'],
            num_activations
        )

    # ==========================================================================
    # SYSTOLIC DRAIN (Output)
    # ==========================================================================
    output_bytes_per_tile = array_size * 4  # FP32 outputs (accumulated)
    total_output_bytes = output_bytes_per_tile * num_tiles

    breakdown.add_event(
        CyclePhase.SYSTOLIC_DRAIN,
        "Output drain (one edge)",
        params['output_drain_pj_per_byte'],
        total_output_bytes
    )

    # ==========================================================================
    # MEMORY ACCESS
    # ==========================================================================
    # TPU uses SRAM buffers + HBM (no traditional cache hierarchy)

    sram_bytes = int(bytes_transferred * ratios.l1_hit)
    hbm_bytes = bytes_transferred - sram_bytes

    is_sram_resident = (mode == OperatingMode.L1_RESIDENT)

    if sram_bytes > 0:
        if is_sram_resident:
            desc = "On-chip SRAM (100% resident)"
        else:
            desc = f"On-chip SRAM ({int(ratios.l1_hit * 100)}% hit)"
        breakdown.add_event(
            CyclePhase.MEM_SRAM,
            desc,
            params['sram_pj_per_byte'],
            sram_bytes
        )

    if hbm_bytes > 0:
        breakdown.add_event(
            CyclePhase.MEM_HBM,
            f"HBM ({hbm_bytes} bytes)",
            params['hbm_pj_per_byte'],
            hbm_bytes
        )

    return breakdown

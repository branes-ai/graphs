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

from typing import Optional

from .base import (
    CyclePhase,
    OperatingMode,
    HitRatios,
    DEFAULT_HIT_RATIOS,
    CycleEnergyBreakdown,
)


# TPU Energy Parameters (based on Google TPU v4 / Coral Edge TPU)
# Process: 7nm
# Frequency: 1.0-1.5 GHz
# Voltage: ~0.75V

TPU_ENERGY_PARAMS = {
    # Systolic array dimensions
    'array_size': 128,                # 128x128 systolic array (TPU v4)

    # Systolic compute (extremely efficient!)
    'mac_pj': 0.1,                    # Per MAC in systolic array
    'accumulator_pj': 0.05,           # Per accumulation
    'activation_pj': 0.2,             # Per activation function

    # Data loading overhead
    'weight_load_pj_per_byte': 0.5,   # Loading weights into systolic array
    'input_feed_pj_per_byte': 0.3,    # Feeding inputs (one edge)
    'output_drain_pj_per_byte': 0.3,  # Draining outputs (one edge)

    # Control overhead (minimal for systolic)
    'control_pj_per_tile': 50.0,      # Control per matrix tile
    'config_pj': 1000.0,              # Configuration overhead

    # Memory hierarchy
    'sram_pj_per_byte': 0.5,          # On-chip SRAM buffer
    'hbm_pj_per_byte': 8.0,           # HBM (more efficient than GPU HBM)

    # Utilization factors
    'weight_reuse': 64,               # Weight reuse factor (batch size effect)
    'input_reuse': 128,               # Input reuse (filter size effect)
}


def build_tpu_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    matrix_size: int = 128,
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
    """
    # TPU has no L3 cache - L3 mode should behave like DRAM mode
    # (SRAM misses go directly to HBM)
    effective_mode = mode
    if mode == OperatingMode.L3_RESIDENT:
        effective_mode = OperatingMode.DRAM_RESIDENT

    ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[effective_mode]
    params = TPU_ENERGY_PARAMS

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

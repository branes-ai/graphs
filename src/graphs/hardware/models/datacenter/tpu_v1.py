"""
TPU v1 Resource Model (ISCA 2017 Paper Architecture)

Based on: "In-Datacenter Performance Analysis of a Tensor Processing Unit"
          Norman P. Jouppi et al., ISCA 2017

Architecture highlights:
- 256×256 systolic array (65,536 MACs)
- 700 MHz clock
- 92 TOPS INT8 peak
- 8 GiB DDR3 Weight Memory (off-chip)
- 256 KiB Weight FIFO (4 tiles × 64 KiB)
- 4 MiB Accumulators (double-buffered)
- 24 MiB Unified Buffer
- 256-byte wide data paths
- PCIe Gen3 x16 host interface

Key characteristics:
- Largest systolic array (256×256 vs 128×128 in v3+)
- Off-chip DDR3 (higher energy than on-chip HBM)
- INT8 only (no BF16 support)
- Roofline knee at ~1350 ops/byte
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ThermalOperatingPoint,
    ComputeFabric,
    get_base_alu_energy,
)
from ...architectural_energy import TPUTileEnergyModel


def tpu_v1_resource_model() -> HardwareResourceModel:
    """
    Google TPU v1 resource model (ISCA 2017 paper architecture).

    The original TPU designed for inference-only workloads.
    Key innovation: Large 256×256 systolic array for matrix operations.

    Architecture:
    - Single Matrix Multiplier Unit (MXU)
    - 256×256 systolic array (65,536 MACs)
    - 92 TOPS INT8 peak (256×256 × 2 ops × 700 MHz)
    - INT8 only (no floating point)

    Returns:
        HardwareResourceModel configured for TPU v1
    """
    # Thermal operating point (datacenter inference accelerator)
    thermal_default = ThermalOperatingPoint(
        name="default",
        tdp_watts=75.0,  # TPU v1 TDP (estimated, not published)
        cooling_solution="active-air",
        performance_specs={}  # Uses precision_profiles for performance
    )

    # ========================================================================
    # Systolic Array Fabric (28nm Standard Cell)
    # ========================================================================
    # TPU v1 has a single large 256×256 systolic array
    # Process: 28nm (first-gen TPU, older process)
    # Clock: 700 MHz
    # Peak INT8: 92 TOPS (256×256 × 2 ops × 700 MHz)
    systolic_fabric = ComputeFabric(
        fabric_type="systolic_array",
        circuit_type="standard_cell",
        num_units=256 * 256,  # 65,536 MACs (256×256 array)
        ops_per_unit_per_clock={
            Precision.INT8: 2,  # MAC = 2 ops (multiply + accumulate)
        },
        core_frequency_hz=700e6,  # 700 MHz
        process_node_nm=28,  # 28nm process
        energy_per_flop_fp32=get_base_alu_energy(28, 'standard_cell'),  # 4.0 pJ
        energy_scaling={
            Precision.INT8: 0.125,  # INT8 is 8× more efficient than FP32
        }
    )

    # TPU v1 tile energy model (for architectural analysis)
    tile_energy_model = TPUTileEnergyModel(
        # Array configuration (v1 has the largest systolic array)
        array_width=256,
        array_height=256,
        num_arrays=1,  # Single MXU

        # Tile configuration (v1 uses 64 KiB tiles)
        weight_tile_size=64 * 1024,  # 64 KiB per tile
        weight_fifo_depth=4,  # 4 tiles buffered (256 KiB total)

        # Pipeline (v1 has the longest pipeline)
        pipeline_fill_cycles=256,  # 256 cycles to fill pipeline
        clock_frequency_hz=700e6,  # 700 MHz

        # Accumulator (4 MiB for roofline knee at ~1350 ops/byte)
        accumulator_size=4 * 1024 * 1024,  # 4 MiB
        accumulator_width=256,  # 256 elements wide

        # Unified Buffer
        unified_buffer_size=24 * 1024 * 1024,  # 24 MiB

        # Energy coefficients (DDR3 era, older process node)
        weight_memory_energy_per_byte=10.0e-12,  # 10 pJ/byte (DDR3, off-chip)
        weight_fifo_energy_per_byte=0.5e-12,  # 0.5 pJ/byte (on-chip SRAM)
        unified_buffer_read_energy_per_byte=0.5e-12,  # 0.5 pJ/byte
        unified_buffer_write_energy_per_byte=0.5e-12,  # 0.5 pJ/byte
        accumulator_write_energy_per_element=0.4e-12,  # 0.4 pJ (32-bit write)
        accumulator_read_energy_per_element=0.3e-12,  # 0.3 pJ (32-bit read)
        weight_shift_in_energy_per_element=0.3e-12,  # 0.3 pJ (shift register)
        activation_stream_energy_per_element=0.2e-12,  # 0.2 pJ (stream)
        mac_energy=0.2e-12,  # 0.2 pJ per INT8 MAC (optimized for INT8)
    )

    # Calculate peak performance
    int8_peak = systolic_fabric.get_peak_ops_per_sec(Precision.INT8)

    model = HardwareResourceModel(
        name="TPU-v1",
        hardware_type=HardwareType.TPU,

        # NEW: Compute fabrics
        compute_fabrics=[systolic_fabric],

        # Legacy fields
        compute_units=1,  # Single MXU (Matrix Multiplier Unit)
        threads_per_unit=256 * 256,  # 256×256 systolic array
        warps_per_unit=256,  # rows in systolic array
        warp_size=256,  # columns in systolic array

        precision_profiles={
            # TPU v1 only supports INT8
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=int8_peak,  # 92 TOPS INT8
                tensor_core_supported=True,  # Systolic array
                relative_speedup=1.0,  # Only mode available
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=34e9,  # 34 GB/s DDR3
        l1_cache_per_unit=64 * 1024,  # 64 KiB weight buffer per tile
        l2_cache_total=256 * 1024,  # 256 KiB Weight FIFO
        main_memory=8 * 1024**3,  # 8 GiB DDR3 Weight Memory
        energy_per_flop_fp32=systolic_fabric.energy_per_flop_fp32,  # 4.0 pJ (28nm standard_cell)
        energy_per_byte=10e-12,  # DDR3 energy
        energy_scaling={
            Precision.INT8: 0.125,  # INT8 is 8× more efficient
        },
        min_occupancy=0.5,  # Systolic arrays need high utilization
        max_concurrent_kernels=1,  # Single large batch
        wave_quantization=1,

        # Thermal profile
        thermal_operating_points={
            "default": thermal_default,
        },
        default_thermal_profile="default",
    )

    # Attach tile energy model to the resource model
    model.tile_energy_model = tile_energy_model

    return model

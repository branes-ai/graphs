"""
Tpu V4 Resource Model hardware resource model.

Extracted from resource_model.py during refactoring.
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ClockDomain,
    ComputeResource,
    TileSpecialization,
    KPUComputeResource,
    PerformanceCharacteristics,
    ThermalOperatingPoint,
    ComputeFabric,
    get_base_alu_energy,
)
from ...architectural_energy import TPUTileEnergyModel


def tpu_v4_resource_model() -> HardwareResourceModel:
    """
    Google TPU v4 resource model.

    Architecture:
    - 2 Matrix Multiplier Units (MXUs)
    - Each MXU: 128×128 systolic array (16,384 MACs)
    - Per MXU: 137.5 TFLOPS BF16, 275 TOPS INT8

    Key characteristics:
    - Optimized for BF16 and INT8
    - 2× INT8 performance vs BF16
    - Very energy efficient
    """
    # Thermal operating point (datacenter TPU pod)
    thermal_default = ThermalOperatingPoint(
        name="default",
        tdp_watts=350.0,  # TPU v4 TDP
        cooling_solution="active-liquid",
        performance_specs={}  # Uses precision_profiles for performance
    )

    # ========================================================================
    # Systolic Array Fabric (7nm Standard Cell)
    # ========================================================================
    # TPU v4 has 2 MXUs with 128×128 systolic arrays each
    # Process: 7nm (4th-gen TPU)
    # Clock: 1050 MHz
    # Peak BF16: 275 TFLOPS (2 MXUs × 128×128 × 2 ops × 1050 MHz)
    # Peak INT8: 550 TOPS (2× BF16)
    systolic_fabric = ComputeFabric(
        fabric_type="systolic_array",
        circuit_type="standard_cell",
        num_units=2 * 128 * 128,  # 2 MXUs × 16,384 MACs = 32,768 total MACs
        ops_per_unit_per_clock={
            Precision.BF16: 2,  # MAC = 2 ops
            Precision.INT8: 2,  # MAC = 2 ops
        },
        core_frequency_hz=1050e6,  # 1.05 GHz
        process_node_nm=7,  # 7nm process
        energy_per_flop_fp32=get_base_alu_energy(7, 'standard_cell'),  # 1.8 pJ
        energy_scaling={
            Precision.FP32: 1.0,
            Precision.BF16: 0.5,  # BF16 is 2× more efficient
            Precision.INT8: 0.125,  # INT8 is 8× more efficient
        }
    )

    # TPU v4 tile energy model (for architectural analysis)
    tile_energy_model = TPUTileEnergyModel(
        # Array configuration (v4 uses 128×128 arrays × 2 MXUs)
        array_width=128,
        array_height=128,
        num_arrays=2,  # 2 MXUs per chip

        # Tile configuration (smaller than v1's 64 KiB)
        weight_tile_size=32 * 1024,  # 32 KiB per tile
        weight_fifo_depth=2,  # 2 tiles buffered (estimated)

        # Pipeline (shorter than v1's 256 cycles)
        pipeline_fill_cycles=128,  # 128 cycles to fill pipeline
        clock_frequency_hz=1050e6,  # 1.05 GHz (estimated from 275 TFLOPS)

        # Accumulator (2 MiB per MXU, sized for roofline knee)
        accumulator_size=2 * 1024 * 1024,  # 2 MiB per MXU
        accumulator_width=128,  # 128 elements wide

        # Unified Buffer (estimated 32 MiB for v4)
        unified_buffer_size=32 * 1024 * 1024,  # 32 MiB

        # Energy coefficients (HBM2e, advanced process node)
        weight_memory_energy_per_byte=10.0e-12,  # 10 pJ/byte (HBM2e)
        weight_fifo_energy_per_byte=0.5e-12,  # 0.5 pJ/byte (on-chip SRAM)
        unified_buffer_read_energy_per_byte=0.5e-12,  # 0.5 pJ/byte
        unified_buffer_write_energy_per_byte=0.5e-12,  # 0.5 pJ/byte
        accumulator_write_energy_per_element=0.4e-12,  # 0.4 pJ (32-bit write)
        accumulator_read_energy_per_element=0.3e-12,  # 0.3 pJ (32-bit read)
        weight_shift_in_energy_per_element=0.3e-12,  # 0.3 pJ (shift register)
        activation_stream_energy_per_element=0.2e-12,  # 0.2 pJ (stream)
        mac_energy=0.25e-12,  # 0.25 pJ per BF16 MAC (slightly higher than INT8)
    )

    # Calculate peak performance
    bf16_peak = systolic_fabric.get_peak_ops_per_sec(Precision.BF16)
    int8_peak = systolic_fabric.get_peak_ops_per_sec(Precision.INT8)

    model = HardwareResourceModel(
        name="TPU-v4",
        hardware_type=HardwareType.TPU,

        # NEW: Compute fabrics
        compute_fabrics=[systolic_fabric],

        # Legacy fields
        compute_units=2,  # 2 MXUs (Matrix Multiplier Units)
        threads_per_unit=128 * 128,  # 128×128 systolic array per MXU
        warps_per_unit=128,  # rows in systolic array
        warp_size=128,  # columns in systolic array

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=bf16_peak / 2,  # Half of BF16 (not native)
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=bf16_peak,  # 275 TFLOPS
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=int8_peak,  # 550 TOPS (2× BF16)
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.BF16,

        peak_bandwidth=1.2e12,  # 1.2 TB/s HBM2e
        l1_cache_per_unit=16 * 1024 * 1024,  # 16 MB per MXU
        l2_cache_total=32 * 1024 * 1024,  # 32 MB shared
        main_memory=32 * 1024**3,  # 32 GB HBM2e
        energy_per_flop_fp32=systolic_fabric.energy_per_flop_fp32,  # 1.8 pJ (7nm standard_cell)
        energy_per_byte=10e-12,
        energy_scaling={
            Precision.FP32: 1.0,
            Precision.BF16: 0.5,
            Precision.INT8: 0.125,
        },
        min_occupancy=0.5,  # Systolic arrays need high utilization
        max_concurrent_kernels=1,  # Typically runs one large batch
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



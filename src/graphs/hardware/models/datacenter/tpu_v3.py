"""
TPU v3 Resource Model (2018, First with HBM)

Architecture highlights:
- 2× 128×128 systolic arrays (2 MXUs)
- ~940 MHz clock
- 123 TFLOPS BF16 per chip (2 MXUs)
- 246 TOPS INT8 per chip
- 16 GB HBM (on-chip, much faster than DDR3)
- 900 GB/s memory bandwidth
- 32 KiB weight tiles (smaller than v1's 64 KiB)
- 128-cycle pipeline (shorter than v1's 256 cycles)

Key innovations vs v1:
- HBM instead of DDR3 (5× lower energy per byte)
- Smaller arrays (128×128) for better utilization
- BF16 floating point support
- 2 MXUs for higher throughput
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


def tpu_v3_resource_model() -> HardwareResourceModel:
    """
    Google TPU v3 resource model.

    First TPU generation with HBM and BF16 support.
    Moved to smaller 128×128 arrays (2 MXUs) for better utilization.

    Architecture:
    - 2 Matrix Multiplier Units (MXUs)
    - Each MXU: 128×128 systolic array (16,384 MACs)
    - Per MXU: ~61.5 TFLOPS BF16, ~123 TOPS INT8

    Returns:
        HardwareResourceModel configured for TPU v3
    """
    # Thermal operating point (datacenter TPU pod)
    thermal_default = ThermalOperatingPoint(
        name="default",
        tdp_watts=200.0,  # TPU v3 TDP (estimated)
        cooling_solution="active-liquid",
        performance_specs={}  # Uses precision_profiles for performance
    )

    # ========================================================================
    # Systolic Array Fabric (16nm Standard Cell)
    # ========================================================================
    # TPU v3 has 2 MXUs with 128×128 systolic arrays each
    # Process: 16nm (3rd-gen TPU)
    # Clock: 940 MHz
    # Peak BF16: 123 TFLOPS (2 MXUs × 128×128 × 2 ops × 940 MHz)
    # Peak INT8: 246 TOPS (2× BF16)
    systolic_fabric = ComputeFabric(
        fabric_type="systolic_array",
        circuit_type="standard_cell",
        num_units=2 * 128 * 128,  # 2 MXUs × 16,384 MACs = 32,768 total MACs
        ops_per_unit_per_clock={
            Precision.BF16: 2,  # MAC = 2 ops
            Precision.INT8: 2,  # MAC = 2 ops
        },
        core_frequency_hz=940e6,  # 940 MHz
        process_node_nm=16,  # 16nm process
        energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell'),  # 2.7 pJ
        energy_scaling={
            Precision.FP32: 1.0,
            Precision.BF16: 0.5,  # BF16 is 2× more efficient
            Precision.INT8: 0.125,  # INT8 is 8× more efficient
        }
    )

    # TPU v3 tile energy model (for architectural analysis)
    tile_energy_model = TPUTileEnergyModel(
        # Array configuration (v3 uses smaller 128×128 arrays × 2)
        array_width=128,
        array_height=128,
        num_arrays=2,  # 2 MXUs per chip

        # Tile configuration (smaller tiles than v1)
        weight_tile_size=32 * 1024,  # 32 KiB per tile
        weight_fifo_depth=2,  # 2 tiles buffered (estimated)

        # Pipeline (shorter than v1)
        pipeline_fill_cycles=128,  # 128 cycles to fill pipeline
        clock_frequency_hz=940e6,  # 940 MHz (estimated from 123 TFLOPS)

        # Accumulator (2 MiB per MXU)
        accumulator_size=2 * 1024 * 1024,  # 2 MiB per MXU (estimated)
        accumulator_width=128,  # 128 elements wide

        # Unified Buffer (estimated larger than v1)
        unified_buffer_size=32 * 1024 * 1024,  # 32 MiB (estimated)

        # Energy coefficients (HBM, advanced process node)
        weight_memory_energy_per_byte=5.0e-12,  # 5 pJ/byte (HBM, lower than DDR3)
        weight_fifo_energy_per_byte=0.5e-12,  # 0.5 pJ/byte (on-chip SRAM)
        unified_buffer_read_energy_per_byte=0.5e-12,  # 0.5 pJ/byte
        unified_buffer_write_energy_per_byte=0.5e-12,  # 0.5 pJ/byte
        accumulator_write_energy_per_element=0.4e-12,  # 0.4 pJ (32-bit write)
        accumulator_read_energy_per_element=0.3e-12,  # 0.3 pJ (32-bit read)
        weight_shift_in_energy_per_element=0.3e-12,  # 0.3 pJ (shift register)
        activation_stream_energy_per_element=0.2e-12,  # 0.2 pJ (stream)
        mac_energy=0.25e-12,  # 0.25 pJ per BF16 MAC
    )

    # Calculate peak performance
    bf16_peak = systolic_fabric.get_peak_ops_per_sec(Precision.BF16)
    int8_peak = systolic_fabric.get_peak_ops_per_sec(Precision.INT8)

    model = HardwareResourceModel(
        name="TPU-v3",
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
                peak_ops_per_sec=bf16_peak,  # 123 TFLOPS
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=int8_peak,  # 246 TOPS (2× BF16)
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.BF16,

        peak_bandwidth=900e9,  # 900 GB/s HBM
        l1_cache_per_unit=16 * 1024 * 1024,  # 16 MB per MXU (estimated)
        l2_cache_total=32 * 1024 * 1024,  # 32 MB shared (estimated)
        main_memory=16 * 1024**3,  # 16 GB HBM
        energy_per_flop_fp32=systolic_fabric.energy_per_flop_fp32,  # 2.7 pJ (16nm standard_cell)
        energy_per_byte=5e-12,  # HBM energy
        energy_scaling={
            Precision.FP32: 1.0,
            Precision.BF16: 0.5,
            Precision.INT8: 0.125,
        },
        min_occupancy=0.5,  # Systolic arrays need high utilization
        max_concurrent_kernels=1,
        wave_quantization=1,

        # Thermal profile
        thermal_operating_points={
            "default": thermal_default,
        },
        default_thermal_profile="default",
    )

    # Attach tile energy model
    model.tile_energy_model = tile_energy_model

    return model

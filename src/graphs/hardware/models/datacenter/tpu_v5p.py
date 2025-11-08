"""
TPU v5p Resource Model (2023, Latest Performance-Optimized)

Architecture highlights:
- Enhanced 128×128 systolic arrays (likely more than 2 MXUs)
- ~1100 MHz clock (estimated)
- 459 TFLOPS BF16 per chip
- FP8 support for transformers
- Sparsity acceleration (dynamic zero-skipping)
- HBM2e/HBM3 memory
- ~1.6-2.0 TB/s bandwidth
- Improved interconnect (ICI) for multi-chip scaling

Key innovations vs v4:
- FP8 precision for lower energy
- Hardware sparsity support (skip zero weights/activations)
- Higher clock speeds
- Better multi-chip scaling
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ThermalOperatingPoint,
)
from ...architectural_energy import TPUTileEnergyModel


def tpu_v5p_resource_model() -> HardwareResourceModel:
    """
    Google TPU v5p resource model (performance-optimized, 2023).

    Latest generation TPU with FP8 and sparsity support.
    Optimized for large-scale transformer training and inference.

    Architecture:
    - Multiple MXUs (128×128 each, exact count not public)
    - 459 TFLOPS BF16 per chip
    - FP8 support (~2× BF16 throughput)
    - Sparsity acceleration

    Returns:
        HardwareResourceModel configured for TPU v5p
    """
    # Thermal operating point (datacenter TPU pod)
    thermal_default = ThermalOperatingPoint(
        name="default",
        tdp_watts=400.0,  # TPU v5p TDP (estimated, higher than v4)
        cooling_solution="active-liquid",
        performance_specs={}  # Uses precision_profiles for performance
    )

    # TPU v5p tile energy model
    tile_energy_model = TPUTileEnergyModel(
        # Array configuration (similar to v4 but more MXUs)
        array_width=128,
        array_height=128,
        num_arrays=2,  # Likely more, but not public (conservative estimate)

        # Tile configuration
        weight_tile_size=32 * 1024,  # 32 KiB per tile
        weight_fifo_depth=2,  # 2 tiles buffered

        # Pipeline
        pipeline_fill_cycles=128,  # 128 cycles to fill pipeline
        clock_frequency_hz=1100e6,  # 1.1 GHz (estimated from 459 TFLOPS)

        # Accumulator
        accumulator_size=2 * 1024 * 1024,  # 2 MiB per MXU
        accumulator_width=128,  # 128 elements wide

        # Unified Buffer
        unified_buffer_size=32 * 1024 * 1024,  # 32 MiB

        # Energy coefficients (HBM3, advanced process node)
        weight_memory_energy_per_byte=8.0e-12,  # 8 pJ/byte (HBM3, lower than HBM2e)
        weight_fifo_energy_per_byte=0.5e-12,  # 0.5 pJ/byte (on-chip SRAM)
        unified_buffer_read_energy_per_byte=0.5e-12,  # 0.5 pJ/byte
        unified_buffer_write_energy_per_byte=0.5e-12,  # 0.5 pJ/byte
        accumulator_write_energy_per_element=0.4e-12,  # 0.4 pJ (32-bit write)
        accumulator_read_energy_per_element=0.3e-12,  # 0.3 pJ (32-bit read)
        weight_shift_in_energy_per_element=0.3e-12,  # 0.3 pJ (shift register)
        activation_stream_energy_per_element=0.2e-12,  # 0.2 pJ (stream)
        mac_energy=0.25e-12,  # 0.25 pJ per BF16 MAC
    )

    model = HardwareResourceModel(
        name="TPU-v5p",
        hardware_type=HardwareType.TPU,
        compute_units=2,  # Conservative estimate (likely more)
        threads_per_unit=128 * 128,  # 128×128 systolic array per MXU
        warps_per_unit=128,  # rows in systolic array
        warp_size=128,  # columns in systolic array

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=229.5e12,  # Half of BF16 (not native)
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=459e12,  # 459 TFLOPS
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.FP8_E4M3: PrecisionProfile(
                precision=Precision.FP8_E4M3,
                peak_ops_per_sec=918e12,  # ~2× BF16 (FP8 support)
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.FP16,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=918e12,  # ~2× BF16
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.BF16,

        peak_bandwidth=1.6e12,  # 1.6 TB/s HBM3 (estimated)
        l1_cache_per_unit=16 * 1024 * 1024,  # 16 MB per MXU
        l2_cache_total=32 * 1024 * 1024,  # 32 MB shared
        main_memory=32 * 1024**3,  # 32 GB HBM3 (estimated)
        energy_per_flop_fp32=0.35e-12,  # Improved efficiency vs v4
        energy_per_byte=8e-12,  # HBM3 energy
        min_occupancy=0.5,
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

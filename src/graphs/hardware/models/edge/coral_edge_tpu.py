"""
Coral Edge Tpu Resource Model hardware resource model.

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
)
from ...architectural_energy import TPUTileEnergyModel


def coral_edge_tpu_resource_model() -> HardwareResourceModel:
    """
    Google Coral Edge TPU resource model.

    Configuration: Single Edge TPU chip (USB/M.2/PCIe variants)
    Architecture: Scaled-down systolic array from Google TPU

    Key characteristics:
    - Ultra-low power edge AI accelerator (0.5-2W)
    - 4 TOPS INT8 (much smaller than datacenter TPU v4)
    - INT8 quantization required (no FP16/FP32 support)
    - Perfect for IoT, embedded systems, battery-powered devices
    - Cost-effective: ~$25-75 depending on form factor

    References:
    - Coral Edge TPU: 4 TOPS @ INT8 only
    - Power: 0.5W idle, 2W peak (USB variant)
    - Target: Ultra-low-power edge inference (IoT, cameras, drones)
    - Limitation: Requires TensorFlow Lite models with full INT8 quantization

    Note: This is NOT the datacenter TPU v4 - it's designed for
    battery-powered edge devices where power is more critical than performance.
    """
    # Performance specs
    int8_tops = 4e12  # 4 TOPS INT8 (only mode supported)
    efficiency = 0.85  # 85% efficiency (well-optimized systolic array)
    effective_tops = int8_tops * efficiency  # 3.4 TOPS effective

    # Power profile (very low power)
    power_avg = 2.0  # Watts (peak during inference)

    # Energy per operation (ultra-efficient due to low power)
    # 2W / 3.4 TOPS = 0.59 pJ/op
    energy_per_flop_fp32 = 0.6e-12  # ~0.6 pJ/FLOP (most efficient!)
    energy_per_byte = 20e-12  # USB bandwidth limited

    # Clock domain (single operating point - no DVFS on Edge TPU)
    clock_domain = ClockDomain(
        base_clock_hz=500e6,  # 500 MHz (estimated)
        max_boost_clock_hz=500e6,
        sustained_clock_hz=500e6,  # No throttling on this low-power device
        dvfs_enabled=False,  # No DVFS on this fixed-frequency device
    )

    # Compute resource
    compute_resource = ComputeResource(
        resource_type="Systolic-Array",
        num_units=1,
        ops_per_unit_per_clock={
            Precision.INT8: 8,  # 4 TOPS / 500 MHz = 8 ops/clock
        },
        clock_domain=clock_domain,
    )

    # Thermal operating point (single profile)
    thermal_operating_points = {
        "2W": ThermalOperatingPoint(
            name="2W",
            tdp_watts=2.0,
            cooling_solution="Passive (heatsink)",
            performance_specs={
                Precision.INT8: PerformanceCharacteristics(
                    precision=Precision.INT8,
                    compute_resource=compute_resource,
                    efficiency_factor=efficiency,  # 0.85 - very efficient systolic array
                    native_acceleration=True,
                    tile_utilization=1.0,
                ),
            },
        ),
    }

    # Coral Edge TPU tile energy model (scaled down from v1)
    tile_energy_model = TPUTileEnergyModel(
        # Array configuration (estimated smaller array)
        array_width=64,  # Estimated (not published)
        array_height=64,
        num_arrays=1,  # Single small systolic array

        # Tile configuration (very small tiles for edge)
        weight_tile_size=4 * 1024,  # 4 KiB (tiny tiles)
        weight_fifo_depth=1,  # Minimal buffering

        # Pipeline (short pipeline)
        pipeline_fill_cycles=64,  # 64 cycles (estimated)
        clock_frequency_hz=500e6,  # 500 MHz

        # Accumulator (minimal for edge)
        accumulator_size=512 * 1024,  # 512 KB (estimated)
        accumulator_width=64,  # 64 elements wide

        # Unified Buffer (uses host memory, minimal on-chip)
        unified_buffer_size=512 * 1024,  # 512 KB on-chip

        # Energy coefficients (ultra-low power for edge)
        weight_memory_energy_per_byte=20.0e-12,  # 20 pJ/byte (USB 3.0, off-chip)
        weight_fifo_energy_per_byte=0.5e-12,  # 0.5 pJ/byte (on-chip SRAM)
        unified_buffer_read_energy_per_byte=0.5e-12,  # 0.5 pJ/byte
        unified_buffer_write_energy_per_byte=0.5e-12,  # 0.5 pJ/byte
        accumulator_write_energy_per_element=0.4e-12,  # 0.4 pJ (32-bit write)
        accumulator_read_energy_per_element=0.3e-12,  # 0.3 pJ (32-bit read)
        weight_shift_in_energy_per_element=0.3e-12,  # 0.3 pJ (shift register)
        activation_stream_energy_per_element=0.2e-12,  # 0.2 pJ (stream)
        mac_energy=0.15e-12,  # 0.15 pJ per INT8 MAC (very efficient)
    )

    model = HardwareResourceModel(
        name="Coral-Edge-TPU",
        hardware_type=HardwareType.TPU,
        compute_units=1,  # Single systolic array
        threads_per_unit=256,  # Systolic array dimension (estimated)
        warps_per_unit=1,
        warp_size=1,

        precision_profiles={
            # Edge TPU ONLY supports INT8 - no FP32/FP16
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=int8_tops,  # 4 TOPS INT8
                tensor_core_supported=True,  # Systolic array acts like tensor cores
                relative_speedup=1.0,  # Only mode available
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=4e9,  # ~4 GB/s (USB 3.0 or PCIe limited)
        l1_cache_per_unit=512 * 1024,  # ~512 KB on-chip memory (estimated)
        l2_cache_total=0,  # No L2, uses host memory
        main_memory=0,  # Uses host CPU memory
        energy_per_flop_fp32=energy_per_flop_fp32,
        energy_per_byte=energy_per_byte,
        energy_scaling={
            Precision.INT8: 1.0,  # Base (only mode)
        },
        min_occupancy=1.0,  # Always fully utilized
        max_concurrent_kernels=1,  # Single model at a time
        wave_quantization=1,
        thermal_operating_points=thermal_operating_points,
        default_thermal_profile="2W",
    )

    # Attach tile energy model
    model.tile_energy_model = tile_energy_model

    return model



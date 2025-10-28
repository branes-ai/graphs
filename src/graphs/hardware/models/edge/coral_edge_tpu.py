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

    return HardwareResourceModel(
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



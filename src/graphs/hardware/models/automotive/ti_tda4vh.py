"""
Ti Tda4Vh Resource Model hardware resource model.

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


def ti_tda4vh_resource_model() -> HardwareResourceModel:
    """
    Texas Instruments TDA4VH (Jacinto 7 Very High Performance) Automotive ADAS Processor.

    ARCHITECTURE:
    - High-performance ADAS for Level 3-4 autonomous driving
    - 8× Cortex-A72 @ 2.0 GHz (vs 2× in TDA4VM)
    - 4× C7x DSP cores @ 1.0 GHz (vs 1× in TDA4VM)
    - 4× MMAv2 accelerators (vs 1× MMAv1 in TDA4VM)
    - Automotive-grade: ASIL-D/SIL-3

    PERFORMANCE:
    - 4× MMAv2: 32 TOPS INT8 @ 1.0 GHz (4× TDA4VM)
    - 4× C7x DSP: 320 GFLOPS FP32
    - Expected effective: ~20-25 TOPS INT8 sustained

    CPU:
    - 8× Cortex-A72 @ 2.0 GHz
    - Multiple R5F cores for safety

    MEMORY:
    - LPDDR5 @ 6400 MT/s
    - Bandwidth: ~100 GB/s (higher than TDA4VM)
    - Capacity: Up to 16GB

    Power Profiles:
    - 20W Mode: Multi-camera Level 2+ ADAS
    - 35W Mode: Full Level 3-4 autonomy stack

    USE CASES:
    - Advanced ADAS Level 3-4
    - 8-12 camera surround view
    - Lidar + radar + camera fusion
    - Highway pilot, urban pilot
    """
    num_dsp_units = 128  # 4× TDA4VM (4× MMAv2)

    clock_20w = ClockDomain(base_clock_hz=700e6, max_boost_clock_hz=1.0e9, sustained_clock_hz=850e6, dvfs_enabled=True)
    compute_resource_20w = ComputeResource(
        resource_type="TI-4xC7x-DSP-4xMMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={Precision.INT8: 250, Precision.INT16: 125, Precision.FP16: 62, Precision.FP32: 31},
        clock_domain=clock_20w,
    )
    thermal_20w = ThermalOperatingPoint(
        name="20W-multi-camera-L2+-ADAS",
        tdp_watts=20.0,
        cooling_solution="automotive-active",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_20w,
                instruction_efficiency=0.93,
                memory_bottleneck_factor=0.75,
                efficiency_factor=0.70,  # Lower due to multi-accelerator coordination
                tile_utilization=0.85,
                native_acceleration=True,
            ),
        }
    )

    clock_35w = ClockDomain(base_clock_hz=800e6, max_boost_clock_hz=1.0e9, sustained_clock_hz=950e6, dvfs_enabled=True)
    compute_resource_35w = ComputeResource(
        resource_type="TI-4xC7x-DSP-4xMMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={Precision.INT8: 250, Precision.INT16: 125, Precision.FP16: 62, Precision.FP32: 31},
        clock_domain=clock_35w,
    )
    thermal_35w = ThermalOperatingPoint(
        name="35W-full-L3-4-autonomy",
        tdp_watts=35.0,
        cooling_solution="automotive-active-enhanced",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_35w,
                instruction_efficiency=0.95,
                memory_bottleneck_factor=0.80,
                efficiency_factor=0.78,
                tile_utilization=0.90,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="TI-TDA4VH-4xC7x-4xMMAv2",
        hardware_type=HardwareType.DSP,
        compute_units=num_dsp_units,
        threads_per_unit=250,
        warps_per_unit=1,
        warp_size=1,
        precision_profiles={
            Precision.INT8: PrecisionProfile(precision=Precision.INT8, peak_ops_per_sec=32.0e12, tensor_core_supported=False, relative_speedup=1.0, bytes_per_element=1, accumulator_precision=Precision.INT32),
            Precision.FP32: PrecisionProfile(precision=Precision.FP32, peak_ops_per_sec=320e9, tensor_core_supported=False, relative_speedup=0.01, bytes_per_element=4),
        },
        default_precision=Precision.INT8,
        peak_bandwidth=100e9,  # LPDDR5 @ 6400 MT/s
        l1_cache_per_unit=48 * 1024,
        l2_cache_total=16 * 1024 * 1024,  # Larger cache for multiple accelerators
        main_memory=16 * 1024**3,
        energy_per_flop_fp32=1.8e-12,
        energy_per_byte=15e-12,  # LPDDR5 more efficient
        energy_scaling={Precision.INT8: 0.15, Precision.INT16: 0.25, Precision.FP16: 0.50, Precision.FP32: 1.0},
        min_occupancy=0.60,  # Lower due to multi-accelerator complexity
        max_concurrent_kernels=8,  # 4× accelerators allow more parallelism
        wave_quantization=8,
        thermal_operating_points={"20W": thermal_20w, "35W": thermal_35w},
        default_thermal_profile="20W",
    )


# ============================================================================
# CEVA NeuPro Neural Processing IP
# ============================================================================


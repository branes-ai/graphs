"""
Jetson Orin Nano Resource Model hardware resource model.

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


def jetson_orin_nano_resource_model() -> HardwareResourceModel:
    """
    NVIDIA Jetson Orin Nano (8GB variant) with realistic DVFS-aware power modeling.

    Configuration: Nano variant (1024 CUDA cores, 16 Ampere SMs, 32 Tensor Cores)

    CRITICAL REALITY CHECK - Performance Specifications:
    - Marketing claim (Super): 67 TOPS INT8 (sparse, all engines)
    - Marketing claim (original): 40 TOPS INT8 (sparse, all engines)
    - Dense networks GPU only: ~21 TOPS INT8 (16 SMs × 512 ops/SM/clock × 650 MHz)
    - Customer empirical data: 2-4% of peak at typical power budgets
    - Root cause: Same as AGX - severe DVFS thermal throttling + memory bottlenecks

    Power Profiles with Realistic DVFS Behavior:
    ============================================

    7W Mode (Low Power - Battery-Powered Drones/Robots):
    - Base clock: 204 MHz (minimum)
    - Boost clock: 918 MHz (datasheet)
    - Sustained clock: 300 MHz (empirical under thermal load)
    - Thermal throttle factor: 33% (severe throttling!)
    - Effective INT8: ~1.5 TOPS (7% of 21 TOPS GPU dense peak)
    - Use case: Battery-powered drones, small robots (avoid thermal shutdown)

    15W Mode (Balanced - Typical Edge AI Deployment):
    - Sustained clock: 500 MHz (54% of boost)
    - Effective INT8: ~4 TOPS (19% of 21 TOPS GPU dense peak)
    - Use case: Edge AI devices with passive cooling

    References:
    - Jetson Orin Nano Super Specs: 67 TOPS, 102 GB/s, 15W max
    - Jetson Orin Nano 8GB Specs: 40 TOPS, 68 GB/s, 7-15W
    - TechPowerUp GPU Database: 1024 CUDA cores, 32 Tensor cores
    """
    # Physical hardware specs (Nano has half the SMs of AGX)
    num_sms = 16  # 1024 CUDA cores / 64 cores per SM
    cuda_cores_per_sm = 64
    int8_ops_per_sm_per_clock = 512  # Tensor Core: 64 × 8
    fp32_ops_per_sm_per_clock = 64   # CUDA core
    fp16_ops_per_sm_per_clock = 512  # Tensor Core FP16

    # ========================================================================
    # 7W MODE: Low power deployment (battery-powered devices)
    # ========================================================================
    clock_7w = ClockDomain(
        base_clock_hz=204e6,
        max_boost_clock_hz=918e6,
        sustained_clock_hz=300e6,  # 33% throttle
        dvfs_enabled=True,
    )

    compute_resource_7w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_7w,
    )

    # Sustained INT8: 16 SMs × 512 ops/SM/clock × 300 MHz = 2.46 TOPS
    # Effective: 2.46 × 0.40 = 0.98 TOPS (4.7% of 21 TOPS dense peak)

    thermal_7w = ThermalOperatingPoint(
        name="7W-battery",
        tdp_watts=7.0,
        cooling_solution="passive-heatsink-small",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.80,
                memory_bottleneck_factor=0.55,
                efficiency_factor=0.40,  # 40% of sustained (4% of peak!)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_7w,
                efficiency_factor=0.35,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_7w,
                efficiency_factor=0.20,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 15W MODE: Balanced configuration (typical edge deployment)
    # ========================================================================
    clock_15w = ClockDomain(
        base_clock_hz=306e6,
        max_boost_clock_hz=918e6,
        sustained_clock_hz=500e6,  # 54% of boost
        dvfs_enabled=True,
    )

    compute_resource_15w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_15w,
    )

    # Sustained INT8: 16 × 512 × 500 MHz = 4.1 TOPS
    # Effective: 4.1 × 0.50 = 2.05 TOPS (9.7% of 21 TOPS dense peak)

    thermal_15w = ThermalOperatingPoint(
        name="15W-standard",
        tdp_watts=15.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_15w,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.60,
                efficiency_factor=0.50,  # 50% of sustained (10% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_15w,
                efficiency_factor=0.45,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_15w,
                efficiency_factor=0.30,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # Hardware Resource Model
    # ========================================================================
    return HardwareResourceModel(
        name="Jetson-Orin-Nano",
        hardware_type=HardwareType.GPU,
        compute_units=num_sms,
        threads_per_unit=64,
        warps_per_unit=2,
        warp_size=32,

        # Thermal operating points with DVFS modeling
        thermal_operating_points={
            "7W": thermal_7w,   # Battery-powered devices
            "15W": thermal_15w,  # Standard edge AI deployment
        },
        default_thermal_profile="7W",  # Most realistic for embodied AI

        # Legacy precision profiles (backward compatibility)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=21e12,  # 16 SMs × 512 ops/clock × 650 MHz (realistic peak)
                tensor_core_supported=True,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=68e9,  # 68 GB/s (original) or 102 GB/s (Super)
        l1_cache_per_unit=128 * 1024,
        l2_cache_total=2 * 1024 * 1024,  # 2 MB (half of AGX)
        main_memory=8 * 1024**3,  # 8 GB
        energy_per_flop_fp32=1.2e-12,  # Slightly worse efficiency than AGX
        energy_per_byte=18e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=4,  # Fewer than AGX
        wave_quantization=2,  # Smaller wave size
    )



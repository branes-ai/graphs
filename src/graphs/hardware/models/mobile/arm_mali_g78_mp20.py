"""
Arm Mali G78 Mp20 Resource Model hardware resource model.

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


def arm_mali_g78_mp20_resource_model() -> HardwareResourceModel:
    """
    ARM Mali-G78 MP20 GPU IP Core.

    ARCHITECTURE:
    - Licensable mobile GPU IP core for SoC integration
    - 2nd generation Valhall architecture
    - 20 shader cores (MP20 configuration, max 24 cores)
    - Unified shader architecture (compute + graphics)
    - Warp width: 16 threads

    PERFORMANCE:
    - Graphics: ~1.94 TFLOPS FP32 @ 848 MHz (~97 GFLOPS per core)
    - Compute: ~2 TOPS INT8 (estimated, not optimized for AI)
    - FP16: ~3.88 TFLOPS (2× FP32)
    - Primarily a graphics GPU with compute capabilities

    PRECISION SUPPORT:
    - FP32: Native, primary mode for graphics
    - FP16: Native, 2× throughput vs FP32
    - INT8: Supported but not optimized (no tensor cores)
    - Note: Mali GPUs are graphics-focused, not AI-optimized

    MEMORY:
    - Configurable L2 cache (512 KB - 2 MB typical)
    - External memory via SoC interconnect
    - Bandwidth depends on SoC integration (20-50 GB/s typical)

    POWER:
    - 3-5W typical TDP @ 848 MHz
    - Power-efficient for mobile graphics
    - DVFS for dynamic power management

    USE CASES:
    - Mobile gaming (flagship smartphones)
    - Computational photography
    - Light AI inference (alongside dedicated NPU)
    - AR/VR applications
    - UI rendering and composition

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on ARM published specs and Google Tensor SoC
    - Used in Google Tensor (Pixel 6/6 Pro)
    - Graphics-optimized, not AI-optimized
    - For serious AI workloads, pair with dedicated NPU

    REFERENCES:
    - ARM Mali-G78 Product Brief (2020)
    - Google Tensor SoC specifications
    - AnandTech Mali-G78 analysis
    - Typical configuration: 848 MHz, 20 cores
    """
    # Model as 20 shader cores
    # Each core: 97 GFLOPS FP32 @ 848 MHz
    # Total: 1.94 TFLOPS FP32
    num_cores = 20

    clock = ClockDomain(
        base_clock_hz=400e6,      # 400 MHz minimum
        max_boost_clock_hz=950e6, # 950 MHz max
        sustained_clock_hz=848e6, # 848 MHz typical (Google Tensor)
        dvfs_enabled=True,
    )

    compute_resource = ComputeResource(
        resource_type="ARM-Mali-G78-MP20",
        num_units=num_cores,
        ops_per_unit_per_clock={
            # Each core: 97 GFLOPS @ 848 MHz → 114 ops/cycle
            Precision.FP32: 114,   # 114 FP32 ops/cycle/core
            Precision.FP16: 228,   # 228 FP16 ops/cycle/core (2× FP32)
            Precision.INT8: 114,   # ~114 INT8 ops/cycle/core (not optimized)
            Precision.INT16: 114,  # Similar to FP16 throughput
        },
        clock_domain=clock,
    )

    # Peak FP32: 20 cores × 114 ops/cycle × 848 MHz = 1.93 TFLOPS ✓

    thermal_5w = ThermalOperatingPoint(
        name="5W-mobile-gaming",
        tdp_watts=5.0,
        cooling_solution="passive-mobile",
        performance_specs={
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource,
                instruction_efficiency=0.85,  # Graphics GPU efficiency
                memory_bottleneck_factor=0.65,  # Mobile bandwidth constraints
                efficiency_factor=0.60,  # Graphics workload optimized
                tile_utilization=0.75,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.65,
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource,
                instruction_efficiency=0.70,  # Not optimized for INT8 AI
                memory_bottleneck_factor=0.65,
                efficiency_factor=0.50,  # Lower for AI workloads
                tile_utilization=0.70,
                native_acceleration=False,  # No tensor cores
            ),
        }
    )

    return HardwareResourceModel(
        name="ARM-Mali-G78-MP20",
        hardware_type=HardwareType.GPU,
        compute_units=num_cores,
        threads_per_unit=256,  # Warp size 16 × 16 execution lanes
        warps_per_unit=16,
        warp_size=16,

        thermal_operating_points={
            "5W": thermal_5w,
        },
        default_thermal_profile="5W",

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=1.94e12,  # 1.94 TFLOPS FP32
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=3.88e12,  # 3.88 TFLOPS FP16
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=1.94e12,  # ~2 TOPS INT8 (not optimized)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.FP16,  # Mobile GPU default

        # Memory (typical mobile SoC integration)
        peak_bandwidth=40e9,  # 40 GB/s (typical for mobile SoC)
        l1_cache_per_unit=32 * 1024,  # 32 KB per core
        l2_cache_total=2 * 1024 * 1024,  # 2 MB shared L2
        main_memory=8 * 1024**3,  # Up to 8 GB

        # Energy (mobile-optimized)
        energy_per_flop_fp32=2.0e-12,  # 2.0 pJ/FLOP
        energy_per_byte=15e-12,  # 15 pJ/byte (mobile DRAM)
        energy_scaling={
            Precision.INT8: 0.20,
            Precision.INT16: 0.30,
            Precision.FP16: 0.50,
            Precision.FP32: 1.0,
        },

        min_occupancy=0.60,
        max_concurrent_kernels=32,  # High parallelism for graphics
        wave_quantization=16,  # Process in groups of 16 threads (warp size)
    )

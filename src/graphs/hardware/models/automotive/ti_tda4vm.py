"""
Ti Tda4Vm Resource Model hardware resource model.

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


def ti_tda4vm_resource_model() -> HardwareResourceModel:
    """
    Texas Instruments TDA4VM (Jacinto 7) Automotive ADAS Processor.

    Based on TI's Jacinto 7 family for automotive advanced driver assistance systems.

    ARCHITECTURE:
    - Heterogeneous compute: Cortex-A72 CPU + C7x DSP + MMA
    - Primary AI accelerator: C7x DSP with Matrix Multiply Accelerator (MMA)
    - Automotive-grade: ASIL-D/SIL-3 safety certification
    - Process: 16nm FinFET (automotive qualified)
    - Temperature: -40°C to 125°C (automotive grade AEC-Q100)

    CRITICAL REALITY CHECK - Performance Specifications:
    - C7x DSP: 80 GFLOPS FP32, 256 GOPS INT16 @ 1.0 GHz
    - MMA (Matrix Multiply Accelerator): 8 TOPS INT8 @ 1.0 GHz
    - Expected effective: ~4-5 TOPS INT8 under sustained 10W operation
    - Root cause: Automotive thermal constraints + memory bandwidth

    CPU CONFIGURATION:
    - 2× Cortex-A72 @ 2.0 GHz (application processing)
    - R5F safety cores for ASIL-D compliance

    DSP (C7x):
    - 1.0 GHz peak clock
    - 80 GFLOPS FP32
    - 256 GOPS INT16
    - Vector processing: 512-bit SIMD
    - L1D: 48 KB (32 KB cache + 16 KB SRAM)

    MMA (Matrix Multiply Accelerator):
    - 8 TOPS INT8 @ 1.0 GHz
    - Integrated with C7x DSP
    - INT8/INT16 native support
    - Optimized for CNNs

    Power Profiles:
    ==============

    10W Mode (Typical Front Camera ADAS):
    - Sustained DSP clock: ~850 MHz (85% of 1.0 GHz)
    - Effective INT8: ~5 TOPS (62% of 8 TOPS peak)
    - Use case: Front camera, lane detection, object detection
    - Thermal: Automotive passive cooling

    20W Mode (Full ADAS System):
    - Sustained DSP clock: ~950 MHz (95% of 1.0 GHz)
    - Effective INT8: ~6.5 TOPS (81% of 8 TOPS peak)
    - Use case: Multi-camera (4-6 cameras), radar fusion, parking assist
    - Thermal: Active cooling in vehicle

    Memory:
    - LPDDR4x @ 3733 MT/s (dual-channel)
    - Bandwidth: ~60 GB/s
    - Capacity: Up to 8GB
    - MSMC: 8 MB on-chip SRAM for DSP

    References:
    - TI TDA4VM Product Brief
    - Jacinto 7 Architecture Overview
    - C7x DSP Core Manual
    - Automotive ADAS specifications
    """
    # ========================================================================
    # C7x DSP + MMA ARCHITECTURE MODELING
    # ========================================================================
    # C7x has:
    # - Vector DSP core: 512-bit SIMD, 1.0 GHz
    # - Matrix Multiply Accelerator (MMA): Dedicated for matrix ops
    # - We model as equivalent "DSP processing elements"

    # 8 TOPS INT8 @ 1.0 GHz
    # → 8e12 ops/sec / 1.0e9 Hz = 8,000 ops/cycle
    # If we model as 32 "DSP processing elements":
    # → 8,000 / 32 = 250 ops/cycle/unit

    num_dsp_units = 32  # Equivalent processing elements (C7x + MMA combined)

    # ========================================================================
    # CLOCK DOMAIN - 10W Automotive Thermal Envelope
    # ========================================================================
    clock_10w = ClockDomain(
        base_clock_hz=600e6,        # 600 MHz minimum
        max_boost_clock_hz=1.0e9,   # 1.0 GHz peak
        sustained_clock_hz=850e6,   # 850 MHz sustained @ 10W (85% of peak)
        dvfs_enabled=True,
    )

    # ========================================================================
    # COMPUTE RESOURCE - 10W Profile
    # ========================================================================
    compute_resource_10w = ComputeResource(
        resource_type="TI-C7x-DSP-MMA",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={
            Precision.INT8: 250,    # 250 INT8 ops/cycle/unit (MMA optimized)
            Precision.INT16: 125,   # 125 INT16 ops/cycle/unit (0.5× INT8)
            Precision.FP16: 62,     # 62 FP16 ops/cycle/unit (slower)
            Precision.FP32: 31,     # 31 FP32 ops/cycle/unit (C7x baseline)
        },
        clock_domain=clock_10w,
    )

    # Peak INT8: 32 units × 250 ops/cycle × 1.0 GHz = 8.0 TOPS ✓
    # Sustained @ 10W: 32 × 250 × 850 MHz = 6.8 TOPS
    # Effective: 6.8 × 0.70 = 4.76 TOPS (60% of 8 TOPS peak)

    # ========================================================================
    # THERMAL PROFILE (10W Front Camera ADAS)
    # ========================================================================
    thermal_10w = ThermalOperatingPoint(
        name="10W-front-camera-ADAS",
        tdp_watts=10.0,
        cooling_solution="automotive-passive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_10w,
                instruction_efficiency=0.90,  # Automotive optimized
                memory_bottleneck_factor=0.75,  # 60 GB/s for 8 TOPS
                efficiency_factor=0.70,  # 70% effective (conservative for automotive)
                tile_utilization=0.85,  # Good MMA utilization
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=compute_resource_10w,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.65,
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_10w,
                instruction_efficiency=0.75,
                memory_bottleneck_factor=0.65,
                efficiency_factor=0.55,
                tile_utilization=0.75,
                native_acceleration=False,  # Emulated via C7x
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_10w,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.60,
                efficiency_factor=0.50,
                tile_utilization=0.70,
                native_acceleration=True,  # C7x native FP32
            ),
        }
    )

    # ========================================================================
    # CLOCK DOMAIN - 20W Full ADAS System
    # ========================================================================
    clock_20w = ClockDomain(
        base_clock_hz=700e6,        # 700 MHz minimum
        max_boost_clock_hz=1.0e9,   # 1.0 GHz peak
        sustained_clock_hz=950e6,   # 950 MHz sustained @ 20W (95% of peak)
        dvfs_enabled=True,
    )

    compute_resource_20w = ComputeResource(
        resource_type="TI-C7x-DSP-MMA",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={
            Precision.INT8: 250,
            Precision.INT16: 125,
            Precision.FP16: 62,
            Precision.FP32: 31,
        },
        clock_domain=clock_20w,
    )

    # Sustained @ 20W: 32 × 250 × 950 MHz = 7.6 TOPS
    # Effective: 7.6 × 0.80 = 6.08 TOPS (76% of 8 TOPS peak)

    thermal_20w = ThermalOperatingPoint(
        name="20W-full-ADAS-system",
        tdp_watts=20.0,
        cooling_solution="automotive-active",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_20w,
                instruction_efficiency=0.92,
                memory_bottleneck_factor=0.80,
                efficiency_factor=0.80,  # Better at higher power
                tile_utilization=0.90,
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=compute_resource_20w,
                instruction_efficiency=0.90,
                memory_bottleneck_factor=0.75,
                efficiency_factor=0.75,
                tile_utilization=0.85,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_20w,
                instruction_efficiency=0.80,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.65,
                tile_utilization=0.80,
                native_acceleration=False,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_20w,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.65,
                efficiency_factor=0.60,
                tile_utilization=0.75,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # HARDWARE RESOURCE MODEL
    # ========================================================================
    return HardwareResourceModel(
        name="TI-TDA4VM-C7x-DSP",
        hardware_type=HardwareType.DSP,
        compute_units=num_dsp_units,
        threads_per_unit=4,  # Vector lanes per processing element
        warps_per_unit=1,
        warp_size=16,  # SIMD width approximation

        # Thermal operating points
        thermal_operating_points={
            "10W": thermal_10w,   # Front camera ADAS
            "20W": thermal_20w,   # Full multi-camera system
        },
        default_thermal_profile="10W",  # Most common automotive deployment

        # Legacy precision profiles (backward compatibility)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=8e12,  # 8 TOPS INT8
                tensor_core_supported=True,  # MMA acts like tensor cores
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=4e12,  # 4 TOPS INT16 (0.5× INT8)
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=2,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=80e9,  # 80 GFLOPS FP32
                tensor_core_supported=False,  # C7x vector, not MMA
                relative_speedup=0.01,
                bytes_per_element=4,
            ),
        },
        default_precision=Precision.INT8,

        # ====================================================================
        # MEMORY HIERARCHY - LPDDR4x External Memory
        # ====================================================================
        # TDA4VM uses LPDDR4x (automotive grade)
        # - Dual channel × 32-bit × 3733 MT/s = ~60 GB/s
        # - MSMC: 8 MB on-chip SRAM dedicated to C7x DSP
        # ====================================================================
        peak_bandwidth=60e9,  # 60 GB/s LPDDR4x @ 3733 MT/s
        l1_cache_per_unit=48 * 1024,  # 48 KB L1D per C7x (32 KB cache + 16 KB SRAM)
        l2_cache_total=8 * 1024 * 1024,  # 8 MB MSMC SRAM
        main_memory=8 * 1024**3,  # Up to 8 GB LPDDR4x

        # Energy (automotive-optimized, conservative)
        energy_per_flop_fp32=2.0e-12,  # 2.0 pJ/FLOP (automotive grade, higher than mobile)
        energy_per_byte=20e-12,  # 20 pJ/byte (LPDDR4x automotive)
        energy_scaling={
            Precision.INT8: 0.15,   # 15% of FP32 energy
            Precision.INT16: 0.25,  # 25% of FP32 energy
            Precision.FP16: 0.50,   # 50% of FP32 energy
            Precision.FP32: 1.0,    # Baseline
        },

        # Scheduling (automotive deterministic scheduling)
        min_occupancy=0.70,  # Automotive requires high utilization
        max_concurrent_kernels=4,  # Limited for determinism
        wave_quantization=4,
    )



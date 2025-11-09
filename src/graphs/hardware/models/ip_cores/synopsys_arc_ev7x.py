"""
Synopsys Arc Ev7X Resource Model hardware resource model.

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


def synopsys_arc_ev7x_resource_model() -> HardwareResourceModel:
    """
    Synopsys ARC EV7x Embedded Vision Processor IP Core.

    ARCHITECTURE:
    - Licensable embedded vision processor IP for SoC integration
    - Heterogeneous: 1-4 Vector Processing Units (VPUs) + DNN accelerator
    - Each VPU: 512-bit wide vector DSP
    - DNN accelerator: 880-14,080 MACs (scalable)
    - ARCv2 RISC ISA base

    PERFORMANCE:
    - Peak: Up to 35 TOPS INT8 @ 16nm FinFET
    - 4× performance of ARC EV6x
    - Configurable 1-4 core

    PRECISION SUPPORT:
    - INT8: Native via DNN accelerator (primary mode)
    - INT16: Native via VPUs
    - INT32: Supported
    - FP32: Via VPU FPU

    MEMORY:
    - Configurable local memory
    - External memory via AXI interconnect
    - Typical: 2-8 MB local memory

    POWER:
    - 3-5W typical for full EV7x @ 1.0 GHz
    - ~7-10 TOPS/W efficiency
    - Automotive-grade power management

    USE CASES:
    - Automotive ADAS (camera processing)
    - Surveillance systems
    - Drone vision
    - AR/VR applications

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Synopsys published specs
    - Need empirical benchmarking on actual silicon

    REFERENCES:
    - Synopsys ARC EV7x Product Brief (2019)
    - EE Times coverage (2019)
    - Synopsys DesignWare IP catalog
    """
    # Model 4-core configuration with full DNN accelerator
    # 35 TOPS @ 1.0 GHz → 35e12 ops/sec → 35,000 ops/cycle
    # Model as 128 equivalent processing elements: 35,000 / 128 = 273 ops/cycle/unit
    num_ev_units = 128

    clock = ClockDomain(
        base_clock_hz=600e6,      # 600 MHz minimum
        max_boost_clock_hz=1.2e9, # 1.2 GHz max
        sustained_clock_hz=1.0e9, # 1.0 GHz sustained
        dvfs_enabled=True,
    )

    compute_resource = ComputeResource(
        resource_type="Synopsys-ARC-EV7x-4core",
        num_units=num_ev_units,
        ops_per_unit_per_clock={
            Precision.INT8: 273,   # 273 INT8 MACs/cycle/unit
            Precision.INT16: 136,  # 136 INT16 MACs/cycle/unit
            Precision.INT32: 68,   # 68 INT32 MACs/cycle/unit
            Precision.FP32: 17,    # ~2.2 GFLOPS per core × 4 cores
        },
        clock_domain=clock,
    )

    # Peak INT8: 128 units × 273 ops/cycle × 1.0 GHz = 34.94 TOPS ≈ 35 TOPS ✓

    thermal_5w = ThermalOperatingPoint(
        name="5W-automotive-vision",
        tdp_watts=5.0,
        cooling_solution="automotive-passive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource,
                instruction_efficiency=0.90,  # Efficient DNN accelerator
                memory_bottleneck_factor=0.72,  # Automotive workloads
                efficiency_factor=0.68,  # 68% effective
                tile_utilization=0.82,
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=compute_resource,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.65,
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.65,
                efficiency_factor=0.60,
                tile_utilization=0.75,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="Synopsys-ARC-EV7x-4core",
        hardware_type=HardwareType.DSP,
        compute_units=num_ev_units,
        threads_per_unit=4,
        warps_per_unit=1,
        warp_size=32,

        thermal_operating_points={
            "5W": thermal_5w,
        },
        default_thermal_profile="5W",

        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=35e12,  # 35 TOPS INT8
                tensor_core_supported=True,  # DNN accelerator
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=17.5e12,  # 17.5 TOPS INT16
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=2,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=8.8e9,  # ~8.8 GFLOPS FP32 (4 cores × 2.2 GFLOPS)
                tensor_core_supported=False,
                relative_speedup=0.00025,
                bytes_per_element=4,
            ),
        },
        default_precision=Precision.INT8,

        # Memory
        peak_bandwidth=60e9,  # 60 GB/s (automotive SoC integration)
        l1_cache_per_unit=32 * 1024,  # 32 KB per unit
        l2_cache_total=4 * 1024 * 1024,  # 4 MB shared cache
        main_memory=8 * 1024**3,  # Up to 8 GB

        # Energy (automotive-optimized)
        energy_per_flop_fp32=1.4e-12,  # 1.4 pJ/FLOP
        energy_per_byte=14e-12,  # 14 pJ/byte
        energy_scaling={
            Precision.INT8: 0.14,
            Precision.INT16: 0.22,
            Precision.FP32: 1.0,
            Precision.INT4: 0.07,
        },

        min_occupancy=0.70,
        max_concurrent_kernels=8,  # 4 VPUs allow good parallelism
        wave_quantization=4,
    )


# ============================================================================
# ARM Mali GPU IP
# ============================================================================


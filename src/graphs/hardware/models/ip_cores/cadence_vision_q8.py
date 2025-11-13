"""
Cadence Vision Q8 Resource Model hardware resource model.

Extracted from resource_model.py during refactoring.
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ComputeFabric,
    get_base_alu_energy,
    ClockDomain,
    ComputeResource,
    TileSpecialization,
    KPUComputeResource,
    PerformanceCharacteristics,
    ThermalOperatingPoint,
)


def cadence_vision_q8_resource_model() -> HardwareResourceModel:
    """
    Cadence Tensilica Vision Q8 DSP IP Core (7th Generation).

    ARCHITECTURE:
    - Licensable vision DSP IP core for SoC integration
    - 7th generation Tensilica Vision DSP (flagship)
    - 1024-bit SIMD engine for vision processing
    - Heterogeneous: Vector + Scalar units

    PERFORMANCE:
    - Peak: 3.8 TOPS (INT8/INT16)
    - 129 GFLOPS FP32
    - 2× performance of Vision Q7 DSP

    PRECISION SUPPORT:
    - INT8/INT16: Native, optimized for vision
    - FP32: 129 GFLOPS
    - FP16: Supported

    MEMORY:
    - Configurable local memory
    - External memory via SoC interconnect
    - Typical: 512 KB - 2 MB local SRAM

    POWER:
    - 0.5-1W typical @ 1.0 GHz
    - ~3-7 TOPS/W efficiency
    - Power-efficient for always-on vision

    USE CASES:
    - Automotive vision (ADAS cameras)
    - Mobile device cameras (ISP + AI)
    - Surveillance cameras
    - AR/VR vision processing

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Cadence published specs
    - Need empirical benchmarking on actual silicon

    REFERENCES:
    - Cadence Tensilica Vision Q8 Product Brief (2021)
    - Tensilica Vision DSP Family specifications
    """
    # Model as 32 equivalent processing elements
    # 3.8 TOPS @ 1.0 GHz → 3.8e12 ops/sec → 118.75 ops/cycle → 32 units × 118.75 ops/unit/cycle
    num_dsp_units = 32
    sustained_clock_hz = 1.0e9  # 1.0 GHz sustained

    # ========================================================================
    # Multi-Fabric Architecture (Cadence Vision Q8 - 1024-bit SIMD DSP)
    # ========================================================================
    # SIMD Fabric (Vision processing - INT8/INT16/FP32/FP16)
    # ========================================================================
    simd_fabric = ComputeFabric(
        fabric_type="vision_q8_simd",
        circuit_type="simd_packed",      # 1024-bit SIMD engine
        num_units=32,                    # 32 SIMD units
        ops_per_unit_per_clock={
            Precision.INT8: 119,         # 119 INT8 ops/cycle/unit
            Precision.INT16: 119,        # 119 INT16 ops/cycle/unit (same as INT8 for vision)
            Precision.FP32: 4,           # 4 FP32 ops/cycle/unit (129 GFLOPS / 32 units / 1 GHz)
            Precision.FP16: 8,           # 8 FP16 ops/cycle/unit (2× FP32)
        },
        core_frequency_hz=sustained_clock_hz,  # 1.0 GHz
        process_node_nm=16,              # 16nm (typical for vision DSP IP)
        energy_per_flop_fp32=get_base_alu_energy(16, 'simd_packed'),  # 2.43 pJ
        energy_scaling={
            Precision.INT8: 0.15,        # INT8
            Precision.INT16: 0.15,       # INT16 (same as INT8)
            Precision.FP16: 0.50,        # FP16
            Precision.FP32: 1.0,         # Baseline
        }
    )

    # SIMD INT8: 32 units × 119 ops/cycle × 1.0 GHz = 3.81 TOPS ≈ 3.8 TOPS ✓
    # SIMD FP32: 32 units × 4 ops/cycle × 1.0 GHz = 128 GFLOPS ≈ 129 GFLOPS ✓

    clock = ClockDomain(
        base_clock_hz=600e6,      # 600 MHz minimum
        max_boost_clock_hz=1.2e9, # 1.2 GHz max
        sustained_clock_hz=1.0e9, # 1.0 GHz sustained
        dvfs_enabled=True,
    )

    compute_resource = ComputeResource(
        resource_type="Cadence-Tensilica-Vision-Q8",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={
            Precision.INT8: 119,   # ~119 INT8 ops/cycle/unit
            Precision.INT16: 119,  # Same for INT16 (vision optimized)
            Precision.FP32: 4,     # 129 GFLOPS / 32 units / 1.0 GHz
            Precision.FP16: 8,     # 2× FP32
        },
        clock_domain=clock,
    )

    thermal_1w = ThermalOperatingPoint(
        name="1W-vision",
        tdp_watts=1.0,
        cooling_solution="passive-mobile",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource,
                instruction_efficiency=0.88,  # Good vision DSP efficiency
                memory_bottleneck_factor=0.70,  # Vision workloads are bandwidth-sensitive
                efficiency_factor=0.65,  # 65% effective
                tile_utilization=0.80,
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
        name="Cadence-Tensilica-Vision-Q8",
        hardware_type=HardwareType.DSP,

        # NEW: Multi-fabric architecture (1024-bit SIMD engine)
        compute_fabrics=[simd_fabric],

        compute_units=num_dsp_units,
        threads_per_unit=4,
        warps_per_unit=1,
        warp_size=32,

        thermal_operating_points={
            "1W": thermal_1w,
        },
        default_thermal_profile="1W",

        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=3.8e12,  # 3.8 TOPS
                tensor_core_supported=False,  # Vector DSP, not tensor cores
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=3.8e12,  # 3.8 TOPS (same as INT8)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=129e9,  # 129 GFLOPS FP32
                tensor_core_supported=False,
                relative_speedup=0.034,
                bytes_per_element=4,
            ),
        },
        default_precision=Precision.INT8,

        # Memory
        peak_bandwidth=40e9,  # 40 GB/s (typical SoC integration)
        l1_cache_per_unit=32 * 1024,  # 32 KB per unit
        l2_cache_total=1 * 1024 * 1024,  # 1 MB shared cache
        main_memory=4 * 1024**3,  # Up to 4 GB

        # Energy
        energy_per_flop_fp32=simd_fabric.energy_per_flop_fp32,  # 2.43 pJ (16nm, simd_packed)
        energy_per_byte=12e-12,  # 12 pJ/byte
        energy_scaling={
            Precision.INT8: 0.15,
            Precision.INT16: 0.15,
            Precision.FP32: 1.0,
            Precision.FP16: 0.50,
        },

        min_occupancy=0.70,
        max_concurrent_kernels=4,
        wave_quantization=4,
    )


# ============================================================================
# Synopsys ARC EV Embedded Vision Processor IP
# ============================================================================


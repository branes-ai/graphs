"""
Ti Tda4Vl Resource Model hardware resource model.

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


def ti_tda4vl_resource_model() -> HardwareResourceModel:
    """
    Texas Instruments TDA4VL (Jacinto 7 Entry-Level) Automotive ADAS Processor.

    ARCHITECTURE:
    - Entry-level ADAS: Lower cost, lower performance than TDA4VM
    - Heterogeneous: Cortex-A72 + C7x DSP + MMAv2 (newer generation)
    - Automotive-grade: ASIL-B/C (lower safety level than TDA4VM)
    - Temperature: -40°C to 125°C

    KEY DIFFERENCES FROM TDA4VM:
    - Half the AI performance: 4 TOPS INT8 vs 8 TOPS
    - Lower CPU frequency: A72 @ 1.2 GHz vs 2.0 GHz
    - Newer MMAv2 architecture (more efficient than MMAv1 in TDA4VM)
    - Lower power envelope: 7-12W typical

    PERFORMANCE:
    - MMAv2: 4 TOPS INT8 @ 1.0 GHz (half of TDA4VM)
    - C7x DSP: 40 GFLOPS FP32 @ 1.0 GHz
    - Expected effective: ~2-3 TOPS INT8 under sustained operation

    CPU:
    - 2× Cortex-A72 @ 1.2 GHz
    - R5F safety cores for ASIL-B/C

    MEMORY:
    - LPDDR4x @ 3733 MT/s
    - Bandwidth: ~60 GB/s
    - Capacity: Up to 4GB

    Power Profiles:
    - 7W Mode: Entry-level ADAS (single camera, lane detection)
    - 12W Mode: Multi-function ADAS (front camera + side cameras)

    USE CASES:
    - Entry-level ADAS (Lane Keep, TSR, basic ACC)
    - Cost-sensitive automotive markets
    - Single front-facing camera systems
    """
    num_dsp_units = 16  # Half of TDA4VM (4 TOPS vs 8 TOPS)
    sustained_clock_hz = 900e6  # 900 MHz sustained @ 12W

    # ========================================================================
    # Multi-Fabric Architecture (TI C7x DSP + MMAv2 Entry-Level)
    # ========================================================================
    # C7x DSP Fabric (General VLIW/SIMD compute: FP32, control flow)
    # ========================================================================
    c7x_fabric = ComputeFabric(
        fabric_type="c7x_dsp",
        circuit_type="simd_packed",    # VLIW/SIMD DSP
        num_units=4,                   # 4 C7x DSP cores (half of TDA4AL)
        ops_per_unit_per_clock={
            Precision.FP32: 10,         # 40 GFLOPS / 4 cores / 1.0 GHz = 10 ops/cycle
            Precision.FP16: 20,         # 2× FP32
        },
        core_frequency_hz=1.0e9,       # 1.0 GHz base
        process_node_nm=28,             # 28nm TSMC
        energy_per_flop_fp32=get_base_alu_energy(28, 'simd_packed'),  # 3.6 pJ
        energy_scaling={
            Precision.FP32: 1.0,        # Baseline
            Precision.FP16: 0.50,       # Half precision
            Precision.INT8: 0.15,       # INT8 (used by MMAv2)
        }
    )

    # ========================================================================
    # MMAv2 Tensor Fabric (Matrix operations: INT8 convolution, matmul)
    # ========================================================================
    mma_fabric = ComputeFabric(
        fabric_type="mma_v2",
        circuit_type="tensor_core",     # Matrix multiply accelerator
        num_units=1,                    # Single MMA unit (half capacity vs TDA4AL)
        ops_per_unit_per_clock={
            Precision.INT8: 4000,       # 4 TOPS / 1.0 GHz = 4000 ops/cycle (half of TDA4AL)
            Precision.INT16: 2000,      # Half of INT8
        },
        core_frequency_hz=1.0e9,        # 1.0 GHz
        process_node_nm=28,
        energy_per_flop_fp32=get_base_alu_energy(28, 'tensor_core'),  # 3.4 pJ (15% better)
        energy_scaling={
            Precision.INT8: 0.15,       # INT8 is very efficient
            Precision.INT16: 0.25,
        }
    )

    # C7x DSP FP32: 4 cores × 10 ops/cycle × 1.0 GHz = 40 GFLOPS ✓
    # MMAv2 INT8: 1 unit × 4000 ops/cycle × 1.0 GHz = 4 TOPS ✓

    # ========================================================================
    # Thermal Operating Points
    # ========================================================================
    clock_7w = ClockDomain(
        base_clock_hz=500e6,
        max_boost_clock_hz=1.0e9,
        sustained_clock_hz=750e6,  # 75% sustained @ 7W
        dvfs_enabled=True,
    )

    compute_resource_7w = ComputeResource(
        resource_type="TI-C7x-DSP-MMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={
            Precision.INT8: 250,
            Precision.INT16: 125,
            Precision.FP16: 62,
            Precision.FP32: 31,
        },
        clock_domain=clock_7w,
    )

    thermal_7w = ThermalOperatingPoint(
        name="7W-entry-ADAS",
        tdp_watts=7.0,
        cooling_solution="automotive-passive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.92,  # MMAv2 more efficient
                memory_bottleneck_factor=0.75,
                efficiency_factor=0.72,
                tile_utilization=0.85,
                native_acceleration=True,
            ),
        }
    )

    clock_12w = ClockDomain(
        base_clock_hz=700e6,
        max_boost_clock_hz=1.0e9,
        sustained_clock_hz=900e6,
        dvfs_enabled=True,
    )

    compute_resource_12w = ComputeResource(
        resource_type="TI-C7x-DSP-MMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={Precision.INT8: 250, Precision.INT16: 125, Precision.FP16: 62, Precision.FP32: 31},
        clock_domain=clock_12w,
    )

    thermal_12w = ThermalOperatingPoint(
        name="12W-multi-function-ADAS",
        tdp_watts=12.0,
        cooling_solution="automotive-passive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_12w,
                instruction_efficiency=0.93,
                memory_bottleneck_factor=0.80,
                efficiency_factor=0.78,
                tile_utilization=0.88,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="TI-TDA4VL-C7x-MMAv2",
        hardware_type=HardwareType.DSP,

        # NEW: Multi-fabric architecture (C7x DSP + MMAv2 Entry-Level)
        compute_fabrics=[c7x_fabric, mma_fabric],

        compute_units=num_dsp_units,
        threads_per_unit=250,
        warps_per_unit=1,
        warp_size=1,
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=4.0e12,  # 4 TOPS INT8
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=40e9,  # 40 GFLOPS FP32
                tensor_core_supported=False,
                relative_speedup=0.01,
                bytes_per_element=4,
            ),
        },
        default_precision=Precision.INT8,
        peak_bandwidth=60e9,
        l1_cache_per_unit=48 * 1024,
        l2_cache_total=8 * 1024 * 1024,
        main_memory=4 * 1024**3,
        # Energy (use C7x DSP fabric as baseline for general-purpose operations)
        energy_per_flop_fp32=c7x_fabric.energy_per_flop_fp32,  # 3.6 pJ (28nm, SIMD packed)
        energy_per_byte=20e-12,
        energy_scaling={Precision.INT8: 0.15, Precision.INT16: 0.25, Precision.FP16: 0.50, Precision.FP32: 1.0},
        min_occupancy=0.70,
        max_concurrent_kernels=4,
        wave_quantization=4,
        thermal_operating_points={"7W": thermal_7w, "12W": thermal_12w},
        default_thermal_profile="7W",
    )



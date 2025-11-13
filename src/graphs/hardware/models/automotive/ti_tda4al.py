"""
Ti Tda4Al Resource Model hardware resource model.

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


def ti_tda4al_resource_model() -> HardwareResourceModel:
    """
    Texas Instruments TDA4AL (Jacinto 7 Advanced Low-Power) Automotive ADAS Processor.

    ARCHITECTURE:
    - Mid-range ADAS: Similar AI performance to TDA4VM but newer architecture
    - Heterogeneous: Dual A72 @ 2.0 GHz + C7x DSP + MMAv2
    - Automotive-grade: ASIL-D/SIL-3
    - Process: Newer node, better power efficiency than TDA4VM

    KEY DIFFERENCES FROM TDA4VM:
    - Same AI performance: 8 TOPS INT8
    - MMAv2 architecture (more efficient than MMAv1)
    - Higher CPU frequency: A72 @ 2.0 GHz (vs TDA4VM's 2.0 GHz)
    - Better power efficiency: 10-18W range

    PERFORMANCE:
    - MMAv2: 8 TOPS INT8 @ 1.0 GHz
    - C7x DSP: 80 GFLOPS FP32 @ 1.0 GHz
    - Expected effective: ~5-6 TOPS INT8 sustained

    MEMORY:
    - LPDDR4x @ 3733 MT/s
    - Bandwidth: ~60 GB/s
    - Capacity: Up to 8GB

    Power Profiles:
    - 10W Mode: Front camera ADAS
    - 18W Mode: Multi-camera ADAS (better than TDA4VM @ 20W due to MMAv2)

    USE CASES:
    - ADAS Level 2-3 (similar to TDA4VM but more efficient)
    - Replaces TDA4VM in newer designs
    """
    num_dsp_units = 32  # Same as TDA4VM
    sustained_clock_hz = 980e6  # 980 MHz sustained @ 18W

    # ========================================================================
    # Multi-Fabric Architecture (TI C7x DSP + MMAv2)
    # ========================================================================
    # C7x DSP Fabric (General VLIW/SIMD compute: FP32, control flow)
    # ========================================================================
    c7x_fabric = ComputeFabric(
        fabric_type="c7x_dsp",
        circuit_type="simd_packed",    # VLIW/SIMD DSP
        num_units=8,                   # 8 C7x DSP cores (estimated)
        ops_per_unit_per_clock={
            Precision.FP32: 10,         # 80 GFLOPS / 8 cores / 1.0 GHz = 10 ops/cycle
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
        num_units=1,                    # Single MMA unit (large systolic array)
        ops_per_unit_per_clock={
            Precision.INT8: 8000,       # 8 TOPS / 1.0 GHz = 8000 ops/cycle
            Precision.INT16: 4000,      # Half of INT8
        },
        core_frequency_hz=1.0e9,        # 1.0 GHz
        process_node_nm=28,
        energy_per_flop_fp32=get_base_alu_energy(28, 'tensor_core'),  # 3.4 pJ (15% better)
        energy_scaling={
            Precision.INT8: 0.15,       # INT8 is very efficient
            Precision.INT16: 0.25,
        }
    )

    # C7x DSP FP32: 8 cores × 10 ops/cycle × 1.0 GHz = 80 GFLOPS ✓
    # MMAv2 INT8: 1 unit × 8000 ops/cycle × 1.0 GHz = 8 TOPS ✓

    # ========================================================================
    # Thermal Operating Points
    # ========================================================================
    clock_10w = ClockDomain(base_clock_hz=600e6, max_boost_clock_hz=1.0e9, sustained_clock_hz=880e6, dvfs_enabled=True)
    compute_resource_10w = ComputeResource(
        resource_type="TI-C7x-DSP-MMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={Precision.INT8: 250, Precision.INT16: 125, Precision.FP16: 62, Precision.FP32: 31},
        clock_domain=clock_10w,
    )
    thermal_10w = ThermalOperatingPoint(
        name="10W-front-camera-ADAS",
        tdp_watts=10.0,
        cooling_solution="automotive-passive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_10w,
                instruction_efficiency=0.94,  # MMAv2 improvement
                memory_bottleneck_factor=0.78,
                efficiency_factor=0.75,  # Better than TDA4VM's 0.70
                tile_utilization=0.88,
                native_acceleration=True,
            ),
        }
    )

    clock_18w = ClockDomain(base_clock_hz=700e6, max_boost_clock_hz=1.0e9, sustained_clock_hz=980e6, dvfs_enabled=True)
    compute_resource_18w = ComputeResource(
        resource_type="TI-C7x-DSP-MMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={Precision.INT8: 250, Precision.INT16: 125, Precision.FP16: 62, Precision.FP32: 31},
        clock_domain=clock_18w,
    )
    thermal_18w = ThermalOperatingPoint(
        name="18W-multi-camera-ADAS",
        tdp_watts=18.0,
        cooling_solution="automotive-active",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_18w,
                instruction_efficiency=0.95,
                memory_bottleneck_factor=0.82,
                efficiency_factor=0.82,
                tile_utilization=0.92,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="TI-TDA4AL-C7x-MMAv2",
        hardware_type=HardwareType.DSP,

        # NEW: Multi-fabric architecture (C7x DSP + MMAv2)
        compute_fabrics=[c7x_fabric, mma_fabric],

        compute_units=num_dsp_units,
        threads_per_unit=250,
        warps_per_unit=1,
        warp_size=1,
        precision_profiles={
            Precision.INT8: PrecisionProfile(precision=Precision.INT8, peak_ops_per_sec=8.0e12, tensor_core_supported=False, relative_speedup=1.0, bytes_per_element=1, accumulator_precision=Precision.INT32),
            Precision.FP32: PrecisionProfile(precision=Precision.FP32, peak_ops_per_sec=80e9, tensor_core_supported=False, relative_speedup=0.01, bytes_per_element=4),
        },
        default_precision=Precision.INT8,
        peak_bandwidth=60e9,
        l1_cache_per_unit=48 * 1024,
        l2_cache_total=8 * 1024 * 1024,
        main_memory=8 * 1024**3,
        # Energy (use C7x DSP fabric as baseline for general-purpose operations)
        energy_per_flop_fp32=c7x_fabric.energy_per_flop_fp32,  # 3.6 pJ (28nm, SIMD packed)
        energy_per_byte=18e-12,
        energy_scaling={Precision.INT8: 0.15, Precision.INT16: 0.25, Precision.FP16: 0.50, Precision.FP32: 1.0},
        min_occupancy=0.70,
        max_concurrent_kernels=4,
        wave_quantization=4,
        thermal_operating_points={"10W": thermal_10w, "18W": thermal_18w},
        default_thermal_profile="10W",
    )



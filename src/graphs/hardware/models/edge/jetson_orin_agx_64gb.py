"""
Jetson Orin AGX 64GB Resource Model hardware resource model.

MEMORY: 64 GB LPDDR5

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
    BOMCostProfile,
)


def jetson_orin_agx_64gb_resource_model() -> HardwareResourceModel:
    """
    NVIDIA Jetson Orin AGX 64GB with realistic DVFS-aware multi-power-profile modeling.

    MEMORY: 64 GB LPDDR5

    Configuration: AGX variant (2048 CUDA cores, 16 Ampere SMs, 64 Tensor Cores)

    CRITICAL REALITY CHECK - Performance Specifications:
    - Marketing claim: 275 TOPS INT8 (sparse networks, all engines: GPU+DLA+PVA)
    - Dense networks total: 138 TOPS INT8 (GPU + 2×DLA)
      - GPU only (dense): 85 TOPS INT8 ← Relevant for PyTorch workloads
      - 2×DLA (dense): 52.5 TOPS INT8
    - GPU sparse: 170 TOPS INT8 (requires specially-prepared sparse networks)
    - Customer empirical data: 2-4% of peak at typical power budgets
    - Root cause: Severe DVFS thermal throttling + memory bottlenecks

    Power Profiles with Realistic DVFS Behavior:
    ============================================

    15W Mode (Passive Cooling - What Customers Actually Deploy):
    - Base clock: 306 MHz (guaranteed minimum)
    - Boost clock: 1.02 GHz (datasheet spec, rarely sustained)
    - Sustained clock: 400 MHz (empirical under thermal load)
    - Thermal throttle factor: 39% (severe throttling!)
    - Effective INT8: ~5 TOPS (6% of 85 TOPS GPU dense peak)
    - Use case: Battery-powered robots, drones (must avoid thermal shutdown)

    30W Mode (Active Cooling - Better but Still Throttles):
    - Sustained clock: 650 MHz (64% of boost)
    - Effective INT8: ~17 TOPS (20% of 85 TOPS GPU dense peak)
    - Use case: Tethered robots with active cooling

    60W Mode (Max Performance - Unrealistic for Embodied AI):
    - Sustained clock: 1.0 GHz (98% of boost)
    - Effective INT8: ~51 TOPS (60% of 85 TOPS GPU dense peak)
    - Use case: Benchtop testing only (too hot for deployment!)

    References:
    - Jetson Orin AGX Datasheet: NVIDIA Technical Brief
    - Empirical measurements: Customer lab data (2-4% of peak @ 15W)
    - DVFS behavior: Observed clock throttling under sustained load
    """
    # Physical hardware specs (constant across power modes)
    # Official specs: 2048 CUDA cores total, Ampere architecture
    num_sms = 16                     # 2048 CUDA cores ÷ 128 cores/SM = 16 SMs
    cuda_cores_per_sm = 128          # Ampere architecture standard
    tensor_cores_per_sm = 4          # 64 Tensor cores ÷ 16 SMs = 4 per SM
    int8_ops_per_sm_per_clock = 512  # Tensor Core capability: 128 × 4
    fp32_ops_per_sm_per_clock = 256  # CUDA core capability: 128 cores × 2 ops (FMA)
    fp16_ops_per_sm_per_clock = 512  # Tensor Core FP16

    # ========================================================================
    # 15W MODE: Realistic deployment configuration (passive cooling)
    # ========================================================================
    clock_15w = ClockDomain(
        base_clock_hz=306e6,         # 306 MHz guaranteed minimum
        max_boost_clock_hz=1.02e9,   # 1.02 GHz datasheet boost
        sustained_clock_hz=400e6,    # 400 MHz empirical (39% throttle!)
        dvfs_enabled=True,
    )

    compute_resource_15w_int8 = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_15w,
    )

    # Peak INT8: 16 SMs × 512 ops/SM/clock × 1.02 GHz = 8.4 TOPS (simplified model)
    # NOTE: Actual GPU dense peak is 85 TOPS at 1.3 GHz (using all 64 Tensor Cores)
    # Sustained INT8: 16 × 512 × 400 MHz = 3.3 TOPS
    # Effective INT8: 3.3 TOPS × 0.47 empirical derate = 1.5 TOPS (1.8% of 85 TOPS GPU dense)

    thermal_15w = ThermalOperatingPoint(
        name="15W-passive",
        tdp_watts=15.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_15w_int8,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.60,
                efficiency_factor=0.47,  # 47% of sustained (3% of peak!)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_15w_int8,
                efficiency_factor=0.40,  # Worse (more memory bound)
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_15w_int8,
                efficiency_factor=0.25,  # Much worse
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 30W MODE: Balanced configuration (active fan cooling)
    # ========================================================================
    clock_30w = ClockDomain(
        base_clock_hz=612e6,
        max_boost_clock_hz=1.15e9,
        sustained_clock_hz=650e6,    # 650 MHz sustained (57% throttle)
        dvfs_enabled=True,
    )

    compute_resource_30w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_30w,
    )

    # Sustained INT8: 16 × 512 × 650 MHz = 5.3 TOPS
    # Effective: 5.3 × 0.60 = 3.2 TOPS (3.8% of 85 TOPS GPU dense peak)

    thermal_30w = ThermalOperatingPoint(
        name="30W-active",
        tdp_watts=30.0,
        cooling_solution="active-fan",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.60,  # Better (10% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.50,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.35,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 60W MODE: Max performance (unrealistic for robots - benchtop only!)
    # ========================================================================
    clock_60w = ClockDomain(
        base_clock_hz=918e6,
        max_boost_clock_hz=1.3e9,
        sustained_clock_hz=1.0e9,    # 1.0 GHz sustained (77% of boost)
        dvfs_enabled=True,
    )

    compute_resource_60w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_60w,
    )

    # Sustained INT8: 32 × 512 × 1.0 GHz = 16.4 TOPS
    # Effective: 16.4 × 0.75 = 12.3 TOPS (7.2% of peak)

    thermal_60w = ThermalOperatingPoint(
        name="60W-max",
        tdp_watts=60.0,
        cooling_solution="active-fan-max",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.75,  # Best case (still only 30% of peak!)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.65,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.50,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # BOM Cost Profile
    # ========================================================================
    bom_cost = BOMCostProfile(
        silicon_die_cost=380.0,
        package_cost=60.0,
        memory_cost=160.0,
        pcb_assembly_cost=35.0,
        thermal_solution_cost=15.0,
        other_costs=20.0,
        total_bom_cost=670.0,
        margin_multiplier=1.34,
        retail_price=899.0,
        volume_tier="10K+",
        process_node="8nm",
        year=2025,
        notes="High-end edge AI module. 64GB LPDDR5. NVIDIA pricing: competitive margin for robotics/automotive. Competes with datacenter inference cards.",
    )

    # ========================================================================
    # Hardware Resource Model (uses NEW thermal operating points)
    # ========================================================================
    return HardwareResourceModel(
        name="Jetson-Orin-AGX-64GB",
        hardware_type=HardwareType.GPU,
        compute_units=num_sms,
        threads_per_unit=64,
        warps_per_unit=2,
        warp_size=32,

        # GPU Microarchitecture (Ampere)
        cuda_cores_per_sm=cuda_cores_per_sm,
        tensor_cores_per_sm=tensor_cores_per_sm,
        ops_per_clock_per_core=2.0,  # FMA: 2 ops/clock for FP32
        sm_boost_clock_hz=1.3e9,     # 1.3 GHz boost (60W mode)
        sm_sustained_clock_hz=650e6,  # 650 MHz sustained (30W mode typical)

        # NEW: Thermal operating points with DVFS modeling
        thermal_operating_points={
            "15W": thermal_15w,  # Realistic deployment
            "30W": thermal_30w,  # Balanced
            "60W": thermal_60w,  # Max performance (unrealistic)
        },
        default_thermal_profile="15W",  # Most realistic for embodied AI

        # Legacy precision profiles (for backward compatibility)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=85e12,  # GPU dense (PyTorch workloads; 170 TOPS sparse)
                tensor_core_supported=True,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=204.8e9,
        l1_cache_per_unit=128 * 1024,
        l2_cache_total=4 * 1024 * 1024,
        main_memory=64 * 1024**3,
        energy_per_flop_fp32=1.0e-12,
        energy_per_byte=15e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=8,
        wave_quantization=4,
        bom_cost_profile=bom_cost,
    )



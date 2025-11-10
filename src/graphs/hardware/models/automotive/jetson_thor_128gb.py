"""
Jetson Thor 128GB Resource Model hardware resource model.

MEMORY: 128 GB LPDDR5X

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


def jetson_thor_128gb_resource_model() -> HardwareResourceModel:
    """
    NVIDIA Jetson Thor 128GB with realistic DVFS modeling (Next-gen edge AI, 2025+).

    MEMORY: 128 GB LPDDR5X

    Configuration: Blackwell-based GPU, 64 SMs, 1000 TOPS INT8 peak (actual datapath)

    CRITICAL REALITY CHECK (Projected based on Orin empirical data):
    - NVIDIA claims: 2000 TOPS INT8 (includes sparsity - workload dependent!)
    - Actual datapath: 1000 TOPS INT8 (speed-of-light without sparsity)
    - Expected reality: 3-5% of peak at deployable power budgets
    - Improved thermal design vs Orin, but still throttles significantly

    Power Profiles (Projected):
    ==========================

    30W Mode (Typical Deployment - Autonomous Vehicles):
    - Better thermal design than Orin
    - Sustained clock: 750 MHz (58% of boost)
    - Effective INT8: ~30 TOPS (3% of peak)
    - Use case: Autonomous vehicles with active cooling

    60W Mode (Max Performance - High-end Robotics):
    - Sustained clock: 1.1 GHz (85% of boost)
    - Effective INT8: ~60 TOPS (6% of peak)
    - Use case: Humanoid robots, industrial AGVs

    100W Mode (Benchtop/Development Only):
    - Sustained clock: 1.25 GHz (96% of boost)
    - Effective INT8: ~100 TOPS (10% of peak)
    - Use case: Development workstations (not deployable)
    """
    # Physical hardware (constant across power modes)
    num_sms = 64  # Estimated for 1000 TOPS actual datapath
    int8_ops_per_sm_per_clock = 256  # Actual datapath (not sparsity-inflated)
    fp32_ops_per_sm_per_clock = 128  # Wider SMs
    fp16_ops_per_sm_per_clock = 256  # Match INT8 without sparsity

    # ========================================================================
    # 30W MODE: Typical deployment (autonomous vehicles)
    # ========================================================================
    clock_30w = ClockDomain(
        base_clock_hz=500e6,
        max_boost_clock_hz=1.3e9,
        sustained_clock_hz=750e6,  # 58% of boost (better than Orin!)
        dvfs_enabled=True,
    )

    compute_resource_30w = ComputeResource(
        resource_type="Blackwell-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_30w,
    )

    # Sustained: 64 × 256 × 750 MHz = 12.3 TOPS
    # Effective: 12.3 × 0.50 = 6.1 TOPS (0.6% of peak!)

    thermal_30w = ThermalOperatingPoint(
        name="30W-active",
        tdp_watts=30.0,
        cooling_solution="active-fan",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.50,  # 50% of sustained (3% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.45,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.30,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 60W MODE: High-performance robotics
    # ========================================================================
    clock_60w = ClockDomain(
        base_clock_hz=800e6,
        max_boost_clock_hz=1.3e9,
        sustained_clock_hz=1.1e9,  # 85% of boost
        dvfs_enabled=True,
    )

    compute_resource_60w = ComputeResource(
        resource_type="Blackwell-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_60w,
    )

    # Sustained: 64 × 256 × 1.1 GHz = 18.0 TOPS
    # Effective: 18.0 × 0.65 = 11.7 TOPS (1.2% of peak)

    thermal_60w = ThermalOperatingPoint(
        name="60W-active",
        tdp_watts=60.0,
        cooling_solution="active-fan-enhanced",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.65,  # Better (6% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.60,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.45,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 100W MODE: Development/benchtop only
    # ========================================================================
    clock_100w = ClockDomain(
        base_clock_hz=1.0e9,
        max_boost_clock_hz=1.3e9,
        sustained_clock_hz=1.25e9,  # 96% of boost
        dvfs_enabled=True,
    )

    compute_resource_100w = ComputeResource(
        resource_type="Blackwell-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_100w,
    )

    # Sustained: 64 × 256 × 1.25 GHz = 20.5 TOPS
    # Effective: 20.5 × 0.80 = 16.4 TOPS (1.6% of peak)

    thermal_100w = ThermalOperatingPoint(
        name="100W-max",
        tdp_watts=100.0,
        cooling_solution="active-fan-max",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_100w,
                efficiency_factor=0.80,  # Best case (10% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_100w,
                efficiency_factor=0.70,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_100w,
                efficiency_factor=0.55,
                native_acceleration=True,
            ),
        }
    )

    # BOM Cost Profile
    bom_cost = BOMCostProfile(
        silicon_die_cost=850.0,
        package_cost=180.0,
        memory_cost=350.0,
        pcb_assembly_cost=90.0,
        thermal_solution_cost=80.0,
        other_costs=50.0,
        total_bom_cost=1600.0,
        margin_multiplier=1.56,
        retail_price=2500.0,
        volume_tier="10K+",
        process_node="4nm",
        year=2025,
        notes="Next-gen automotive AI platform. 128GB HBM3, Blackwell architecture, 4nm process. Advanced CoWoS-like packaging. Liquid cooling capable. Target: autonomous vehicles, humanoid robots.",
    )

    return HardwareResourceModel(
        name="Jetson-Thor-128GB",
        hardware_type=HardwareType.GPU,
        compute_units=num_sms,
        threads_per_unit=128,
        warps_per_unit=4,
        warp_size=32,

        # NEW: Thermal operating points
        thermal_operating_points={
            "30W": thermal_30w,  # Typical deployment
            "60W": thermal_60w,  # High-performance
            "100W": thermal_100w,  # Development only
        },
        default_thermal_profile="30W",

        # Legacy for backward compat (uses realistic dense workload peak, ~10% of datapath)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=100e12,  # Dense workload peak (~10% of 1000 TOPS datapath)
                tensor_core_supported=True,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=450e9,
        l1_cache_per_unit=256 * 1024,
        l2_cache_total=8 * 1024 * 1024,
        main_memory=128 * 1024**3,
        energy_per_flop_fp32=0.8e-12,
        energy_per_byte=12e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=16,
        wave_quantization=4,
        bom_cost_profile=bom_cost,
    )



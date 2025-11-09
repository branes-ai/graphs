"""
Qualcomm SA8775P Snapdragon Ride Resource Model - Mid-range Automotive ADAS SoC

5nm automotive processor with Hexagon NPU (dual HMX co-processors).
Target: ADAS L2+/L3, cockpit compute, automotive vision, robotics.

Configuration:
- 32 TOPS mixed precision (estimated, INT8/INT16/FP16)
- Hexagon DSP with dual HMX (Hexagon Matrix eXtensions)
- 5nm TSMC process (ASIL D safety certified)
- 20-45W TDP range

Competitor to: KPU-T256, Jetson Orin AGX, Mobileye EyeQ5H
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ClockDomain,
    ComputeResource,
    PerformanceCharacteristics,
    ThermalOperatingPoint,
    BOMCostProfile,
)


def qualcomm_sa8775p_resource_model() -> HardwareResourceModel:
    """
    Qualcomm SA8775P Snapdragon Ride with Hexagon NPU (dual HMX).

    ARCHITECTURE:
    - Hexagon DSP with quad HVX (Vector) + dual HMX (Matrix) co-processors
    - 5nm TSMC automotive-grade process (ASIL D certified)
    - Mixed precision: INT4/INT8/INT16/FP16
    - Safety features: Lockstep cores, ECC memory, functional safety island

    PERFORMANCE:
    - 32 TOPS mixed precision (estimated based on market positioning)
    - Realistic: ~16-20 TOPS INT8 sustained (50-60% efficiency)
    - Better than Jetson Orin AGX @ 30W (5-10 TOPS, 2-6%)
    - Lower than KPU-T256 (64 TOPS peak, ~45 TOPS effective @ 70%)

    POWER PROFILES:
    - 20W Mode: Passive cooling (cockpit compute)
    - 30W Mode: Active cooling (ADAS + cockpit)
    - 45W Mode: Max performance (L3 autonomous driving)

    USE CASES:
    - Automotive cockpit (HMI, voice, driver monitoring)
    - ADAS Level 2+/3 (highway pilot, parking assist)
    - Robotics (AMR, inspection robots, AGVs)

    CALIBRATION STATUS: ⚠️ ESTIMATED
    - TOPS estimate based on automotive SoC market analysis (30-40 TOPS typical)
    - Hexagon architecture suggests 50-60% efficiency (better than GPU, worse than KPU)
    - Cross-validated against 5nm process capabilities
    """
    # Physical hardware (constant across power modes)
    # Dual HMX (Matrix) + Quad HVX (Vector)
    # Estimate: 32 processing units total
    num_processing_units = 32  # HMX + HVX combined
    int8_ops_per_unit_per_clock = 256  # Matrix accelerator capabilities

    # 32 TOPS INT8 @ 2.0 GHz sustained
    # → 32e12 / (32 units × 2.0e9) = 500 ops/unit/clock
    # Using 256 ops/unit/clock requires higher count or clock

    # ========================================================================
    # 20W MODE: Passive cooling (cockpit compute)
    # ========================================================================
    clock_20w = ClockDomain(
        base_clock_hz=1.0e9,        # 1.0 GHz base
        max_boost_clock_hz=2.4e9,   # 2.4 GHz boost
        sustained_clock_hz=1.6e9,   # 1.6 GHz sustained (67% throttle)
        dvfs_enabled=True,
    )

    compute_resource_20w = ComputeResource(
        resource_type="Hexagon-HMX-Matrix-Accelerator",
        num_units=num_processing_units,
        ops_per_unit_per_clock={
            Precision.INT8: 256,
            Precision.INT4: 512,
            Precision.INT16: 128,
            Precision.FP16: 128,
        },
        clock_domain=clock_20w,
    )

    # Sustained INT8: 32 × 256 × 1.6 GHz = 13.1 TOPS
    # Need to boost either unit count or ops/clock to hit 32 TOPS target

    thermal_20w = ThermalOperatingPoint(
        name="20W-passive",
        tdp_watts=20.0,
        cooling_solution="passive-heatsink-automotive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_20w,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.65,  # Automotive memory (DDR5)
                efficiency_factor=0.50,  # 50% → ~6.5 TOPS effective
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=compute_resource_20w,
                efficiency_factor=0.45,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_20w,
                efficiency_factor=0.42,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 30W MODE: Active cooling (ADAS + cockpit)
    # ========================================================================
    clock_30w = ClockDomain(
        base_clock_hz=1.5e9,
        max_boost_clock_hz=2.4e9,
        sustained_clock_hz=2.0e9,   # 2.0 GHz sustained (83% of boost)
        dvfs_enabled=True,
    )

    compute_resource_30w = ComputeResource(
        resource_type="Hexagon-HMX-Matrix-Accelerator",
        num_units=num_processing_units,
        ops_per_unit_per_clock={
            Precision.INT8: 500,  # Recalibrated for 32 TOPS target
            Precision.INT4: 1000,
            Precision.INT16: 250,
            Precision.FP16: 250,
        },
        clock_domain=clock_30w,
    )

    # Sustained INT8: 32 × 500 × 2.0 GHz = 32 TOPS ✓

    thermal_30w = ThermalOperatingPoint(
        name="30W-active",
        tdp_watts=30.0,
        cooling_solution="active-fan-automotive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_30w,
                instruction_efficiency=0.90,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.58,  # 58% → ~18.5 TOPS effective
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.52,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.50,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 45W MODE: Max performance (L3 autonomous driving)
    # ========================================================================
    clock_45w = ClockDomain(
        base_clock_hz=1.8e9,
        max_boost_clock_hz=2.4e9,
        sustained_clock_hz=2.3e9,   # 2.3 GHz sustained (96% of boost)
        dvfs_enabled=True,
    )

    compute_resource_45w = ComputeResource(
        resource_type="Hexagon-HMX-Matrix-Accelerator",
        num_units=num_processing_units,
        ops_per_unit_per_clock={
            Precision.INT8: 500,
            Precision.INT4: 1000,
            Precision.INT16: 250,
            Precision.FP16: 250,
        },
        clock_domain=clock_45w,
    )

    # Peak INT8: 32 × 500 × 2.4 GHz = 38.4 TOPS (overclocked)
    # Sustained: 32 × 500 × 2.3 GHz = 36.8 TOPS

    thermal_45w = ThermalOperatingPoint(
        name="45W-max",
        tdp_watts=45.0,
        cooling_solution="active-fan-enhanced",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_45w,
                instruction_efficiency=0.92,
                memory_bottleneck_factor=0.75,
                efficiency_factor=0.65,  # 65% → ~23.9 TOPS effective
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=compute_resource_45w,
                efficiency_factor=0.60,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_45w,
                efficiency_factor=0.55,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # BOM COST PROFILE (Estimated @ 10K units, automotive-grade)
    # ========================================================================
    bom_cost = BOMCostProfile(
        silicon_die_cost=180.0,      # 5nm automotive (ASIL D) - expensive!
        package_cost=35.0,            # Advanced automotive package
        memory_cost=80.0,             # 16GB LPDDR5 (automotive-grade)
        pcb_assembly_cost=25.0,       # Automotive PCB with safety features
        thermal_solution_cost=15.0,   # Enhanced thermal for automotive
        other_costs=15.0,             # Testing, certification, safety
        total_bom_cost=0,             # Auto-calculated
        margin_multiplier=2.2,        # Lower automotive margin (B2B)
        retail_price=0,               # Auto-calculated
        volume_tier="10K+",
        process_node="5nm",
        year=2025,
        notes="Automotive-grade SoC with ASIL D certification. Higher BOM due to safety features and testing."
    )

    # BOM: $180 + $35 + $80 + $25 + $15 + $15 = $350
    # Retail: $350 × 2.2 = $770 (competitive with automotive market)

    # ========================================================================
    # Hardware Resource Model
    # ========================================================================
    return HardwareResourceModel(
        name="SA8775P-Snapdragon-Ride",
        hardware_type=HardwareType.DSP,
        compute_units=num_processing_units,
        threads_per_unit=128,  # Matrix accelerator threads
        warps_per_unit=4,
        warp_size=32,

        # Thermal operating points
        thermal_operating_points={
            "20W": thermal_20w,  # Cockpit compute
            "30W": thermal_30w,  # ADAS + cockpit
            "45W": thermal_45w,  # L3 autonomous
        },
        default_thermal_profile="30W",  # Most common ADAS deployment

        # Legacy precision profiles
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=32e12,  # 32 TOPS INT8 (estimated peak)
                tensor_core_supported=True,  # HMX matrix accelerator
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=64e12,  # 64 TOPS INT4
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=0.5,
            ),
        },
        default_precision=Precision.INT8,

        # Memory hierarchy (automotive DDR5)
        peak_bandwidth=90e9,  # ~90 GB/s (automotive LPDDR5)
        l1_cache_per_unit=128 * 1024,  # 128 KB per unit (estimated)
        l2_cache_total=8 * 1024 * 1024,  # 8 MB L2 (estimated)
        main_memory=16 * 1024**3,  # 16 GB LPDDR5 (typical automotive)

        # Energy (5nm automotive process)
        energy_per_flop_fp32=0.7e-12,  # 0.7 pJ/FLOP (5nm efficient)
        energy_per_byte=10e-12,         # 10 pJ/byte (LPDDR5)

        # Scheduling
        min_occupancy=0.5,
        max_concurrent_kernels=16,  # Automotive supports multi-task
        wave_quantization=4,

        # BOM cost
        bom_cost_profile=bom_cost,
    )

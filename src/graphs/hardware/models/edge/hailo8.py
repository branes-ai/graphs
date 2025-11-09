"""
Hailo-8 Resource Model - Computer Vision AI Accelerator

Dataflow architecture optimized for convolutional neural networks.
Target: Edge AI cameras, drones, robots, embedded vision systems.

Configuration:
- 26 TOPS INT8 (dense workloads, not sparse)
- Dataflow architecture with distributed on-chip memory
- All-on-chip design (no external DRAM)
- 16nm process
- 2.5W typical power consumption

Competitor to: KPU-T64, Google Coral Edge TPU, Qualcomm QCS6490
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


def hailo8_resource_model() -> HardwareResourceModel:
    """
    Hailo-8 Computer Vision AI Accelerator.

    ARCHITECTURE:
    - Structure-driven dataflow architecture (similar to KPU concept)
    - Distributed on-chip memory fabric (no Von Neumann bottleneck)
    - Network-specific compilation (custom dataflow per model)
    - All computations on-chip (minimizes external memory access)
    - 16nm TSMC process

    PERFORMANCE:
    - 26 TOPS INT8 (marketed, achievable)
    - Realistic: ~22 TOPS INT8 sustained (85% efficiency)
    - Excellent efficiency due to dataflow architecture
    - No DVFS throttling (low power, excellent thermal design)

    POWER PROFILE:
    - 2.5W typical (single operating point, no DVFS needed)
    - Passive cooling sufficient
    - Best power efficiency in class (10.4 TOPS/W)

    USE CASES:
    - Edge AI cameras (YOLOv5, YOLOv8 detection)
    - Drones (real-time object tracking)
    - Industrial inspection (defect detection)
    - Automotive ADAS (parking assist, lane detection)

    CALIBRATION STATUS: ✅ WELL-DOCUMENTED
    - Hailo publishes detailed benchmarks
    - 85% efficiency typical for CNNs
    - Real-world deployments confirm performance
    """
    # Physical hardware
    num_dataflow_units = 32  # Estimated dataflow processing elements
    int8_ops_per_unit_per_clock = 500  # High ops/clock for dataflow

    # ========================================================================
    # 2.5W MODE: Single operating point (no DVFS, well-designed thermal)
    # ========================================================================
    clock_2_5w = ClockDomain(
        base_clock_hz=1.6e9,        # 1.6 GHz (estimated)
        max_boost_clock_hz=1.6e9,   # No boost, constant frequency
        sustained_clock_hz=1.6e9,   # No throttling
        dvfs_enabled=False,          # Fixed frequency, excellent thermal
    )

    compute_resource_2_5w = ComputeResource(
        resource_type="Hailo-Dataflow-Architecture",
        num_units=num_dataflow_units,
        ops_per_unit_per_clock={
            Precision.INT8: 500,  # 32 units × 500 ops/clock × 1.6 GHz ≈ 25.6 TOPS
            Precision.INT4: 1000,  # 2× for INT4 (not primary use case)
        },
        clock_domain=clock_2_5w,
    )

    # Sustained INT8: 32 × 500 × 1.6 GHz = 25.6 TOPS ≈ 26 TOPS ✓

    thermal_2_5w = ThermalOperatingPoint(
        name="2.5W-passive",
        tdp_watts=2.5,
        cooling_solution="passive-heatsink-small",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_2_5w,
                instruction_efficiency=0.95,  # Dataflow is very efficient
                memory_bottleneck_factor=0.90,  # On-chip memory minimizes bottleneck
                efficiency_factor=0.85,  # 85% → ~22 TOPS effective (excellent!)
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=compute_resource_2_5w,
                efficiency_factor=0.80,  # Slightly lower for INT4
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # BOM COST PROFILE (Estimated @ 10K units)
    # ========================================================================
    bom_cost = BOMCostProfile(
        silicon_die_cost=25.0,       # 16nm die (small, efficient)
        package_cost=8.0,             # Standard BGA package
        memory_cost=0.0,              # All on-chip SRAM (no external DRAM)
        pcb_assembly_cost=4.0,        # Minimal external components
        thermal_solution_cost=1.0,    # Tiny heatsink (2.5W)
        other_costs=2.0,              # Testing, connectors
        total_bom_cost=0,             # Auto-calculated: $40
        margin_multiplier=4.0,        # High margin for specialized product
        retail_price=160.0,           # Known retail: $150-180 for M.2 module
        volume_tier="10K+",
        process_node="16nm",
        year=2025,
        notes="Ultra-efficient edge AI accelerator. Low BOM due to all-on-chip design. "
              "Highest TOPS/$ and TOPS/W in entry-level segment. M.2 module retails at $160."
    )

    # BOM: $25 + $8 + $0 + $4 + $1 + $2 = $40
    # Retail: $160 (actual market pricing)
    # Cost structure: $40 BOM, $120 margin (75% gross margin)

    # ========================================================================
    # Hardware Resource Model
    # ========================================================================
    return HardwareResourceModel(
        name="Hailo-8",
        hardware_type=HardwareType.KPU,  # Dataflow architecture (similar to KPU)
        compute_units=num_dataflow_units,
        threads_per_unit=128,  # Dataflow "threads" per unit
        warps_per_unit=1,
        warp_size=1,

        # Thermal operating points (single profile, no DVFS needed)
        thermal_operating_points={
            "2.5W": thermal_2_5w,
        },
        default_thermal_profile="2.5W",

        # Legacy precision profiles
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=26e12,  # 26 TOPS INT8 (marketed, achievable)
                tensor_core_supported=True,  # Dataflow acts like specialized hardware
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=52e12,  # 52 TOPS INT4 (theoretical)
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=0.5,
            ),
        },
        default_precision=Precision.INT8,

        # Memory hierarchy (all on-chip)
        peak_bandwidth=200e9,  # ~200 GB/s on-chip bandwidth (estimated)
        l1_cache_per_unit=512 * 1024,  # 512 KB per unit (on-chip SRAM)
        l2_cache_total=8 * 1024 * 1024,  # 8 MB total on-chip (estimated)
        main_memory=0,  # No external DRAM (uses host memory for I/O only)

        # Energy (16nm, highly optimized dataflow)
        energy_per_flop_fp32=0.5e-12,  # 0.5 pJ/FLOP (very efficient)
        energy_per_byte=2e-12,          # 2 pJ/byte (on-chip SRAM)

        # Scheduling
        min_occupancy=0.8,  # Dataflow compiler ensures high occupancy
        max_concurrent_kernels=1,  # Single model execution
        wave_quantization=1,

        # BOM cost
        bom_cost_profile=bom_cost,
    )

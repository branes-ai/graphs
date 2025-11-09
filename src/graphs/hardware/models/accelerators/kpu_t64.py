"""
Kpu T64 Resource Model hardware resource model.

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


def kpu_t64_resource_model() -> HardwareResourceModel:
    """
    Stillwater KPU-T64 with 64 heterogeneous tiles for embodied AI / edge devices.

    ============================================================================
    EDGE AI / EMBODIED AI WORKLOAD ALLOCATION (64 tiles)
    ============================================================================

    Target: Battery-powered drones, robots, edge AI devices
    - Same 70/20/10 ratio as T100/T300, scaled to 64 tiles
    - 44 INT8 tiles (69%): Computer vision, object detection
    - 13 BF16 tiles (20%): Sensor fusion, lightweight transformers
    - 7 Matrix tiles (11%): Classification heads, embeddings

    Architecture: 8×8 Checkerboard
    - 64 compute tiles arranged in 8×8 grid
    - 64 L3 memory tiles (256KB each) for distributed memory
    - Low-latency interconnect for tile-to-tile communication
    - Power-optimized for 3-6W operation

    Power Profiles for Edge AI:
    - 3W: Ultra-low power (battery-powered drones)
    - 6W: Standard edge AI (default)
    - 10W: High performance edge (max sustainable)

    Key Advantages vs Jetson Orin Nano:
    ✓ Better TOPS/W efficiency (10.6 vs 2.7 TOPS/W)
    ✓ Higher efficiency_factor (65-70% vs 4-10%)
    ✓ No DVFS throttling (well-designed thermal)
    ✓ Distributed on-chip memory (lower latency)
    """
    # Clock domain for T64 (power-optimized)
    t64_clock = ClockDomain(
        base_clock_hz=800e6,
        max_boost_clock_hz=900e6,
        sustained_clock_hz=850e6,  # 94% of boost at 3W
        dvfs_enabled=True,
    )

    # ========================================================================
    # T64 TILE ALLOCATION (64 tiles total, ~70/20/10 ratio)
    # ========================================================================
    t64_int8_tiles = TileSpecialization(
        tile_type="INT8-primary",
        num_tiles=44,  # 69% of 64
        array_dimensions=(16, 8),
        pe_configuration="INT8-MAC",
        ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
        optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
        clock_domain=t64_clock,
    )

    t64_bf16_tiles = TileSpecialization(
        tile_type="BF16-primary",
        num_tiles=13,  # 20% of 64
        array_dimensions=(16, 8),
        pe_configuration="BF16-FMA",
        ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
        optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
        clock_domain=t64_clock,
    )

    t64_matrix_tiles = TileSpecialization(
        tile_type="Matrix-8x8",
        num_tiles=7,  # 11% of 64
        array_dimensions=(8, 8),
        pe_configuration="Mixed-INT8-BF16-Matrix",
        ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
        optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
        clock_domain=t64_clock,
    )

    t64_compute = KPUComputeResource(
        total_tiles=64,
        tile_specializations=[t64_int8_tiles, t64_bf16_tiles, t64_matrix_tiles],
    )

    # ========================================================================
    # THERMAL PROFILES for T64 (Edge AI SKUs)
    # ========================================================================

    # 3W Profile: Ultra-low power (battery-powered drones, wearables)
    thermal_3w = ThermalOperatingPoint(
        name="3W-battery",
        tdp_watts=3.0,
        cooling_solution="passive-heatsink-tiny",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t64_compute,
                efficiency_factor=0.60,  # 60%! (vs Jetson Orin Nano's 4%)
                tile_utilization=0.90,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t64_compute,
                efficiency_factor=0.55,
                tile_utilization=0.85,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t64_compute,
                efficiency_factor=0.65,
                tile_utilization=0.92,
                native_acceleration=True,
            ),
        }
    )

    # 6W Profile: Standard edge AI (default)
    t64_clock_6w = ClockDomain(
        base_clock_hz=850e6,
        max_boost_clock_hz=950e6,
        sustained_clock_hz=900e6,  # 95% of boost
        dvfs_enabled=True,
    )
    t64_compute_6w = KPUComputeResource(
        total_tiles=64,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=44, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t64_clock_6w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=13, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t64_clock_6w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=7, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t64_clock_6w,
            ),
        ],
    )

    thermal_6w = ThermalOperatingPoint(
        name="6W-standard",
        tdp_watts=6.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t64_compute_6w,
                efficiency_factor=0.65,  # Excellent efficiency
                tile_utilization=0.93,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t64_compute_6w,
                efficiency_factor=0.60,
                tile_utilization=0.88,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t64_compute_6w,
                efficiency_factor=0.70,
                tile_utilization=0.95,
                native_acceleration=True,
            ),
        }
    )

    # 10W Profile: High performance edge (active cooling)
    t64_clock_10w = ClockDomain(
        base_clock_hz=900e6,
        max_boost_clock_hz=1.0e9,
        sustained_clock_hz=950e6,  # 95% of boost
        dvfs_enabled=True,
    )
    t64_compute_10w = KPUComputeResource(
        total_tiles=64,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=44, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t64_clock_10w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=13, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t64_clock_10w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=7, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t64_clock_10w,
            ),
        ],
    )

    thermal_10w = ThermalOperatingPoint(
        name="10W-performance",
        tdp_watts=10.0,
        cooling_solution="active-fan-small",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t64_compute_10w,
                efficiency_factor=0.70,  # Peak efficiency with cooling
                tile_utilization=0.95,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t64_compute_10w,
                efficiency_factor=0.65,
                tile_utilization=0.90,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t64_compute_10w,
                efficiency_factor=0.75,
                tile_utilization=0.97,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="Stillwater KPU-T64",
        hardware_type=HardwareType.KPU,
        compute_units=64,
        threads_per_unit=256,
        warps_per_unit=0,  # KPU uses tiles, not warps
        warp_size=0,

        # Thermal operating points
        thermal_operating_points={
            "3W": thermal_3w,   # Battery-powered
            "6W": thermal_6w,   # Standard edge AI
            "10W": thermal_10w,  # High performance
        },
        default_thermal_profile="6W",

        # Legacy precision profiles
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=6.9e12,  # 6.9 TOPS @ 900 MHz
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=3.3e12,  # 3.3 TFLOPS
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=8.8e12,  # 8.8 TOPS
                tensor_core_supported=True,
                relative_speedup=2.5,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT16,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=64e9,  # 64 GB/s LPDDR5
        l1_cache_per_unit=256 * 1024,  # 256 KB per tile
        l2_cache_total=4 * 1024 * 1024,  # 4 MB shared L2
        main_memory=8 * 1024**3,  # 8 GB LPDDR5
        energy_per_flop_fp32=0.10e-12,
        energy_per_byte=12e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=2,
        wave_quantization=1,

        # BOM cost profile (estimated @ 10K units, 2025)
        bom_cost_profile=BOMCostProfile(
            silicon_die_cost=75.0,        # 16nm TSMC (64 tiles, smaller die than T256)
            package_cost=15.0,             # Flip-chip BGA
            memory_cost=20.0,              # 2GB LPDDR4X on-package (16MB L3 on-die)
            pcb_assembly_cost=8.0,         # SMT assembly, passives
            thermal_solution_cost=2.0,     # Small heatsink (3-10W)
            other_costs=5.0,               # Testing, connectors
            total_bom_cost=0,              # Auto-calculated: $125
            margin_multiplier=2.4,         # Stillwater margin target
            retail_price=0,                # Auto-calculated: $300
            volume_tier="10K+",
            process_node="16nm",
            year=2025,
            notes="Entry-level KPU for edge AI. Competitive with Hailo-8 ($40-60 BOM) "
                  "but higher BOM due to integrated memory. Superior to Jetson Orin Nano ($200 BOM)."
        ),
    )



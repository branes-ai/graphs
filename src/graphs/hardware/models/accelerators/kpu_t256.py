"""
Kpu T256 Resource Model hardware resource model.

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


def kpu_t256_resource_model() -> HardwareResourceModel:
    """
    Stillwater KPU-T256 with 256 heterogeneous tiles for high-performance Embodied AI.

    ============================================================================
    HIGH-PERFORMANCE EDGE AI WORKLOAD ALLOCATION (256 tiles)
    ============================================================================

    Target: High-throughput edge servers, autonomous vehicles, drones
    - Designed for real-time multi-modal AI workloads
    - Optimized for power efficiency (15W-50W TDP range)
    - Heterogeneous tile allocation to match workload profiles
    - Tiles are 16x16 KPU compute fabrics, scaled to 256 tiles
    - Distributed L3 memory for low-latency data access
    - 70/20/10 tile ratio for INT8/BF16/Matrix workloads
    - 179 INT8 tiles (70%): Massive parallel vision processing
    - 51 BF16 tiles (20%): Multi-modal fusion, transformers
    - 26 Matrix tiles (10%): Large-scale embeddings, LLM inference

    Architecture: 16x16 Checkerboard
    - 256 compute tiles arranged in 16x16 grid
    - 256 L3 memory tiles (256KB each) for distributed memory
    - High-bandwidth interconnect (2D torus)
    - Power profiles: 15W, 30W, 50W

    Power Profiles:
    - 15W: Efficient mode (edge servers)
    - 30W: Balanced mode (datacenter inference)
    - 50W: Performance mode (max throughput)
     
    Competition: 
    - Jetson Orin AGX model: 16 SMs, 128 CUDA cores each, 1.5 GHz base clock = 16 * 192GOPS peak = 3.072 TOPS 
    - KPU T256 model: 256 tiles, 16x16 = 256 cores each, 512 ops/clock at 1.5 GHz = 256 * 768GOPS peak = 196 TOPS

    Key Advantages vs Jetson Orin AGX:
    ✓ 65x more raw performance than Jetson Orin AGX (196 TOPS vs 3.0 TOPS)
    ✓ Higher efficiency_factor (70-80% vs 5-12%)
    ✓ Better memory hierarchy (distributed L3)
    ✓ Predictable performance (no DVFS throttling)
    """
    # Clock domain for T256
    t256_clock = ClockDomain(
        base_clock_hz=1.0e9,
        max_boost_clock_hz=1.15e9,
        sustained_clock_hz=1.1e9,  # 95% of boost at 15W
        dvfs_enabled=True,
    )

    # ========================================================================
    # T256 TILE ALLOCATION (256 tiles total, 70/20/10 ratio)
    # ========================================================================
    t256_int8_tiles = TileSpecialization(
        tile_type="INT8-primary",
        num_tiles=179,  # 70% of 256
        array_dimensions=(16, 8),
        pe_configuration="INT8-MAC",
        ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
        optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
        clock_domain=t256_clock,
    )

    t256_bf16_tiles = TileSpecialization(
        tile_type="BF16-primary",
        num_tiles=51,  # 20% of 256
        array_dimensions=(16, 8),
        pe_configuration="BF16-FMA",
        ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
        optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
        clock_domain=t256_clock,
    )

    t256_matrix_tiles = TileSpecialization(
        tile_type="Matrix-8x8",
        num_tiles=26,  # 10% of 256
        array_dimensions=(8, 8),
        pe_configuration="Mixed-INT8-BF16-Matrix",
        ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
        optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
        clock_domain=t256_clock,
    )

    t256_compute = KPUComputeResource(
        total_tiles=256,
        tile_specializations=[t256_int8_tiles, t256_bf16_tiles, t256_matrix_tiles],
    )

    # ========================================================================
    # THERMAL PROFILES for T256
    # ========================================================================

    # 15W Profile: Efficient mode
    thermal_15w = ThermalOperatingPoint(
        name="15W-efficient",
        tdp_watts=15.0,
        cooling_solution="passive-heatsink-large",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t256_compute,
                efficiency_factor=0.68,
                tile_utilization=0.93,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t256_compute,
                efficiency_factor=0.63,
                tile_utilization=0.88,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t256_compute,
                efficiency_factor=0.73,
                tile_utilization=0.95,
                native_acceleration=True,
            ),
        }
    )

    # 30W Profile: Balanced mode
    t256_clock_30w = ClockDomain(
        base_clock_hz=1.350e6,
        max_boost_clock_hz=1.5e9,
        sustained_clock_hz=1.4e9,  # 95% of boost
        dvfs_enabled=True,
    )
    t256_compute_30w = KPUComputeResource(
        total_tiles=256,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=179, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t256_clock_30w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=51, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t256_clock_30w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=26, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t256_clock_30w,
            ),
        ],
    )

    thermal_30w = ThermalOperatingPoint(
        name="30W-balanced",
        tdp_watts=30.0,
        cooling_solution="active-fan",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t256_compute_30w,
                efficiency_factor=0.73,
                tile_utilization=0.95,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t256_compute_30w,
                efficiency_factor=0.68,
                tile_utilization=0.90,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t256_compute_30w,
                efficiency_factor=0.78,
                tile_utilization=0.97,
                native_acceleration=True,
            ),
        }
    )

    # 50W Profile: Performance mode
    t256_clock_50w = ClockDomain(
        base_clock_hz=1.5e9,
        max_boost_clock_hz=1.75e9,
        sustained_clock_hz=1.65e9,  # 96% of boost
        dvfs_enabled=True,
    )
    t256_compute_50w = KPUComputeResource(
        total_tiles=256,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=179, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t256_clock_50w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=51, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t256_clock_50w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=26, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t256_clock_50w,
            ),
        ],
    )

    thermal_50w = ThermalOperatingPoint(
        name="50W-performance",
        tdp_watts=50.0,
        cooling_solution="active-fan-max",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t256_compute_50w,
                efficiency_factor=0.78,  # Peak efficiency
                tile_utilization=0.97,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t256_compute_50w,
                efficiency_factor=0.73,
                tile_utilization=0.92,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t256_compute_50w,
                efficiency_factor=0.83,
                tile_utilization=0.98,
                native_acceleration=True,
            ),
        }
    )

    # BOM Cost Profile
    bom_cost = BOMCostProfile(
        silicon_die_cost=280.0,
        package_cost=45.0,
        memory_cost=90.0,
        pcb_assembly_cost=20.0,
        thermal_solution_cost=8.0,
        other_costs=12.0,
        total_bom_cost=455.0,
        margin_multiplier=2.4,
        retail_price=1092.0,
        volume_tier="10K+",
        process_node="16nm",
        year=2025,
        notes="Mid-range edge AI accelerator. 256 tiles, advanced flip-chip packaging. Competitive with high-end edge GPUs but better efficiency and predictability.",
    )

    return HardwareResourceModel(
        name="Stillwater KPU-T256",
        hardware_type=HardwareType.KPU,
        compute_units=256,
        threads_per_unit=256,
        warps_per_unit=0,
        warp_size=0,

        # Thermal operating points
        thermal_operating_points={
            "15W": thermal_15w,
            "30W": thermal_30w,
            "50W": thermal_50w,
        },
        default_thermal_profile="30W",

        # Legacy precision profiles
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=33.8e12,  # 33.8 TOPS @ 1.05 GHz
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=16.3e12,  # 16.3 TFLOPS
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=43.3e12,  # 43.3 TOPS
                tensor_core_supported=True,
                relative_speedup=2.5,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT16,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=256e9,  # 256 GB/s (4×LPDDR5 or DDR5)
        l1_cache_per_unit=256 * 1024,  # 256 KB per tile
        l2_cache_total=16 * 1024 * 1024,  # 16 MB shared L2
        main_memory=32 * 1024**3,  # 32 GB
        energy_per_flop_fp32=0.9e-12,  # Fixed: was 0.09e-12, now matches mac_energy_fp32 in tile model
        energy_per_byte=11e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=4,
        wave_quantization=2,
        bom_cost_profile=bom_cost,
    )



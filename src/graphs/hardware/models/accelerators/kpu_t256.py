"""
Kpu T256 Resource Model hardware resource model.

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
    TileScheduleClass,
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
    - Jetson Orin AGX model: 16 SMs, 128 CUDA cores each, 1.3 GHz base clock = 16 * 192GOPS peak = 3.072 TOPS 
                             64 TensorCores, 1 4x4x4 matmul per clock, 1.3 GHz base clock = 64 * 64 * 1.3Ghz peak = 5.3248 TOPS
    - KPU T256 model: 256 tiles, 16x16 = 256 cores each, 512 ops/clock at 1.5 GHz = 256 * 768GOPS peak = 196 TOPS

    Key Advantages vs Jetson Orin AGX:
    ✓ 65x more raw performance than Jetson Orin AGX (196 TOPS vs 3.0 TOPS)
    ✓ Higher efficiency_factor (70-80% vs 5-12%)
    ✓ Better memory hierarchy (distributed L3)
    ✓ Predictable performance (no DVFS throttling)
    """
    # ========================================================================
    # COMPUTE FABRICS (256 tiles, 70/20/10 allocation: INT8/BF16/Matrix)
    # ========================================================================
    # Use 30W mode as baseline (1.4 GHz sustained)
    baseline_freq_hz = 1.4e9  # 1.4 GHz sustained (30W mode)

    # M0.5 revision: T256 uses 20x20 PE arrays (from 16x16) to give it
    # meaningfully more peak compute than T128. With 16x16 x 256 tiles,
    # T256 had fewer total PEs (65,536) than T128 (73,728) - making it
    # commercially unjustifiable at its 2.5x TDP. 20x20 x 256 = 102,400
    # PEs gives T256 a genuine ~2x peak advantage (287 TOPS vs 148 TOPS).
    # Inverse-scaling story still holds (T64/T128 = 32x32 > T256 = 20x20).

    # INT8 Tile Fabric (Standard Cell - 179 tiles @ 70%)
    # 20x20 PE array = 400 PEs/tile x 2 ops/PE/cycle = 800 INT8 ops/tile/clock
    int8_fabric = ComputeFabric(
        fabric_type="kpu_int8_tile",
        circuit_type="standard_cell",
        num_units=179,  # 70% of 256 tiles
        ops_per_unit_per_clock={
            Precision.INT8: 800,
            Precision.INT4: 1600,
            Precision.BF16: 400,
        },
        core_frequency_hz=baseline_freq_hz,  # 1.4 GHz (30W)
        process_node_nm=16,        # TSMC 16nm
        energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell'),  # 2.7 pJ
        energy_scaling={
            Precision.INT8: 0.125,  # INT8 = 1/8 of FP32 energy
            Precision.INT4: 0.0625, # INT4 = 1/16 of FP32 energy
            Precision.BF16: 0.5,
        }
    )

    # BF16 Tile Fabric (Standard Cell - 51 tiles @ 20%)
    bf16_fabric = ComputeFabric(
        fabric_type="kpu_bf16_tile",
        circuit_type="standard_cell",
        num_units=51,   # 20% of 256 tiles
        ops_per_unit_per_clock={
            Precision.BF16: 400,
            Precision.FP32: 200,
            Precision.INT8: 800,   # INT8 fallback
        },
        core_frequency_hz=baseline_freq_hz,
        process_node_nm=16,
        energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell'),  # 2.7 pJ
        energy_scaling={
            Precision.BF16: 0.5,
            Precision.FP32: 1.0,
            Precision.INT8: 0.125,
        }
    )

    # Matrix Tile Fabric (26 tiles @ 10%) - uniform 20x20 array, no tensor-
    # core density claim. KPU positions on energy per op via domain-flow.
    matrix_fabric = ComputeFabric(
        fabric_type="kpu_matrix_tile",
        circuit_type="standard_cell",
        num_units=26,   # 10% of 256 tiles
        ops_per_unit_per_clock={
            Precision.INT8: 800,   # 20x20 PEs x 2 ops/PE/cycle
            Precision.BF16: 400,
        },
        core_frequency_hz=baseline_freq_hz,
        process_node_nm=16,
        energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell'),
        energy_scaling={
            Precision.INT8: 0.15,
            Precision.BF16: 0.50,
        }
    )

    # Legacy clock domain for thermal profiles
    t256_clock = ClockDomain(
        base_clock_hz=1.0e9,
        max_boost_clock_hz=1.15e9,
        sustained_clock_hz=1.1e9,  # 95% of boost at 15W
        dvfs_enabled=True,
    )

    # ========================================================================
    # T256 TILE ALLOCATION (256 tiles total, 70/20/10 ratio)
    # M0.5: homogeneous 20x20 PE array per tile; larger engine uses smaller
    # tile size to keep per-tile utilization high across many tiles.
    # Scheduling is OUTPUT_STATIONARY (distributed domain-flow fabric).
    # ========================================================================
    _T256_PE_ARRAY = (20, 20)
    t256_int8_tiles = TileSpecialization(
        tile_type="INT8-primary",
        num_tiles=179,  # 70% of 256
        array_dimensions=_T256_PE_ARRAY,
        pe_configuration="INT8-MAC",
        ops_per_tile_per_clock={Precision.INT8: 800, Precision.INT4: 1600, Precision.BF16: 400},
        optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
        clock_domain=t256_clock,
        schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
        pipeline_fill_cycles=20,   # one column sweep through a 20-wide array
        pipeline_drain_cycles=20,
    )

    t256_bf16_tiles = TileSpecialization(
        tile_type="BF16-primary",
        num_tiles=51,  # 20% of 256
        array_dimensions=_T256_PE_ARRAY,
        pe_configuration="BF16-FMA",
        ops_per_tile_per_clock={Precision.BF16: 400, Precision.FP32: 200, Precision.INT8: 800},
        optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
        clock_domain=t256_clock,
        schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
        pipeline_fill_cycles=20,
        pipeline_drain_cycles=20,
    )

    t256_matrix_tiles = TileSpecialization(
        tile_type="Matrix",
        num_tiles=26,  # 10% of 256
        array_dimensions=_T256_PE_ARRAY,
        pe_configuration="Mixed-INT8-BF16-Matrix",
        ops_per_tile_per_clock={Precision.INT8: 800, Precision.BF16: 400},
        optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
        clock_domain=t256_clock,
        schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
        pipeline_fill_cycles=20,
        pipeline_drain_cycles=20,
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
                tile_type="INT8-primary", num_tiles=179, array_dimensions=_T256_PE_ARRAY,
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 800, Precision.INT4: 1600, Precision.BF16: 400},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t256_clock_30w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=20, pipeline_drain_cycles=20,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=51, array_dimensions=_T256_PE_ARRAY,
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 400, Precision.FP32: 200, Precision.INT8: 800},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t256_clock_30w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=20, pipeline_drain_cycles=20,
            ),
            TileSpecialization(
                tile_type="Matrix", num_tiles=26, array_dimensions=_T256_PE_ARRAY,
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 800, Precision.BF16: 400},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t256_clock_30w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=20, pipeline_drain_cycles=20,
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
                tile_type="INT8-primary", num_tiles=179, array_dimensions=_T256_PE_ARRAY,
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 800, Precision.INT4: 1600, Precision.BF16: 400},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t256_clock_50w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=20, pipeline_drain_cycles=20,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=51, array_dimensions=_T256_PE_ARRAY,
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 400, Precision.FP32: 200, Precision.INT8: 800},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t256_clock_50w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=20, pipeline_drain_cycles=20,
            ),
            TileSpecialization(
                tile_type="Matrix", num_tiles=26, array_dimensions=_T256_PE_ARRAY,
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 800, Precision.BF16: 400},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t256_clock_50w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=20, pipeline_drain_cycles=20,
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

    # ========================================================================
    # Legacy Precision Profiles (calculated from fabrics for backward compatibility)
    # ========================================================================
    # Calculate peak ops using fabrics (use 30W sustained @ 1.4 GHz as baseline)
    # INT8 peak includes all tile classes that can run INT8 (native
    # INT8 tiles + BF16 tiles running INT8 fallback + matrix tiles).
    # Matches the doc claim of ~287 TOPS; omitting BF16 fallback
    # undercounts to ~230 TOPS.
    int8_peak_from_int8_tiles = int8_fabric.get_peak_ops_per_sec(Precision.INT8)
    int8_peak_from_bf16_tiles = bf16_fabric.get_peak_ops_per_sec(Precision.INT8)
    int8_peak_from_matrix_tiles = matrix_fabric.get_peak_ops_per_sec(Precision.INT8)
    total_int8_peak = (
        int8_peak_from_int8_tiles
        + int8_peak_from_bf16_tiles
        + int8_peak_from_matrix_tiles
    )

    bf16_peak_from_bf16_tiles = bf16_fabric.get_peak_ops_per_sec(Precision.BF16)
    bf16_peak_from_matrix_tiles = matrix_fabric.get_peak_ops_per_sec(Precision.BF16)
    total_bf16_peak = bf16_peak_from_bf16_tiles + bf16_peak_from_matrix_tiles

    int4_peak = int8_fabric.get_peak_ops_per_sec(Precision.INT4)
    fp32_peak = bf16_fabric.get_peak_ops_per_sec(Precision.FP32)

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

    model = HardwareResourceModel(
        name="Stillwater KPU-T256",
        hardware_type=HardwareType.KPU,

        # NEW: Compute fabrics
        compute_fabrics=[int8_fabric, bf16_fabric, matrix_fabric],

        # Legacy fields (for backward compatibility)
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

        # Legacy precision profiles (calculated from fabrics)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=total_int8_peak,  # INT8 tiles + Matrix tiles
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=total_bf16_peak,  # BF16 tiles + Matrix tiles
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=fp32_peak,  # BF16 tiles only
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=4,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=int4_peak,  # INT8 tiles only
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

        # Legacy energy (use BF16 fabric as FP32 baseline)
        energy_per_flop_fp32=bf16_fabric.energy_per_flop_fp32,  # 2.7 pJ
        energy_per_byte=11e-12,       # 11 pJ/byte (LPDDR5)
        energy_scaling={
            Precision.FP32: 1.0,
            Precision.BF16: 0.5,
            Precision.INT8: 0.125,
            Precision.INT4: 0.0625,
        },

        min_occupancy=0.3,
        max_concurrent_kernels=4,
        wave_quantization=2,
        bom_cost_profile=bom_cost,
    )

    # Add tile energy model for detailed energy analysis
    from ...architectural_energy import KPUTileEnergyModel

    tile_energy_model = KPUTileEnergyModel(
        # Product configuration (T256-specific)
        num_tiles=256,
        pes_per_tile=_T256_PE_ARRAY[0] * _T256_PE_ARRAY[1],  # 20x20 = 400
        tile_mesh_dimensions=(16, 16),  # 16x16 checkerboard

        # Memory hierarchy (4-stage)
        dram_bandwidth_gb_s=204.8,    # LPDDR5-6400 (16× channels)
        l3_size_per_tile=256 * 1024,  # 256 KiB per tile
        l2_size_per_tile=32 * 1024,   # 32 KiB per tile
        l1_size_per_pe=4 * 1024,      # 4 KiB per PE

        # Clock frequency (30W profile default)
        clock_frequency_hz=1.05e9,  # 1.05 GHz

        # Memory hierarchy energy (DDR4-based)
        dram_read_energy_per_byte=10e-12,   # 10 pJ (DDR4)
        dram_write_energy_per_byte=12e-12,  # 12 pJ (DDR4)
        l3_read_energy_per_byte=2.0e-12,    # 2.0 pJ (distributed SRAM)
        l3_write_energy_per_byte=2.5e-12,   # 2.5 pJ
        l2_read_energy_per_byte=0.8e-12,    # 0.8 pJ (tile-local SRAM)
        l2_write_energy_per_byte=1.0e-12,   # 1.0 pJ
        l1_read_energy_per_byte=0.3e-12,    # 0.3 pJ (PE-local SRAM)
        l1_write_energy_per_byte=0.4e-12,   # 0.4 pJ

        # Computation energy (BLAS operators)
        # MAC energies at 16nm optimized domain-flow (M0.5 revision).
        # Reference: 16nm 1-bit full adder ~0.01 pJ.
        mac_energy_int8=0.10e-12,  # 0.10 pJ @ 16nm optimized domain-flow
        mac_energy_bf16=0.16e-12,  # 0.16 pJ
        mac_energy_fp32=0.30e-12,  # 0.30 pJ
    )

    model.tile_energy_model = tile_energy_model

    return model



"""
Kpu T768 Resource Model hardware resource model.

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
    BOMCostProfile,
)


def kpu_t768_resource_model() -> HardwareResourceModel:
    """
    Stillwater KPU-T768 with 768 heterogeneous tiles for datacenter AI inference.

    ============================================================================
    DATACENTER AI INFERENCE WORKLOAD ALLOCATION (768 tiles)
    ============================================================================

    Target: Datacenter inference, high-throughput AI serving, LLM inference
    - Same 70/20/10 ratio as T64/T256, scaled to 768 tiles
    - 537 INT8 tiles (70%): Massive parallel vision, object detection
    - 154 BF16 tiles (20%): Transformer layers, multi-modal fusion
    - 77 Matrix tiles (10%): Large matmuls, LLM token generation

    Architecture: 32x24 Grid (768 tiles)
    - 768 compute tiles arranged in optimized grid
    - 768 L3 memory tiles (256KB each) for distributed memory
    - Ultra-high-bandwidth interconnect (2D torus + express channels)
    - Power profiles: 30W, 60W, 100W

    Power Profiles:
    - 30W: Efficient datacenter (max efficiency)
    - 60W: Balanced datacenter (default)
    - 100W: Performance mode (max throughput)

    Key Advantages:
    ✓ 3x more tiles than T256
    ✓ Datacenter-class throughput (100+ TOPS effective)
    ✓ Excellent efficiency_factor (75-85%)
    ✓ Distributed memory architecture
    ✓ Optimized for batch inference
    """
    sustained_clock_hz = 1.2e9  # 1.2 GHz sustained @ 60W

    # ========================================================================
    # Multi-Fabric Architecture (KPU Tile-Based Accelerator - Datacenter)
    # ========================================================================
    # INT8 Tile Fabric (Primary for computer vision)
    # ========================================================================
    int8_tile_fabric = ComputeFabric(
        fabric_type="kpu_int8_tile",
        circuit_type="standard_cell",   # Tile-based dataflow accelerator
        num_units=537,                   # 537 INT8 tiles (70% of 768)
        ops_per_unit_per_clock={
            Precision.INT8: 512,         # 512 INT8 ops/tile/cycle
            Precision.INT4: 1024,        # 1024 INT4 ops/tile/cycle
            Precision.BF16: 256,         # 256 BF16 ops/tile/cycle (fallback)
        },
        core_frequency_hz=sustained_clock_hz,  # 1.2 GHz
        process_node_nm=12,              # 12nm TSMC (datacenter)
        energy_per_flop_fp32=get_base_alu_energy(12, 'standard_cell'),  # 2.5 pJ
        energy_scaling={
            Precision.INT8: 0.15,        # INT8 is very efficient
            Precision.INT4: 0.08,        # INT4 even more efficient
            Precision.BF16: 0.50,        # BF16 less efficient on INT8 tiles
        }
    )

    # ========================================================================
    # BF16 Tile Fabric (Sensor fusion, lightweight transformers)
    # ========================================================================
    bf16_tile_fabric = ComputeFabric(
        fabric_type="kpu_bf16_tile",
        circuit_type="standard_cell",
        num_units=154,                   # 154 BF16 tiles (20% of 768)
        ops_per_unit_per_clock={
            Precision.BF16: 256,         # 256 BF16 ops/tile/cycle (optimized)
            Precision.FP32: 128,         # 128 FP32 ops/tile/cycle
            Precision.INT8: 512,         # 512 INT8 ops/tile/cycle (fallback)
        },
        core_frequency_hz=sustained_clock_hz,
        process_node_nm=12,
        energy_per_flop_fp32=get_base_alu_energy(12, 'standard_cell'),  # 2.5 pJ
        energy_scaling={
            Precision.BF16: 0.50,        # BF16 baseline
            Precision.FP32: 1.0,         # FP32 full energy
            Precision.INT8: 0.15,        # INT8 efficient
        }
    )

    # ========================================================================
    # Matrix Tile Fabric (Classification heads, embeddings)
    # ========================================================================
    matrix_tile_fabric = ComputeFabric(
        fabric_type="kpu_matrix_tile",
        circuit_type="tensor_core",      # Matrix tiles are tensor-optimized
        num_units=77,                    # 77 matrix tiles (10% of 768)
        ops_per_unit_per_clock={
            Precision.INT8: 8192,        # 8192 INT8 ops/tile/cycle (matrix multiply)
            Precision.BF16: 4096,        # 4096 BF16 ops/tile/cycle
        },
        core_frequency_hz=sustained_clock_hz,
        process_node_nm=12,
        energy_per_flop_fp32=get_base_alu_energy(12, 'tensor_core'),  # 2.1 pJ (15% better)
        energy_scaling={
            Precision.INT8: 0.15,
            Precision.BF16: 0.50,
        }
    )

    # INT8 tiles: 537 × 512 × 1.2 GHz = 330 TOPS INT8
    # BF16 tiles: 154 × 256 × 1.2 GHz = 47.3 TFLOPS BF16
    # Matrix tiles: 77 × 8192 × 1.2 GHz = 757 TOPS INT8
    # Total INT8 peak: 330 + 757 = 1087 TOPS INT8 (theoretical max, mixed workload)

    # ========================================================================
    # Clock Domains and Tile Specializations (Legacy for KPUComputeResource)
    # ========================================================================
    # Clock domain for T768 (datacenter-optimized)
    t768_clock = ClockDomain(
        base_clock_hz=1.0e9,
        max_boost_clock_hz=1.2e9,
        sustained_clock_hz=1.1e9,  # 92% of boost at 30W
        dvfs_enabled=True,
    )

    # ========================================================================
    # T768 TILE ALLOCATION (768 tiles total, 70/20/10 ratio)
    # ========================================================================
    t768_int8_tiles = TileSpecialization(
        tile_type="INT8-primary",
        num_tiles=537,  # 70% of 768
        array_dimensions=(16, 8),
        pe_configuration="INT8-MAC",
        ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
        optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
        clock_domain=t768_clock,
    )

    t768_bf16_tiles = TileSpecialization(
        tile_type="BF16-primary",
        num_tiles=154,  # 20% of 768
        array_dimensions=(16, 8),
        pe_configuration="BF16-FMA",
        ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
        optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
        clock_domain=t768_clock,
    )

    t768_matrix_tiles = TileSpecialization(
        tile_type="Matrix-8x8",
        num_tiles=77,  # 10% of 768
        array_dimensions=(8, 8),
        pe_configuration="Mixed-INT8-BF16-Matrix",
        ops_per_tile_per_clock={Precision.INT8: 8192, Precision.BF16: 4096},
        optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
        clock_domain=t768_clock,
    )

    t768_compute = KPUComputeResource(
        total_tiles=768,
        tile_specializations=[t768_int8_tiles, t768_bf16_tiles, t768_matrix_tiles],
    )

    # ========================================================================
    # THERMAL PROFILES for T768 (Datacenter SKUs)
    # ========================================================================

    # 30W Profile: Efficient datacenter
    thermal_30w = ThermalOperatingPoint(
        name="30W-efficient",
        tdp_watts=30.0,
        cooling_solution="active-datacenter-1U",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t768_compute,
                efficiency_factor=0.75,  # 75% efficiency
                tile_utilization=0.95,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t768_compute,
                efficiency_factor=0.70,
                tile_utilization=0.92,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t768_compute,
                efficiency_factor=0.78,
                tile_utilization=0.96,
                native_acceleration=True,
            ),
        }
    )

    # 60W Profile: Balanced datacenter (default)
    t768_clock_60w = ClockDomain(
        base_clock_hz=1.1e9,
        max_boost_clock_hz=1.3e9,
        sustained_clock_hz=1.2e9,  # 92% of boost
        dvfs_enabled=True,
    )
    t768_compute_60w = KPUComputeResource(
        total_tiles=768,
        tile_specializations=[
            TileSpecialization("INT8-primary", 537, (16, 16), "INT8-MAC",
                             {Precision.INT8: 512, Precision.INT4: 1024, Precision.BF16: 256},
                             {Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                             t768_clock_60w),
            TileSpecialization("BF16-primary", 154, (16, 16), "BF16-FMA",
                             {Precision.BF16: 256, Precision.FP32: 128, Precision.INT8: 512},
                             {Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                             t768_clock_60w),
            TileSpecialization("Matrix-8x8", 77, (8, 8), "Mixed-INT8-BF16-Matrix",
                             {Precision.INT8: 8192, Precision.BF16: 4096},
                             {Precision.INT8: 1.0, Precision.BF16: 1.0},
                             t768_clock_60w),
        ],
    )

    thermal_60w = ThermalOperatingPoint(
        name="60W-balanced",
        tdp_watts=60.0,
        cooling_solution="active-datacenter-2U",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t768_compute_60w,
                efficiency_factor=0.80,  # 80% efficiency at higher power
                tile_utilization=0.96,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t768_compute_60w,
                efficiency_factor=0.75,
                tile_utilization=0.94,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t768_compute_60w,
                efficiency_factor=0.82,
                tile_utilization=0.97,
                native_acceleration=True,
            ),
        }
    )

    # 100W Profile: Performance mode
    t768_clock_100w = ClockDomain(
        base_clock_hz=1.2e9,
        max_boost_clock_hz=1.4e9,
        sustained_clock_hz=1.35e9,  # 96% of boost
        dvfs_enabled=True,
    )
    t768_compute_100w = KPUComputeResource(
        total_tiles=768,
        tile_specializations=[
            TileSpecialization("INT8-primary", 537, (16, 8), "INT8-MAC",
                             {Precision.INT8: 512, Precision.INT4: 1024, Precision.BF16: 256},
                             {Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                             t768_clock_100w),
            TileSpecialization("BF16-primary", 154, (16, 8), "BF16-FMA",
                             {Precision.BF16: 256, Precision.FP32: 128, Precision.INT8: 512},
                             {Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                             t768_clock_100w),
            TileSpecialization("Matrix-8x8", 77, (8, 8), "Mixed-INT8-BF16-Matrix",
                             {Precision.INT8: 8192, Precision.BF16: 4096},
                             {Precision.INT8: 1.0, Precision.BF16: 1.0},
                             t768_clock_100w),
        ],
    )

    thermal_100w = ThermalOperatingPoint(
        name="100W-performance",
        tdp_watts=100.0,
        cooling_solution="active-datacenter-2U-enhanced",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t768_compute_100w,
                efficiency_factor=0.85,  # 85% efficiency at max power
                tile_utilization=0.98,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t768_compute_100w,
                efficiency_factor=0.80,
                tile_utilization=0.96,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t768_compute_100w,
                efficiency_factor=0.87,
                tile_utilization=0.99,
                native_acceleration=True,
            ),
        }
    )

    # BOM Cost Profile
    bom_cost = BOMCostProfile(
        silicon_die_cost=680.0,
        package_cost=120.0,
        memory_cost=280.0,
        pcb_assembly_cost=65.0,
        thermal_solution_cost=45.0,
        other_costs=35.0,
        total_bom_cost=1225.0,
        margin_multiplier=2.4,
        retail_price=2940.0,
        volume_tier="10K+",
        process_node="7nm",
        year=2025,
        notes="High-end Autonomous Driving AI accelerator. 768 tiles, 7nm process, multi-chip or interposer packaging. HBM2e/LPDDR5X memory. Liquid cooling capable.",
    )

    # Build resource model
    model = HardwareResourceModel(
        name="Stillwater KPU-T768",
        hardware_type=HardwareType.KPU,

        # NEW: Multi-fabric architecture (INT8 + BF16 + Matrix tiles)
        compute_fabrics=[int8_tile_fabric, bf16_tile_fabric, matrix_tile_fabric],

        compute_units=768,
        threads_per_unit=256,
        warps_per_unit=0,  # KPU uses tiles, not warps
        warp_size=0,

        thermal_operating_points={
            "30W": thermal_30w,
            "60W": thermal_60w,
            "100W": thermal_100w,
        },
        default_thermal_profile="60W",

        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=130.1e12,  # 130.1 TOPS @ 60W
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=48.9e12,  # 48.9 TFLOPS @ 60W
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=260.2e12,  # 260.2 TOPS @ 60W
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT16,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=1638e9,  # 2x819GB/s (HBM)
        l1_cache_per_unit=256 * 1024,  # 256 KB per tile
        l2_cache_total=32 * 1024 * 1024,  # 32 MB shared L2
        main_memory=64 * 1024**3,  # 64 GB
        # Energy (use BF16 tile fabric as baseline for general-purpose FP32 operations)
        energy_per_flop_fp32=bf16_tile_fabric.energy_per_flop_fp32,  # 2.5 pJ (12nm, standard cell)
        energy_per_byte=10e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=8,
        wave_quantization=2,
        bom_cost_profile=bom_cost,
    )

    # Add tile energy model for detailed energy analysis
    from ...architectural_energy import KPUTileEnergyModel

    tile_energy_model = KPUTileEnergyModel(
        # Product configuration (T768-specific)
        num_tiles=768,
        pes_per_tile=256,
        tile_mesh_dimensions=(32, 24),  # 32×24 rectangular grid (768 tiles)

        # Memory hierarchy (4-stage, datacenter optimized)
        dram_bandwidth_gb_s=1638.0,   # HBM3
        l3_size_per_tile=256 * 1024,  # 256 KiB per tile
        l2_size_per_tile=32 * 1024,   # 32 KiB per tile
        l1_size_per_pe=4 * 1024,      # 4 KiB per PE

        # Clock frequency (60W profile default)
        clock_frequency_hz=1.3e9,  # 1.3 GHz

        # Memory hierarchy energy (HBM3-based, datacenter-optimized)
        dram_read_energy_per_byte=8.0e-12,   # 8.0 pJ (HBM3, more efficient than DDR4)
        dram_write_energy_per_byte=10.0e-12,  # 10 pJ (HBM3)
        l3_read_energy_per_byte=1.2e-12,    # 1.2 pJ (distributed SRAM, 7nm)
        l3_write_energy_per_byte=1.5e-12,   # 1.5 pJ
        l2_read_energy_per_byte=0.5e-12,    # 0.5 pJ (tile-local SRAM, 7nm)
        l2_write_energy_per_byte=0.6e-12,   # 0.6 pJ
        l1_read_energy_per_byte=0.15e-12,   # 0.15 pJ (PE-local SRAM, 7nm)
        l1_write_energy_per_byte=0.2e-12,   # 0.2 pJ

        # Computation energy (BLAS operators, 7nm datacenter)
        mac_energy_int8=0.2e-12,   # 0.2 pJ (INT8, 7nm optimized)
        mac_energy_bf16=0.35e-12,  # 0.35 pJ (BF16)
        mac_energy_fp32=0.5e-12,   # 0.5 pJ (FP32)
    )

    model.tile_energy_model = tile_energy_model

    return model



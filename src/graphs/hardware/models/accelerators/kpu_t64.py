"""
Kpu T64 Resource Model hardware resource model.

Extracted from resource_model.py during refactoring.
"""

from ...fabric_model import SoCFabricModel, Topology
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

    Architecture: 8x8 Checkerboard
    - 64 compute tiles arranged in 8x8 grid
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
    sustained_clock_hz = 900e6  # 900 MHz sustained @ 6W

    # ========================================================================
    # Multi-Fabric Architecture (KPU Tile-Based Accelerator)
    # ========================================================================
    # INT8 Tile Fabric (Primary for computer vision)
    # ========================================================================
    int8_tile_fabric = ComputeFabric(
        fabric_type="kpu_int8_tile",
        circuit_type="standard_cell",   # Tile-based domain-flow accelerator
        num_units=44,                    # 44 INT8 tiles (69% of 64)
        ops_per_unit_per_clock={
            Precision.INT8: 512,         # 512 INT8 ops/tile/cycle
            Precision.INT4: 1024,        # 1024 INT4 ops/tile/cycle
            Precision.BF16: 256,         # 256 BF16 ops/tile/cycle (fallback)
        },
        core_frequency_hz=sustained_clock_hz,  # 900 MHz
        process_node_nm=16,              # 16nm TSMC
        energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell'),  # 2.7 pJ
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
        num_units=13,                    # 13 BF16 tiles (20% of 64)
        ops_per_unit_per_clock={
            Precision.BF16: 512,         # 512 BF16 ops/tile/cycle (optimized)
            Precision.FP32: 128,         # 128 FP32 ops/tile/cycle
            Precision.INT8: 512,         # 512 INT8 ops/tile/cycle (fallback)
        },
        core_frequency_hz=sustained_clock_hz,
        process_node_nm=16,
        energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell'),  # 2.7 pJ
        energy_scaling={
            Precision.BF16: 0.50,        # BF16 baseline
            Precision.FP32: 1.0,         # FP32 full energy
            Precision.INT8: 0.15,        # INT8 efficient
        }
    )

    # ========================================================================
    # Matrix Tile Fabric (Classification heads, embeddings)
    # ========================================================================
    # M0.5: matrix tile uses the SKU's uniform 32x32 PE array. No
    # tensor-core density claim; KPU positions on energy per op.
    matrix_tile_fabric = ComputeFabric(
        fabric_type="kpu_matrix_tile",
        circuit_type="standard_cell",
        num_units=7,                     # 7 matrix tiles (11% of 64)
        ops_per_unit_per_clock={
            Precision.INT8: 2048,        # 32x32 PEs x 2 ops/PE/cycle
            Precision.BF16: 1024,
        },
        core_frequency_hz=sustained_clock_hz,
        process_node_nm=16,
        energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell'),
        energy_scaling={
            Precision.INT8: 0.15,
            Precision.BF16: 0.50,
        }
    )

    # With uniform 32x32 tiles: 64 tiles x 2048 INT8 x 900 MHz ~= 118 TOPS INT8
    # T64 positioning targets competitive TOPS/W at edge TDP envelopes; the
    # larger tile exceeds the original 6W ALU envelope, so T64 is now sized
    # for a higher thermal profile. See cli/check_tdp_feasibility.py.

    # ========================================================================
    # Clock Domains and Tile Specializations (Legacy for KPUComputeResource)
    # ========================================================================
    # Clock domain for T64 (power-optimized)
    t64_clock = ClockDomain(
        base_clock_hz=800e6,
        max_boost_clock_hz=900e6,
        sustained_clock_hz=850e6,  # 94% of boost at 3W
        dvfs_enabled=True,
    )

    # ========================================================================
    # T64 TILE ALLOCATION (64 tiles total, ~70/20/10 ratio)
    # M0.5: homogeneous 32x32 PE array per tile; smaller engine uses larger
    # tile size to amortize pipeline fill/drain across the workload.
    # Scheduling is OUTPUT_STATIONARY (distributed domain-flow fabric).
    # ========================================================================
    _T64_PE_ARRAY = (32, 32)
    # 32x32 array at 2 INT8 ops/PE/clock = 2048 INT8 ops/tile/clock
    t64_int8_tiles = TileSpecialization(
        tile_type="INT8-primary",
        num_tiles=44,  # 69% of 64
        array_dimensions=_T64_PE_ARRAY,
        pe_configuration="INT8-MAC",
        ops_per_tile_per_clock={Precision.INT8: 2048, Precision.INT4: 4096, Precision.BF16: 1024},
        optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
        clock_domain=t64_clock,
        schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
        pipeline_fill_cycles=32,   # one column sweep through a 32-wide array
        pipeline_drain_cycles=32,
    )

    t64_bf16_tiles = TileSpecialization(
        tile_type="BF16-primary",
        num_tiles=13,  # 20% of 64
        array_dimensions=_T64_PE_ARRAY,
        pe_configuration="BF16-FMA",
        ops_per_tile_per_clock={Precision.BF16: 1024, Precision.FP32: 512, Precision.INT8: 2048},
        optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
        clock_domain=t64_clock,
        schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
        pipeline_fill_cycles=32,
        pipeline_drain_cycles=32,
    )

    t64_matrix_tiles = TileSpecialization(
        tile_type="Matrix",
        num_tiles=7,  # 11% of 64
        array_dimensions=_T64_PE_ARRAY,
        pe_configuration="Mixed-INT8-BF16-Matrix",
        ops_per_tile_per_clock={Precision.INT8: 2048, Precision.BF16: 1024},
        optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
        clock_domain=t64_clock,
        schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
        pipeline_fill_cycles=32,
        pipeline_drain_cycles=32,
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
                tile_type="INT8-primary", num_tiles=44, array_dimensions=_T64_PE_ARRAY,
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 2048, Precision.INT4: 4096, Precision.BF16: 1024},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t64_clock_6w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=32, pipeline_drain_cycles=32,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=13, array_dimensions=_T64_PE_ARRAY,
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 1024, Precision.FP32: 512, Precision.INT8: 2048},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t64_clock_6w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=32, pipeline_drain_cycles=32,
            ),
            TileSpecialization(
                tile_type="Matrix", num_tiles=7, array_dimensions=_T64_PE_ARRAY,
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 2048, Precision.BF16: 1024},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t64_clock_6w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=32, pipeline_drain_cycles=32,
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
                tile_type="INT8-primary", num_tiles=44, array_dimensions=_T64_PE_ARRAY,
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 2048, Precision.INT4: 4096, Precision.BF16: 1024},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t64_clock_10w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=32, pipeline_drain_cycles=32,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=13, array_dimensions=_T64_PE_ARRAY,
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 1024, Precision.FP32: 512, Precision.INT8: 2048},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t64_clock_10w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=32, pipeline_drain_cycles=32,
            ),
            TileSpecialization(
                tile_type="Matrix", num_tiles=7, array_dimensions=_T64_PE_ARRAY,
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 2048, Precision.BF16: 1024},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t64_clock_10w,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=32, pipeline_drain_cycles=32,
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

    model = HardwareResourceModel(
        name="Stillwater KPU-T64",
        hardware_type=HardwareType.KPU,

        # NEW: Multi-fabric architecture (INT8 + BF16 + Matrix tiles)
        compute_fabrics=[int8_tile_fabric, bf16_tile_fabric, matrix_tile_fabric],

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
                # 36,864 PEs * 2 ops * 900 MHz ~= 66.4 TOPS INT8 peak
                peak_ops_per_sec=66.4e12,
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=33.2e12,  # 33.2 TFLOPS BF16
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            # FP16 runs on the same BF16 tile fabric (issue #53). Without an
            # explicit entry, get_peak_ops() silently falls back to INT8 and
            # makes -p fp16 byte-identical to -p int8 in the analyzer output.
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=33.2e12,  # FP16 >= BF16 throughput on KPU
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            # FP32 is supported but at much lower throughput; matches the
            # 1.5 TFLOPS already reported by `mcp specs stillwater_kpu_t64`.
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=1.5e12,
                tensor_core_supported=True,
                relative_speedup=0.045,  # vs INT8
                bytes_per_element=4,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=132.8e12,  # 132.8 TOPS INT4
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

        # On-chip bandwidth peaks (#61). Stillwater KPU-T64 vendor
        # architecture spec, with the math consistent with the rest of
        # this module (32x32 PE array, 900 MHz sustained clock):
        #
        # Per-tile L1 (scratchpad) bandwidth: each tile contains a 32x32
        # PE mesh (1024 PEs) with 256 KB SRAM. The "feed every PE every
        # cycle" peak demand at bf16 (2 bytes/elem, 2 operands/FMA) is
        # 1024 * 2 * 2 = 4096 B/cycle. At ``sustained_clock_hz`` =
        # 900 MHz that's ~3.69 TB/s/tile of theoretical peak demand.
        #
        # However spatial dataflow re-uses operands via PE-local
        # registers and inter-PE wires rather than re-fetching from L1
        # every cycle. Steady-state L1 demand is far below the per-PE
        # operand rate -- typically ~40% of peak for output-stationary
        # schedules, since each operand is fetched once per K-loop
        # iteration and reused across the inner mesh. The figure used
        # here (1.5 TB/s/tile = 41% of 3.69 TB/s peak) captures that
        # steady-state demand. Aggregate L1 BW: ~96 TB/s across 64 tiles.
        #
        # Shared L2 bandwidth: the 4 MB shared L2 sits behind the inter-
        # tile NoC and feeds the tile mesh at the NoC's bisection BW.
        # Vendor spec for T64 is 200 GB/s aggregate L2; this is the
        # bottleneck for cross-tile data sharing in the dataflow schedule.
        l1_bandwidth_per_unit_bps=1.5e12,
        l2_bandwidth_bps=200e9,
        # Energy (use BF16 tile fabric as baseline for general-purpose FP32 operations)
        energy_per_flop_fp32=bf16_tile_fabric.energy_per_flop_fp32,  # 2.7 pJ (16nm, standard cell)
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

    # Add tile energy model for detailed energy analysis
    from ...architectural_energy import KPUTileEnergyModel

    tile_energy_model = KPUTileEnergyModel(
        # Product configuration (T64-specific)
        num_tiles=64,
        pes_per_tile=_T64_PE_ARRAY[0] * _T64_PE_ARRAY[1],  # 32x32 = 1024
        tile_mesh_dimensions=(8, 8),  # 8x8 checkerboard

        # Memory hierarchy (4-stage, edge AI optimized)
        dram_bandwidth_gb_s=64.0,     # LPDDR5 (64 GB/s)
        l3_size_per_tile=256 * 1024,  # 256 KiB per tile
        l2_size_per_tile=32 * 1024,   # 32 KiB per tile
        l1_size_per_pe=4 * 1024,      # 4 KiB per PE

        # Clock frequency (6W profile default)
        clock_frequency_hz=900e6,  # 900 MHz

        # Memory hierarchy energy (LPDDR5-based, lower than DDR4)
        dram_read_energy_per_byte=12e-12,   # 12 pJ (LPDDR5, more efficient)
        dram_write_energy_per_byte=15e-12,  # 15 pJ (LPDDR5)
        l3_read_energy_per_byte=1.5e-12,    # 1.5 pJ (distributed SRAM)
        l3_write_energy_per_byte=1.8e-12,   # 1.8 pJ
        l2_read_energy_per_byte=0.6e-12,    # 0.6 pJ (tile-local SRAM)
        l2_write_energy_per_byte=0.8e-12,   # 0.8 pJ
        l1_read_energy_per_byte=0.2e-12,    # 0.2 pJ (PE-local SRAM)
        l1_write_energy_per_byte=0.3e-12,   # 0.3 pJ

        # Computation energy (BLAS operators, edge-optimized)
        # MAC energies at 16nm optimized domain-flow silicon (M0.5 revision).
        # Reference: 16nm 1-bit full adder dynamic energy ~= 0.01 pJ.
        # INT8 MAC requires ~8 FA-equivalents + array/register overhead ->
        # 0.10 pJ/MAC is aggressive-but-defensible for a pre-scheduled
        # dataflow fabric with no instruction fetch.
        mac_energy_int8=0.10e-12,  # 0.10 pJ @ 16nm optimized domain-flow
        mac_energy_bf16=0.16e-12,  # 0.16 pJ (approx 1.6x INT8)
        mac_energy_fp32=0.30e-12,  # 0.30 pJ (approx 3x INT8)
    )

    model.tile_energy_model = tile_energy_model

    # M3 Layer 3: tile-local SRAM is the L1-equivalent under the
    # M0.5 dataflow-tile abstraction.
    model.l1_storage_kind = "scratchpad"

    # M4 Layer 4: per-tile L2 from M0.5 KPUTileEnergyModel.
    model.l2_cache_per_unit = tile_energy_model.l2_size_per_tile
    model.l2_topology = "per-unit"

    # M5 Layer 5: distributed per-tile L3 scratchpad (M0.5 abstraction).
    # Coherence stays 'none' -- compiler-managed, not a coherent cache.
    model.l3_present = True
    model.l3_cache_total = (
        tile_energy_model.l3_size_per_tile * tile_energy_model.num_tiles
    )
    model.coherence_protocol = "none"

    # M7 Layer 7: 64 GB/s LPDDR5 on-package.
    model.memory_technology = "LPDDR5"
    model.memory_read_energy_per_byte_pj = 12.0
    model.memory_write_energy_per_byte_pj = 14.4

    # M6 Layer 6: 2D mesh fabric tied to M0.5 tile abstraction.
    model.soc_fabric = SoCFabricModel(
        topology=Topology.MESH_2D,
        hop_latency_ns=1.0,
        pj_per_flit_per_hop=0.5,
        bisection_bandwidth_gbps=64.0,
        controller_count=tile_energy_model.num_tiles,
        flit_size_bytes=16,
        mesh_dimensions=tile_energy_model.tile_mesh_dimensions,
        routing_distance_factor=1.2,
        provenance=("Stillwater KPU-T64 tile mesh (M0.5 "
                    "dataflow-tile abstraction)"),
    )
    tile_energy_model.soc_fabric = model.soc_fabric

    from graphs.core.confidence import EstimationConfidence
    model.set_provenance(
        "l1_cache_per_unit",
        EstimationConfidence.theoretical(
            score=0.85,
            source=("Stillwater KPU-T64 datasheet: 256 KB tile-local "
                    "SRAM per dataflow tile (M0.5 tile abstraction)"),
        ),
    )
    model.set_provenance(
        "l1_storage_kind",
        EstimationConfidence.theoretical(
            score=0.95,
            source="KPU domain-flow architecture: tile-local SRAM, no L1 cache",
        ),
    )

    # M4 Layer 4 provenance
    model.set_provenance(
        "l2_cache_per_unit",
        EstimationConfidence.theoretical(
            score=0.85,
            source=("Stillwater KPU-T64: per-tile L2 SRAM read from "
                    "KPUTileEnergyModel.l2_size_per_tile (M0.5 abstraction)"),
        ),
    )
    model.set_provenance(
        "l2_topology",
        EstimationConfidence.theoretical(
            score=0.95,
            source="KPU domain-flow architecture: private per-tile L2",
        ),
    )

    # M5 Layer 5 provenance
    model.set_provenance(
        "l3_present",
        EstimationConfidence.theoretical(
            score=0.95,
            source=("KPU domain-flow: distributed per-tile L3 SRAM "
                    "scratchpad (M0.5 KPUTileEnergyModel)"),
        ),
    )
    model.set_provenance(
        "l3_cache_total",
        EstimationConfidence.theoretical(
            score=0.85,
            source=("Stillwater KPU-T64: per-tile L3 SRAM size x "
                    "num_tiles, read from KPUTileEnergyModel"),
        ),
    )
    model.set_provenance(
        "coherence_protocol",
        EstimationConfidence.theoretical(
            score=0.95,
            source=("KPU domain-flow: software-managed distributed L3 "
                    "scratchpad; no inter-tile coherence protocol"),
        ),
    )

    # M6 Layer 6 provenance
    model.set_provenance(
        "soc_fabric",
        EstimationConfidence.theoretical(
            score=0.85,
            source=("Stillwater KPU-T64: tile mesh tied to M0.5 "
                    "KPUTileEnergyModel"),
        ),
    )

    # M7 Layer 7 provenance
    for key in ("memory_technology",
                "memory_read_energy_per_byte_pj",
                "memory_write_energy_per_byte_pj"):
        model.set_provenance(
            key,
            EstimationConfidence.theoretical(
                score=0.80,
                source="Stillwater KPU-T64: 64 GB/s LPDDR5 on-package",
            ),
        )

    return model



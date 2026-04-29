"""
KPU T128 Resource Model - mid-range SKU at 128 tiles with 32x32 PE arrays.

Introduced in M0.5 to sit between T64 (64 tiles, 32x32 PEs) and T256
(256 tiles, 20x20 PEs). The PE-array-size / tile-count tradeoff is a
core KPU design knob: smaller engines benefit from larger per-tile PE
arrays (amortize fill/drain over more wavefront cycles), while larger
engines move toward smaller tiles to preserve per-tile utilization
across many concurrent tiles.

Future work: heterogeneous tile sizes on a single SoC (mix of 8x8
through 64x64 on one die). This file treats the SoC as homogeneous
with a single PE-array size; the ``TileSpecialization`` abstraction
already supports heterogeneity for later exploration.

The KPU is a distributed domain-flow machine capable of direct
execution of systems of affine recurrence equations. Scheduling is
``OUTPUT_STATIONARY`` across the whole fabric - the KPU's signature
advantage. See ``docs/hardware/kpu_domainflow_tile_model.md``.
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ComputeFabric,
    get_base_alu_energy,
    ClockDomain,
    TileSpecialization,
    TileScheduleClass,
    KPUComputeResource,
    PerformanceCharacteristics,
    ThermalOperatingPoint,
    BOMCostProfile,
)


def kpu_t128_resource_model() -> HardwareResourceModel:
    """
    Stillwater KPU-T128 with 128 homogeneous tiles, 32x32 PE array per tile.

    ============================================================================
    MID-RANGE EDGE AI / EMBODIED AI ALLOCATION (128 tiles)
    ============================================================================

    Target: Higher-throughput robotics, embodied AI, vision-rich edge
    - 70/20/10 role split (INT8/BF16/Matrix) preserved for comparison
      with T64 and T256; tile array size is uniform at 32x32
    - 89 INT8 tiles (69%): vision, detection, convolutional workloads
    - 26 BF16 tiles (20%): normalization, attention, sensor fusion
    - 13 Matrix tiles (11%): classification heads, embeddings

    Power Profiles for embodied AI:
    - 6W: battery-friendly embedded
    - 12W: sustained edge (default)
    - 18W: active-cooled peak

    Architectural signature:
      - Distributed domain-flow fabric, output-stationary scheduling
      - Per-PE steady-state MAC energy well below Tensor Core
      - Fill/drain overlaps across adjacent tiles -> effective pipeline
        utilization -> 1.0 at approximately 12+ workload tiles
    """
    sustained_clock_hz = 1.0e9  # 1.0 GHz sustained at 12W (default)

    _T128_PE_ARRAY = (32, 32)
    # 32x32 PE array at 2 INT8 ops/PE/clock = 2048 INT8 ops/tile/clock
    # (ratio 2048/1024 = 2 INT8 ops per PE per cycle)
    _PE_COUNT = _T128_PE_ARRAY[0] * _T128_PE_ARRAY[1]
    _INT8_OPS_PER_TILE = _PE_COUNT * 2    # 2048
    _BF16_OPS_PER_TILE = _PE_COUNT * 1    # 1024
    _INT4_OPS_PER_TILE = _PE_COUNT * 4    # 4096

    int8_tile_fabric = ComputeFabric(
        fabric_type="kpu_int8_tile",
        circuit_type="standard_cell",
        num_units=89,  # 69.5% of 128
        ops_per_unit_per_clock={
            Precision.INT8: _INT8_OPS_PER_TILE,
            Precision.INT4: _INT4_OPS_PER_TILE,
            Precision.BF16: _BF16_OPS_PER_TILE,
        },
        core_frequency_hz=sustained_clock_hz,
        process_node_nm=16,
        energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell'),
        energy_scaling={
            Precision.INT8: 0.15,
            Precision.INT4: 0.08,
            Precision.BF16: 0.50,
        },
    )

    bf16_tile_fabric = ComputeFabric(
        fabric_type="kpu_bf16_tile",
        circuit_type="standard_cell",
        num_units=26,  # 20% of 128
        ops_per_unit_per_clock={
            Precision.BF16: _BF16_OPS_PER_TILE,
            Precision.FP32: _PE_COUNT // 2,  # half-throughput at FP32
            Precision.INT8: _INT8_OPS_PER_TILE,
        },
        core_frequency_hz=sustained_clock_hz,
        process_node_nm=16,
        energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell'),
        energy_scaling={
            Precision.BF16: 0.50,
            Precision.FP32: 1.0,
            Precision.INT8: 0.15,
        },
    )

    matrix_tile_fabric = ComputeFabric(
        fabric_type="kpu_matrix_tile",
        circuit_type="standard_cell",  # domain-flow, not tensor-core emulation
        num_units=13,  # 10% of 128
        ops_per_unit_per_clock={
            Precision.INT8: _INT8_OPS_PER_TILE,
            Precision.BF16: _BF16_OPS_PER_TILE,
        },
        core_frequency_hz=sustained_clock_hz,
        process_node_nm=16,
        energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell'),
        energy_scaling={
            Precision.INT8: 0.15,
            Precision.BF16: 0.50,
        },
    )

    # ------------------------------------------------------------------
    # Clock domains + tile specializations (shared across thermal profiles
    # with only clock_domain varying).
    # ------------------------------------------------------------------
    _FILL_CYCLES = 32  # one column sweep through a 32-wide PE array
    _DRAIN_CYCLES = 32

    def _make_tiles(clock: ClockDomain):
        return [
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=89, array_dimensions=_T128_PE_ARRAY,
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={
                    Precision.INT8: _INT8_OPS_PER_TILE,
                    Precision.INT4: _INT4_OPS_PER_TILE,
                    Precision.BF16: _BF16_OPS_PER_TILE,
                },
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=clock,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=_FILL_CYCLES,
                pipeline_drain_cycles=_DRAIN_CYCLES,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=26, array_dimensions=_T128_PE_ARRAY,
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={
                    Precision.BF16: _BF16_OPS_PER_TILE,
                    Precision.FP32: _PE_COUNT // 2,
                    Precision.INT8: _INT8_OPS_PER_TILE,
                },
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=clock,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=_FILL_CYCLES,
                pipeline_drain_cycles=_DRAIN_CYCLES,
            ),
            TileSpecialization(
                tile_type="Matrix", num_tiles=13, array_dimensions=_T128_PE_ARRAY,
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={
                    Precision.INT8: _INT8_OPS_PER_TILE,
                    Precision.BF16: _BF16_OPS_PER_TILE,
                },
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=clock,
                schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
                pipeline_fill_cycles=_FILL_CYCLES,
                pipeline_drain_cycles=_DRAIN_CYCLES,
            ),
        ]

    # 6W Profile: embedded / battery-friendly
    t128_clock_6w = ClockDomain(
        base_clock_hz=700e6, max_boost_clock_hz=850e6,
        sustained_clock_hz=800e6, dvfs_enabled=True,
    )
    t128_compute_6w = KPUComputeResource(
        total_tiles=128, tile_specializations=_make_tiles(t128_clock_6w),
    )
    thermal_6w = ThermalOperatingPoint(
        name="6W-embedded", tdp_watts=6.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8, compute_resource=t128_compute_6w,
                efficiency_factor=0.63, tile_utilization=0.91, native_acceleration=True),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16, compute_resource=t128_compute_6w,
                efficiency_factor=0.58, tile_utilization=0.86, native_acceleration=True),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4, compute_resource=t128_compute_6w,
                efficiency_factor=0.68, tile_utilization=0.93, native_acceleration=True),
        }
    )

    # 12W Profile: sustained edge (default)
    t128_clock_12w = ClockDomain(
        base_clock_hz=950e6, max_boost_clock_hz=1.1e9,
        sustained_clock_hz=1.0e9, dvfs_enabled=True,
    )
    t128_compute_12w = KPUComputeResource(
        total_tiles=128, tile_specializations=_make_tiles(t128_clock_12w),
    )
    thermal_12w = ThermalOperatingPoint(
        name="12W-sustained", tdp_watts=12.0,
        cooling_solution="passive-heatsink-large",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8, compute_resource=t128_compute_12w,
                efficiency_factor=0.68, tile_utilization=0.94, native_acceleration=True),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16, compute_resource=t128_compute_12w,
                efficiency_factor=0.63, tile_utilization=0.89, native_acceleration=True),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4, compute_resource=t128_compute_12w,
                efficiency_factor=0.73, tile_utilization=0.96, native_acceleration=True),
        }
    )

    # 18W Profile: active-cooled peak
    t128_clock_18w = ClockDomain(
        base_clock_hz=1.05e9, max_boost_clock_hz=1.2e9,
        sustained_clock_hz=1.15e9, dvfs_enabled=True,
    )
    t128_compute_18w = KPUComputeResource(
        total_tiles=128, tile_specializations=_make_tiles(t128_clock_18w),
    )
    thermal_18w = ThermalOperatingPoint(
        name="18W-performance", tdp_watts=18.0,
        cooling_solution="active-fan-small",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8, compute_resource=t128_compute_18w,
                efficiency_factor=0.72, tile_utilization=0.96, native_acceleration=True),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16, compute_resource=t128_compute_18w,
                efficiency_factor=0.67, tile_utilization=0.91, native_acceleration=True),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4, compute_resource=t128_compute_18w,
                efficiency_factor=0.77, tile_utilization=0.97, native_acceleration=True),
        }
    )

    model = HardwareResourceModel(
        name="Stillwater KPU-T128",
        hardware_type=HardwareType.KPU,
        compute_fabrics=[int8_tile_fabric, bf16_tile_fabric, matrix_tile_fabric],
        compute_units=128,
        threads_per_unit=_PE_COUNT,
        warps_per_unit=0,
        warp_size=0,
        thermal_operating_points={
            "6W": thermal_6w,
            "12W": thermal_12w,
            "18W": thermal_18w,
        },
        default_thermal_profile="12W",
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                # 128 tiles x 2048 INT8 ops/tile x 1.0 GHz ~= 262 TOPS peak
                peak_ops_per_sec=128 * _INT8_OPS_PER_TILE * sustained_clock_hz,
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=128 * _BF16_OPS_PER_TILE * sustained_clock_hz,
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=128 * _INT4_OPS_PER_TILE * sustained_clock_hz,
                tensor_core_supported=True,
                relative_speedup=2.5,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT16,
            ),
        },
        default_precision=Precision.INT8,
        peak_bandwidth=96e9,  # 96 GB/s LPDDR5
        l1_cache_per_unit=256 * 1024,     # 256 KB per tile
        l2_cache_total=8 * 1024 * 1024,   # 8 MB shared L2
        main_memory=16 * 1024**3,         # 16 GB LPDDR5
        energy_per_flop_fp32=bf16_tile_fabric.energy_per_flop_fp32,
        energy_per_byte=12e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=2,
        wave_quantization=1,
        bom_cost_profile=BOMCostProfile(
            silicon_die_cost=100.0,       # 16nm TSMC, between T64 and T256
            package_cost=18.0,
            memory_cost=30.0,              # 4GB LPDDR5 on-package
            pcb_assembly_cost=10.0,
            thermal_solution_cost=3.0,
            other_costs=6.0,
            total_bom_cost=0,              # auto-calculated
            margin_multiplier=2.4,
            retail_price=0,                # auto-calculated
            volume_tier="10K+",
            process_node="16nm",
            year=2026,
            notes="Mid-range KPU for embodied AI. 32x32 PE array per tile "
                  "balances pipeline utilization against tile count.",
        ),
    )

    # Attach tile energy model (shared structure with T64 / T256;
    # numeric coefficients scale with PE count and clock).
    from ...architectural_energy import KPUTileEnergyModel

    tile_energy_model = KPUTileEnergyModel(
        num_tiles=128,
        pes_per_tile=_PE_COUNT,
        tile_mesh_dimensions=(16, 8),      # 128 tiles in a 16x8 mesh
        dram_bandwidth_gb_s=96.0,          # LPDDR5 96 GB/s
        l3_size_per_tile=256 * 1024,
        l2_size_per_tile=32 * 1024,
        l1_size_per_pe=4 * 1024,
        clock_frequency_hz=sustained_clock_hz,
        dram_read_energy_per_byte=12e-12,
        dram_write_energy_per_byte=15e-12,
        l3_read_energy_per_byte=1.5e-12,
        l3_write_energy_per_byte=1.8e-12,
        l2_read_energy_per_byte=0.6e-12,
        l2_write_energy_per_byte=0.8e-12,
        l1_read_energy_per_byte=0.2e-12,
        l1_write_energy_per_byte=0.3e-12,
        # MAC energies at 16nm optimized domain-flow (M0.5 revision).
        # Reference: 16nm 1-bit full adder ~0.01 pJ; INT8 MAC = ~8 FAs
        # + overhead -> 0.10 pJ/MAC in optimized silicon.
        mac_energy_int8=0.10e-12,   # 0.10 pJ @ 16nm optimized domain-flow
        mac_energy_bf16=0.16e-12,   # 0.16 pJ
        mac_energy_fp32=0.30e-12,   # 0.30 pJ
    )

    model.tile_energy_model = tile_energy_model

    # M3 Layer 3: tile-local SRAM is the L1-equivalent under the
    # M0.5 dataflow-tile abstraction. Software-managed (the host
    # compiler stages tiles), so hit rate is deterministic 1.0.
    model.l1_storage_kind = "scratchpad"

    # M4 Layer 4: per-tile L2 SRAM, modeled in M0.5 KPUTileEnergyModel
    # as 32 KiB. Read directly to avoid drift from the energy model.
    model.l2_cache_per_unit = tile_energy_model.l2_size_per_tile
    model.l2_topology = "per-unit"

    # M5 Layer 5: distributed per-tile L3 scratchpad. The M0.5
    # KPUTileEnergyModel models an L3 layer (256 KiB / tile, with
    # explicit l3_*_energy_per_byte charges); reflect that here so
    # the Layer 5 panel and the energy model agree. Coherence stays
    # 'none' because the L3 is software-managed scratchpad routed
    # via the mesh, not a coherent shared cache.
    model.l3_present = True
    model.l3_cache_total = (
        tile_energy_model.l3_size_per_tile * tile_energy_model.num_tiles
    )
    model.coherence_protocol = "none"

    from graphs.core.confidence import EstimationConfidence
    model.set_provenance(
        "l1_cache_per_unit",
        EstimationConfidence.theoretical(
            score=0.85,
            source=("Stillwater KPU-T128 datasheet: 256 KB tile-local "
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

    # M4 Layer 4 provenance: per-tile L2 from M0.5 tile energy model
    model.set_provenance(
        "l2_cache_per_unit",
        EstimationConfidence.theoretical(
            score=0.85,
            source=("Stillwater KPU-T128: 32 KiB per-tile L2 SRAM, "
                    "read from KPUTileEnergyModel.l2_size_per_tile "
                    "(M0.5 abstraction)"),
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
            source=("Stillwater KPU-T128: 256 KiB per-tile L3 SRAM x "
                    "128 tiles = 32 MiB distributed L3 (read from "
                    "KPUTileEnergyModel.l3_size_per_tile * num_tiles)"),
        ),
    )
    model.set_provenance(
        "coherence_protocol",
        EstimationConfidence.theoretical(
            score=0.95,
            source=("KPU domain-flow: software-managed distributed L3 "
                    "scratchpad; no inter-tile coherence protocol "
                    "(data routing is Layer 6 transport)"),
        ),
    )

    return model

"""KPU resource-model loader: YAML -> HardwareResourceModel.

Phase 4a infrastructure. Reads a ``KPUEntry`` from the
``embodied-schemas`` data catalog and constructs a
``HardwareResourceModel`` of the same shape the existing
``kpu_t{64,128,256,768}_resource_model()`` factories produce. The
mappings are documented inline so the migration path (Phase 4b --
collapsing the four hand-coded factories to thin wrappers around this
loader) is mechanical.

This loader does NOT replace the existing factories yet. It is
scaffolding that lets new code paths (mappers, validators, downstream
analyses) construct resource models from YAML without depending on the
hand-coded values, AND lets us verify YAML <-> factory equivalence
before swapping the factories over. See ``docs/designs/
kpu-sku-and-process-node-plan.md`` Stage 4b for the migration steps.

Mapping summary:

  KPUEntry field                    -> HardwareResourceModel field
  --------------------------------    ------------------------------
  name                                name
  -                                   hardware_type = HardwareType.KPU
  kpu_architecture.total_tiles        compute_units
  max(tile.pes_per_tile)              threads_per_unit
  -                                   warps_per_unit = 0
  kpu_architecture.tiles[]            compute_fabrics[] (one per class)
  power.thermal_profiles[]            thermal_operating_points (dict)
  power.default_thermal_profile       default_thermal_profile
  derived from compute_fabrics        precision_profiles
  -                                   default_precision = INT8
  memory.memory_bandwidth_gbps        peak_bandwidth (B/s)
  memory.l3_kib_per_tile              l1_cache_per_unit (per-tile L3)
  memory.l3 * total_tiles             l2_cache_total (chip-wide)
  memory.memory_size_gb               main_memory (B)
  process_node.energy_per_op_pj       energy_per_flop_fp32 (J/op)

Fields the YAML doesn't carry are left at their dataclass defaults
(BOMCostProfile, soc_fabric, tile_energy_model, set_provenance entries).
The hand-coded factories add those after calling the loader -- that's
the Phase 4b shape.
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas import (
    KPUEntry,
    KPUTileScheduleClass as YamlScheduleClass,
    load_kpus,
    load_process_nodes,
)
from embodied_schemas.process_node import CircuitClass, ProcessNodeEntry

from ...resource_model import (
    ClockDomain,
    ComputeFabric,
    HardwareResourceModel,
    HardwareType,
    KPUComputeResource,
    PerformanceCharacteristics,
    Precision,
    PrecisionProfile,
    ThermalOperatingPoint,
    TileScheduleClass,
    TileSpecialization,
    get_base_alu_energy,
)


class KPUYamlLoaderError(Exception):
    """Raised when a KPU YAML cannot be turned into a HardwareResourceModel."""


# ---------------------------------------------------------------------------
# Precision-name -> Precision enum
# ---------------------------------------------------------------------------

# YAML keys are lowercase strings; Precision enum values vary slightly.
# Translate explicitly so the loader fails loudly on unknown precisions.
_PRECISION_BY_YAML_KEY: dict[str, Precision] = {
    "fp64": Precision.FP64,
    "fp32": Precision.FP32,
    "tf32": Precision.TF32,
    "fp16": Precision.FP16,
    "bf16": Precision.BF16,
    "fp8_e4m3": Precision.FP8_E4M3,
    "fp8_e5m2": Precision.FP8_E5M2,
    "fp4": Precision.FP4,
    "int32": Precision.INT32,
    "int16": Precision.INT16,
    "int8": Precision.INT8,
    "int4": Precision.INT4,
}


def _precision_dict_from_yaml(
    ops_per_clock: dict[str, float]
) -> dict[Precision, int]:
    """Translate {'int8': 2048, ...} -> {Precision.INT8: 2048, ...}.

    Skips unknown precision names with no error -- forward-compatible
    against new precision values added to the YAML schema before the
    Precision enum is updated.
    """
    out: dict[Precision, int] = {}
    for k, v in ops_per_clock.items():
        precision = _PRECISION_BY_YAML_KEY.get(k.lower())
        if precision is None:
            continue
        out[precision] = int(v)
    return out


# ---------------------------------------------------------------------------
# YAML schedule class -> resource_model schedule class
# ---------------------------------------------------------------------------

_SCHEDULE_CLASS_MAP: dict[YamlScheduleClass, TileScheduleClass] = {
    YamlScheduleClass.OUTPUT_STATIONARY: TileScheduleClass.OUTPUT_STATIONARY,
    YamlScheduleClass.WEIGHT_STATIONARY: TileScheduleClass.WEIGHT_STATIONARY,
    YamlScheduleClass.INPUT_STATIONARY: TileScheduleClass.UNSPECIFIED,
    YamlScheduleClass.NO_LOCAL_REUSE: TileScheduleClass.UNSPECIFIED,
}


# ---------------------------------------------------------------------------
# CircuitClass -> ComputeFabric circuit_type label
# ---------------------------------------------------------------------------

_CIRCUIT_TYPE_LABEL: dict[CircuitClass, str] = {
    CircuitClass.HP_LOGIC: "standard_cell",
    CircuitClass.BALANCED_LOGIC: "standard_cell",
    CircuitClass.LP_LOGIC: "standard_cell",
    CircuitClass.ULL_LOGIC: "standard_cell",
    CircuitClass.SRAM_HD: "standard_cell",
    CircuitClass.SRAM_HC: "standard_cell",
    CircuitClass.SRAM_HP: "standard_cell",
    CircuitClass.ANALOG: "custom_datacenter",
    CircuitClass.IO: "custom_datacenter",
    CircuitClass.MIXED: "standard_cell",
}


# ---------------------------------------------------------------------------
# Energy lookups
# ---------------------------------------------------------------------------

def _fabric_energy_per_fp32_j(
    node: ProcessNodeEntry, circuit_class: CircuitClass
) -> float:
    """Energy per FP32 op in joules, looked up from process_node.

    Falls back to ``get_base_alu_energy(node_nm, circuit_type)`` (the
    legacy default-table) when the YAML doesn't list an energy for
    this (class, fp32) pair. Returning a non-zero default keeps
    ComputeFabric.__post_init__ from invoking its own fallback path
    (which would key on circuit_type strings rather than the YAML's
    finer CircuitClass).
    """
    key = f"{circuit_class.value}:fp32"
    pj = node.energy_per_op_pj.get(key)
    if pj and pj > 0:
        return pj * 1e-12
    # Fallback to the existing energy table.
    return get_base_alu_energy(node.node_nm, "standard_cell")


def _fabric_energy_scaling(
    node: ProcessNodeEntry,
    circuit_class: CircuitClass,
    precisions: list[Precision],
) -> dict[Precision, float]:
    """Per-precision scaling factors (multipliers on the FP32 base energy).

    Reads ``process_node.energy_per_op_pj["{circuit_class}:{precision}"]``
    for each precision the fabric supports, divides by the FP32 baseline,
    and stores the ratio. Falls back to the resource_model default table
    when the YAML doesn't list a value.
    """
    fp32_pj = node.energy_per_op_pj.get(f"{circuit_class.value}:fp32")
    out: dict[Precision, float] = {}
    for precision in precisions:
        # Use lowercase enum-value name to compose the key.
        key = f"{circuit_class.value}:{precision.value}"
        prec_pj = node.energy_per_op_pj.get(key)
        if prec_pj and prec_pj > 0 and fp32_pj and fp32_pj > 0:
            out[precision] = prec_pj / fp32_pj
        # Otherwise leave the default scaling (HardwareResourceModel
        # has a per-precision default table) alone.
    return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_kpu_resource_model_from_yaml(
    base_id: str,
    *,
    kpus: Optional[dict[str, KPUEntry]] = None,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
) -> HardwareResourceModel:
    """Build a ``HardwareResourceModel`` for ``base_id`` from the YAML catalog.

    Args:
        base_id: KPU SKU id, e.g., ``"stillwater_kpu_t256"``.
        kpus / process_nodes: Optional pre-loaded catalogs (tests pass
            in-memory dicts to avoid disk I/O). Defaults load from the
            installed ``embodied-schemas`` package.

    Raises:
        KPUYamlLoaderError: when ``base_id`` is not in the catalog or its
            ``process_node_id`` doesn't resolve.
    """
    if kpus is None:
        kpus = load_kpus()
    if process_nodes is None:
        process_nodes = load_process_nodes()

    sku = kpus.get(base_id)
    if sku is None:
        raise KPUYamlLoaderError(
            f"no KPU SKU with id={base_id!r}. Available: "
            f"{', '.join(sorted(kpus))}"
        )
    node = process_nodes.get(sku.process_node_id)
    if node is None:
        raise KPUYamlLoaderError(
            f"SKU {base_id!r} references process_node_id="
            f"{sku.process_node_id!r} which does not resolve"
        )

    # Default profile gives the sustained clock used for fabric core_frequency_hz
    # and for ClockDomain.sustained_clock_hz.
    default_profile = next(
        (p for p in sku.power.thermal_profiles
         if p.name == sku.power.default_thermal_profile),
        None,
    )
    if default_profile is None:
        raise KPUYamlLoaderError(
            f"SKU {base_id!r}: default_thermal_profile "
            f"{sku.power.default_thermal_profile!r} is not in "
            f"thermal_profiles"
        )
    default_clock_hz = default_profile.clock_mhz * 1e6
    boost_clock_hz = sku.clocks.boost_clock_mhz * 1e6
    base_clock_hz = sku.clocks.base_clock_mhz * 1e6

    # ------------------------------------------------------------------
    # Build ComputeFabric per tile class (peak throughput vehicle)
    # ------------------------------------------------------------------
    compute_fabrics: list[ComputeFabric] = []
    for tile in sku.kpu_architecture.tiles:
        ops_dict = _precision_dict_from_yaml(tile.ops_per_tile_per_clock)
        if not ops_dict:
            continue
        fabric = ComputeFabric(
            fabric_type=f"kpu_{tile.tile_type.lower().replace('-', '_')}",
            circuit_type=_CIRCUIT_TYPE_LABEL.get(
                tile.pe_circuit_class, "standard_cell"
            ),
            num_units=tile.num_tiles,
            ops_per_unit_per_clock=ops_dict,
            core_frequency_hz=default_clock_hz,
            process_node_nm=node.node_nm,
            energy_per_flop_fp32=_fabric_energy_per_fp32_j(
                node, tile.pe_circuit_class
            ),
            energy_scaling=_fabric_energy_scaling(
                node, tile.pe_circuit_class, list(ops_dict.keys())
            ),
        )
        compute_fabrics.append(fabric)

    if not compute_fabrics:
        raise KPUYamlLoaderError(
            f"SKU {base_id!r}: no tile class produced a valid ComputeFabric "
            f"(check ops_per_tile_per_clock entries)."
        )

    # ------------------------------------------------------------------
    # Build TileSpecialization-based KPUComputeResource. ThermalOperatingPoints
    # then reference it via per-precision PerformanceCharacteristics.
    # ------------------------------------------------------------------
    clock_domain = ClockDomain(
        base_clock_hz=base_clock_hz,
        max_boost_clock_hz=boost_clock_hz,
        sustained_clock_hz=default_clock_hz,
        dvfs_enabled=True,
    )
    tile_specializations: list[TileSpecialization] = []
    for tile in sku.kpu_architecture.tiles:
        ops_dict = _precision_dict_from_yaml(tile.ops_per_tile_per_clock)
        if not ops_dict:
            continue
        # No optimization_level information in the YAML at v1; default
        # to 1.0 for every supported precision.
        optimization_level = {p: 1.0 for p in ops_dict}
        tile_specializations.append(
            TileSpecialization(
                tile_type=tile.tile_type,
                num_tiles=tile.num_tiles,
                ops_per_tile_per_clock=ops_dict,
                optimization_level=optimization_level,
                clock_domain=clock_domain,
                array_dimensions=(tile.pe_array_rows, tile.pe_array_cols),
                pe_configuration=tile.tile_type,
                schedule_class=_SCHEDULE_CLASS_MAP.get(
                    tile.schedule_class, TileScheduleClass.UNSPECIFIED
                ),
                pipeline_fill_cycles=tile.pipeline_fill_cycles,
                pipeline_drain_cycles=tile.pipeline_drain_cycles,
            )
        )
    kpu_compute = KPUComputeResource(
        total_tiles=sku.kpu_architecture.total_tiles,
        tile_specializations=tile_specializations,
    )

    # ------------------------------------------------------------------
    # ThermalOperatingPoints from YAML thermal_profiles
    # ------------------------------------------------------------------
    # Determine which precisions any tile claims to support.
    supported_precisions: set[Precision] = set()
    for tile in sku.kpu_architecture.tiles:
        supported_precisions.update(
            _precision_dict_from_yaml(tile.ops_per_tile_per_clock).keys()
        )

    thermal_operating_points: dict[str, ThermalOperatingPoint] = {}
    for profile in sku.power.thermal_profiles:
        # Per-profile clock domain so calc_peak / calc_sustained reflect
        # the profile's actual frequency.
        profile_clock_domain = ClockDomain(
            base_clock_hz=base_clock_hz,
            max_boost_clock_hz=profile.clock_mhz * 1e6,
            sustained_clock_hz=profile.clock_mhz * 1e6,
            dvfs_enabled=True,
        )
        # Build a per-profile KPUComputeResource so calc_*_ops uses the
        # profile-specific clock.
        profile_specializations = []
        for spec in tile_specializations:
            profile_specializations.append(
                TileSpecialization(
                    tile_type=spec.tile_type,
                    num_tiles=spec.num_tiles,
                    ops_per_tile_per_clock=spec.ops_per_tile_per_clock,
                    optimization_level=spec.optimization_level,
                    clock_domain=profile_clock_domain,
                    array_dimensions=spec.array_dimensions,
                    pe_configuration=spec.pe_configuration,
                    schedule_class=spec.schedule_class,
                    pipeline_fill_cycles=spec.pipeline_fill_cycles,
                    pipeline_drain_cycles=spec.pipeline_drain_cycles,
                )
            )
        profile_compute = KPUComputeResource(
            total_tiles=sku.kpu_architecture.total_tiles,
            tile_specializations=profile_specializations,
        )

        performance_specs: dict[Precision, PerformanceCharacteristics] = {}
        for precision in supported_precisions:
            performance_specs[precision] = PerformanceCharacteristics(
                precision=precision,
                compute_resource=profile_compute,
                # v1: assume KPU achieves ~70% efficiency at the default
                # profile, ~60% at the highest, ~80% at the lowest. These
                # are placeholders -- Phase 4b will read measured
                # efficiency factors from a calibration source.
                efficiency_factor=0.70,
                tile_utilization=0.95,
                native_acceleration=True,
            )

        thermal_operating_points[profile.name] = ThermalOperatingPoint(
            name=profile.name,
            tdp_watts=profile.tdp_watts,
            cooling_solution=profile.cooling_solution_id,
            performance_specs=performance_specs,
        )

    # ------------------------------------------------------------------
    # PrecisionProfile dict -- one per supported precision, peak from fabrics
    # ------------------------------------------------------------------
    precision_profiles: dict[Precision, PrecisionProfile] = {}
    bytes_by_precision: dict[Precision, float] = {
        Precision.FP64: 8, Precision.FP32: 4, Precision.TF32: 4,
        Precision.FP16: 2, Precision.BF16: 2,
        Precision.FP8_E4M3: 1, Precision.FP8_E5M2: 1,
        Precision.FP4: 0.5,
        Precision.INT32: 4, Precision.INT16: 2,
        Precision.INT8: 1, Precision.INT4: 0.5,
    }
    for precision in supported_precisions:
        peak = sum(
            f.get_peak_ops_per_sec(precision) for f in compute_fabrics
        )
        if peak <= 0:
            continue
        precision_profiles[precision] = PrecisionProfile(
            precision=precision,
            peak_ops_per_sec=peak,
            tensor_core_supported=True,
            relative_speedup=1.0,
            bytes_per_element=bytes_by_precision.get(precision, 4),
        )

    # ------------------------------------------------------------------
    # Memory + cache fields
    # ------------------------------------------------------------------
    mem = sku.kpu_architecture.memory
    peak_bandwidth_bps = mem.memory_bandwidth_gbps * 1e9
    main_memory_bytes = int(mem.memory_size_gb * 1024**3)
    l3_per_tile_bytes = mem.l3_kib_per_tile * 1024
    l3_total_bytes = l3_per_tile_bytes * sku.kpu_architecture.total_tiles
    # The historical "l1_cache_per_unit" on KPU resource models is the
    # per-tile scratchpad (= L3 in the per-tile-L3 vocabulary). Map there
    # so existing consumers see the same field semantics.
    l1_cache_per_unit_bytes = l3_per_tile_bytes
    # "l2_cache_total" historically held the chip-wide shared SRAM
    # rolled to one number. Use total L3 in this loader for parity.
    l2_cache_total_bytes = l3_total_bytes

    # Per-PE thread count proxy: the largest tile-class PE array. Used
    # by mappers that compute parallelism budgets.
    threads_per_unit = max(
        (t.pe_array_rows * t.pe_array_cols
         for t in sku.kpu_architecture.tiles),
        default=1,
    )

    # ------------------------------------------------------------------
    # Energy roll-ups
    # ------------------------------------------------------------------
    # Use the BALANCED_LOGIC FP32 figure as the FP32 baseline; falls
    # back to the legacy table.
    energy_per_flop_fp32 = _fabric_energy_per_fp32_j(
        node, CircuitClass.BALANCED_LOGIC
    )
    # Energy per byte from the off-chip memory PHY. v1 placeholder; the
    # generator's _MEM_PHY_PJ_PER_BYTE_BY_TYPE table has the same numbers.
    _MEM_PJ_PER_BYTE = {
        "lpddr5": 10.0, "lpddr5x": 9.5, "lpddr4": 12.0, "lpddr4x": 11.0,
        "ddr5": 13.0, "hbm2": 7.0, "hbm2e": 6.5,
        "hbm3": 6.0, "hbm3e": 5.5, "gddr6": 8.0, "gddr6x": 7.5,
    }
    energy_per_byte = _MEM_PJ_PER_BYTE.get(mem.memory_type.value, 10.0) * 1e-12

    # ------------------------------------------------------------------
    # Construct the model
    # ------------------------------------------------------------------
    return HardwareResourceModel(
        name=sku.name,
        hardware_type=HardwareType.KPU,
        compute_units=sku.kpu_architecture.total_tiles,
        threads_per_unit=threads_per_unit,
        warps_per_unit=0,
        peak_bandwidth=peak_bandwidth_bps,
        l1_cache_per_unit=l1_cache_per_unit_bytes,
        l2_cache_total=l2_cache_total_bytes,
        main_memory=main_memory_bytes,
        energy_per_flop_fp32=energy_per_flop_fp32,
        energy_per_byte=energy_per_byte,
        warp_size=0,
        compute_fabrics=compute_fabrics,
        precision_profiles=precision_profiles,
        default_precision=Precision.INT8,
        thermal_operating_points=thermal_operating_points,
        default_thermal_profile=sku.power.default_thermal_profile,
        min_occupancy=0.3,
        max_concurrent_kernels=4,
        wave_quantization=2,
    )

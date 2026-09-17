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
    ComputeProduct,
    KPUTileScheduleClass as YamlScheduleClass,
    load_process_nodes,
)
from embodied_schemas.compute_product import KPUBlock
from embodied_schemas.datapath import AbsoluteEnergy, RelativeEnergy
from embodied_schemas.kpu import FixedFunctionTile, KPUTileSpec, SystolicTile
from embodied_schemas.power_domain import PowerDomainKind
from embodied_schemas.process_node import CircuitClass, ProcessNodeEntry

from ...architectural_energy import KPUTileEnergyModel
from ...compute_product_loader import load_compute_products_unified
from ...kpu_access import KPUBlockLookupError, kpu_block_of, kpu_die_of
from ...fabric_model import SoCFabricModel, Topology
from ...resource_model import (
    ClockDomain,
    ComputeFabric,
    FixedFunctionUnit,
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


def _kpu_block(cp: ComputeProduct) -> KPUBlock:
    """The product's single KPUBlock, located by kind (see
    ``graphs.hardware.kpu_access``). Raises ``KPUYamlLoaderError`` if the
    product has no KPU block or more than one."""
    try:
        return kpu_block_of(cp)
    except KPUBlockLookupError as exc:
        raise KPUYamlLoaderError(str(exc)) from exc


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
# Per-memory-type DRAM PHY energy (pJ per byte). Mirrors the table in
# silicon_math, kept here so the loader can build tile_energy_model
# without importing sku_validators (avoids a circular dep).
# ---------------------------------------------------------------------------

_DRAM_READ_PJ_PER_BYTE: dict[str, float] = {
    "lpddr4": 12.0,
    "lpddr4x": 11.0,
    "lpddr5": 10.0,
    "lpddr5x": 9.5,
    "ddr5": 13.0,
    "gddr6": 8.0,
    "gddr6x": 7.5,
    "hbm2": 7.0,
    "hbm2e": 6.5,
    "hbm3": 6.0,
    "hbm3e": 5.5,
    "unified": 10.0,
}
# Write PHY energy is typically ~20 % above read for DDR-style DRAM and
# ~15 % above read for HBM. Apply a single 1.2x heuristic; the
# difference vs measured silicon is well below the loader's other
# uncertainties and matches the existing factory convention.
_DRAM_WRITE_RATIO = 1.2


# ---------------------------------------------------------------------------
# On-chip SRAM energy (pJ per byte read / write). Node-agnostic v1
# placeholders mirroring the values the existing factories hard-code
# (which themselves are 16 nm ballpark numbers used uniformly across
# T64 / T256 / T768). When PDK SRAM characterization data lands, swap
# this for a per-node table.
# ---------------------------------------------------------------------------

_L3_READ_PJ_PER_BYTE = 2.0
_L3_WRITE_PJ_PER_BYTE = 2.5
_L2_READ_PJ_PER_BYTE = 0.8
_L2_WRITE_PJ_PER_BYTE = 1.0
_L1_READ_PJ_PER_BYTE = 0.3
_L1_WRITE_PJ_PER_BYTE = 0.4


# ---------------------------------------------------------------------------
# YAML NoC topology -> resource_model Topology enum
# ---------------------------------------------------------------------------

_NOC_TOPOLOGY_MAP: dict[str, Topology] = {
    "mesh_2d": Topology.MESH_2D,
    "torus_2d": Topology.MESH_2D,  # closest fit; no TORUS_2D enum value
    "crossbar": Topology.CROSSBAR,
    "ring": Topology.RING,
    "clos": Topology.CLOS,
    "full_mesh": Topology.FULL_MESH,
}

# Topology strings whose mapping above is *lossy* -- we approximate
# them with the closest enum value but downstream consumers should
# treat the result as low-confidence (different bisection bandwidth,
# different hop-count formula, etc.). Keep this set narrow: only
# entries where the resolved Topology genuinely misrepresents the
# underlying interconnect.
_LOSSY_TOPOLOGY_MAPPINGS: frozenset[str] = frozenset({"torus_2d"})


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


def _node_energy_per_flop_fp32_j(
    node: ProcessNodeEntry, circuit_class: CircuitClass
) -> float:
    """Energy per FP32 *FLOP* in joules, single-sourced from the process node.

    ``process_node.energy_per_op_pj`` is keyed per **MAC** -- the YAML rollup is
    "MAC + register + clock", and ``_build_tile_energy_model`` consumes the very
    same figures as per-MAC ``mac_energy_*``. The EnergyAnalyzer, however,
    multiplies by ``sg.flops`` which counts **2 FLOPs per MAC**, so the per-FLOP
    base is the per-MAC node energy divided by 2. Feeding the per-MAC value in
    un-halved double-counts dynamic compute energy by 2x (issue #81).

    Falls back to the legacy ``get_base_alu_energy`` table (already FP32-FLOP
    keyed) when the node lacks the entry -- preserves behavior for nodes whose
    ``energy_per_op_pj`` table is sparse.
    """
    key = f"{circuit_class.value}:fp32"
    pj = node.energy_per_op_pj.get(key)
    if pj and pj > 0:
        return pj * 1e-12 / 2.0  # per-MAC -> per-FLOP
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

# ---------------------------------------------------------------------------
# Tile kinds (graphs#268 C5)
# ---------------------------------------------------------------------------

def _programmable(tile) -> bool:
    """Tile kinds with PEs / cells and precision-typed ops (pe_fabric,
    systolic). Fixed-function tiles become ``fixed_function_units``."""
    return isinstance(tile, (KPUTileSpec, SystolicTile))


def _array_dims(tile) -> tuple[int, int]:
    if isinstance(tile, SystolicTile):
        return (tile.array_rows, tile.array_cols)
    return (tile.pe_array_rows, tile.pe_array_cols)


def _circuit_class(tile) -> CircuitClass:
    return tile.circuit_class if isinstance(tile, SystolicTile) else tile.pe_circuit_class


def _domain_op(cp: ComputeProduct, profile, tile):
    """The tile class's power-domain operating point in ``profile`` (its
    ``power_domain_id``, else a tile_class domain listing it), or None."""
    ops = profile.domain_operating_points or {}
    if not ops:
        return None
    domain = tile.power_domain_id
    if domain is None:
        domain = next(
            (d.domain_id for d in _kpu_block(cp).power_domains or []
             if d.kind == PowerDomainKind.TILE_CLASS and tile.tile_class_id in d.members),
            None,
        )
    return ops.get(domain) if domain is not None else None


def _class_clock_hz(cp: ComputeProduct, profile, tile) -> float:
    op = _domain_op(cp, profile, tile)
    return (op.clock_mhz if op is not None and op.clock_mhz else profile.clock_mhz) * 1e6


def _class_gated(cp: ComputeProduct, profile, tile) -> bool:
    op = _domain_op(cp, profile, tile)
    return bool(op is not None and op.gated)


def _datapath_mac_energy_pj(tile, node: ProcessNodeEntry, precision: str) -> Optional[float]:
    """Per-MAC energy the tile's datapath declares for ``precision`` (the
    MAC / FMA mode on that format), in this loader's convention: the node's
    ``energy_per_op_pj`` anchor is one MAC, so a RelativeEnergy is
    ``ratio x anchor``. None when nothing is declared or resolvable."""
    if isinstance(tile, SystolicTile):
        units = [tile.mac]
    elif isinstance(tile, KPUTileSpec) and tile.datapath is not None:
        units = tile.datapath.functional_units
    else:
        return None
    for unit in units:
        if unit.op.value not in ("mac", "fma"):
            continue
        for mode in unit.modes:
            if mode.operand_format != precision:
                continue
            e = mode.energy
            if isinstance(e, RelativeEnergy) and e.anchor in node.energy_per_op_pj:
                return e.ratio * node.energy_per_op_pj[e.anchor]
            if isinstance(e, AbsoluteEnergy):
                return e.pj  # published at its reference node; not rescaled here
    return None


def _build_tile_energy_model(
    cp: ComputeProduct,
    node: ProcessNodeEntry,
    *,
    default_clock_hz: float,
    tile=None,
) -> KPUTileEnergyModel:
    """Construct a KPUTileEnergyModel from the YAML SKU + process node.

    Mappings:

    * Architectural shape (num_tiles, mesh, L1/L2/L3 sizes,
      DRAM bandwidth, clock) -- direct read from
      ``_kpu_block(cp)``.
    * ``pes_per_tile`` -- the dominant tile class's PE count
      (``num_tiles`` x array). The energy model uses this for routing,
      L1 sizing, and per-tile compute energy; the dominant class's
      footprint is the right ballpark for those calculations.
    * ``mac_energy_int8/bf16/fp32`` -- from
      ``node.energy_per_op_pj["balanced_logic:<precision>"]``. Falls
      back to ``get_base_alu_energy(node_nm, "standard_cell")`` scaled
      by precision-typical ratios when the YAML doesn't list a value.
    * ``dram_read/write_energy_per_byte`` -- from the per-memory-type
      table above.
    * ``l1/l2/l3_read/write_energy_per_byte`` -- node-agnostic v1
      placeholders matching the existing factory values; PR (later)
      will swap for a per-process-node SRAM table when PDK data lands.
    """
    arch = _kpu_block(cp)
    mem = arch.memory

    # Dominant tile class -- the programmable one with the highest
    # num_tiles. In all hand-authored Stillwater SKUs this is INT8-primary.
    # With ``tile`` set, the model is built for that class instead
    # (graphs#268 C5 per-class models): its PE count, and its datapath /
    # library energies.
    per_class = tile is not None
    if tile is None:
        tile = max((t for t in arch.tiles if _programmable(t)), key=lambda t: t.num_tiles)
    pes_per_tile = tile.pes_per_tile

    # MAC energies from the process node's per-(class, precision) table.
    # Use BALANCED_LOGIC as the default class (the PEs of the dominant
    # tile class are typically balanced; the matrix tile uses HP_LOGIC
    # but the model only carries one set of MAC energies, so the
    # majority class wins).
    #
    # Fallback strategy: when the YAML's energy_per_op_pj table is
    # sparse for a precision, scale ``get_base_alu_energy(node_nm,
    # "standard_cell")`` by precision-typical ratios. This keeps a
    # node like TSMC N7 (low base energy) from collapsing back to a
    # node-agnostic constant; sparse PDK tables stay node-aware.
    base_alu_energy_j = get_base_alu_energy(node.node_nm, "standard_cell")
    base_alu_energy_pj = base_alu_energy_j * 1e12
    # Ratios are PE-datapath-width typicals: INT8 ~12% of FP32 (from
    # observed factory ratios across N16/N7), BF16 ~50%, FP32 = 1.0
    # (the alu energy table is FP32-keyed by convention).
    _PRECISION_FALLBACK_RATIO = {"int8": 0.12, "bf16": 0.50, "fp32": 1.0}

    library = _circuit_class(tile) if per_class else CircuitClass.BALANCED_LOGIC

    def _mac_energy_pj(precision: str) -> float:
        if per_class:
            declared = _datapath_mac_energy_pj(tile, node, precision)
            if declared is not None and declared > 0:
                return declared
        key = f"{library.value}:{precision}"
        pj = node.energy_per_op_pj.get(key)
        if pj is not None and pj > 0:
            return pj
        ratio = _PRECISION_FALLBACK_RATIO.get(precision, 1.0)
        return base_alu_energy_pj * ratio

    # DRAM PHY energy from the memory-type table (with 1.2x ratio for write).
    dram_read_pj = _DRAM_READ_PJ_PER_BYTE.get(mem.memory_type.value, 10.0)
    dram_write_pj = dram_read_pj * _DRAM_WRITE_RATIO

    return KPUTileEnergyModel(
        num_tiles=arch.total_tiles,
        pes_per_tile=pes_per_tile,
        tile_mesh_dimensions=(arch.noc.mesh_rows, arch.noc.mesh_cols),
        dram_bandwidth_gb_s=mem.memory_bandwidth_gbps,
        l3_size_per_tile=mem.l3_kib_per_tile * 1024,
        l2_size_per_tile=mem.l2_kib_per_tile * 1024,
        l1_size_per_pe=mem.l1_kib_per_tile * 1024,
        clock_frequency_hz=default_clock_hz,
        # DRAM PHY energy
        dram_read_energy_per_byte=dram_read_pj * 1e-12,
        dram_write_energy_per_byte=dram_write_pj * 1e-12,
        # On-chip SRAM energies (node-agnostic v1)
        l3_read_energy_per_byte=_L3_READ_PJ_PER_BYTE * 1e-12,
        l3_write_energy_per_byte=_L3_WRITE_PJ_PER_BYTE * 1e-12,
        l2_read_energy_per_byte=_L2_READ_PJ_PER_BYTE * 1e-12,
        l2_write_energy_per_byte=_L2_WRITE_PJ_PER_BYTE * 1e-12,
        l1_read_energy_per_byte=_L1_READ_PJ_PER_BYTE * 1e-12,
        l1_write_energy_per_byte=_L1_WRITE_PJ_PER_BYTE * 1e-12,
        # MAC energies from PDK; node-scaled fallback when YAML is sparse
        mac_energy_int8=_mac_energy_pj("int8") * 1e-12,
        mac_energy_bf16=_mac_energy_pj("bf16") * 1e-12,
        mac_energy_fp32=_mac_energy_pj("fp32") * 1e-12,
    )


def _build_soc_fabric(
    cp: ComputeProduct, node: ProcessNodeEntry
) -> SoCFabricModel:
    """Construct a SoCFabricModel from the YAML SKU's NoC declaration.

    Maps ``kpu_architecture.noc`` directly to the resource_model fabric:

    * ``topology`` -- string -> ``Topology`` enum via _NOC_TOPOLOGY_MAP.
      Unknown topology strings produce ``Topology.UNKNOWN`` with
      ``low_confidence=True`` so downstream consumers see the gap.
    * ``mesh_dimensions`` -- ``(mesh_rows, mesh_cols)``.
    * ``flit_size_bytes`` -- direct from ``noc.flit_bytes``.
    * ``bisection_bandwidth_gbps`` -- direct, with 0.0 fallback when
      the YAML doesn't declare it.
    * ``controller_count`` -- ``total_tiles`` (each tile is a NoC node).
    * ``hop_latency_ns``, ``pj_per_flit_per_hop`` -- node-agnostic v1
      placeholders matching the values the existing factories use; the
      analyzer treats them as approximations and is robust to
      modest differences.
    """
    arch = _kpu_block(cp)
    noc = arch.noc

    topology_key = noc.topology.lower()
    topology = _NOC_TOPOLOGY_MAP.get(topology_key, Topology.UNKNOWN)
    # Mark low_confidence for both UNKNOWN topologies AND lossy
    # mappings (e.g., torus_2d -> mesh_2d) so downstream consumers
    # see the approximation instead of treating it as exact.
    low_confidence = (
        topology is Topology.UNKNOWN
        or topology_key in _LOSSY_TOPOLOGY_MAPPINGS
    )

    return SoCFabricModel(
        topology=topology,
        hop_latency_ns=1.0,           # 16 nm-ish typical; see factory comments
        pj_per_flit_per_hop=0.5,      # ditto
        bisection_bandwidth_gbps=noc.bisection_bandwidth_gbps or 0.0,
        controller_count=arch.total_tiles,
        flit_size_bytes=noc.flit_bytes,
        mesh_dimensions=(noc.mesh_rows, noc.mesh_cols),
        routing_distance_factor=1.2,  # typical XY routing with detours
        low_confidence=low_confidence,
        provenance=(
            f"{cp.name} NoC from "
            f"embodied-schemas:kpus/{cp.vendor}/{cp.id}.yaml "
            f"(kpu_architecture.noc); on-chip SRAM + per-hop energy "
            f"are node-agnostic v1 placeholders"
        ),
    )


def _fixed_function_units(
    cp: ComputeProduct, node: ProcessNodeEntry, profile, process_nodes
) -> tuple[FixedFunctionUnit, ...]:
    """The KPU's fixed-function tiles as ``FixedFunctionUnit``s: work units
    per second at the class's clock in ``profile``, energy per unit
    retargeted to ``node`` (graphs#268 C5)."""
    from ...kpu_power_model import fixed_function_pj_per_unit

    units = []
    for t in _kpu_block(cp).tiles:
        if not isinstance(t, FixedFunctionTile):
            continue
        core = t.core
        pj = fixed_function_pj_per_unit(core, node, process_nodes)
        units.append(FixedFunctionUnit(
            function_id=core.function_id,
            tile_class_id=t.tile_class_id,
            num_tiles=t.num_tiles,
            work_unit=core.throughput.unit.value,
            units_per_second=core.units_per_clock * t.num_tiles * _class_clock_hz(cp, profile, t),
            energy_per_unit_j=pj * 1e-12 if pj is not None else None,
            input_bytes_per_unit=core.io.input_bytes_per_unit if core.io else 0.0,
            output_bytes_per_unit=core.io.output_bytes_per_unit if core.io else 0.0,
        ))
    return tuple(units)


def load_kpu_resource_model_from_yaml(
    base_id: str,
    *,
    kpus: Optional[dict[str, ComputeProduct]] = None,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
) -> HardwareResourceModel:
    """Build a ``HardwareResourceModel`` for ``base_id`` from the YAML catalog.

    Args:
        base_id: KPU SKU id, e.g., ``"kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"``.
        kpus / process_nodes: Optional pre-loaded catalogs (tests pass
            in-memory dicts to avoid disk I/O). Defaults load from the
            installed ``embodied-schemas`` package.

    Raises:
        KPUYamlLoaderError: when ``base_id`` is not in the catalog or its
            ``process_node_id`` doesn't resolve.
    """
    if kpus is None:
        kpus = load_compute_products_unified()
    if process_nodes is None:
        process_nodes = load_process_nodes()

    cp = kpus.get(base_id)
    if cp is None:
        raise KPUYamlLoaderError(
            f"no KPU SKU with id={base_id!r}. Available: "
            f"{', '.join(sorted(kpus))}"
        )
    try:
        kpu_die = kpu_die_of(cp)
    except KPUBlockLookupError as exc:
        raise KPUYamlLoaderError(f"SKU {base_id!r}: {exc}") from exc
    node = process_nodes.get(kpu_die.process_node_id)
    if node is None:
        raise KPUYamlLoaderError(
            f"SKU {base_id!r} references process_node_id="
            f"{kpu_die.process_node_id!r} which does not resolve"
        )

    # Default profile gives the sustained clock used for fabric core_frequency_hz
    # and for ClockDomain.sustained_clock_hz.
    default_profile = next(
        (p for p in cp.power.thermal_profiles
         if p.name == cp.power.default_thermal_profile),
        None,
    )
    if default_profile is None:
        raise KPUYamlLoaderError(
            f"SKU {base_id!r}: default_thermal_profile "
            f"{cp.power.default_thermal_profile!r} is not in "
            f"thermal_profiles"
        )
    default_clock_hz = default_profile.clock_mhz * 1e6
    boost_clock_hz = kpu_die.clocks.boost_clock_mhz * 1e6
    base_clock_hz = kpu_die.clocks.base_clock_mhz * 1e6

    # ------------------------------------------------------------------
    # Build ComputeFabric per tile class (peak throughput vehicle)
    # ------------------------------------------------------------------
    # One per programmable tile class, keyed by tile_class_id (for every
    # catalog SKU identical to the old tile_type-derived name). Fixed-
    # function tiles have no precision-typed ops; they become
    # ``fixed_function_units`` below.
    compute_fabrics: list[ComputeFabric] = []
    for tile in _kpu_block(cp).tiles:
        if not _programmable(tile):
            continue
        ops_dict = _precision_dict_from_yaml(tile.ops_per_tile_per_clock)
        if not ops_dict:
            continue
        cc = _circuit_class(tile)
        fabric = ComputeFabric(
            fabric_type=f"kpu_{tile.tile_class_id}",
            circuit_type=_CIRCUIT_TYPE_LABEL.get(cc, "standard_cell"),
            num_units=tile.num_tiles,
            ops_per_unit_per_clock=ops_dict,
            core_frequency_hz=_class_clock_hz(cp, default_profile, tile),
            process_node_nm=node.node_nm,
            energy_per_flop_fp32=_fabric_energy_per_fp32_j(node, cc),
            energy_scaling=_fabric_energy_scaling(node, cc, list(ops_dict.keys())),
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
    specialized_tiles = []  # the YAML tile behind each specialization
    for tile in _kpu_block(cp).tiles:
        if not _programmable(tile):
            continue
        ops_dict = _precision_dict_from_yaml(tile.ops_per_tile_per_clock)
        if not ops_dict:
            continue
        # No optimization_level information in the YAML at v1; default
        # to 1.0 for every supported precision.
        optimization_level = {p: 1.0 for p in ops_dict}
        # The class's own clock when its power domain sets one (graphs#268
        # C5); the chip clock domain otherwise (every catalog SKU).
        class_hz = _class_clock_hz(cp, default_profile, tile)
        tile_clock = clock_domain if class_hz == default_clock_hz else ClockDomain(
            base_clock_hz=base_clock_hz,
            max_boost_clock_hz=boost_clock_hz,
            sustained_clock_hz=class_hz,
            dvfs_enabled=True,
        )
        specialized_tiles.append(tile)
        tile_specializations.append(
            TileSpecialization(
                tile_type=tile.tile_type,
                num_tiles=tile.num_tiles,
                ops_per_tile_per_clock=ops_dict,
                optimization_level=optimization_level,
                clock_domain=tile_clock,
                array_dimensions=_array_dims(tile),
                pe_configuration=tile.tile_type,
                schedule_class=_SCHEDULE_CLASS_MAP.get(
                    tile.dataflow if isinstance(tile, SystolicTile) else tile.schedule_class,
                    TileScheduleClass.UNSPECIFIED,
                ),
                pipeline_fill_cycles=tile.pipeline_fill_cycles,
                pipeline_drain_cycles=tile.pipeline_drain_cycles,
            )
        )
    # Each ThermalOperatingPoint builds its own profile-specific
    # KPUComputeResource below (with the profile clock baked in), so
    # there's no chip-level KPUComputeResource to retain here.

    # ------------------------------------------------------------------
    # ThermalOperatingPoints from YAML thermal_profiles
    # ------------------------------------------------------------------
    # Determine which precisions any tile claims to support.
    supported_precisions: set[Precision] = set()
    for tile in _kpu_block(cp).tiles:
        if not _programmable(tile):
            continue
        supported_precisions.update(
            _precision_dict_from_yaml(tile.ops_per_tile_per_clock).keys()
        )

    thermal_operating_points: dict[str, ThermalOperatingPoint] = {}
    for profile in cp.power.thermal_profiles:
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
        for spec, tile in zip(tile_specializations, specialized_tiles):
            # A gated power domain takes its classes out of this profile;
            # a domain clock replaces the profile clock (graphs#268 C5).
            if _class_gated(cp, profile, tile):
                continue
            class_hz = _class_clock_hz(cp, profile, tile)
            spec_clock = profile_clock_domain if class_hz == profile.clock_mhz * 1e6 else (
                ClockDomain(
                    base_clock_hz=base_clock_hz,
                    max_boost_clock_hz=class_hz,
                    sustained_clock_hz=class_hz,
                    dvfs_enabled=True,
                )
            )
            profile_specializations.append(
                TileSpecialization(
                    tile_type=spec.tile_type,
                    num_tiles=spec.num_tiles,
                    ops_per_tile_per_clock=spec.ops_per_tile_per_clock,
                    optimization_level=spec.optimization_level,
                    clock_domain=spec_clock,
                    array_dimensions=spec.array_dimensions,
                    pe_configuration=spec.pe_configuration,
                    schedule_class=spec.schedule_class,
                    pipeline_fill_cycles=spec.pipeline_fill_cycles,
                    pipeline_drain_cycles=spec.pipeline_drain_cycles,
                )
            )
        profile_compute = KPUComputeResource(
            total_tiles=_kpu_block(cp).total_tiles,
            tile_specializations=profile_specializations,
        )

        # Per-(profile, precision) calibration data lives on the YAML's
        # KPUThermalProfile (Phase 4b PR 3, embodied-schemas#10). When
        # absent, fall back to the historical placeholders -- the field
        # is Optional precisely so external user YAMLs don't need to
        # backfill for the loader to function.
        eff_by_prec = profile.efficiency_factor_by_precision or {}
        util_by_prec = profile.tile_utilization_by_precision or {}

        # Only the precisions this profile can actually run: a gated class
        # takes its precisions with it (graphs#268 C5).
        profile_precisions = {
            p for spec in profile_specializations for p in spec.ops_per_tile_per_clock
        }
        performance_specs: dict[Precision, PerformanceCharacteristics] = {}
        for precision in profile_precisions:
            performance_specs[precision] = PerformanceCharacteristics(
                precision=precision,
                compute_resource=profile_compute,
                efficiency_factor=eff_by_prec.get(precision.value, 0.70),
                tile_utilization=util_by_prec.get(precision.value, 0.95),
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
        # Round to 0.1 TOPS / TFLOPS so the loader matches the
        # cleanly-published numbers downstream consumers (and the
        # hand-coded factory literals) use. The raw fabric x clock
        # arithmetic produces 117.9648e12 for T64 INT8; published
        # value is 118.0 TOPS. Rounding to 1 decimal place in the
        # T/G unit matches both the YAML's performance.* roll-ups
        # and the legacy factory literals to within 0.1%.
        peak_rounded = round(peak / 1e12, 1) * 1e12
        precision_profiles[precision] = PrecisionProfile(
            precision=precision,
            peak_ops_per_sec=peak_rounded,
            tensor_core_supported=True,
            relative_speedup=1.0,
            bytes_per_element=bytes_by_precision.get(precision, 4),
        )

    # ------------------------------------------------------------------
    # Memory + cache fields
    # ------------------------------------------------------------------
    mem = _kpu_block(cp).memory
    peak_bandwidth_bps = mem.memory_bandwidth_gbps * 1e9
    main_memory_bytes = int(mem.memory_size_gb * 1024**3)
    l3_per_tile_bytes = mem.l3_kib_per_tile * 1024
    # The historical "l1_cache_per_unit" on KPU resource models is the
    # per-tile scratchpad (= L3 in the per-tile-L3 vocabulary). Map there
    # so existing consumers see the same field semantics.
    l1_cache_per_unit_bytes = l3_per_tile_bytes
    # "l2_cache_total" on the resource_model is read by KPUMapper as
    # ADDITIVE to ``num_tiles * l1_cache_per_unit`` to compute total
    # on-chip capacity, so it must NOT also include the per-tile L3
    # (that would double-count). The hand-coded factories pre-Phase-4b
    # used per-SKU literal values matching this convention; the loader
    # mirrors them here so PR 5 (factory collapse) is purely mechanical.
    # Architectural note: M0.5 has no chip-wide L2 above per-tile L3;
    # these values are legacy-architectural placeholders. Eventual
    # cleanup tracked in docs/designs/kpu-test-contract-snapshot.md.
    _LEGACY_L2_CACHE_TOTAL_MB = {
        "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp":  4,
        "kpu_t128_32x32_lp5x8_16nm_tsmc_ffp": 8,
        "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp": 16,
        "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc": 32,
    }
    l2_cache_total_bytes = (
        _LEGACY_L2_CACHE_TOTAL_MB.get(base_id, 0) * 1024 * 1024
    )

    # Per-PE thread count proxy: the largest tile-class PE array. Used
    # by mappers that compute parallelism budgets.
    threads_per_unit = max(
        (t.pes_per_tile for t in _kpu_block(cp).tiles if _programmable(t)),
        default=1,
    )

    # ------------------------------------------------------------------
    # Energy roll-ups
    # ------------------------------------------------------------------
    # FP32 per-FLOP baseline, single-sourced from the process node's per-MAC
    # energy_per_op_pj (halved to per-FLOP; see _node_energy_per_flop_fp32_j).
    # The model-level energy_scaling ratios are applied after construction.
    energy_per_flop_fp32 = _node_energy_per_flop_fp32_j(
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
    # KPUTileEnergyModel + SoCFabricModel (Phase 4b PR 2)
    # ------------------------------------------------------------------
    tile_energy_model = _build_tile_energy_model(
        cp, node, default_clock_hz=default_clock_hz
    )
    soc_fabric = _build_soc_fabric(cp, node)

    # ------------------------------------------------------------------
    # Construct the model
    # ------------------------------------------------------------------
    model = HardwareResourceModel(
        name=cp.name,
        hardware_type=HardwareType.KPU,
        compute_units=_kpu_block(cp).total_tiles,
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
        default_thermal_profile=cp.power.default_thermal_profile,
        min_occupancy=0.3,
        max_concurrent_kernels=4,
        wave_quantization=2,
    )

    # tile_energy_model and soc_fabric aren't constructor args on
    # HardwareResourceModel -- the existing factories attach them as
    # post-construction attributes. Match that convention so downstream
    # consumers (analyzers, the KPUMapper energy path) see the same shape.
    model.tile_energy_model = tile_energy_model
    model.soc_fabric = soc_fabric

    # Per tile class (graphs#268 C5): an energy model per programmable
    # class (``tile_energy_model`` stays the dominant class's, as before),
    # and the fixed-function units.
    model.tile_energy_models = {
        t.tile_class_id: _build_tile_energy_model(
            cp, node, default_clock_hz=_class_clock_hz(cp, default_profile, t), tile=t
        )
        for t in _kpu_block(cp).tiles
        if _programmable(t)
    }
    # TileSpecialization carries the human label; reports resolve it to the
    # class id to pick that class's energy model.
    model.tile_class_by_tile_type = {
        t.tile_type: t.tile_class_id for t in _kpu_block(cp).tiles if _programmable(t)
    }
    # The mapper builds its capability-aware tile pools from the kind
    # (graphs#268 C5b): a GEMM prefers the systolic classes, and a uniform
    # pe_fabric chip keeps the legacy flat pool.
    model.tile_kind_by_tile_type = {
        t.tile_type: t.tile_kind.value for t in _kpu_block(cp).tiles if _programmable(t)
    }
    model.fixed_function_units = _fixed_function_units(cp, node, default_profile, process_nodes)

    # Single-source the per-precision energy_scaling from the process node's
    # energy_per_op_pj ratios (issue #81), overriding the hardcoded dataclass
    # defaults. Ratios are unit-independent (prec_pj / fp32_pj), so they apply
    # cleanly to the per-FLOP base above. Precisions the node doesn't list keep
    # their default ratio. The EnergyAnalyzer reads model.energy_scaling.
    model.energy_scaling.update(
        _fabric_energy_scaling(
            node, CircuitClass.BALANCED_LOGIC, list(precision_profiles.keys())
        )
    )

    # M3-M7 layer attributes -- KPU-architecture constants (not in YAML)
    # plus values derivable from kpu_architecture.memory. Matches what
    # the hand-coded factories were setting post-construction; pushed
    # into the loader as part of Phase 4b PR 5 so the factory wrappers
    # don't need to repeat them.
    model.l1_storage_kind = "scratchpad"
    model.l2_cache_per_unit = mem.l2_kib_per_tile * 1024
    model.l2_topology = "per-unit"
    model.l3_present = True
    model.l3_cache_total = (
        mem.l3_kib_per_tile * 1024 * _kpu_block(cp).total_tiles
    )
    model.coherence_protocol = "none"

    # M7 Layer 7 -- DRAM PHY. Memory technology + per-byte read/write
    # energy in pJ. Reads from the per-memory-type table mirroring the
    # generator's _MEM_PHY_PJ_PER_BYTE_BY_TYPE.
    model.memory_technology = mem.memory_type.value.upper()
    model.memory_read_energy_per_byte_pj = _DRAM_READ_PJ_PER_BYTE.get(
        mem.memory_type.value, 10.0
    )
    model.memory_write_energy_per_byte_pj = (
        model.memory_read_energy_per_byte_pj * _DRAM_WRITE_RATIO
    )

    # Provenance for the M3-M7 attributes set above. Source citations
    # tie back to the YAML file path and the M0.5 KPU dataflow-tile
    # abstraction that drives these conventions.
    yaml_source = (
        f"embodied-schemas:kpus/{cp.vendor}/{cp.id}.yaml"
    )
    from graphs.core.confidence import EstimationConfidence
    _PROVENANCE = EstimationConfidence.theoretical(
        score=0.85,
        source=(
            f"{cp.name} M0.5 dataflow-tile abstraction; derived from {yaml_source}"
        ),
    )
    for key in (
        "l1_cache_per_unit", "l1_storage_kind",
        "l2_cache_per_unit", "l2_topology",
        "l3_present", "l3_cache_total", "coherence_protocol",
        "soc_fabric",
        "memory_technology",
        "memory_read_energy_per_byte_pj",
        "memory_write_energy_per_byte_pj",
    ):
        model.set_provenance(key, _PROVENANCE)

    return model

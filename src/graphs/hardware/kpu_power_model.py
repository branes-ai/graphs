"""KPU thermal-design-power (TDP) derivation.

Given a SKU spec, a thermal profile (clock + cooling), a process node, and
a workload assumption, derive the sustained worst-case power dissipation.
The architect chooses clocks and cooling; TDP is the consequence.

The power model is a roll-up of five terms:

  1. PE compute power      = peak_compute_w x utilization x duty_cycle
  2. L2 SRAM access        = (1 - L1_hit) x byte_rate x sram_pj_per_byte
  3. L3 SRAM access        = (1 - L2_hit) x L2_byte_rate x sram_pj_per_byte
  4. NoC traversal         = noc_flit_rate x avg_hops x noc_pj_per_flit
  5. DRAM PHY              = (1 - L3_hit) x L3_byte_rate x dram_pj_per_byte
  + chip leakage           (already on the SKU)

PE compute power uses ``ProcessNode.energy_per_op_pj`` (already in the
schema) and assumes L1 access energy is rolled into it. Memory and NoC
terms use the new ProcessNode fields ``sram_access_pj_per_byte``,
``dram_io_pj_per_byte``, ``noc_pj_per_flit_per_hop``.

Workload assumption: the architect sizes the chip for a target workload,
not for sustained worst-case GEMM. The default ``WorkloadAssumption``
captures DNN inference (well-tiled, ~15% compute duty cycle, modest
cache miss rates). Architects can override per spec or per profile.

The TDP picks the worst-case precision the chip supports -- if the chip
runs FP32 at higher power than INT8, FP32 sets TDP. Activity factor is
applied uniformly across precisions (the workload duty cycle is a chip
property, not a precision-specific one).

Heterogeneous KPUs (graphs#268 Phase C2): a profile whose chip has a
systolic or fixed-function tile, a ``tdp_scenario`` or per-domain operating
points goes through ``compute_heterogeneous_tdp_breakdown``, which adds:

* **Per-kind compute energy.** pe_fabric / systolic ops are charged from the
  datapath ``EnergyRef`` when declared (node anchor otherwise). Fixed-
  function tiles are charged ``units/s x pj_per_unit`` scaled from their
  reference node by the logic / SRAM energy ratios.
* **Per power domain V/f and gating.** A tile class runs at its domain's
  clock / Vdd / activity; a gated domain draws neither dynamic power nor
  leakage. Memory / NoC terms use the uncore domain's Vdd.
* **``tdp_scenario``.** Each tile class runs at its declared activity, in
  its own worst-power mode, instead of one chip-wide precision and duty
  cycle.

Every legacy-shaped profile (pe_fabric tiles only, no scenario, no domain
operating points) keeps the original formula exactly (KPU golden gate);
the heterogeneous engine reproduces it for those SKUs as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from embodied_schemas import (
    ComputeProduct,
    Die,
    DieRole,
    KPUBlock,
    Market,
    Packaging,
    PackagingKind,
    Power,
    ProductKind,
)
from embodied_schemas.datapath import (
    LEGACY_PRECISION_OPS,
    AbsoluteEnergy,
    FunctionalUnit,
    RelativeEnergy,
)
from embodied_schemas.kpu import (
    FixedFunctionTile,
    KPUTheoreticalPerformance,
    KPUThermalProfile,
    KPUTileSpec,
    SystolicTile,
)
from embodied_schemas.power_domain import PowerDomainKind
from embodied_schemas.process_node import CircuitClass, ProcessNodeEntry

from .kpu_sku_input import KPUSKUInputSpec
from .sku_validators import silicon_math as sm
from .sku_validators.silicon_math import total_chip_leakage_w


@dataclass(frozen=True)
class WorkloadAssumption:
    """Workload knobs that turn peak compute throughput into sustained power.

    Defaults model well-tiled DNN inference: high cache hit rates,
    moderate duty cycle (the chip is rate-limited by something, usually
    DRAM bandwidth or DMA, not compute). Architects override this when
    sizing for a different workload class (training, sparse, attention,
    etc.).
    """

    # Fraction of cycles in which a PE actually fires its datapath. The
    # rest are clock-gated stalls (DMA waits, pipeline bubbles, control).
    # Inference: ~0.15. Training: closer to 0.4. Memory-bound LLM: ~0.05.
    compute_duty_cycle: float = 0.15

    # Bytes touched per op (operand load + result store). 1.5 captures
    # weight reuse: weights stay resident, only activations stream.
    bytes_per_op: float = 1.5

    # Cache hit rates. Per-level miss rate = 1 - hit. L1 misses go to L2,
    # L2 to L3, L3 to DRAM.
    l1_hit_rate: float = 0.95
    l2_hit_rate: float = 0.85
    l3_hit_rate: float = 0.70

    # NoC: fraction of L1 accesses that traverse the on-chip mesh, times
    # the average number of hops. Tile-local data movement (within-tile
    # PE-to-PE) is rolled into PE compute energy.
    noc_traversal_rate_per_op: float = 0.05
    avg_noc_hops: float = 4.0


DEFAULT_WORKLOAD = WorkloadAssumption()


@dataclass
class TDPBreakdown:
    """Per-term decomposition for one thermal profile."""

    profile_name: str
    clock_mhz: float
    worst_precision: str
    pe_compute_w: float
    l2_sram_w: float
    l3_sram_w: float
    noc_w: float
    dram_phy_w: float
    leakage_w: float

    @property
    def dynamic_w(self) -> float:
        return self.pe_compute_w + self.l2_sram_w + self.l3_sram_w + self.noc_w + self.dram_phy_w

    @property
    def total_tdp_w(self) -> float:
        return self.dynamic_w + self.leakage_w


def _peak_compute_w_for_precision(
    spec: KPUSKUInputSpec,
    node: ProcessNodeEntry,
    precision: str,
    clock_mhz: float,
) -> float:
    """Peak compute power assuming every PE that supports `precision` is
    firing every clock. Sums per-tile-class contributions using the
    tile's ``pe_circuit_class`` to look up energy_per_op_pj."""
    clock_hz = clock_mhz * 1e6
    total_pj_per_clock = 0.0
    for tile in spec.kpu_architecture.tiles:
        ops_per_clock = tile.ops_per_tile_per_clock.get(precision, 0.0) * tile.num_tiles
        if ops_per_clock <= 0:
            continue
        key = f"{tile.pe_circuit_class.value}:{precision}"
        e_pj = node.energy_per_op_pj.get(key)
        if e_pj is None:
            continue  # unsupported precision on this library; skip silently
        total_pj_per_clock += ops_per_clock * e_pj
    return total_pj_per_clock * clock_hz * 1e-12  # W


def _peak_ops_per_s_for_precision(
    spec: KPUSKUInputSpec,
    precision: str,
    clock_mhz: float,
) -> float:
    """Total ops/sec across the chip if every supporting tile fires every clock."""
    clock_hz = clock_mhz * 1e6
    return clock_hz * sum(
        tile.num_tiles * tile.ops_per_tile_per_clock.get(precision, 0.0)
        for tile in spec.kpu_architecture.tiles
    )


def _sram_pj_per_byte(node: ProcessNodeEntry, lib: CircuitClass, default: float) -> float:
    """Look up the node's sram_access_pj_per_byte for `lib`, falling back
    to ``default`` if unset (older ProcessNode YAMLs)."""
    return node.sram_access_pj_per_byte.get(lib, default)


def _noc_pj_per_flit(node: ProcessNodeEntry, lib: CircuitClass, default: float) -> float:
    return node.noc_pj_per_flit_per_hop.get(lib, default)


def compute_thermal_profile_tdp_breakdown(
    spec: KPUSKUInputSpec,
    profile: KPUThermalProfile,
    node: ProcessNodeEntry,
    workload: WorkloadAssumption | None = None,
    nodes: Optional[Mapping[str, ProcessNodeEntry]] = None,
) -> TDPBreakdown:
    """Derive a per-term TDP breakdown for one thermal profile.

    Picks the precision that produces the highest sustained dynamic
    power and reports it as ``worst_precision``. TDP is the sum of all
    five terms plus chip leakage.

    A heterogeneous profile (see ``is_legacy_shaped``) is derived by
    ``compute_heterogeneous_tdp_breakdown`` instead; ``nodes`` resolves the
    reference process nodes of fixed-function figures there.
    """
    if not is_legacy_shaped(spec, profile):
        return compute_heterogeneous_tdp_breakdown(spec, profile, node, workload, nodes)
    workload = workload or DEFAULT_WORKLOAD
    arch = spec.kpu_architecture

    # Build a placeholder ComputeProduct so total_chip_leakage_w can walk
    # the silicon_bin (it expects a sized chip). The transistor / die
    # fields don't affect leakage math -- only the per-block area does.
    placeholder_cp = _placeholder_compute_product(spec, profile, node)
    leakage_w = max(0.0, total_chip_leakage_w(placeholder_cp, node))

    # Voltage scaling: dynamic power scales by (V/Vnom)^2. Defaults to
    # nominal_vdd if the profile doesn't pick a Vdd. This is what spreads
    # TDP across DVFS operating points (Orin-style: lower-power modes
    # drop both clock AND Vdd).
    vdd = profile.vdd_v if profile.vdd_v is not None else node.nominal_vdd_v
    voltage_scale = (vdd / node.nominal_vdd_v) ** 2

    # Leakage Vdd-scaling (#154): leakage_w_per_mm2 is characterized AT nominal
    # Vdd, so total_chip_leakage_w returns the nominal-Vdd figure. Sub-threshold
    # and gate leakage both fall steeply with Vdd -- holding leakage flat
    # overstates it by 2-3x on low-power profiles (e.g. 0.60 V vs 0.80 V
    # nominal). Scale by (V/Vnom)^leakage_vdd_exponent when the node carries the
    # exponent; otherwise leave flat (legacy behavior on older ProcessNode YAMLs).
    if node.leakage_vdd_exponent is not None and vdd != node.nominal_vdd_v:
        leakage_w *= (vdd / node.nominal_vdd_v) ** node.leakage_vdd_exponent

    # Per-tile L2 + L3 use sram_hd by convention (matches catalog silicon_bin).
    sram_pj = _sram_pj_per_byte(node, CircuitClass.SRAM_HD, default=0.5)
    dram_pj = node.dram_io_pj_per_byte if node.dram_io_pj_per_byte is not None else 7.0
    noc_pj = _noc_pj_per_flit(node, arch.noc.router_circuit_class, default=1.0)
    flit_bytes = arch.noc.flit_bytes
    duty = workload.compute_duty_cycle
    bpo = workload.bytes_per_op

    best: TDPBreakdown | None = None
    precisions = set()
    for tile in arch.tiles:
        precisions.update(tile.ops_per_tile_per_clock.keys())

    for precision in precisions:
        peak_compute_w = _peak_compute_w_for_precision(
            spec, node, precision, profile.clock_mhz
        )
        if peak_compute_w <= 0:
            continue
        # Tile utilization for this precision (default 0.95 if unset).
        tile_util = (
            profile.tile_utilization_by_precision.get(precision, 0.95)
            if profile.tile_utilization_by_precision
            else 0.95
        )
        # Optional per-profile activity_factor multiplies the chip-wide
        # workload duty cycle. Lets architects tune low-power profiles
        # without changing the workload model.
        profile_activity = (
            profile.activity_factor if profile.activity_factor is not None else 1.0
        )
        effective_duty = duty * profile_activity
        sustained_compute_w = peak_compute_w * tile_util * effective_duty

        # Memory traffic at this precision.
        peak_ops_per_s = _peak_ops_per_s_for_precision(
            spec, precision, profile.clock_mhz
        )
        sustained_ops_per_s = peak_ops_per_s * tile_util * effective_duty
        l1_byte_rate = sustained_ops_per_s * bpo
        l2_byte_rate = l1_byte_rate * (1.0 - workload.l1_hit_rate)
        l3_byte_rate = l2_byte_rate * (1.0 - workload.l2_hit_rate)
        dram_byte_rate = l3_byte_rate * (1.0 - workload.l3_hit_rate)

        # All dynamic terms scale with V^2 (charging/discharging C*V^2*f).
        # Leakage scales separately by (V/Vnom)^leakage_vdd_exponent and was
        # already applied to leakage_w above (#154).
        sustained_compute_w *= voltage_scale
        l2_w = l2_byte_rate * sram_pj * 1e-12 * voltage_scale
        l3_w = l3_byte_rate * sram_pj * 1e-12 * voltage_scale
        dram_w = dram_byte_rate * dram_pj * 1e-12 * voltage_scale

        # NoC: a fraction of L1 byte traffic crosses the mesh, packed
        # into flits, traversing avg_hops on average.
        noc_byte_rate = l1_byte_rate * workload.noc_traversal_rate_per_op
        noc_flit_rate = noc_byte_rate / flit_bytes
        noc_w = noc_flit_rate * workload.avg_noc_hops * noc_pj * 1e-12 * voltage_scale

        bd = TDPBreakdown(
            profile_name=profile.name,
            clock_mhz=profile.clock_mhz,
            worst_precision=precision,
            pe_compute_w=sustained_compute_w,
            l2_sram_w=l2_w,
            l3_sram_w=l3_w,
            noc_w=noc_w,
            dram_phy_w=dram_w,
            leakage_w=leakage_w,
        )
        if best is None or bd.total_tdp_w > best.total_tdp_w:
            best = bd

    if best is None:
        # Spec has no precisions / no usable tiles; report leakage-only.
        return TDPBreakdown(
            profile_name=profile.name,
            clock_mhz=profile.clock_mhz,
            worst_precision="(none)",
            pe_compute_w=0.0, l2_sram_w=0.0, l3_sram_w=0.0,
            noc_w=0.0, dram_phy_w=0.0, leakage_w=leakage_w,
        )
    return best


def compute_thermal_profile_tdp_w(
    spec: KPUSKUInputSpec,
    profile: KPUThermalProfile,
    node: ProcessNodeEntry,
    workload: WorkloadAssumption | None = None,
    nodes: Optional[Mapping[str, ProcessNodeEntry]] = None,
) -> float:
    """Convenience wrapper: just the rounded TDP in watts."""
    return round(
        compute_thermal_profile_tdp_breakdown(spec, profile, node, workload, nodes).total_tdp_w,
        1,
    )


def _placeholder_compute_product(
    spec: KPUSKUInputSpec,
    profile: KPUThermalProfile,
    node: ProcessNodeEntry,
) -> ComputeProduct:
    """Build a minimum-viable ComputeProduct so silicon_math helpers
    (which walk a SKU rather than a spec) can run during TDP derivation.
    The die / performance / power roll-ups are placeholders -- only the
    architecture and silicon_bin matter for the math we use here."""
    perf = KPUTheoreticalPerformance(int8_tops=0.0, bf16_tflops=0.0, fp32_tflops=0.0)
    placeholder_tdp = profile.tdp_watts if profile.tdp_watts > 0 else 1.0
    return ComputeProduct(
        id=spec.id,
        name=spec.name,
        vendor=spec.vendor,
        kind=ProductKind.CHIP,
        packaging=Packaging(
            kind=PackagingKind.MONOLITHIC,
            num_dies=1,
            package_type="monolithic",
        ),
        dies=[
            Die(
                die_id="kpu_compute",
                die_role=DieRole.COMPUTE,
                process_node_id=spec.process_node_id,
                die_size_mm2=1.0,           # placeholder
                transistors_billion=1.0,    # placeholder
                silicon_bin=spec.silicon_bin,
                clocks=spec.clocks,
                blocks=[KPUBlock.from_architecture(spec.kpu_architecture)],
                interconnects=[],
            )
        ],
        performance=perf,
        power=Power(
            tdp_watts=placeholder_tdp,
            max_power_watts=placeholder_tdp,
            min_power_watts=placeholder_tdp,
            default_thermal_profile=profile.name,
            thermal_profiles=[profile],
        ),
        market=Market(
            launch_date=spec.market.launch_date,
            launch_msrp_usd=spec.market.launch_msrp_usd,
            target_market=spec.market.target_market,
            product_family=spec.market.product_family,
            model_tier=spec.market.model_tier,
            is_available=spec.market.is_available,
        ),
        notes=spec.notes,
        datasheet_url=spec.datasheet_url,
        last_updated=spec.last_updated,
    )


# ---------------------------------------------------------------------------
# Heterogeneous KPUs (graphs#268 Phase C2)
# ---------------------------------------------------------------------------

# A ProcessNode energy anchor ``"<class>:<format>"`` is the node's MAC / FMA
# at that format. This model charges ``energy_per_op_pj`` per op (2 per MAC),
# so one anchor invocation is 2 ops' energy. A RelativeEnergy ``ratio`` is
# per invocation: ratio 1.0 on an INT8 MAC costs exactly the legacy charge,
# and a lerp at 1.3 costs 1.3 FMA invocations.
ANCHOR_OPS_PER_INVOCATION = 2.0


@dataclass
class HeterogeneousTDPBreakdown(TDPBreakdown):
    """``TDPBreakdown`` for a heterogeneous profile.

    ``pe_compute_w`` covers the programmable kinds (pe_fabric, systolic);
    ``fixed_function_w`` the fixed-function tiles. ``compute_w_by_tile_class``
    splits both by tile class. ``worst_precision`` is ``"scenario"`` when the
    profile's ``tdp_scenario`` set the activities.
    """

    fixed_function_w: float = 0.0
    compute_w_by_tile_class: dict = field(default_factory=dict)
    gated_tile_classes: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def dynamic_w(self) -> float:
        return super().dynamic_w + self.fixed_function_w


def is_legacy_shaped(spec: KPUSKUInputSpec, profile: KPUThermalProfile) -> bool:
    """Whether the original TDP formula applies unchanged: every tile is a
    pe_fabric tile and the profile declares no ``tdp_scenario`` and no
    per-domain operating points."""
    return (
        all(isinstance(t, KPUTileSpec) for t in spec.kpu_architecture.tiles)
        and profile.tdp_scenario is None
        and not profile.domain_operating_points
    )


@dataclass
class _Option:
    """One way a tile class can run: a legacy precision or a custom datapath
    key, with whole-class ops and energy per clock (pJ, at nominal Vdd)."""

    key: str
    ops_per_clock: float
    pj_per_clock: float


@dataclass
class _ClassModel:
    tile_class_id: str
    clock_mhz: float
    vdd_v: float
    gated: bool
    activity: Optional[float]  # domain activity (replaces the profile's), if set
    legacy: dict = field(default_factory=dict)  # precision -> _Option
    custom: list = field(default_factory=list)  # [_Option] of non-MAC/FMA keys
    ff_pj_per_clock: float = 0.0  # fixed function (whole class)
    ff_io_bytes_per_clock: float = 0.0

    def worst(self) -> Optional[_Option]:
        opts = list(self.legacy.values()) + self.custom
        return max(opts, key=lambda o: o.pj_per_clock) if opts else None


def _node_lookup(ref_id: str, nodes: Optional[Mapping[str, ProcessNodeEntry]]):
    if nodes is None:
        from embodied_schemas import load_process_nodes

        nodes = load_process_nodes()
    return nodes.get(ref_id)


def _logic_ratio(node: ProcessNodeEntry, ref: ProcessNodeEntry, cc: CircuitClass) -> float:
    """Energy ratio target / reference for logic of library ``cc`` (its INT8
    anchor, falling back to balanced logic)."""
    for lib in (cc.value, CircuitClass.BALANCED_LOGIC.value):
        a, b = node.energy_per_op_pj.get(f"{lib}:int8"), ref.energy_per_op_pj.get(f"{lib}:int8")
        if a and b:
            return a / b
    return 1.0


def _sram_ratio(node: ProcessNodeEntry, ref: ProcessNodeEntry) -> Optional[float]:
    a = node.sram_access_pj_per_byte.get(CircuitClass.SRAM_HD)
    b = ref.sram_access_pj_per_byte.get(CircuitClass.SRAM_HD)
    return a / b if a and b else None


def _invocation_pj(
    unit: FunctionalUnit, mode, cc: CircuitClass, node: ProcessNodeEntry, nodes, notes: list
) -> Optional[float]:
    """Energy of one invocation of ``unit`` in ``mode`` at ``node`` (pJ), or
    None when the mode declares no energy."""
    e = mode.energy
    if isinstance(e, RelativeEnergy):
        anchor = node.energy_per_op_pj.get(e.anchor)
        if anchor is None:
            notes.append(f"{unit.unit_id}:{mode.operand_format}: node lacks anchor {e.anchor!r}")
            return 0.0
        return e.ratio * ANCHOR_OPS_PER_INVOCATION * anchor
    if isinstance(e, AbsoluteEnergy):
        ref = _node_lookup(e.ref_node_id, nodes)
        if ref is None:
            notes.append(f"{unit.unit_id}: reference node {e.ref_node_id!r} not in catalog")
            return e.pj
        return e.pj * _logic_ratio(node, ref, e.circuit_class)
    return None


def _datapath_options(units, pes: int, cc: CircuitClass, node, nodes, notes):
    """(legacy precision -> _Option, [custom _Option]) for PE-array units.

    Units sharing a ``"<op>:<format>"`` key run concurrently (their energy
    adds); different keys are alternative modes. A legacy precision takes the
    larger of its MAC / FMA keys, as the datapath's legacy projection does.
    A mode without declared energy is charged the node anchor per op.
    """
    by_key: dict[str, list] = {}
    for unit in units:
        opi = unit.resolved_ops_per_invocation
        for mode in unit.modes:
            key = f"{unit.op.value}:{mode.operand_format}"
            rate = mode.lanes / mode.issue_interval_cycles  # invocations per clock
            e_inv = _invocation_pj(unit, mode, cc, node, nodes, notes)
            if e_inv is None:
                anchor = node.energy_per_op_pj.get(f"{cc.value}:{mode.operand_format}")
                if anchor is None:
                    notes.append(
                        f"{unit.unit_id}:{mode.operand_format}: node lacks energy key "
                        f"{cc.value}:{mode.operand_format}; compute energy counted as 0"
                    )
                e_inv = opi * (anchor or 0.0)
            ops, pj = by_key.get(key, (0.0, 0.0))
            by_key[key] = (ops + rate * opi * pes, pj + rate * e_inv * pes)
    legacy: dict[str, _Option] = {}
    custom: list[_Option] = []
    legacy_ops = {op.value for op in LEGACY_PRECISION_OPS}
    for key, (ops, pj) in by_key.items():
        op, fmt = key.split(":", 1)
        if op in legacy_ops:
            if fmt not in legacy or ops > legacy[fmt].ops_per_clock:
                legacy[fmt] = _Option(fmt, ops, pj)
        else:
            custom.append(_Option(key, ops, pj))
    return legacy, custom


def _tile_domains(arch) -> dict[str, str]:
    """tile_class_id -> power domain id (explicit ``power_domain_id``, else a
    tile_class domain listing the class)."""
    out: dict[str, str] = {}
    for d in arch.power_domains or []:
        if d.kind == PowerDomainKind.TILE_CLASS:
            for m in d.members:
                out[m] = d.domain_id
    for t in arch.tiles:
        if t.power_domain_id is not None:
            out[t.tile_class_id] = t.power_domain_id
    return out


def _class_models(spec, profile, node, nodes, notes) -> list[_ClassModel]:
    arch = spec.kpu_architecture
    ops_points = profile.domain_operating_points or {}
    domains = _tile_domains(arch)
    base_vdd = profile.vdd_v if profile.vdd_v is not None else node.nominal_vdd_v
    models = []
    for t in arch.tiles:
        op = ops_points.get(domains.get(t.tile_class_id, ""))
        m = _ClassModel(
            tile_class_id=t.tile_class_id,
            clock_mhz=op.clock_mhz if op and op.clock_mhz else profile.clock_mhz,
            vdd_v=op.vdd_v if op and op.vdd_v else base_vdd,
            gated=bool(op and op.gated),
            activity=op.activity if op else None,
        )
        if isinstance(t, KPUTileSpec):
            cc = t.pe_circuit_class
            if t.datapath is not None:
                m.legacy, m.custom = _datapath_options(
                    t.datapath.functional_units, t.total_pes,
                    t.datapath.circuit_class or cc, node, nodes, notes,
                )
            else:
                for prec, ops in t.ops_per_tile_per_clock.items():
                    e = node.energy_per_op_pj.get(f"{cc.value}:{prec}")
                    if e is None:
                        # As the legacy formula: no compute energy, but the
                        # ops still drive memory / NoC / DRAM traffic.
                        notes.append(
                            f"{t.tile_class_id}: node lacks energy key "
                            f"{cc.value}:{prec}; compute energy counted as 0"
                        )
                        e = 0.0
                    m.legacy[prec] = _Option(prec, ops * t.num_tiles, ops * t.num_tiles * e)
        elif isinstance(t, SystolicTile):
            m.legacy, m.custom = _datapath_options(
                [t.mac], t.total_pes, t.circuit_class, node, nodes, notes
            )
        elif isinstance(t, FixedFunctionTile):
            core = t.core
            ref = _node_lookup(core.energy.ref_node_id, nodes)
            if ref is None:
                notes.append(
                    f"{t.tile_class_id}: reference node {core.energy.ref_node_id!r} not in "
                    f"catalog; energy not scaled"
                )
                scale = 1.0
            else:
                r_logic = _logic_ratio(node, ref, CircuitClass.BALANCED_LOGIC)
                r_sram = _sram_ratio(node, ref) or r_logic
                sf = core.energy.sram_fraction
                scale = (1.0 - sf) * r_logic + sf * r_sram
            units = core.units_per_clock * t.num_tiles
            m.ff_pj_per_clock = units * core.energy.pj_per_unit * scale
            if core.io is not None:
                io = core.io.input_bytes_per_unit + core.io.output_bytes_per_unit
                m.ff_io_bytes_per_clock = units * io
        models.append(m)
    return models


def _leakage_w(spec, profile, node, nodes, models, uncore_vdd) -> float:
    """Chip leakage with per-part Vdd scaling: each tile class's silicon at
    its domain Vdd (nothing for a gated class), the rest at the uncore Vdd."""
    cp = _placeholder_compute_product(spec, profile, node)
    by_class = {m.tile_class_id: m for m in models}
    exp = node.leakage_vdd_exponent

    def scaled(w: float, vdd: float) -> float:
        if exp is not None and vdd != node.nominal_vdd_v:
            return w * (vdd / node.nominal_vdd_v) ** exp
        return w

    def owner(tile_class_id):
        return by_class.get(tile_class_id) if tile_class_id else None

    total = 0.0
    for b in spec.silicon_bin.blocks:
        w = sm.estimate_block_leakage_w(b, cp, node)
        m = None
        ts = b.transistor_source
        if ts.kind.value == "per_pe" and ts.count_ref:
            try:
                m = owner(sm.resolve_tile_ref(cp, ts.count_ref.removeprefix("tile.")).tile_class_id)
            except sm.SiliconMathError:
                m = None
        total += 0.0 if (m and m.gated) else scaled(w, m.vdd_v if m else uncore_vdd)
    for ba in sm.resolve_carried_areas(cp, node, nodes):
        w = node.leakage_w_per_mm2.get(ba.circuit_class, 0.0) * ba.area_mm2
        cid = ba.name.split(".", 1)[0]
        m = owner(cid) if not ba.name.startswith("noc_overlay.") else None
        total += 0.0 if (m and m.gated) else scaled(w, m.vdd_v if m else uncore_vdd)
    return max(0.0, total)


def compute_heterogeneous_tdp_breakdown(
    spec: KPUSKUInputSpec,
    profile: KPUThermalProfile,
    node: ProcessNodeEntry,
    workload: WorkloadAssumption | None = None,
    nodes: Optional[Mapping[str, ProcessNodeEntry]] = None,
) -> HeterogeneousTDPBreakdown:
    """TDP breakdown for any KPU, heterogeneous or not.

    * **Without ``tdp_scenario``:** the original formula, generalized. Every
      tile class runs at the workload duty cycle times its activity (its
      domain's ``activity``, else the profile's ``activity_factor``). One
      chip-wide precision is swept for the worst total. Classes without a
      MAC / FMA precision run their worst custom mode, and fixed-function
      tiles run at the same duty.
    * **With ``tdp_scenario``:** each class runs at its scenario activity in
      its own worst-power mode (``worst_precision`` is ``"scenario"``).

    Clock / Vdd come from the tile class's power domain operating point
    (else the profile's); a gated class contributes nothing. Memory and NoC
    traffic come from the programmable ops (plus fixed-function IO bytes,
    which ride the NoC) and are charged at the uncore domain's Vdd.
    """
    workload = workload or DEFAULT_WORKLOAD
    arch = spec.kpu_architecture
    if nodes is None:
        from embodied_schemas import load_process_nodes

        nodes = load_process_nodes()  # once, for every reference-node lookup below
    notes: list[str] = []
    models = _class_models(spec, profile, node, nodes, notes)
    live = [m for m in models if not m.gated]
    gated = sorted(m.tile_class_id for m in models if m.gated)

    base_vdd = profile.vdd_v if profile.vdd_v is not None else node.nominal_vdd_v
    uncore_vdd = base_vdd
    ops_points = profile.domain_operating_points or {}
    for d in arch.power_domains or []:
        if d.kind == PowerDomainKind.UNCORE and d.domain_id in ops_points:
            op = ops_points[d.domain_id]
            uncore_vdd = op.vdd_v or base_vdd
            break
    uncore_scale = (uncore_vdd / node.nominal_vdd_v) ** 2
    leakage_w = _leakage_w(spec, profile, node, nodes, models, uncore_vdd)

    sram_pj = _sram_pj_per_byte(node, CircuitClass.SRAM_HD, default=0.5)
    dram_pj = node.dram_io_pj_per_byte if node.dram_io_pj_per_byte is not None else 7.0
    noc_pj = _noc_pj_per_flit(node, arch.noc.router_circuit_class, default=1.0)
    profile_activity = profile.activity_factor if profile.activity_factor is not None else 1.0
    scenario = profile.tdp_scenario

    def evaluate(precision: Optional[str]) -> HeterogeneousTDPBreakdown:
        prog_w = ff_w = ops_per_s = ff_io_bytes_per_s = 0.0
        by_class: dict[str, float] = {}
        for m in live:
            hz = m.clock_mhz * 1e6
            vscale = (m.vdd_v / node.nominal_vdd_v) ** 2
            if scenario is not None:
                duty = scenario.get(m.tile_class_id, 0.0)
                opt, util = m.worst(), 1.0
            else:
                duty = workload.compute_duty_cycle * (
                    m.activity if m.activity is not None else profile_activity
                )
                if precision in m.legacy:
                    opt = m.legacy[precision]
                    util = (
                        profile.tile_utilization_by_precision.get(precision, 0.95)
                        if profile.tile_utilization_by_precision
                        else 0.95
                    )
                else:
                    opt = max(m.custom, key=lambda o: o.pj_per_clock) if (
                        m.custom and not m.legacy
                    ) else None
                    util = 0.95
            w = 0.0
            if opt is not None:
                w = opt.pj_per_clock * hz * 1e-12 * util * duty * vscale
                prog_w += w
                ops_per_s += opt.ops_per_clock * hz * util * duty
            if m.ff_pj_per_clock:
                fw = m.ff_pj_per_clock * hz * 1e-12 * duty * vscale
                ff_w += fw
                w += fw
                ff_io_bytes_per_s += m.ff_io_bytes_per_clock * hz * duty
            by_class[m.tile_class_id] = w
        l1_byte_rate = ops_per_s * workload.bytes_per_op
        l2_byte_rate = l1_byte_rate * (1.0 - workload.l1_hit_rate)
        l3_byte_rate = l2_byte_rate * (1.0 - workload.l2_hit_rate)
        dram_byte_rate = l3_byte_rate * (1.0 - workload.l3_hit_rate)
        noc_byte_rate = l1_byte_rate * workload.noc_traversal_rate_per_op + ff_io_bytes_per_s
        noc_flit_rate = noc_byte_rate / arch.noc.flit_bytes
        return HeterogeneousTDPBreakdown(
            profile_name=profile.name,
            clock_mhz=profile.clock_mhz,
            worst_precision="scenario" if scenario is not None else (precision or "(none)"),
            pe_compute_w=prog_w,
            l2_sram_w=l2_byte_rate * sram_pj * 1e-12 * uncore_scale,
            l3_sram_w=l3_byte_rate * sram_pj * 1e-12 * uncore_scale,
            noc_w=noc_flit_rate * workload.avg_noc_hops * noc_pj * 1e-12 * uncore_scale,
            dram_phy_w=dram_byte_rate * dram_pj * 1e-12 * uncore_scale,
            leakage_w=leakage_w,
            fixed_function_w=ff_w,
            compute_w_by_tile_class=by_class,
            gated_tile_classes=gated,
            notes=list(notes),
        )

    if scenario is not None:
        return evaluate(None)
    precisions = sorted({p for m in live for p in m.legacy})
    candidates = [evaluate(p) for p in precisions] or [evaluate(None)]
    # Legacy semantics: a precision with no compute power is not a candidate
    # (its ops may still carry traffic when another precision is chosen).
    candidates = [
        c for c in candidates if c.pe_compute_w + c.fixed_function_w > 0
    ] or candidates[:1]
    return max(candidates, key=lambda c: c.total_tdp_w)

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
"""

from __future__ import annotations

from dataclasses import dataclass

from embodied_schemas.kpu import KPUEntry, KPUThermalProfile
from embodied_schemas.process_node import CircuitClass, ProcessNodeEntry

from .kpu_sku_input import KPUSKUInputSpec
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
) -> TDPBreakdown:
    """Derive a per-term TDP breakdown for one thermal profile.

    Picks the precision that produces the highest sustained dynamic
    power and reports it as ``worst_precision``. TDP is the sum of all
    five terms plus chip leakage.
    """
    workload = workload or DEFAULT_WORKLOAD
    arch = spec.kpu_architecture

    # Build a placeholder KPUEntry so total_chip_leakage_w can walk the
    # silicon_bin (it expects a sized chip). The transistor / die fields
    # don't affect leakage math -- only the per-block area does.
    placeholder = _placeholder_entry(spec, profile, node)
    leakage_w = max(0.0, total_chip_leakage_w(placeholder, node))

    # Voltage scaling: dynamic power scales by (V/Vnom)^2. Defaults to
    # nominal_vdd if the profile doesn't pick a Vdd. This is what spreads
    # TDP across DVFS operating points (Orin-style: lower-power modes
    # drop both clock AND Vdd).
    vdd = profile.vdd_v if profile.vdd_v is not None else node.nominal_vdd_v
    voltage_scale = (vdd / node.nominal_vdd_v) ** 2

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
        # Leakage doesn't scale here (model limitation: should also drop
        # with Vdd, but our leakage_w_per_mm2 is single-Vdd in the schema).
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
) -> float:
    """Convenience wrapper: just the rounded TDP in watts."""
    return round(
        compute_thermal_profile_tdp_breakdown(spec, profile, node, workload).total_tdp_w,
        1,
    )


def _placeholder_entry(
    spec: KPUSKUInputSpec,
    profile: KPUThermalProfile,
    node: ProcessNodeEntry,
) -> KPUEntry:
    """Build a minimum-viable KPUEntry so silicon_math helpers (which
    walk a SKU rather than a spec) can run during TDP derivation. The
    die / performance / power roll-ups are placeholders -- only the
    architecture and silicon_bin matter for the math we use here."""
    from embodied_schemas import KPUDieSpec, KPUPowerSpec, KPUTheoreticalPerformance

    die = KPUDieSpec(
        architecture="KPU Tile",
        foundry=node.foundry,
        process_nm=node.node_nm,
        process_name=node.node_name,
        transistors_billion=1.0,  # placeholder
        die_size_mm2=1.0,         # placeholder
    )
    perf = KPUTheoreticalPerformance(int8_tops=0.0, bf16_tflops=0.0, fp32_tflops=0.0)
    power = KPUPowerSpec(
        tdp_watts=profile.tdp_watts if profile.tdp_watts > 0 else 1.0,
        max_power_watts=profile.tdp_watts if profile.tdp_watts > 0 else 1.0,
        min_power_watts=profile.tdp_watts if profile.tdp_watts > 0 else 1.0,
        default_thermal_profile=profile.name,
        thermal_profiles=[profile],
    )
    return KPUEntry(
        id=spec.id,
        name=spec.name,
        vendor=spec.vendor,
        process_node_id=spec.process_node_id,
        die=die,
        kpu_architecture=spec.kpu_architecture,
        silicon_bin=spec.silicon_bin,
        clocks=spec.clocks,
        performance=perf,
        power=power,
        market=spec.market,
        notes=spec.notes,
        datasheet_url=spec.datasheet_url,
        last_updated=spec.last_updated,
    )

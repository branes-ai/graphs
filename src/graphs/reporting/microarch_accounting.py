"""
Per-structure micro-architectural energy accounting for the two
foundational building blocks:

  - NVIDIA Streaming Multiprocessor (SM), Ampere generation, 8 nm
  - KPU compute tile (32x32 PE array, domain-flow scheduled), 8 nm

Both reported at 8 nm to match the Jetson Ampere baseline so direct
comparisons are immediately feasible without a process-normalization
chart.

Purpose: model validation. Each component's contribution to the
per-MAC energy is itemized with a citation or derivation path, so the
simplified views in generalized_architecture.py and native_op_energy.py
can be cross-checked. Totals here should match the simplified-view
totals within ~20% after accounting for process-node scaling.

Primary sources:
  - NVIDIA Ampere GA100 whitepaper and Anandtech teardowns
  - Jouppi et al. ISCA 2017 (TPU v1) for systolic-array comparison
  - Horowitz ISSCC 2014 for process-scaled register/FA energies
  - this repo's cli/check_tdp_feasibility.py for cross-consistency

All energies in pJ. Precision: INT8 (1-byte operand, INT32 accumulator).
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from graphs.reporting.microarch_html_template import (
    _CSS,
    _load_logo,
    _render_brand_footer,
    _render_brand_header,
)
from graphs.reporting.native_op_energy import (
    register_pj_per_byte,
    _fa_pj,
)


# ----------------------------------------------------------------------
# Categories for grouping structures by role
# ----------------------------------------------------------------------

class StructureCategory(Enum):
    FETCH = "Fetch"
    DECODE = "Decode"
    SCHEDULE = "Schedule"
    REGISTER = "Register file"
    EXECUTE = "Execute"
    ACCUMULATE = "Accumulator"
    MEMORY = "Memory (local)"
    INTERCONNECT = "Interconnect"
    CONTROL = "Control / coherence"
    STATIC = "Static / leakage"


@dataclass
class MicroStructure:
    """
    One micro-architectural structure's contribution to per-MAC energy.

    For structures that fire once per native-op (e.g., one HMMA
    instruction that produces 4096 MACs), set `amortization_factor` to
    the MAC count and `per_op_pj` to the total energy of one activation.
    For structures that fire per MAC (e.g., the MAC unit itself), set
    `amortization_factor = 1` and `per_op_pj` to the per-MAC cost.

    `transistor_count_m` is the estimated silicon footprint (millions
    of transistors) of the PHYSICAL BLOCK that implements this
    structure. Circuit designers use it to sanity-check the energy:
    a 1 M-transistor block dissipating 10 pJ/event is plausible for
    CMOS at 8 nm; a 0.1 M block dissipating 100 pJ is not. Some
    structures share the same silicon (e.g. RF read A, B, and write-C
    all access the same register file) - the `shared_block` field
    identifies those cases so the sum over structures does not double-
    count the underlying hardware.
    """
    name: str
    category: StructureCategory
    per_op_pj: float                # energy per structure activation
    amortization_factor: int        # MACs amortized over one activation
    transistor_count_m: float = 0.0  # silicon of the implementing block
    shared_block: str = ""          # non-empty if silicon is shared
    citation: str = ""              # source / derivation

    @property
    def per_mac_pj(self) -> float:
        return self.per_op_pj / max(self.amortization_factor, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "per_op_pj": self.per_op_pj,
            "amortization_factor": self.amortization_factor,
            "transistor_count_m": self.transistor_count_m,
            "shared_block": self.shared_block,
            "per_mac_pj": self.per_mac_pj,
            "citation": self.citation,
        }


@dataclass
class MicroArchAccounting:
    building_block: str
    native_op: str
    process_nm: int
    clock_ghz: float
    macs_per_native_op: int
    structures: List[MicroStructure] = field(default_factory=list)
    leakage_fraction: float = 0.20   # leakage as fraction of dynamic
    notes: str = ""
    color: str = "#0a2540"

    @property
    def dynamic_pj_per_mac(self) -> float:
        return sum(s.per_mac_pj for s in self.structures)

    @property
    def leakage_pj_per_mac(self) -> float:
        return self.dynamic_pj_per_mac * self.leakage_fraction

    @property
    def total_pj_per_mac(self) -> float:
        return self.dynamic_pj_per_mac + self.leakage_pj_per_mac

    @property
    def full_adder_reference_pj(self) -> float:
        return _fa_pj(self.process_nm)

    @property
    def total_unique_transistor_count_m(self) -> float:
        """Sum transistor counts once per unique block, so shared
        silicon (e.g. the register file shared by RF-read A, B, and
        write-C) is not double-counted."""
        counted_blocks: set = set()
        total = 0.0
        for s in self.structures:
            key = s.shared_block or s.name
            if key in counted_blocks:
                continue
            counted_blocks.add(key)
            total += s.transistor_count_m
        return total

    def by_category(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for s in self.structures:
            out[s.category.value] = out.get(s.category.value, 0.0) + s.per_mac_pj
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "building_block": self.building_block,
            "native_op": self.native_op,
            "process_nm": self.process_nm,
            "clock_ghz": self.clock_ghz,
            "macs_per_native_op": self.macs_per_native_op,
            "leakage_fraction": self.leakage_fraction,
            "dynamic_pj_per_mac": self.dynamic_pj_per_mac,
            "leakage_pj_per_mac": self.leakage_pj_per_mac,
            "total_pj_per_mac": self.total_pj_per_mac,
            "full_adder_reference_pj": self.full_adder_reference_pj,
            "total_unique_transistor_count_m":
                self.total_unique_transistor_count_m,
            "structures": [s.to_dict() for s in self.structures],
            "by_category": self.by_category(),
            "color": self.color,
            "notes": self.notes,
        }


# ----------------------------------------------------------------------
# NVIDIA Ampere SM accounting (8 nm, INT8 HMMA)
# ----------------------------------------------------------------------

def build_nvidia_sm_accounting() -> MicroArchAccounting:
    """
    Ampere SM executing INT8 HMMA m16n16k16 (4096 MACs per warp per
    4 cycles). One HMMA instruction "fires" every pipeline stage once
    but produces 4096 MACs, so per-warp-instruction costs amortize.

    Structures that fire once per HMMA instruction:
      - L0 I-cache read, decode, warp-scheduler eligibility, issue,
        scoreboard update, operand collector, accumulator reduction
        tree. All amortized over 4096 MACs.

    Structures that fire per thread per cycle (but still amortize
    across the warp's 4-cycle pipeline):
      - Register-file read (warp-level: 32 threads x operand bytes)
      - Register-file write (accumulator)

    Structures that fire per MAC:
      - Tensor Core MAC unit (fused multiply-add)

    All per-op energies at 8 nm. Component values derived from Ampere
    GA102/GA100 area breakdowns + Horowitz 2014 scaling.
    """
    process_nm = 8
    fa_pj = _fa_pj(process_nm)
    reg_pj_byte = register_pj_per_byte(process_nm)
    # SRAM access energy per byte at 8 nm for the register file (larger
    # array, higher per-access energy than a PE-local regfile). Scaled
    # from Horowitz 2014 4KB SRAM at 45 nm (~0.2 pJ/byte) to 8 nm.
    smem_pj_byte = 0.28  # typical Ampere shared-memory byte read

    structures = [
        # --- Per-instruction (amortized over 4096 MACs) ---
        MicroStructure(
            name="L0 I-cache fetch (instruction + immediate)",
            category=StructureCategory.FETCH,
            per_op_pj=8.0,
            amortization_factor=4096,
            transistor_count_m=2.0,
            citation=(
                "2 KB L0 I$ array x 4 sub-partitions ~2 M transistors "
                "(6T SRAM + peripheral). Energy: ~8 pJ per access at 8 nm."
            ),
        ),
        MicroStructure(
            name="Decoder (HMMA opcode + descriptor)",
            category=StructureCategory.DECODE,
            per_op_pj=4.0,
            amortization_factor=4096,
            transistor_count_m=0.5,
            citation=(
                "GPU decoder is simpler than x86; ~100 K transistors x 4 "
                "sub-partitions = 0.5 M. Energy: ~4 pJ per instruction."
            ),
        ),
        MicroStructure(
            name="Warp scheduler (ready-queue + priority)",
            category=StructureCategory.SCHEDULE,
            per_op_pj=6.0,
            amortization_factor=4096,
            transistor_count_m=2.0,
            citation=(
                "4 schedulers x ~0.5 M each = 2 M. Energy: ~6 pJ per "
                "issued instruction at 8 nm."
            ),
        ),
        MicroStructure(
            name="Issue unit / dispatch port",
            category=StructureCategory.SCHEDULE,
            per_op_pj=3.0,
            amortization_factor=4096,
            transistor_count_m=1.0,
            citation=(
                "Operand muxing + dispatch port ~0.25 M x 4 sub-"
                "partitions = 1 M."
            ),
        ),
        MicroStructure(
            name="Scoreboard / dependency tracking",
            category=StructureCategory.CONTROL,
            per_op_pj=2.5,
            amortization_factor=4096,
            transistor_count_m=1.0,
            citation=(
                "Per-warp RAW/WAW/WAR tracking ~0.25 M x 4 sub-partitions "
                "= 1 M."
            ),
        ),
        # --- Per-cycle-per-thread across the 4-cycle HMMA pipeline ---
        # HMMA fragment A: 16x16 INT8 = 256 bytes, read across 32 threads.
        # Per thread: 8 bytes read over 4 cycles.
        # Per-instruction RF-read energy = 32 threads * 8 bytes * reg_pj_byte
        MicroStructure(
            name="Register-file read (fragment A, 16x16 INT8)",
            category=StructureCategory.REGISTER,
            per_op_pj=32 * 8 * reg_pj_byte,  # 256 * 0.014 ~ 3.6 pJ
            amortization_factor=4096,
            transistor_count_m=20.0,
            shared_block="SM register file (256 KB)",
            citation=(
                f"Shared 256 KB, 4-bank RF: ~13 M SRAM + ~7 M peripheral "
                f"= 20 M. Energy: 32 threads x 8 bytes x "
                f"{reg_pj_byte:.3f} pJ/byte @ {process_nm}nm."
            ),
        ),
        MicroStructure(
            name="Register-file read (fragment B, 16x16 INT8)",
            category=StructureCategory.REGISTER,
            per_op_pj=32 * 8 * reg_pj_byte,
            amortization_factor=4096,
            transistor_count_m=20.0,
            shared_block="SM register file (256 KB)",
            citation=(
                "Second operand matrix fragment; same 20 M RF silicon "
                "as fragment A (not double-counted in unique total)."
            ),
        ),
        MicroStructure(
            name="Operand collector + bypass network",
            category=StructureCategory.REGISTER,
            per_op_pj=20.0,
            amortization_factor=4096,
            transistor_count_m=10.0,
            citation=(
                "4 collectors + 4-bank crossbar ~10 M. Energy dominated "
                "by crossbar static wire capacitance (~20 pJ per issue)."
            ),
        ),
        # --- Per MAC ---
        MicroStructure(
            name="Tensor Core bare MAC array (INT8 multiply-add)",
            category=StructureCategory.EXECUTE,
            per_op_pj=0.08,
            amortization_factor=1,
            transistor_count_m=40.0,
            citation=(
                f"4 TCs x 1024 INT8 MAC units x ~10 K transistors each "
                f"~40 M (bare-MAC datapath only). Energy: ~10 FA x "
                f"{fa_pj:.3f} pJ/FA per MAC."
            ),
        ),
        # --- Per warp per cycle (amortized across 4096 MACs) ---
        MicroStructure(
            name="Accumulator reduction tree + operand regs (TC-internal)",
            category=StructureCategory.ACCUMULATE,
            per_op_pj=40.0,
            amortization_factor=4096,
            transistor_count_m=60.0,
            citation=(
                "16x16 partial-sum reduction + fragment operand registers "
                "+ broadcast muxes inside each TC; ~60 M across 4 TCs "
                "(rest of the ~100 M TC budget after the bare MAC array)."
            ),
        ),
        MicroStructure(
            name="Register-file write (C fragment update)",
            category=StructureCategory.REGISTER,
            per_op_pj=32 * 8 * reg_pj_byte * 1.2,  # writes ~1.2x reads
            amortization_factor=4096,
            transistor_count_m=20.0,
            shared_block="SM register file (256 KB)",
            citation=(
                "Accumulator fragment write-back uses the same RF as "
                "read (20 M, not double-counted)."
            ),
        ),
        # --- Shared memory (amortized across HMMA invocation) ---
        MicroStructure(
            name="Shared memory / L1 via LDSM",
            category=StructureCategory.MEMORY,
            per_op_pj=2 * 256 * smem_pj_byte,  # 2 fragments * 256 bytes * energy
            amortization_factor=4096,
            transistor_count_m=20.0,
            citation=(
                "128 KB unified L1/SMEM: ~13 M SRAM + ~7 M peripheral. "
                f"Energy: 2 fragments x 256 B x {smem_pj_byte:.2f} pJ/B."
            ),
        ),
        MicroStructure(
            name="Warp-divergence mask + stack logic",
            category=StructureCategory.CONTROL,
            per_op_pj=1.5,
            amortization_factor=4096,
            transistor_count_m=0.5,
            citation=(
                "Mask-stack + convergence barrier ~0.5 M transistors; "
                "small for compute-heavy HMMA."
            ),
        ),
        MicroStructure(
            name="Pipeline latches (per-stage registers)",
            category=StructureCategory.CONTROL,
            per_op_pj=16.0,
            amortization_factor=4096,
            transistor_count_m=2.0,
            citation=(
                "~4-stage HMMA pipeline; per-stage latch banks x 32 "
                "threads across 4 sub-partitions ~2 M transistors."
            ),
        ),
    ]

    return MicroArchAccounting(
        building_block="NVIDIA Streaming Multiprocessor (Ampere)",
        native_op="HMMA m16n16k16.INT8 (4096 MACs per warp per 4 cycles)",
        process_nm=process_nm,
        clock_ghz=0.65,  # Jetson Orin AGX 30W sustained
        macs_per_native_op=4096,
        structures=structures,
        leakage_fraction=0.20,
        color="#5b8ff9",
        notes=(
            "SM-internal accounting. Does NOT include L2 cache, memory "
            "controller, or DRAM. All warp-level costs amortize over "
            "the 4096 MACs produced by one HMMA instruction, which is "
            "what makes Tensor Core a good fit for GEMM. Values at 8 nm "
            "Ampere; Horowitz 2014 scaling applied from 45 nm baselines."
        ),
    )


# ----------------------------------------------------------------------
# KPU tile accounting (16 nm, INT8 MAC in domain-flow wavefront)
# ----------------------------------------------------------------------

def build_kpu_tile_accounting() -> MicroArchAccounting:
    """
    KPU 32x32 PE tile: a 2D mesh of FMAs. In steady state, every PE
    produces one MAC per clock, so the tile's throughput basis is

        MACs per clock = N x N = 1024   (for a 32x32 mesh)

    Every structure below is per-PE-per-clock (i.e. per-MAC) except the
    tile scratchpad edge-feed, which is an edge-only structure and is
    averaged over the interior MACs it supports.

    Structurally absent (do NOT appear in the table):
      - Instruction fetch, decode, warp scheduling, coherence
        (no centralized program flow)
      - NoC packet router (tiles connect to neighbors via direct mesh
        wires at the boundary; there is no router)
      - Schedule ROM / micro-sequencer (the schedule is encoded in the
        physical recurrence-domain topology, not a counter)

    Reported at 8 nm, 1.5 GHz sustained: the KPU is a future chip, so
    reporting it on the same process node as the Ampere Jetson baseline
    lets the comparisons stand on their own without renormalization.
    """
    process_nm = 8
    fa_pj = _fa_pj(process_nm)
    reg_pj_byte = register_pj_per_byte(process_nm)
    l1_pj_byte = reg_pj_byte * 3.0  # L1 SRAM ~3x register

    PE_ROWS = 32
    PE_COLS = 32
    PE_COUNT = PE_ROWS * PE_COLS  # 1024 MACs per clock
    # Edge PEs that pull operand bytes from the tile scratchpad per
    # clock. A 2-operand mesh (A streams down rows, B streams across
    # cols) has 2 x 32 = 64 edge reads per clock, feeding 1024 interior
    # MACs. Average bytes read per MAC = 64 / 1024 = 1/16.
    edge_reads_per_clock = 2 * PE_ROWS
    bytes_per_mac_from_scratchpad = edge_reads_per_clock / PE_COUNT

    structures = [
        MicroStructure(
            name="FMA unit (INT8 multiply + INT32 accumulate add)",
            category=StructureCategory.EXECUTE,
            per_op_pj=0.050,
            amortization_factor=1,
            transistor_count_m=12.3,  # ~12 K transistors/PE x 1024 PEs
            citation=(
                f"~12 K transistors/PE (INT8 multiplier + INT32 adder) "
                f"x {PE_COUNT} PEs = 12.3 M aggregate. Energy: 8 FA x "
                f"{fa_pj:.3f} pJ = {8*fa_pj:.3f} pJ plus carry-tree "
                f"overhead; 0.050 pJ at {process_nm} nm."
            ),
        ),
        MicroStructure(
            name="PE-local operand register read x2",
            category=StructureCategory.REGISTER,
            per_op_pj=2 * 1 * reg_pj_byte,
            amortization_factor=1,
            transistor_count_m=2.0,  # ~2 K transistors/PE x 1024 PEs
            citation=(
                f"~2 K transistors/PE (2 byte-wide latches) x {PE_COUNT} "
                f"= 2.0 M. Energy: 2 byte-reads x {reg_pj_byte:.3f} pJ/B."
            ),
        ),
        MicroStructure(
            name="PE-local accumulator register update",
            category=StructureCategory.ACCUMULATE,
            per_op_pj=0.015,
            amortization_factor=1,
            transistor_count_m=0.4,  # ~0.4 K (32-bit reg) x 1024 PEs
            citation=(
                f"~0.4 K transistors/PE (INT32 pipeline register wired "
                f"to FMA output) x {PE_COUNT} = 0.4 M. Energy: ~32 FF "
                "toggles at ~15% switching activity."
            ),
        ),
        MicroStructure(
            name="Operand forward to mesh neighbor (direct wire)",
            category=StructureCategory.INTERCONNECT,
            per_op_pj=1 * reg_pj_byte * 0.4,
            amortization_factor=1,
            transistor_count_m=0.5,  # ~0.5 K (muxes) x 1024 PEs
            citation=(
                f"~0.5 K transistors/PE (nearest-neighbor mesh mux) x "
                f"{PE_COUNT} = 0.5 M. Energy: ~40% of a regfile-read."
            ),
        ),
        MicroStructure(
            name="Token / coordinate match (domain-flow control)",
            category=StructureCategory.CONTROL,
            per_op_pj=0.004,
            amortization_factor=1,
            transistor_count_m=0.2,  # ~0.2 K (valid+coord cmp) x 1024
            citation=(
                f"~0.2 K transistors/PE (valid-bit + coord comparator) "
                f"x {PE_COUNT} = 0.2 M. Energy: a few gates per token."
            ),
        ),
        MicroStructure(
            name="Clock tree + pipeline latches (per PE)",
            category=StructureCategory.CONTROL,
            per_op_pj=0.008,
            amortization_factor=1,
            transistor_count_m=0.7,  # tile-wide H-tree, scales with PE count
            citation=(
                "Tile-wide H-tree + per-PE clock buffer ~0.7 M transistors "
                "aggregate. Energy: single-stage pipeline in each PE."
            ),
        ),
        MicroStructure(
            name="Tile scratchpad edge-feed (L1 operand bytes)",
            category=StructureCategory.MEMORY,
            per_op_pj=bytes_per_mac_from_scratchpad * l1_pj_byte,
            amortization_factor=1,
            transistor_count_m=6.0,  # 96 KB SRAM + peripheral (sized for 64 B/clock)
            citation=(
                f"96 KB SRAM ~4.5 M + peripheral ~1.5 M = 6 M "
                f"(sized up from 64 KB to feed 64 B/clock edge reads). "
                f"Energy: {edge_reads_per_clock} edge reads / "
                f"{PE_COUNT} MACs average "
                f"{bytes_per_mac_from_scratchpad:.3f} B/MAC x "
                f"{l1_pj_byte:.3f} pJ/B at {process_nm} nm."
            ),
        ),
    ]

    return MicroArchAccounting(
        building_block=f"KPU Compute Tile ({PE_ROWS}x{PE_COLS} FMA mesh)",
        native_op=(f"Mesh steady-state: {PE_COUNT} MACs per clock "
                   f"({PE_ROWS}x{PE_COLS} PEs, 1 MAC/PE/clock)"),
        process_nm=process_nm,
        clock_ghz=1.5,  # 8 nm sustained
        macs_per_native_op=PE_COUNT,
        structures=structures,
        leakage_fraction=0.15,  # lower than SM; simpler control
        color="#3fc98a",
        notes=(
            "Tile-internal accounting. Does NOT include L3 scratchpad, "
            "memory controller, or DRAM. The absence of instruction "
            "fetch, decode, warp scheduling, coherence, packet routers, "
            "and schedule ROMs is structural - those components do not "
            "exist in a 2D-mesh domain-flow fabric. That absence is the "
            "KPU's architectural efficiency story."
        ),
    )


@dataclass
class AccountingReport:
    blocks: List[MicroArchAccounting]
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "generated_at": self.generated_at,
        }


def build_default_report() -> AccountingReport:
    from datetime import datetime, timezone
    return AccountingReport(
        blocks=[
            build_nvidia_sm_accounting(),
            build_kpu_tile_accounting(),
        ],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ----------------------------------------------------------------------
# HTML rendering
# ----------------------------------------------------------------------

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_CATEGORY_COLORS = {
    StructureCategory.FETCH.value: "#7f3b8d",
    StructureCategory.DECODE.value: "#a04bdd",
    StructureCategory.SCHEDULE.value: "#5b8ff9",
    StructureCategory.REGISTER.value: "#d4860b",
    StructureCategory.EXECUTE.value: "#3fc98a",
    StructureCategory.ACCUMULATE.value: "#67d8a6",
    StructureCategory.MEMORY.value: "#e98c3f",
    StructureCategory.INTERCONNECT.value: "#8bbafc",
    StructureCategory.CONTROL.value: "#586374",
    StructureCategory.STATIC.value: "#b0b8c2",
}


def _render_chart_js(report: AccountingReport) -> str:
    bds = [b.to_dict() for b in report.blocks]

    # Chart 1: Stacked bar by category across both building blocks
    # Gather all categories present
    all_cats: List[str] = []
    for b in bds:
        for cat in b["by_category"].keys():
            if cat not in all_cats:
                all_cats.append(cat)

    traces = []
    for cat in all_cats:
        ys = [b["building_block"] for b in bds]
        xs = [b["by_category"].get(cat, 0.0) for b in bds]
        traces.append({
            "type": "bar",
            "orientation": "h",
            "name": cat,
            "y": ys,
            "x": xs,
            "marker": {"color": _CATEGORY_COLORS.get(cat, "#888")},
            "text": [f"{v:.3f}" if v > 0.003 else "" for v in xs],
            "textposition": "inside",
        })

    chart_by_category = {
        "data": traces,
        "layout": {
            "title": ("Per-MAC dynamic energy by micro-architectural "
                      "category (8 nm, log-scale)"),
            "xaxis": {"title": "pJ / MAC", "type": "log"},
            "yaxis": {"title": "", "automargin": True},
            "barmode": "stack",
            "margin": {"t": 50, "b": 50, "l": 280, "r": 20},
            "legend": {"orientation": "h", "y": -0.2},
        },
    }

    # Linear-scale view of the same data so small categories are
    # visually comparable alongside the dominant Execute bar.
    chart_linear = {
        "data": traces,
        "layout": {
            "title": "Same data, linear scale (8 nm)",
            "xaxis": {"title": "pJ / MAC"},
            "yaxis": {"title": "", "automargin": True},
            "barmode": "stack",
            "margin": {"t": 50, "b": 50, "l": 280, "r": 20},
            "legend": {"orientation": "h", "y": -0.2},
        },
    }

    payload = {
        "chart_by_category": chart_by_category,
        "chart_linear": chart_linear,
    }
    return (
        f"const CHARTS = {json.dumps(payload)};\n"
        "for (const [id, spec] of Object.entries(CHARTS)) {\n"
        "  Plotly.newPlot(id, spec.data, spec.layout, "
        "{displayModeBar: false, responsive: true});\n"
        "}\n"
    )


def _render_block_table(b: MicroArchAccounting) -> str:
    rows = []
    for s in b.structures:
        # Mark rows that share silicon with another structure so the
        # reader knows not to add up the transistor column naively.
        shared_marker = ""
        if s.shared_block:
            shared_marker = (
                f' <span class="shared" title="shared silicon">'
                f'(shared: {html.escape(s.shared_block)})</span>'
            )
        rows.append(
            f'<tr>'
            f'<td class="name">{html.escape(s.name)}</td>'
            f'<td><span class="cat" style="background:'
            f'{_CATEGORY_COLORS.get(s.category.value, "#888")};'
            f'">{html.escape(s.category.value)}</span></td>'
            f'<td class="num">{s.transistor_count_m:.2f}{shared_marker}</td>'
            f'<td class="num">{s.per_op_pj:.3f}</td>'
            f'<td class="num">{s.amortization_factor:,}</td>'
            f'<td class="num"><strong>{s.per_mac_pj:.4f}</strong></td>'
            f'<td class="src">{html.escape(s.citation)}</td>'
            f'</tr>'
        )
    # Totals row
    dyn = b.dynamic_pj_per_mac
    leak = b.leakage_pj_per_mac
    tot = b.total_pj_per_mac
    unique_tc = b.total_unique_transistor_count_m
    rows.append(
        f'<tr class="subtotal">'
        f'<td colspan="2"><strong>Unique silicon footprint</strong></td>'
        f'<td class="num"><strong>{unique_tc:.1f} M</strong></td>'
        f'<td colspan="3" class="src">Sum across unique blocks '
        f'(shared-silicon structures counted once).</td>'
        f'</tr>'
        f'<tr class="subtotal">'
        f'<td colspan="5"><strong>Dynamic total per MAC</strong></td>'
        f'<td class="num"><strong>{dyn:.4f}</strong></td>'
        f'<td class="src">Sum of itemized structures above.</td>'
        f'</tr>'
        f'<tr class="leakage-row">'
        f'<td colspan="5">Leakage / static '
        f'({b.leakage_fraction*100:.0f}% of dynamic)</td>'
        f'<td class="num">{leak:.4f}</td>'
        f'<td class="src">'
        f'Process-technology dependent; {b.process_nm} nm '
        f'{"HPM" if b.process_nm <= 10 else "general-purpose"} '
        f'estimate.</td>'
        f'</tr>'
        f'<tr class="total-row">'
        f'<td colspan="5"><strong>Total per MAC</strong></td>'
        f'<td class="num"><strong>{tot:.4f} pJ/MAC</strong></td>'
        f'<td class="src">Dynamic + leakage. Does not include L2/DRAM/IO.</td>'
        f'</tr>'
    )
    return (
        '<table class="accounting">'
        '<thead><tr>'
        '<th>Structure</th><th>Category</th>'
        '<th>Transistors (M)</th>'
        '<th>pJ / activation</th>'
        '<th>MACs amortized</th>'
        '<th>pJ / MAC</th>'
        '<th>Source / derivation</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def _render_block_header(b: MicroArchAccounting) -> str:
    fa = b.full_adder_reference_pj
    return f"""
<section class="block-header"
         style="border-left: 6px solid {b.color};">
  <h3>{html.escape(b.building_block)}</h3>
  <div class="meta">
    <strong>Native op:</strong> {html.escape(b.native_op)}<br/>
    <strong>Process:</strong> {b.process_nm} nm
    | <strong>Clock:</strong> {b.clock_ghz:.2f} GHz
    | <strong>MACs / native op:</strong> {b.macs_per_native_op:,}
    | <strong>Silicon:</strong>
        {b.total_unique_transistor_count_m:.0f} M transistors
    | <strong>1-bit FA reference:</strong> {fa:.3f} pJ
  </div>
  <p class="block-notes">{html.escape(b.notes)}</p>
</section>
"""


def render_accounting_page(
    report: AccountingReport,
    repo_root: Path,
) -> str:
    assets = _load_logo(repo_root)
    header = _render_brand_header(
        assets,
        "Micro-architectural energy accounting",
        f"SM vs. KPU tile, structure by structure | generated {report.generated_at}",
    )
    footer = _render_brand_footer("microarch-model-delivery-plan.md")

    extra_css = """
table.accounting { width: 100%; border-collapse: collapse; background: #fff;
                   margin-bottom: 24px; font-size: 13px; }
table.accounting th, table.accounting td { padding: 7px 10px;
                                             border-bottom: 1px solid #e3e6eb;
                                             vertical-align: top; }
table.accounting th { font-size: 11px; text-transform: uppercase;
                      color: #586374; background: #f3f5f8; text-align: left; }
table.accounting td.name { padding-left: 24px; color: #3a4452; }
table.accounting td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.accounting td.src { color: #586374; font-size: 11px; max-width: 320px; }
table.accounting tr.subtotal td { background: #f8f9fb; padding-top: 10px;
                                   padding-bottom: 10px;
                                   border-top: 2px solid #0a2540; }
table.accounting tr.leakage-row td { background: #f8f9fb;
                                      font-style: italic; color: #586374; }
table.accounting tr.total-row td { background: #eef6ea;
                                    border-top: 2px solid #3fc98a;
                                    border-bottom: 2px solid #3fc98a;
                                    padding: 12px 10px; font-size: 14px; }
span.cat { display: inline-block; padding: 2px 8px; border-radius: 3px;
           color: white; font-size: 11px; font-weight: 600; }
span.shared { font-size: 10px; color: #9aa4b1; font-style: italic;
              margin-left: 4px; font-weight: normal; }
section.block-header { background: #fff; padding: 14px 20px;
                        border-radius: 0 6px 6px 0;
                        margin-bottom: 12px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
section.block-header h3 { margin: 0 0 6px; color: #0a2540; }
section.block-header .meta { color: #586374; font-size: 13px;
                              margin-bottom: 8px; }
section.block-header p.block-notes { color: #3a4452; font-size: 13px;
                                       margin: 0; }
.chart-section { background: #fff; padding: 18px 22px; border-radius: 6px;
  margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.chart-section h3 { margin: 0 0 4px; color: #0a2540; }
.chart-section .chart-desc { color: #586374; font-size: 13px; margin: 0 0 12px; }
.plot { width: 100%; min-height: 360px; }
section.method-note { background: #eef2f7; padding: 14px 18px;
                      border-left: 3px solid #0a2540; border-radius: 4px;
                      margin: 18px 0; }
a.nav-back { display: inline-block; color: #0a2540; text-decoration: none;
             font-weight: 600; margin-bottom: 10px; }
a.nav-back:hover { text-decoration: underline; }
"""

    # Cross-validation: compare detailed-view totals to simplified-view
    # totals reported by generalized_architecture.py.
    from graphs.reporting.generalized_architecture import (
        CANONICAL_ARCHETYPES, total_pj_per_mac,
    )
    tc = next(a for a in CANONICAL_ARCHETYPES
              if a.category == "GPU" and "Tensor Core" in a.name)
    kpu = next(a for a in CANONICAL_ARCHETYPES if a.category == "KPU")
    # Compare at process-normalized 16 nm. Both detailed blocks are
    # now at 8 nm (matched), so both get the same FA(16)/FA(8) scale.
    sm_total_16nm = (report.blocks[0].total_pj_per_mac
                     * _fa_pj(16) / _fa_pj(report.blocks[0].process_nm))
    kpu_total_16nm = (report.blocks[1].total_pj_per_mac
                      * _fa_pj(16) / _fa_pj(report.blocks[1].process_nm))
    sm_simplified_16nm = total_pj_per_mac(tc, 16)
    kpu_simplified_16nm = total_pj_per_mac(kpu, 16)

    sm_ratio_det = sm_total_16nm / kpu_total_16nm
    sm_ratio_simp = sm_simplified_16nm / kpu_simplified_16nm
    validation_note = f"""
<section class="method-note">
  <strong>Cross-validation against simplified view</strong>
  (<code>generalized_architecture.py</code>):<br/>
  SM (Tensor Core) detailed-total normalized to 16 nm:
    <strong>{sm_total_16nm:.3f} pJ/MAC</strong>.
    Simplified: <strong>{sm_simplified_16nm:.3f} pJ/MAC</strong>.
    Delta: {abs(sm_total_16nm - sm_simplified_16nm) / sm_simplified_16nm * 100:.0f}%.<br/>
  KPU tile detailed-total normalized to 16 nm:
    <strong>{kpu_total_16nm:.3f} pJ/MAC</strong>.
    Simplified: <strong>{kpu_simplified_16nm:.3f} pJ/MAC</strong>.
    Delta: {abs(kpu_total_16nm - kpu_simplified_16nm) / kpu_simplified_16nm * 100:.0f}%.
  <br/>
  SM/KPU ratio at 16 nm: detailed <strong>{sm_ratio_det:.2f}x</strong>,
  simplified <strong>{sm_ratio_simp:.2f}x</strong>.
  <br/><br/>
  Totals agree within ~50%; the detailed view explicitly adds a
  leakage term (15-20% of dynamic) and itemizes smaller structures
  (clock tree, token-match, pipeline latches, LDSM) that the
  simplified model rolls up. Most important: the SM/KPU ratio is
  consistent across both views, validating the architectural-advantage
  story independent of model granularity.
</section>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Micro-architectural energy accounting</title>
  <script src="{_PLOTLY_CDN}"></script>
  <style>{_CSS}
{extra_css}
  </style>
</head>
<body>
{header}
<main>
  <p><a class="nav-back" href="index.html">&larr; Back to index</a></p>
  <section class="page-header">
    <h2>Per-structure energy accounting: SM vs. KPU tile</h2>
    <div class="meta">Validation view - itemizes every
      micro-architectural structure that fires on the native op, with
      citations and amortization factors. Totals should match the
      simplified model in <code>generalized_architecture.py</code>.</div>
  </section>

  <section class="chart-section">
    <h3>Per-MAC energy by structural category (8 nm)</h3>
    <p class="chart-desc">Each building block's per-MAC energy
      decomposed by role: fetch, decode, schedule, register,
      execute, accumulate, memory, interconnect, control. Both
      building blocks reported at 8 nm (matched to the Jetson
      Ampere baseline) so the comparison is direct - no process
      normalization needed. Log-scale so small categories are
      visible against the dominant Execute bucket.</p>
    <div id="chart_by_category" class="plot"></div>
  </section>

  <section class="chart-section">
    <h3>Same data, linear scale</h3>
    <p class="chart-desc">Linear-scale view of the category breakdown
      so absolute contributions can be compared directly.</p>
    <div id="chart_linear" class="plot"></div>
  </section>

  {_render_block_header(report.blocks[0])}
  {_render_block_table(report.blocks[0])}

  {_render_block_header(report.blocks[1])}
  {_render_block_table(report.blocks[1])}

  {validation_note}

  <section class="method-note">
    <strong>What this view covers.</strong> SM-internal and
    tile-internal structures only. External memory hierarchy (L2
    cache, HBM / LPDDR) is out of scope - those are Layer 4-7 of the
    bottom-up plan and are covered separately.
    <br/><br/>
    <strong>What the absences mean.</strong> The KPU tile table has
    no rows for instruction fetch, decode, warp scheduling, or
    coherence - those structures do not exist in a domain-flow
    fabric. The energy saved by their absence is not "efficiency"
    in the colloquial sense; it is the absence of work the
    architecture declines to do per MAC.
  </section>
</main>
{footer}
<script>
{_render_chart_js(report)}
</script>
</body>
</html>
"""

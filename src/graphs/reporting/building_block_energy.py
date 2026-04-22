"""
Building-block (per-clock) energy accounting for the two foundational
compute engines - NVIDIA SM and KPU compute tile - as they are deployed
on the SoC.

This complements ``microarch_accounting.py`` (per-MAC view). The two
views reconcile through:

    pJ/MAC = total_pJ_per_clock / native_MACs_per_clock

Use cases:

- **Per-MAC view** (microarch_accounting.py): understand performance
  under a fixed power constraint. Answers "how much compute can I get
  per watt given workload X on architecture Y?"

- **Per-clock view** (this module): drives SoC composition and
  super-cluster designs. Answers "if I deploy N building blocks at
  frequency F with utilization U, how much power and how much compute
  will the chip deliver?"

Each component carries its absolute per-clock energy contribution so
a budget can be built up directly without any per-MAC normalization.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from graphs.reporting.microarch_accounting import StructureCategory
from graphs.reporting.microarch_html_template import (
    _CSS,
    _load_logo,
    _render_brand_footer,
    _render_brand_header,
)
from graphs.reporting.native_op_energy import _fa_pj


@dataclass
class EngineComponent:
    """
    One major component inside a compute engine (e.g. register file,
    tensor cores, L1 scratchpad). Energy is expressed directly in
    pJ/clock as the component appears on silicon - no per-MAC
    normalization.
    """
    name: str
    category: StructureCategory
    count: int                    # number of instances inside the engine
    size_or_spec: str             # "256 KB, 4 banks" / "576 PE (24x24)"
    per_clock_pj: float           # absolute energy contribution per clock
    activity_note: str            # steady-state activity assumption
    citation: str                 # derivation / source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "count": self.count,
            "size_or_spec": self.size_or_spec,
            "per_clock_pj": self.per_clock_pj,
            "activity_note": self.activity_note,
            "citation": self.citation,
        }


@dataclass
class BuildingBlock:
    """
    One replicated compute engine on a SoC (SM, KPU tile, etc.).
    Aggregates components and exposes engine-level metrics.
    """
    name: str
    process_nm: int
    clock_ghz: float
    native_macs_per_clock: int
    components: List[EngineComponent] = field(default_factory=list)
    color: str = "#0a2540"
    notes: str = ""

    @property
    def total_pj_per_clock(self) -> float:
        return sum(c.per_clock_pj for c in self.components)

    @property
    def execute_pj_per_clock(self) -> float:
        return sum(c.per_clock_pj for c in self.components
                   if c.category == StructureCategory.EXECUTE)

    @property
    def execute_fraction(self) -> float:
        tot = self.total_pj_per_clock
        return self.execute_pj_per_clock / tot if tot > 0 else 0.0

    @property
    def power_mw(self) -> float:
        return self.total_pj_per_clock * self.clock_ghz

    @property
    def derived_pj_per_mac(self) -> float:
        return (self.total_pj_per_clock / self.native_macs_per_clock
                if self.native_macs_per_clock > 0 else 0.0)

    def by_category(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in self.components:
            out[c.category.value] = out.get(c.category.value, 0.0) + c.per_clock_pj
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "process_nm": self.process_nm,
            "clock_ghz": self.clock_ghz,
            "native_macs_per_clock": self.native_macs_per_clock,
            "total_pj_per_clock": self.total_pj_per_clock,
            "execute_pj_per_clock": self.execute_pj_per_clock,
            "execute_fraction": self.execute_fraction,
            "power_mw": self.power_mw,
            "derived_pj_per_mac": self.derived_pj_per_mac,
            "components": [c.to_dict() for c in self.components],
            "by_category": self.by_category(),
            "color": self.color,
            "notes": self.notes,
        }


@dataclass
class SocComposition:
    """
    Composes a SoC by replicating a building block N times plus a
    process-dependent overhead (memory controller, L3, IO, etc.).
    """
    block: BuildingBlock
    block_count: int
    utilization: float             # fraction of blocks active per clock
    overhead_mw: float             # fixed SoC overhead at this freq

    @property
    def active_blocks(self) -> float:
        return self.block_count * self.utilization

    @property
    def compute_power_mw(self) -> float:
        return self.active_blocks * self.block.power_mw

    @property
    def total_power_w(self) -> float:
        return (self.compute_power_mw + self.overhead_mw) / 1000.0

    @property
    def peak_tops_int8(self) -> float:
        """Peak INT8 TOPS assuming 2 ops per MAC (multiply + add)."""
        macs_per_sec = (self.block_count * self.block.native_macs_per_clock
                        * self.block.clock_ghz * 1e9)
        return macs_per_sec * 2 / 1e12

    @property
    def sustained_tops_int8(self) -> float:
        return self.peak_tops_int8 * self.utilization

    @property
    def sustained_tops_per_watt(self) -> float:
        tot_w = self.total_power_w
        return self.sustained_tops_int8 / tot_w if tot_w > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_name": self.block.name,
            "block_count": self.block_count,
            "utilization": self.utilization,
            "overhead_mw": self.overhead_mw,
            "compute_power_mw": self.compute_power_mw,
            "total_power_w": self.total_power_w,
            "peak_tops_int8": self.peak_tops_int8,
            "sustained_tops_int8": self.sustained_tops_int8,
            "sustained_tops_per_watt": self.sustained_tops_per_watt,
        }


# ----------------------------------------------------------------------
# NVIDIA Ampere SM as deployed on Orin AGX (GA10x @ 8 nm, ~0.65 GHz)
# ----------------------------------------------------------------------

def build_nvidia_sm_building_block() -> BuildingBlock:
    """
    One Ampere SM running an INT8 HMMA workload at steady state.
    Values calibrated against the published 15-30 W Orin AGX SoC TDP
    and the 16-SM layout; per-SM ~0.5 W at sustained load.
    """
    components = [
        EngineComponent(
            name="Register file",
            category=StructureCategory.MEMORY,
            count=1,
            size_or_spec="64K x 32b = 256 KB, 4 banks (16 KB each)",
            per_clock_pj=150.0,
            activity_note=(
                "~4 warp-wide operand reads + ~2 writes per clock "
                "across 4 sub-partitions"
            ),
            citation=(
                "Warp-wide RF access ~35 pJ at 8 nm (32 threads x "
                "operand bytes x 0.014 pJ/byte, Horowitz-scaled)."
            ),
        ),
        EngineComponent(
            name="Instruction pipeline (L0 I$ + decode + IB)",
            category=StructureCategory.FETCH,
            count=4,
            size_or_spec="4 sub-partitions, 1 inst/clk each",
            per_clock_pj=60.0,
            activity_note="4 schedulers issue 1 instruction/clock each",
            citation=(
                "L0 I-cache ~8 pJ/fetch x 4, decoder ~4 pJ x 4, "
                "instruction-buffer latches ~12 pJ/clock."
            ),
        ),
        EngineComponent(
            name="Warp scheduler + scoreboard",
            category=StructureCategory.SCHEDULE,
            count=4,
            size_or_spec="1 scheduler per SM partition",
            per_clock_pj=35.0,
            activity_note="Ready-queue check + dependency tracking per issue",
            citation=(
                "Ampere SM scheduler ~6 pJ + scoreboard ~2.5 pJ per "
                "issued instruction, x 4 schedulers."
            ),
        ),
        EngineComponent(
            name="Operand collector + bypass network",
            category=StructureCategory.REGISTER,
            count=4,
            size_or_spec="1 collector per sub-partition, 4-bank xbar",
            per_clock_pj=80.0,
            activity_note="4 issues/clock each pass through collector",
            citation=(
                "~20 pJ per issue through the 4-bank operand crossbar "
                "(static network energy dominates)."
            ),
        ),
        EngineComponent(
            name="CUDA cores (FP32/INT32)",
            category=StructureCategory.EXECUTE,
            count=128,
            size_or_spec="128 lanes = 4 partitions x 32",
            per_clock_pj=15.0,
            activity_note="Idle / clock-gated during HMMA workload",
            citation=(
                "Clock-tree + residual leakage; ~5% duty on dense "
                "HMMA kernels."
            ),
        ),
        EngineComponent(
            name="Special Function Units (SFUs)",
            category=StructureCategory.EXECUTE,
            count=16,
            size_or_spec="4 SFUs per SM partition",
            per_clock_pj=10.0,
            activity_note="Mostly idle on dense GEMM; used for activations",
            citation="Low activity on pure GEMM; clock-tree dominates.",
        ),
        EngineComponent(
            name="Tensor Cores",
            category=StructureCategory.EXECUTE,
            count=4,
            size_or_spec="1 TC per SM partition, 16x16x16 HMMA",
            per_clock_pj=370.0,
            activity_note="1024 MACs/clock/TC x 4 = 4096 MACs/clock",
            citation=(
                "4 TCs x 1024 MACs/clock x ~0.08 pJ bare-MAC + "
                "~40 pJ/clock accumulator-reduction tree; published "
                "Ampere TC energy."
            ),
        ),
        EngineComponent(
            name="Clock tree + leakage",
            category=StructureCategory.CONTROL,
            count=1,
            size_or_spec="SM-wide (~5 mm^2 at 8 nm)",
            per_clock_pj=60.0,
            activity_note="Always-on; H-tree distribution + static power",
            citation="~7% of dynamic engine power at 8 nm HPM.",
        ),
    ]

    return BuildingBlock(
        name="NVIDIA Streaming Multiprocessor (Ampere GA10x)",
        process_nm=8,
        clock_ghz=0.65,
        native_macs_per_clock=4096,  # 4 TCs x 1024 MACs/clock
        components=components,
        color="#5b8ff9",
        notes=(
            "SM-level budget during INT8 HMMA. Excludes L2 cache, "
            "memory controller, and DRAM (those are SoC-level). At "
            "0.65 GHz sustained, ~0.51 W/SM; Orin AGX deploys 16 SMs, "
            "giving ~8 W just from SMs in the 15-30 W SoC envelope."
        ),
    )


# ----------------------------------------------------------------------
# KPU compute tile as deployed on T128 (16 nm, 1.0 GHz sustained)
# ----------------------------------------------------------------------

def build_kpu_tile_building_block() -> BuildingBlock:
    """
    One KPU compute tile: a 2D mesh of 576 FMAs (24x24), fed by a
    tile-local L1 scratchpad at its edges. Values consistent with the
    per-MAC view in microarch_accounting.py:
    576 MACs/clock x 0.213 pJ/MAC (fabric) + overheads.
    """
    PE_ROWS = 24
    PE_COLS = 24
    PE_COUNT = PE_ROWS * PE_COLS  # 576
    # Per-PE per-clock fabric cost derived from microarch_accounting:
    # FMA 0.10 + regs 0.05 + accum 0.03 + forward 0.01 + token 0.008
    # + clock 0.015 = 0.213 pJ/PE/clock.
    pe_fabric_pj_per_clock = 0.213 * PE_COUNT  # ~122.7 pJ/clock

    components = [
        EngineComponent(
            name="L1 scratchpad (tile-local)",
            category=StructureCategory.MEMORY,
            count=1,
            size_or_spec="~64 KB SRAM, banked for edge feed",
            per_clock_pj=20.0,
            activity_note=(
                f"{2 * PE_ROWS} edge byte-reads/clock (A-edge + B-edge) "
                f"plus periodic output writeback"
            ),
            citation=(
                "~0.3 pJ/byte at 16 nm for a 64 KB L1 array; 48 edge "
                "reads x 0.3 pJ + ~6 pJ result writeback."
            ),
        ),
        EngineComponent(
            name="2D mesh of FMAs",
            category=StructureCategory.EXECUTE,
            count=PE_COUNT,
            size_or_spec=f"{PE_ROWS}x{PE_COLS} PE grid, 1 MAC/PE/clock",
            per_clock_pj=pe_fabric_pj_per_clock,
            activity_note=f"All {PE_COUNT} PEs fire in steady-state wavefront",
            citation=(
                "0.213 pJ/PE/clock = FMA 0.10 + 2-operand reg read 0.05 "
                "+ accumulator 0.030 + neighbor forward 0.010 + token "
                "match 0.008 + clock latch 0.015. See "
                "microarch_accounting.build_kpu_tile_accounting()."
            ),
        ),
        EngineComponent(
            name="Token / coordinate generators (mesh edges)",
            category=StructureCategory.CONTROL,
            count=2 * PE_ROWS,
            size_or_spec="Injectors at A-edge + B-edge; a few gates each",
            per_clock_pj=5.0,
            activity_note="Generate + dispatch one token per edge per clock",
            citation=(
                "Edge-only control: 48 injectors x ~0.1 pJ per token "
                "generation at 16 nm."
            ),
        ),
        EngineComponent(
            name="Clock tree + leakage",
            category=StructureCategory.CONTROL,
            count=1,
            size_or_spec="Tile-wide (~1 mm^2 at 16 nm)",
            per_clock_pj=20.0,
            activity_note="Simple H-tree distribution; no OOO machinery",
            citation="~15% of dynamic fabric power at 16 nm.",
        ),
    ]

    return BuildingBlock(
        name="KPU Compute Tile (24x24 FMA mesh)",
        process_nm=16,
        clock_ghz=1.0,
        native_macs_per_clock=PE_COUNT,  # 576
        components=components,
        color="#3fc98a",
        notes=(
            "Tile-level budget in steady-state mesh throughput. "
            "Excludes L3 scratchpad and memory controller (those are "
            "SoC-level). At 1.0 GHz, ~0.168 W/tile; T128 deploys 128 "
            "tiles, giving ~21 W at 100% fabric duty (implies ~55% "
            "realistic utilization inside the 12 W TDP envelope)."
        ),
    )


# ----------------------------------------------------------------------
# Default SoC compositions (illustrative, used in the HTML page)
# ----------------------------------------------------------------------

def default_soc_compositions() -> List[SocComposition]:
    sm = build_nvidia_sm_building_block()
    tile = build_kpu_tile_building_block()
    return [
        SocComposition(
            block=sm, block_count=16, utilization=0.50, overhead_mw=7000.0,
        ),
        SocComposition(
            block=tile, block_count=128, utilization=0.55, overhead_mw=2500.0,
        ),
        SocComposition(
            block=tile, block_count=64, utilization=0.55, overhead_mw=1500.0,
        ),
        SocComposition(
            block=tile, block_count=256, utilization=0.55, overhead_mw=4500.0,
        ),
    ]


@dataclass
class BuildingBlockReport:
    blocks: List[BuildingBlock]
    socs: List[SocComposition]
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "socs": [s.to_dict() for s in self.socs],
            "generated_at": self.generated_at,
        }


def build_default_report() -> BuildingBlockReport:
    from datetime import datetime, timezone
    return BuildingBlockReport(
        blocks=[
            build_nvidia_sm_building_block(),
            build_kpu_tile_building_block(),
        ],
        socs=default_soc_compositions(),
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


def _render_chart_js(report: BuildingBlockReport) -> str:
    bds = [b.to_dict() for b in report.blocks]
    all_cats: List[str] = []
    for b in bds:
        for cat in b["by_category"].keys():
            if cat not in all_cats:
                all_cats.append(cat)

    traces = []
    for cat in all_cats:
        ys = [b["name"] for b in bds]
        xs = [b["by_category"].get(cat, 0.0) for b in bds]
        traces.append({
            "type": "bar", "orientation": "h",
            "name": cat, "y": ys, "x": xs,
            "marker": {"color": _CATEGORY_COLORS.get(cat, "#888")},
            "text": [f"{v:.0f}" if v > 1 else "" for v in xs],
            "textposition": "inside",
        })
    chart_per_clock = {
        "data": traces,
        "layout": {
            "title": "Engine energy per clock by component category (pJ/clock)",
            "xaxis": {"title": "pJ / clock"},
            "yaxis": {"title": "", "automargin": True},
            "barmode": "stack",
            "margin": {"t": 50, "b": 50, "l": 280, "r": 20},
            "legend": {"orientation": "h", "y": -0.2},
        },
    }

    # SoC composition bar: total power per SoC
    soc_labels = [f"{s.block.name.split('(')[0].strip()} x{s.block_count} "
                  f"@ U={s.utilization:.0%}" for s in report.socs]
    soc_power = [s.total_power_w for s in report.socs]
    soc_tops = [s.sustained_tops_int8 for s in report.socs]
    soc_eff = [s.sustained_tops_per_watt for s in report.socs]
    chart_soc = {
        "data": [
            {
                "type": "bar", "name": "Power (W)",
                "x": soc_labels, "y": soc_power,
                "marker": {"color": "#586374"},
                "text": [f"{p:.1f} W" for p in soc_power],
                "textposition": "outside",
            },
            {
                "type": "bar", "name": "Sustained TOPS (INT8)",
                "x": soc_labels, "y": soc_tops,
                "yaxis": "y2",
                "marker": {"color": "#3fc98a"},
                "text": [f"{t:.0f}" for t in soc_tops],
                "textposition": "outside",
            },
        ],
        "layout": {
            "title": "SoC composition: power & sustained INT8 TOPS",
            "xaxis": {"title": "Configuration"},
            "yaxis": {"title": "Power (W)", "side": "left"},
            "yaxis2": {"title": "Sustained TOPS",
                       "overlaying": "y", "side": "right"},
            "barmode": "group",
            "margin": {"t": 50, "b": 120, "l": 60, "r": 60},
        },
    }

    payload = {
        "chart_per_clock": chart_per_clock,
        "chart_soc": chart_soc,
    }
    return (
        f"const CHARTS = {json.dumps(payload)};\n"
        "for (const [id, spec] of Object.entries(CHARTS)) {\n"
        "  Plotly.newPlot(id, spec.data, spec.layout, "
        "{displayModeBar: false, responsive: true});\n"
        "}\n"
    )


def _render_block_table(b: BuildingBlock) -> str:
    rows = []
    tot = b.total_pj_per_clock
    for c in b.components:
        pct = (c.per_clock_pj / tot * 100.0) if tot > 0 else 0.0
        rows.append(
            f'<tr>'
            f'<td class="name">{html.escape(c.name)}</td>'
            f'<td><span class="cat" style="background:'
            f'{_CATEGORY_COLORS.get(c.category.value, "#888")};'
            f'">{html.escape(c.category.value)}</span></td>'
            f'<td class="num">{c.count}</td>'
            f'<td class="spec">{html.escape(c.size_or_spec)}</td>'
            f'<td class="num"><strong>{c.per_clock_pj:.1f}</strong></td>'
            f'<td class="num">{pct:.1f}%</td>'
            f'<td class="src">{html.escape(c.activity_note)}<br/>'
            f'<em>{html.escape(c.citation)}</em></td>'
            f'</tr>'
        )
    exec_pct = b.execute_fraction * 100.0
    rows.append(
        f'<tr class="total-row">'
        f'<td colspan="4"><strong>Engine total per clock</strong></td>'
        f'<td class="num"><strong>{tot:.1f} pJ/clock</strong></td>'
        f'<td class="num"><strong>100%</strong></td>'
        f'<td class="src">'
        f'Execute fraction: <strong>{exec_pct:.1f}%</strong>. '
        f'Power at {b.clock_ghz:.2f} GHz: <strong>{b.power_mw:.0f} mW</strong>. '
        f'Derived pJ/MAC (cross-check): <strong>'
        f'{b.derived_pj_per_mac:.3f}</strong> (native throughput '
        f'{b.native_macs_per_clock:,} MACs/clock).'
        f'</td>'
        f'</tr>'
    )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>Component</th><th>Category</th><th>Count</th>'
        '<th>Size / spec</th>'
        '<th>pJ / clock</th><th>% of engine</th>'
        '<th>Activity assumption / source</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def _render_block_header(b: BuildingBlock) -> str:
    return f"""
<section class="block-header"
         style="border-left: 6px solid {b.color};">
  <h3>{html.escape(b.name)}</h3>
  <div class="meta">
    <strong>Process:</strong> {b.process_nm} nm
    | <strong>Clock:</strong> {b.clock_ghz:.2f} GHz
    | <strong>Native throughput:</strong>
        {b.native_macs_per_clock:,} MACs/clock
    | <strong>Power:</strong> {b.power_mw:.0f} mW
    | <strong>pJ/MAC (cross-check):</strong>
        {b.derived_pj_per_mac:.3f}
  </div>
  <p class="block-notes">{html.escape(b.notes)}</p>
</section>
"""


def _render_soc_table(report: BuildingBlockReport) -> str:
    rows = []
    for s in report.socs:
        rows.append(
            f'<tr>'
            f'<td>{html.escape(s.block.name)}</td>'
            f'<td class="num">{s.block_count}</td>'
            f'<td class="num">{s.utilization:.0%}</td>'
            f'<td class="num">{s.block.clock_ghz:.2f} GHz</td>'
            f'<td class="num">{s.compute_power_mw/1000:.2f} W</td>'
            f'<td class="num">{s.overhead_mw/1000:.2f} W</td>'
            f'<td class="num"><strong>{s.total_power_w:.2f} W</strong></td>'
            f'<td class="num">{s.peak_tops_int8:.1f}</td>'
            f'<td class="num">{s.sustained_tops_int8:.1f}</td>'
            f'<td class="num"><strong>{s.sustained_tops_per_watt:.2f}</strong></td>'
            f'</tr>'
        )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>Building block</th>'
        '<th># blocks</th><th>Utilization</th><th>Clock</th>'
        '<th>Compute power</th><th>SoC overhead</th>'
        '<th>Total SoC power</th>'
        '<th>Peak TOPS</th><th>Sustained TOPS</th>'
        '<th>Sustained TOPS/W</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_building_block_page(
    report: BuildingBlockReport, repo_root: Path,
) -> str:
    assets = _load_logo(repo_root)
    header = _render_brand_header(
        assets,
        "Building-block energy accounting (per clock)",
        f"Engine-level budget for SM vs. KPU tile "
        f"| generated {report.generated_at}",
    )
    footer = _render_brand_footer("microarch-model-delivery-plan.md")

    extra_css = """
table.blocks { width: 100%; border-collapse: collapse; background: #fff;
               margin-bottom: 18px; font-size: 13px; }
table.blocks th, table.blocks td { padding: 7px 10px;
                                     border-bottom: 1px solid #e3e6eb;
                                     vertical-align: top; }
table.blocks th { font-size: 11px; text-transform: uppercase;
                  color: #586374; background: #f3f5f8; text-align: left; }
table.blocks td.name { font-weight: 600; color: #0a2540; }
table.blocks td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.blocks td.spec { color: #3a4452; font-size: 12px; }
table.blocks td.src { color: #586374; font-size: 11px; max-width: 320px; }
table.blocks tr.total-row td { background: #eef6ea;
                                border-top: 2px solid #3fc98a;
                                border-bottom: 2px solid #3fc98a;
                                padding: 12px 10px; font-size: 14px; }
span.cat { display: inline-block; padding: 2px 8px; border-radius: 3px;
           color: white; font-size: 11px; font-weight: 600; }
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

    # Cross-validation between per-clock and per-MAC views
    from graphs.reporting.microarch_accounting import build_default_report as mar
    mar_report = mar()
    mar_blocks = {b.building_block: b.total_pj_per_mac for b in mar_report.blocks}
    rows_cv = []
    for b in report.blocks:
        per_clock = b.total_pj_per_clock
        derived = b.derived_pj_per_mac
        # Match on substring - names differ slightly
        matched_name = None
        matched_val = None
        for mname, mval in mar_blocks.items():
            if ("KPU" in b.name and "KPU" in mname) or (
                "Multiprocessor" in b.name and "Multiprocessor" in mname
            ):
                matched_name = mname
                matched_val = mval
                break
        if matched_val is None:
            continue
        delta = (abs(derived - matched_val) / matched_val * 100.0
                 if matched_val > 0 else 0.0)
        rows_cv.append(
            f'<tr>'
            f'<td>{html.escape(b.name)}</td>'
            f'<td class="num">{per_clock:.1f}</td>'
            f'<td class="num">{b.native_macs_per_clock:,}</td>'
            f'<td class="num"><strong>{derived:.3f}</strong></td>'
            f'<td class="num">{matched_val:.3f}</td>'
            f'<td class="num">{delta:.0f}%</td>'
            f'</tr>'
        )
    cv_table = (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>Engine</th><th>Per-clock total (pJ)</th>'
        '<th>Native MACs/clock</th>'
        '<th>Derived pJ/MAC</th>'
        '<th>Independent pJ/MAC (microarch_accounting)</th>'
        '<th>Delta</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_cv)}</tbody>'
        '</table>'
    )

    blocks_html = "\n".join(
        _render_block_header(b) + _render_block_table(b)
        for b in report.blocks
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Building-block energy accounting</title>
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
    <h2>Engine-level energy budget: SM vs. KPU tile</h2>
    <div class="meta">Per-clock energy of every major component as it
      is deployed on the SoC. Drives SoC composition and super-cluster
      designs. Complements the
      <a href="microarch_accounting.html">per-MAC view</a>; the two
      reconcile via <code>pJ/MAC = pJ_per_clock /
      MACs_per_clock</code>.</div>
  </section>

  <section class="chart-section">
    <h3>Engine energy per clock by component category</h3>
    <p class="chart-desc">Absolute pJ/clock at native process and
      steady-state clock frequency. No per-MAC normalization - this is
      the raw silicon budget the SoC must supply every cycle.</p>
    <div id="chart_per_clock" class="plot"></div>
  </section>

  {blocks_html}

  <section class="method-note">
    <h3 style="margin-top:0;">SoC composition (illustrative)</h3>
    <p>Aggregating building blocks into a SoC: <strong>total power =
      count x utilization x power-per-block + SoC overhead</strong>.
      The overhead term covers L2/L3 caches, memory controller,
      NVLink/PCIe, display, and other non-compute IP. Utilization is
      the fraction of blocks active per clock on realistic workloads.</p>
    {_render_soc_table(report)}
    <div id="chart_soc" class="plot" style="margin-top: 12px;"></div>
  </section>

  <section class="method-note">
    <h3 style="margin-top:0;">Cross-validation with per-MAC view</h3>
    <p>The derived pJ/MAC from this per-clock view must agree with the
      independent per-MAC accounting in
      <code>microarch_accounting.py</code>. Agreement validates that
      both analytical models converge on the same silicon cost.</p>
    {cv_table}
  </section>

  <section class="method-note">
    <strong>Which view for which question?</strong>
    <ul>
      <li><strong>pJ/MAC view</strong> (microarch_accounting.py):
        performance under a fixed power budget. "At 12 W, how much
        compute will this engine deliver on workload X?"</li>
      <li><strong>pJ/clock view</strong> (this page): SoC composition
        and super-cluster design. "If I deploy N engines at F GHz with
        U utilization, what is the total power and sustained TOPS?"</li>
    </ul>
  </section>
</main>
{footer}
<script>
{_render_chart_js(report)}
</script>
</body>
</html>
"""

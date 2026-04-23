"""
Silicon speed-of-light analysis.

This module answers a blunt question: on a standard-sized die at a
given process node, with a specific ALU as the only thing on the die,
what is the theoretical peak performance and how does it compare to
shipping products?

The point is to expose the gap between what silicon can do and what
incumbents ship. Investors and customers often anchor on shipping
products (Orin AGX, H100, etc.) as if those set the upper bound.
They do not. An Orin AGX delivers ~275 INT8 TOPS at 30 W because
DVFS clamps the SM clock to fit the envelope; the silicon itself
can run the same SMs 2-3x faster if you give it the power budget.
This analysis makes that headroom quantitative.

Key identity (from which everything else follows):

    TOPS / W = 2 * ops_per_MAC / pJ_per_MAC  (clock- and count-independent)

    => a 0.05 pJ/MAC INT8 ALU has a silicon ceiling of 2000/0.05 = 40 TOPS/W.
       no amount of clocking, scaling, or packaging changes that.

The TDP-constrained view instead asks: given a fixed die full of
these ALUs, at what clock does the die power equal the TDP budget?

    clock_for_tdp(TDP) = TDP / (num_alus * pJ_per_MAC * 1e-3)    [GHz]
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


# Representative TDP envelopes across embedded-through-datacenter
# deployments. Used to sweep clock feasibility.
DEFAULT_TDP_TARGETS_W: tuple = (5.0, 15.0, 30.0, 60.0, 150.0, 300.0)

# Default die area used for the "on a 250 mm^2 die" column. 250 mm^2
# is representative of a mid-range embodied-AI accelerator - smaller
# than H100 (814 mm^2) but larger than a Jetson module SoC (~180 mm^2).
DEFAULT_DIE_AREA_MM2: float = 250.0

# Silicon-capability clock at 8 nm used for the default peak-TOPS
# column. GA106 desktop ships at 1.78 GHz boost, GA107 mobile
# sustains 1.35 GHz; 1.5 GHz is a conservative midpoint and matches
# the KPU tile's target clock.
DEFAULT_SILICON_CLOCK_GHZ: float = 1.5


@dataclass
class BareALU:
    """
    A single multiply-add datapath - the irreducible compute element.

    `area_mm2` and `transistor_count_m` are for ONE ALU (not the whole
    array). `pj_per_clock` is the steady-state dynamic energy per op
    at the process node.
    """
    name: str
    precision: str                # "INT8", "FP32", "BF16", ...
    process_nm: int
    area_mm2: float               # area of this single ALU
    transistor_count_k: float     # thousands of transistors
    pj_per_clock: float           # dynamic pJ per activation
    ops_per_mac: int = 2          # 1 FMA = 2 ops (multiply + add)
    citation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "precision": self.precision,
            "process_nm": self.process_nm,
            "area_mm2": self.area_mm2,
            "transistor_count_k": self.transistor_count_k,
            "pj_per_clock": self.pj_per_clock,
            "ops_per_mac": self.ops_per_mac,
            "tops_per_watt_ceiling": self.tops_per_watt_ceiling,
            "citation": self.citation,
        }

    @property
    def tops_per_watt_ceiling(self) -> float:
        """
        Silicon TOPS/W ceiling for this ALU. Clock- and count-
        independent because both TOPS and power scale linearly with
        (count x clock).

        Derivation: at clock f and N ALUs:
          TOPS = N x f x ops_per_mac x 1e-3      (with f in GHz)
          W    = N x pJ_per_clock x f x 1e-3
          TOPS/W = ops_per_mac / pJ_per_clock    (N and f cancel)

        For a 0.050 pJ/clock INT8 MAC: 2 / 0.050 = 40 TOPS/W.
        """
        return self.ops_per_mac / self.pj_per_clock


@dataclass
class SoLAnalysis:
    """
    Speed-of-light analysis: one ALU tiled across a die of
    `die_area_mm2`, swept across clocks and TDPs.
    """
    alu: BareALU
    die_area_mm2: float = DEFAULT_DIE_AREA_MM2
    silicon_clock_ghz: float = DEFAULT_SILICON_CLOCK_GHZ
    tdp_targets_w: tuple = DEFAULT_TDP_TARGETS_W

    @property
    def num_alus(self) -> int:
        return int(self.die_area_mm2 / self.alu.area_mm2)

    @property
    def die_transistor_count_m(self) -> float:
        """Millions of transistors on the die (ALUs only)."""
        return self.num_alus * self.alu.transistor_count_k / 1000.0

    @property
    def die_energy_pj_per_clock(self) -> float:
        """Dynamic energy per clock when every ALU fires."""
        return self.num_alus * self.alu.pj_per_clock

    def peak_tops(self, clock_ghz: float) -> float:
        """Peak TOPS at a given clock (every ALU firing every cycle)."""
        macs_per_sec = self.num_alus * clock_ghz * 1e9
        return macs_per_sec * self.alu.ops_per_mac / 1e12

    def die_power_w(self, clock_ghz: float) -> float:
        """Dynamic die power in watts at a given clock."""
        return self.die_energy_pj_per_clock * clock_ghz * 1e-3

    def clock_for_tdp(self, tdp_w: float) -> float:
        """Clock (GHz) at which die power equals the given TDP."""
        denom = self.num_alus * self.alu.pj_per_clock * 1e-3
        return tdp_w / denom if denom > 0 else 0.0

    def tdp_sweep(self) -> List[Dict[str, float]]:
        """One row per TDP target with the clock it sustains and the
        peak TOPS at that clock. Rows where the required clock
        exceeds `silicon_clock_ghz` indicate the die is silicon-
        limited, not thermally limited."""
        out: List[Dict[str, float]] = []
        for tdp in self.tdp_targets_w:
            f = self.clock_for_tdp(tdp)
            silicon_limited = f > self.silicon_clock_ghz
            effective_f = min(f, self.silicon_clock_ghz)
            out.append({
                "tdp_w": tdp,
                "clock_for_tdp_ghz": f,
                "effective_clock_ghz": effective_f,
                "silicon_limited": silicon_limited,
                "peak_tops": self.peak_tops(effective_f),
                "effective_power_w": self.die_power_w(effective_f),
            })
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alu": self.alu.to_dict(),
            "die_area_mm2": self.die_area_mm2,
            "num_alus": self.num_alus,
            "die_transistor_count_m": self.die_transistor_count_m,
            "die_energy_pj_per_clock": self.die_energy_pj_per_clock,
            "silicon_clock_ghz": self.silicon_clock_ghz,
            "peak_tops_at_silicon_clock": self.peak_tops(
                self.silicon_clock_ghz
            ),
            "die_power_at_silicon_clock_w": self.die_power_w(
                self.silicon_clock_ghz
            ),
            "tops_per_watt_ceiling": self.alu.tops_per_watt_ceiling,
            "tdp_sweep": self.tdp_sweep(),
        }


@dataclass
class ProductReference:
    """A shipping-product reference point for the gap-to-SoL table."""
    name: str
    process_nm: int
    die_area_mm2: float
    peak_int8_tops: float
    tdp_w: float

    @property
    def tops_per_watt(self) -> float:
        return self.peak_int8_tops / self.tdp_w if self.tdp_w > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "process_nm": self.process_nm,
            "die_area_mm2": self.die_area_mm2,
            "peak_int8_tops": self.peak_int8_tops,
            "tdp_w": self.tdp_w,
            "tops_per_watt": self.tops_per_watt,
        }


# ----------------------------------------------------------------------
# Default catalog
# ----------------------------------------------------------------------
#
# Per-ALU values derived from microarch_accounting.py / building_block_
# energy.py at 8 nm:
#
#   KPU PE FMA (INT8):
#     12.3 M transistors across 1024 PEs = 12.0 K trans/PE
#     mesh area 0.19 mm^2 across 1024 PEs = 0.000185 mm^2/PE
#     0.050 pJ/clock per PE (bare FMA only)
#
#   Ampere TC bare MAC (INT8):
#     40 M transistors across 4096 MACs = 9.8 K trans/MAC
#     TC MAC-array area 0.50 mm^2 across 4096 = 0.000122 mm^2/MAC
#     0.080 pJ/clock per MAC (bare FMA only; operand broadcast + accum
#     charged separately)
#
#   Ampere CUDA-core FMA (FP32):
#     10 M transistors across 128 lanes = 78 K trans/lane
#     area 0.22 mm^2 / 128 lanes = 0.00172 mm^2/lane
#     2.5 pJ/clock per FMA at full activity
# ----------------------------------------------------------------------


def default_alu_catalog() -> List[BareALU]:
    return [
        BareALU(
            name="KPU PE FMA (INT8)",
            precision="INT8",
            process_nm=8,
            area_mm2=0.000185,
            transistor_count_k=12.0,
            pj_per_clock=0.050,
            ops_per_mac=2,
            citation=(
                "microarch_accounting.build_kpu_tile_accounting(): "
                "FMA unit row, 12 M transistors across 1024 PEs, "
                "0.050 pJ at 8 nm (8 FA x 0.005 pJ + carry-tree)."
            ),
        ),
        BareALU(
            name="Ampere TC bare MAC (INT8)",
            precision="INT8",
            process_nm=8,
            area_mm2=0.000122,
            transistor_count_k=9.8,
            pj_per_clock=0.080,
            ops_per_mac=2,
            citation=(
                "microarch_accounting.build_nvidia_sm_accounting(): "
                "Tensor Core bare MAC row, 40 M transistors across "
                "4 TCs x 1024 MACs = 4096 MACs, 0.080 pJ (~10 FA x "
                "0.005 + operand mux within TC)."
            ),
        ),
        BareALU(
            name="Ampere CUDA-core FMA (FP32)",
            precision="FP32",
            process_nm=8,
            area_mm2=0.00172,
            transistor_count_k=78.0,
            pj_per_clock=2.5,
            ops_per_mac=2,
            citation=(
                "building_block_energy.build_nvidia_sm_cuda_path_"
                "building_block(): CUDA-core row, 10 M transistors "
                "across 128 FP32 FMA lanes, 320 pJ/clock total at "
                "full activity. Per-lane: 78 K transistors, 2.5 pJ "
                "(~23 FA for 23-bit mantissa multiply + 24-bit add)."
            ),
        ),
    ]


def default_product_references() -> List[ProductReference]:
    """Shipping products at known TDPs for gap-to-SoL comparison.

    All figures are DENSE INT8 unless labelled sparse. NVIDIA's
    "275 TOPS" headline for Jetson AGX Orin is sparse INT8 at
    MAXN (~60 W) summed across GPU + DLA + PVA; dividing 275 by a
    lower TDP (e.g., 30 W) double-counts the sparsity factor and
    the power-mode. Dense numbers below are derived from NVIDIA's
    published specs and scale linearly with the clock that the
    sustained-TDP envelope permits.
    """
    return [
        # Orin AGX 64GB, dense INT8 across GPU + DLA + PVA engines.
        # TOPS/W ~= 2.3 across all power modes (dynamic efficiency
        # is clock-independent; lower modes trade TOPS for watts
        # proportionally).
        ProductReference(
            name="Jetson AGX Orin MAXN (60 W, dense INT8)",
            process_nm=8,
            die_area_mm2=180.0,   # full SoC
            peak_int8_tops=137.0,
            tdp_w=60.0,
        ),
        ProductReference(
            name="Jetson AGX Orin (50 W, dense INT8)",
            process_nm=8,
            die_area_mm2=180.0,
            peak_int8_tops=105.0,
            tdp_w=50.0,
        ),
        ProductReference(
            name="Jetson AGX Orin (30 W, dense INT8)",
            process_nm=8,
            die_area_mm2=180.0,
            peak_int8_tops=68.0,
            tdp_w=30.0,
        ),
        ProductReference(
            name="Jetson AGX Orin (15 W, dense INT8)",
            process_nm=8,
            die_area_mm2=180.0,
            peak_int8_tops=34.0,
            tdp_w=15.0,
        ),
        # Same silicon at MAXN with structured sparsity: ~2x the
        # dense TOPS, same 60 W envelope. Included so the sparsity
        # marketing trick is visible.
        ProductReference(
            name="Jetson AGX Orin MAXN (60 W, sparse INT8 marketing)",
            process_nm=8,
            die_area_mm2=180.0,
            peak_int8_tops=275.0,
            tdp_w=60.0,
        ),
        ProductReference(
            name="H100 SXM5 (700 W, dense INT8)",
            process_nm=4,
            die_area_mm2=814.0,
            peak_int8_tops=1979.0,
            tdp_w=700.0,
        ),
        ProductReference(
            name="KPU T128 (hypothetical, 12 W)",
            process_nm=8,
            die_area_mm2=80.0,   # 128 x 0.29 mm^2 tile + fabric
            peak_int8_tops=262.0,
            tdp_w=12.0,
        ),
    ]


@dataclass
class SoLReport:
    alus: List[BareALU]
    analyses: List[SoLAnalysis]
    products: List[ProductReference]
    die_area_mm2: float = DEFAULT_DIE_AREA_MM2
    silicon_clock_ghz: float = DEFAULT_SILICON_CLOCK_GHZ
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alus": [a.to_dict() for a in self.alus],
            "analyses": [a.to_dict() for a in self.analyses],
            "products": [p.to_dict() for p in self.products],
            "die_area_mm2": self.die_area_mm2,
            "silicon_clock_ghz": self.silicon_clock_ghz,
            "generated_at": self.generated_at,
        }


def build_default_sol_report(
    die_area_mm2: float = DEFAULT_DIE_AREA_MM2,
    silicon_clock_ghz: float = DEFAULT_SILICON_CLOCK_GHZ,
) -> SoLReport:
    from datetime import datetime, timezone
    alus = default_alu_catalog()
    analyses = [
        SoLAnalysis(
            alu=a,
            die_area_mm2=die_area_mm2,
            silicon_clock_ghz=silicon_clock_ghz,
        )
        for a in alus
    ]
    return SoLReport(
        alus=alus,
        analyses=analyses,
        products=default_product_references(),
        die_area_mm2=die_area_mm2,
        silicon_clock_ghz=silicon_clock_ghz,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ----------------------------------------------------------------------
# HTML rendering helpers (consumed by building_block_energy.py)
# ----------------------------------------------------------------------


def render_alu_catalog_table(alus: List[BareALU]) -> str:
    rows = []
    for alu in alus:
        rows.append(
            f'<tr>'
            f'<td class="name">{html.escape(alu.name)}</td>'
            f'<td>{html.escape(alu.precision)}</td>'
            f'<td class="num">{alu.process_nm}</td>'
            f'<td class="num">{alu.area_mm2*1e6:.0f} μm²</td>'
            f'<td class="num">{alu.transistor_count_k:.1f}</td>'
            f'<td class="num"><strong>{alu.pj_per_clock:.3f}</strong></td>'
            f'<td class="num"><strong>'
            f'{alu.tops_per_watt_ceiling:.0f}</strong></td>'
            f'<td class="src"><em>{html.escape(alu.citation)}</em></td>'
            f'</tr>'
        )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>Bare ALU</th><th>Precision</th><th>Process (nm)</th>'
        '<th>Area / ALU</th><th>Trans. / ALU (K)</th>'
        '<th>pJ / clock</th>'
        '<th>TOPS / W ceiling</th>'
        '<th>Derivation</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_sol_summary_table(
    analyses: List[SoLAnalysis], die_area_mm2: float,
) -> str:
    rows = []
    for a in analyses:
        f = a.silicon_clock_ghz
        rows.append(
            f'<tr>'
            f'<td class="name">{html.escape(a.alu.name)}</td>'
            f'<td class="num">{a.num_alus:,}</td>'
            f'<td class="num">{a.die_transistor_count_m:.0f}</td>'
            f'<td class="num">{a.die_energy_pj_per_clock:.0f}</td>'
            f'<td class="num"><strong>'
            f'{a.peak_tops(f):,.0f}</strong></td>'
            f'<td class="num">{a.die_power_w(f):.0f}</td>'
            f'<td class="num"><strong>'
            f'{a.alu.tops_per_watt_ceiling:.0f}</strong></td>'
            f'</tr>'
        )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        f'<th>Bare ALU (on {die_area_mm2:.0f} mm² die)</th>'
        '<th># ALUs</th><th>Die trans. (M)</th>'
        '<th>pJ / clock</th>'
        f'<th>Peak TOPS @ silicon clock</th>'
        '<th>Die power (W)</th>'
        '<th>TOPS / W ceiling</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_tdp_sweep_table(
    analyses: List[SoLAnalysis], tdp_targets: tuple,
) -> str:
    header_cols = "".join(
        f'<th class="num" colspan="2">{html.escape(a.alu.name)}</th>'
        for a in analyses
    )
    sub_cols = "".join(
        '<th class="num">Clock (GHz)</th>'
        '<th class="num">Peak TOPS</th>'
        for _ in analyses
    )
    rows = []
    for tdp in tdp_targets:
        cells = []
        for a in analyses:
            swept = a.tdp_sweep()
            rec = next(r for r in swept if abs(r["tdp_w"] - tdp) < 1e-9)
            clock_text = f'{rec["effective_clock_ghz"]:.2f}'
            if rec["silicon_limited"]:
                clock_text += " *"  # silicon-limited, not TDP-limited
            cells.append(
                f'<td class="num">{clock_text}</td>'
                f'<td class="num"><strong>'
                f'{rec["peak_tops"]:,.0f}</strong></td>'
            )
        rows.append(
            f'<tr><td class="name">{tdp:.0f} W</td>{"".join(cells)}</tr>'
        )
    return (
        '<table class="blocks">'
        f'<thead><tr><th rowspan="2">TDP target</th>{header_cols}</tr>'
        f'<tr>{sub_cols}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_gap_to_products_table(
    products: List[ProductReference], analyses: List[SoLAnalysis],
) -> str:
    """One row per shipping product; one column per ALU type showing
    the SoL peak at that product's TDP vs the product's actual peak."""
    alu_cols = "".join(
        f'<th class="num" colspan="2">{html.escape(a.alu.name)}<br/>'
        f'<span style="font-weight:normal; font-size:10px; color:#586374;">'
        f'SoL @ product TDP</span></th>'
        for a in analyses
    )
    sub_cols = "".join(
        '<th class="num">SoL TOPS</th><th class="num">Actual / SoL</th>'
        for _ in analyses
    )

    rows = []
    for p in products:
        cells = []
        for a in analyses:
            f_for_tdp = a.clock_for_tdp(p.tdp_w)
            effective_f = min(f_for_tdp, a.silicon_clock_ghz)
            sol_tops = a.peak_tops(effective_f)
            ratio = p.peak_int8_tops / sol_tops if sol_tops > 0 else 0.0
            cells.append(
                f'<td class="num">{sol_tops:,.0f}</td>'
                f'<td class="num"><strong>{ratio*100:.0f}%</strong></td>'
            )
        rows.append(
            f'<tr>'
            f'<td class="name">{html.escape(p.name)}</td>'
            f'<td class="num">{p.process_nm}</td>'
            f'<td class="num">{p.die_area_mm2:.0f}</td>'
            f'<td class="num">{p.tdp_w:.0f} W</td>'
            f'<td class="num"><strong>{p.peak_int8_tops:,.0f}</strong></td>'
            f'<td class="num">{p.tops_per_watt:.1f}</td>'
            f'{"".join(cells)}'
            f'</tr>'
        )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        '<th rowspan="2">Product</th><th rowspan="2">Process (nm)</th>'
        '<th rowspan="2">Die (mm²)</th>'
        '<th rowspan="2">TDP (W)</th>'
        '<th rowspan="2">Actual TOPS</th>'
        '<th rowspan="2">Actual TOPS/W</th>'
        f'{alu_cols}'
        '</tr>'
        f'<tr>{sub_cols}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )

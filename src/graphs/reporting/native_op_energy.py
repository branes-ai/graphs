"""
Native-op energy breakdown: progressive per-MAC energy floor per
architecture, broken down by ALU -> +Register -> +L1 memory layer.

Purpose: give a theoretical-floor comparison of energy per MAC for
each architecture's native operation, independent of workload. The
progression shows how energy grows as you descend into the memory
hierarchy during the native op's steady-state execution.

The "native op" for each archetype:
  - KPU (domain flow):      one PE MAC, SARE-scheduled, 1-cycle steady state
  - TPU (systolic):         one PE MAC in a weight-stationary wavefront
  - Tensor Core (SIMT):     one MAC within a warp-level mma (HMMA) instruction
  - CPU SIMD (not modeled here but schema supports it)

All values are normalized per MAC in steady state. Process node and a
1-bit full-adder reference are carried alongside for calibration. See
cli/check_tdp_feasibility.py for the FA reference table.

References:
  - Horowitz, "Computing's Energy Problem", ISSCC 2014 - register
    access energy scaling across process nodes.
  - Jouppi et al., "In-Datacenter Performance Analysis of a TPU",
    ISCA 2017 - double-buffered weight-stationary systolic.
  - NVIDIA Ampere whitepaper / "A100 Tensor Core GPU Architecture" -
    HMMA 16x16x16 native op, 4-cycle latency per warp.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from graphs.hardware.resource_model import Precision
from graphs.reporting.microarch_html_template import (
    _CSS,
    _load_logo,
    _render_brand_footer,
    _render_brand_header,
)


# ----------------------------------------------------------------------
# Register-access energy scaling (Horowitz 2014 baseline, pJ/byte).
# Values represent per-byte dynamic energy to read (or write) a register
# at nominal Vdd. Process-node scaled from 45nm baseline of ~0.12 pJ/byte.
# ----------------------------------------------------------------------
REG_ACCESS_PJ_PER_BYTE_BY_PROCESS = {
    45: 0.120,  # Horowitz 2014 reference
    28: 0.060,
    22: 0.045,
    16: 0.025,
    14: 0.022,
    12: 0.018,
    10: 0.016,
    8:  0.014,
    7:  0.012,
    5:  0.009,
    3:  0.006,
}


def register_pj_per_byte(process_nm: int) -> float:
    """Per-byte register-file access energy at the given process node."""
    if process_nm in REG_ACCESS_PJ_PER_BYTE_BY_PROCESS:
        return REG_ACCESS_PJ_PER_BYTE_BY_PROCESS[process_nm]
    nearest = min(REG_ACCESS_PJ_PER_BYTE_BY_PROCESS.keys(),
                  key=lambda k: abs(k - process_nm))
    return REG_ACCESS_PJ_PER_BYTE_BY_PROCESS[nearest]


# Full adder reference (duplicated from check_tdp_feasibility for
# self-containment; kept in sync).
FULL_ADDER_PJ_BY_PROCESS = {
    28: 0.025, 22: 0.018, 16: 0.010, 14: 0.009, 12: 0.007,
    10: 0.006, 8: 0.005, 7: 0.004, 5: 0.003, 3: 0.002,
}


def _fa_pj(process_nm: int) -> float:
    if process_nm in FULL_ADDER_PJ_BY_PROCESS:
        return FULL_ADDER_PJ_BY_PROCESS[process_nm]
    nearest = min(FULL_ADDER_PJ_BY_PROCESS.keys(),
                  key=lambda k: abs(k - process_nm))
    return FULL_ADDER_PJ_BY_PROCESS[nearest]


def _precision_byte_width(precision: Precision) -> float:
    mapping = {
        Precision.FP64: 8.0, Precision.FP32: 4.0, Precision.TF32: 4.0,
        Precision.BF16: 2.0, Precision.FP16: 2.0, Precision.FP8: 1.0,
        Precision.INT8: 1.0, Precision.INT4: 0.5,
    }
    return mapping.get(precision, 1.0)


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------

@dataclass
class NativeOpLayer:
    """One additive layer in the per-MAC energy breakdown."""
    name: str                       # "ALU (bare MAC)", "+ Register", "+ L1"
    energy_pj_per_mac: float        # incremental cost at this layer
    cumulative_pj_per_mac: float    # running total including this layer
    source: str = ""                # derivation / citation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "energy_pj_per_mac": self.energy_pj_per_mac,
            "cumulative_pj_per_mac": self.cumulative_pj_per_mac,
            "source": self.source,
        }


@dataclass
class NativeOpBreakdown:
    """Native-op energy floor breakdown for one archetype."""
    archetype: str                  # "Domain Flow (KPU)"
    display_name: str               # "KPU T128"
    sku: str
    color: str
    native_op_name: str             # "PE MAC (SARE-scheduled)"
    precision: str                  # "INT8"
    process_node_nm: int
    full_adder_energy_pj: float
    clock_hz: float
    latency_cycles: int             # native op latency in cycles
    macs_per_native_op: int         # MACs produced per native op
    layers: List[NativeOpLayer] = field(default_factory=list)
    notes: str = ""

    @property
    def total_energy_per_mac_pj(self) -> float:
        return self.layers[-1].cumulative_pj_per_mac if self.layers else 0.0

    @property
    def cycle_time_ns(self) -> float:
        return 1e9 / self.clock_hz if self.clock_hz > 0 else 0.0

    @property
    def native_op_time_ns(self) -> float:
        return self.latency_cycles * self.cycle_time_ns

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archetype": self.archetype,
            "display_name": self.display_name,
            "sku": self.sku,
            "color": self.color,
            "native_op_name": self.native_op_name,
            "precision": self.precision,
            "process_node_nm": self.process_node_nm,
            "full_adder_energy_pj": self.full_adder_energy_pj,
            "clock_hz": self.clock_hz,
            "latency_cycles": self.latency_cycles,
            "macs_per_native_op": self.macs_per_native_op,
            "layers": [l.to_dict() for l in self.layers],
            "total_energy_per_mac_pj": self.total_energy_per_mac_pj,
            "cycle_time_ns": self.cycle_time_ns,
            "native_op_time_ns": self.native_op_time_ns,
            "notes": self.notes,
        }


@dataclass
class NativeOpComparison:
    breakdowns: List[NativeOpBreakdown]
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "breakdowns": [b.to_dict() for b in self.breakdowns],
            "generated_at": self.generated_at,
        }


# ----------------------------------------------------------------------
# Per-archetype builders
# ----------------------------------------------------------------------

def build_kpu_native_op(
    sku: str = "Stillwater-KPU-T128",
    precision: Precision = Precision.INT8,
) -> NativeOpBreakdown:
    """KPU native op: one PE MAC in a SARE-scheduled wavefront.

    Layer breakdown (INT8 @ 16nm):
      ALU:        mac_energy_int8 from the tile energy model
      +Register:  3 PE-local register byte-accesses (2 reads + 1 write
                  of accumulator) scaled by process node
      +L1:        tile-local scratchpad read, amortized over the
                  operand's reuse factor (PE array column dim)
    """
    from graphs.hardware.mappers import get_mapper_by_name
    mapper = get_mapper_by_name(sku)
    if mapper is None:
        raise RuntimeError(f"Mapper {sku!r} not in registry")
    rm = mapper.resource_model
    tp = rm.thermal_operating_points[rm.default_thermal_profile]
    cr = tp.performance_specs[precision].compute_resource
    spec = cr.tile_specializations[0]
    tem = rm.tile_energy_model
    fabrics = getattr(rm, "compute_fabrics", []) or []
    process_nm = fabrics[0].process_node_nm if fabrics else 16

    # ALU
    if precision == Precision.INT8:
        alu_pj = tem.mac_energy_int8 * 1e12
    elif precision == Precision.INT4:
        # INT4 is not a first-class entry in KPUTileEnergyModel; derive
        # from INT8 using the SKU's own energy-scaling ratio (typically
        # ~0.5x INT8 for 4-bit datapaths). Avoids falling through to
        # FP32 and massively overcounting.
        int8_scale = rm.energy_scaling.get(Precision.INT8, 1.0)
        int4_scale = rm.energy_scaling.get(Precision.INT4, int8_scale * 0.5)
        ratio = int4_scale / int8_scale if int8_scale > 0 else 0.5
        alu_pj = tem.mac_energy_int8 * ratio * 1e12
    elif precision == Precision.BF16:
        alu_pj = tem.mac_energy_bf16 * 1e12
    else:
        alu_pj = tem.mac_energy_fp32 * 1e12

    # Register: 3 byte-accesses at process-scaled register-access energy
    bytes_per_op = _precision_byte_width(precision)
    reg_pj = 3 * bytes_per_op * register_pj_per_byte(process_nm)

    # L1 (tile scratchpad): 2 operand bytes fetched per MAC, but each
    # operand is reused along one dimension of the PE array (column
    # reuse in a typical output-stationary dataflow). Effective per-MAC
    # L1 cost = (2 * L1_read_pj_per_byte * bytes) / reuse_factor.
    rows, cols = spec.array_dimensions[0], spec.array_dimensions[1]
    reuse_factor = cols
    l1_pj = (2 * bytes_per_op * tem.l1_read_energy_per_byte * 1e12) / reuse_factor

    layers = [
        NativeOpLayer(
            name="ALU (bare MAC)",
            energy_pj_per_mac=alu_pj,
            cumulative_pj_per_mac=alu_pj,
            source=(f"tile_energy_model.mac_energy_{precision.value} "
                    f"@ {process_nm}nm domain-flow"),
        ),
        NativeOpLayer(
            name="+ Register",
            energy_pj_per_mac=reg_pj,
            cumulative_pj_per_mac=alu_pj + reg_pj,
            source=(f"3 byte-accesses x {register_pj_per_byte(process_nm):.3f} pJ/byte "
                    f"(Horowitz 2014 scaled to {process_nm}nm)"),
        ),
        NativeOpLayer(
            name="+ L1 scratchpad",
            energy_pj_per_mac=l1_pj,
            cumulative_pj_per_mac=alu_pj + reg_pj + l1_pj,
            source=(f"2 operand bytes x "
                    f"{tem.l1_read_energy_per_byte*1e12:.2f} pJ/byte / "
                    f"{reuse_factor}x reuse along PE column"),
        ),
    ]

    return NativeOpBreakdown(
        archetype="Domain Flow (KPU)",
        display_name=sku.replace("Stillwater-", "").replace("-", " "),
        sku=sku,
        color="#3fc98a",
        native_op_name="PE MAC (SARE-scheduled)",
        precision=precision.value,
        process_node_nm=process_nm,
        full_adder_energy_pj=_fa_pj(process_nm),
        clock_hz=spec.clock_domain.sustained_clock_hz,
        latency_cycles=1,   # steady-state, pre-scheduled
        macs_per_native_op=1,
        layers=layers,
        notes=("Domain-flow fabric executes a statically scheduled SARE - "
               "no instruction fetch, no coherence, no dynamic scheduling "
               "per MAC. Register file is PE-local (3 accesses per MAC: "
               "two operand reads + accumulator update). L1 scratchpad "
               "access is amortized along the systolic column reuse."),
    )


def build_tensor_core_native_op(
    precision: Precision = Precision.INT8,
) -> NativeOpBreakdown:
    """Ampere Tensor Core native op: one MAC within an HMMA instruction.

    HMMA m16n16k16 produces 4096 MACs per warp per 4 cycles (1024
    MACs/cycle/warp). Per-MAC energy includes warp register-file access
    (larger than PE-local) and a small amortized shared-memory cost
    when the matrix fragment is reloaded.
    """
    process_nm = 8
    bytes_per_op = _precision_byte_width(precision)

    # ALU: Ampere Tensor Core bare MAC @ 8nm.
    #   FP16/BF16: ~0.40 pJ (derived from published TC ~1.6 pJ/op / 2
    #              ops-per-MAC, backing out register + coherence overhead).
    #   INT8:      ~0.25 pJ (narrower datapath - roughly 8 FA-eq vs 16
    #              for FP16, scaled by the same register/coherence tax).
    alu_pj = 0.40 if precision in (Precision.FP16, Precision.BF16) else 0.25

    # Register: warp register file is substantially bigger than PE-local
    # regs; per-byte access energy is higher. Use 2x the Horowitz baseline
    # at 8nm to reflect the bigger SRAM array.
    reg_pj_per_byte = 2.0 * register_pj_per_byte(process_nm)
    reg_pj = 3 * bytes_per_op * reg_pj_per_byte

    # L1 / shared memory: LDSM loads a 16x16 matrix fragment. Shared
    # memory access ~ 0.5 pJ/byte at 8nm; reuse factor is the fragment
    # column dim (16). Per-MAC: 2 * bytes * 0.5 / 16.
    smem_pj_per_byte = 0.5
    reuse_factor = 16
    l1_pj = (2 * bytes_per_op * smem_pj_per_byte) / reuse_factor

    # Scheduling / coherence overhead per MAC (warp scheduler,
    # coherence, memory ordering). Amortized across the warp's MACs.
    overhead_pj = 0.05

    cum = 0.0
    layers = []
    for name, e_pj, src in [
        ("ALU (bare MAC)", alu_pj,
         f"Ampere HMMA bare MAC @ 8nm, derived from published ~1.6 pJ/op"),
        ("+ Register", reg_pj,
         f"3 byte-accesses x 2x Horowitz baseline at 8nm "
         f"({reg_pj_per_byte:.3f} pJ/byte, warp regfile)"),
        ("+ L1 (shared mem)", l1_pj,
         f"2 bytes x {smem_pj_per_byte:.2f} pJ/byte / {reuse_factor}x frag reuse"),
        ("+ Warp sched / coherence", overhead_pj,
         "Dynamic warp scheduling + memory coherence, amortized"),
    ]:
        cum += e_pj
        layers.append(NativeOpLayer(name, e_pj, cum, src))

    return NativeOpBreakdown(
        archetype="SIMT + Tensor Core",
        sku="Jetson-Orin-AGX-64GB",
        display_name="Jetson Orin AGX (TC)",
        color="#5b8ff9",
        native_op_name=f"HMMA m16n16k16.{precision.value}",
        precision=precision.value,
        process_node_nm=process_nm,
        full_adder_energy_pj=_fa_pj(process_nm),
        clock_hz=650e6,  # sustained at 30W profile
        latency_cycles=4,  # 4-cycle HMMA pipeline
        macs_per_native_op=4096,  # 16 x 16 x 16
        layers=layers,
        notes=("Native op is the warp-level HMMA matrix-multiply-"
               "accumulate instruction. Energy per MAC is amortized "
               "across the 4096 MACs per HMMA, but includes warp "
               "register-file access, shared-memory loads, and "
               "scheduling overhead which have no analogue on a "
               "statically-scheduled domain-flow fabric."),
    )


def build_tpu_native_op(
    precision: Precision = Precision.INT8,
) -> NativeOpBreakdown:
    """Coral Edge TPU native op: one PE MAC in a weight-stationary
    systolic wavefront."""
    process_nm = 14
    bytes_per_op = _precision_byte_width(precision)

    # ALU: 14nm INT8 MAC. Coral published 0.15 pJ/MAC is the compute-
    # energy floor for a well-tuned kernel. Bare MAC is ~0.07 pJ;
    # the remainder is register + operand delivery. Use 0.07 for ALU.
    alu_pj = 0.07

    # Register: weight stays in PE register (loaded once), input byte
    # enters once per cycle. Per-MAC: 2 reads (weight + input) + 1
    # write (accumulator). Weight read amortizes over K reductions ->
    # effectively 1 input read + 1 acc write per MAC.
    reg_pj_per_byte = register_pj_per_byte(process_nm)
    reg_pj = 2 * bytes_per_op * reg_pj_per_byte  # input read + acc update

    # L1 (UB buffer for weights/inputs): weight loaded once per K x K
    # tile, input streamed in. Effective per-MAC L1 cost = (input
    # bytes * L1_energy) / row_reuse. Row reuse ~ PE rows = 64.
    ub_pj_per_byte = 0.10  # representative 14nm UB SRAM
    l1_pj = (bytes_per_op * ub_pj_per_byte) / 64

    cum = 0.0
    layers = []
    for name, e_pj, src in [
        ("ALU (bare MAC)", alu_pj,
         f"14nm INT8 MAC bare-cell energy"),
        ("+ Register (PE-local)", reg_pj,
         f"Weight stationary: 1 input read + 1 acc update per MAC @ "
         f"{reg_pj_per_byte:.3f} pJ/byte"),
        ("+ L1 (UB)", l1_pj,
         f"{bytes_per_op} byte x {ub_pj_per_byte:.2f} pJ/byte / 64x row reuse "
         f"(weight amortized over K)"),
    ]:
        cum += e_pj
        layers.append(NativeOpLayer(name, e_pj, cum, src))

    return NativeOpBreakdown(
        archetype="Systolic (TPU)",
        sku="Google-Coral-Edge-TPU",
        display_name="Coral Edge TPU",
        color="#e98c3f",
        native_op_name="Systolic PE MAC (weight-stationary)",
        precision=precision.value,
        process_node_nm=process_nm,
        full_adder_energy_pj=_fa_pj(process_nm),
        clock_hz=500e6,
        latency_cycles=1,
        macs_per_native_op=1,
        layers=layers,
        notes=("Weight-stationary systolic: weight held in PE register, "
               "input streams in, accumulator updated per cycle. The "
               "bare MAC energy floor is low; the main cost beyond the "
               "ALU is the input-delivery register access. Real "
               "workloads (not native-op floor) add shape-mismatch and "
               "bandwidth-bound layer penalties."),
    )


def build_default_comparison(
    precision: Precision = Precision.INT8,
    kpu_sku: str = "Stillwater-KPU-T128",
) -> NativeOpComparison:
    from datetime import datetime, timezone
    return NativeOpComparison(
        breakdowns=[
            build_tensor_core_native_op(precision),
            build_tpu_native_op(precision),
            build_kpu_native_op(kpu_sku, precision),
        ],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ----------------------------------------------------------------------
# HTML rendering
# ----------------------------------------------------------------------

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _render_chart_js(report: NativeOpComparison) -> str:
    """Stacked-bar chart: one bar per archetype, segments by layer.

    Archetypes may have different numbers of layers (e.g., Tensor Core
    adds a warp-scheduling layer that KPU/TPU don't have). We collect
    the union of layer names preserving per-archetype order and
    zero-fill missing layers.
    """
    bds = [b.to_dict() for b in report.breakdowns]

    # Collect union of layer names preserving order of first appearance
    all_layer_names: List[str] = []
    for b in bds:
        for l in b["layers"]:
            if l["name"] not in all_layer_names:
                all_layer_names.append(l["name"])

    palette = ["#5b8ff9", "#d4860b", "#e98c3f", "#7f3b8d"]
    traces = []
    for i, layer_name in enumerate(all_layer_names):
        ys = [b["display_name"] for b in bds]
        xs = []
        for b in bds:
            found = next((l for l in b["layers"] if l["name"] == layer_name), None)
            xs.append(found["energy_pj_per_mac"] if found else 0.0)
        texts = [f"{v:.3f}" if v > 0 else "" for v in xs]
        traces.append({
            "type": "bar",
            "orientation": "h",
            "name": layer_name,
            "y": ys,
            "x": xs,
            "text": texts,
            "textposition": "inside",
            "marker": {"color": palette[i % len(palette)]},
        })

    # Chart 1: stacked bar of native op energy progression
    chart_progression = {
        "data": traces,
        "layout": {
            "title": "Native-op energy floor per MAC (cumulative by layer)",
            "xaxis": {"title": "pJ / MAC"},
            "yaxis": {"title": "", "automargin": True},
            "barmode": "stack",
            "margin": {"t": 50, "b": 50, "l": 120, "r": 20},
            "legend": {"orientation": "h", "y": -0.2},
        },
    }

    # Chart 2: cumulative progression line (ALU -> +Register -> +L1 -> ...)
    # X-axis indexed by layer position (each archetype keeps its own
    # layer ordering; x-tick labels reflect the first archetype's names
    # with an added "(+N more layers)" if some archetypes go deeper).
    cum_traces = []
    max_depth = max((len(b["layers"]) for b in bds), default=0)
    x_positions = list(range(1, max_depth + 1))
    for b in bds:
        ys = [l["cumulative_pj_per_mac"] for l in b["layers"]]
        xs_local = list(range(1, len(ys) + 1))
        cum_traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": b["display_name"],
            "x": xs_local,
            "y": ys,
            "line": {"color": b["color"], "width": 3},
            "marker": {"size": 10},
        })
    chart_cumulative = {
        "data": cum_traces,
        "layout": {
            "title": "Energy accumulation through the memory hierarchy",
            "xaxis": {
                "title": "Layer depth (progressive)",
                "tickmode": "array",
                "tickvals": x_positions,
                "ticktext": [f"Layer {i}" for i in x_positions],
            },
            "yaxis": {"title": "Cumulative pJ / MAC"},
            "margin": {"t": 50, "b": 50, "l": 60, "r": 20},
            "legend": {"orientation": "h", "y": -0.2},
        },
    }

    payload = {"chart_progression": chart_progression, "chart_cumulative": chart_cumulative}
    return (
        f"const CHARTS = {json.dumps(payload)};\n"
        "for (const [id, spec] of Object.entries(CHARTS)) {\n"
        "  Plotly.newPlot(id, spec.data, spec.layout, "
        "{displayModeBar: false, responsive: true});\n"
        "}\n"
    )


def _render_breakdown_table(report: NativeOpComparison) -> str:
    rows = []
    for b in report.breakdowns:
        d = b.to_dict()
        rows.append(
            f'<tr><td colspan="4" class="group-header" '
            f'style="border-left: 4px solid {d["color"]};">'
            f'<strong>{html.escape(d["display_name"])}</strong> '
            f'-- {html.escape(d["archetype"])}<br/>'
            f'<span class="meta">Native op: {html.escape(d["native_op_name"])} '
            f'({d["macs_per_native_op"]} MACs, {d["latency_cycles"]} cycles @ '
            f'{d["clock_hz"]/1e9:.2f} GHz = {d["native_op_time_ns"]:.2f} ns) '
            f'| {d["process_node_nm"]} nm '
            f'| 1-bit FA ref {d["full_adder_energy_pj"]:.3f} pJ'
            f'</span></td></tr>'
        )
        for layer in d["layers"]:
            rows.append(
                f'<tr>'
                f'<td class="layer-name">{html.escape(layer["name"])}</td>'
                f'<td class="num">{layer["energy_pj_per_mac"]:.4f}</td>'
                f'<td class="num"><strong>{layer["cumulative_pj_per_mac"]:.4f}</strong></td>'
                f'<td class="src">{html.escape(layer["source"])}</td>'
                f'</tr>'
            )
        rows.append(
            f'<tr class="total-row">'
            f'<td><strong>Total native-op energy floor</strong></td>'
            f'<td></td>'
            f'<td class="num"><strong>{d["total_energy_per_mac_pj"]:.4f} pJ/MAC</strong></td>'
            f'<td></td>'
            f'</tr>'
        )
    return (
        '<table class="breakdown">'
        '<thead><tr><th>Layer</th><th>Incremental (pJ/MAC)</th>'
        '<th>Cumulative (pJ/MAC)</th><th>Source</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_native_op_page(
    report: NativeOpComparison,
    repo_root: Path,
) -> str:
    """Render the native-op energy breakdown page."""
    assets = _load_logo(repo_root)
    header = _render_brand_header(
        assets,
        "Native-op energy breakdown",
        f"Theoretical floor per MAC | precision {report.breakdowns[0].precision if report.breakdowns else 'N/A'} "
        f"| generated {report.generated_at}",
    )
    footer = _render_brand_footer("microarch-model-delivery-plan.md")

    extra_css = """
table.breakdown { width: 100%; border-collapse: collapse; background: #fff;
                  margin-bottom: 18px; }
table.breakdown th { font-size: 12px; text-transform: uppercase;
                     color: #586374; background: #f3f5f8;
                     padding: 8px 10px; text-align: left; }
table.breakdown td { padding: 6px 10px; border-bottom: 1px solid #e3e6eb;
                     vertical-align: top; font-size: 13px; }
table.breakdown td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.breakdown td.src { color: #586374; font-size: 12px; }
table.breakdown td.layer-name { padding-left: 24px; color: #3a4452; }
table.breakdown td.group-header { padding: 10px 12px; background: #f8f9fb;
                                  border-left-width: 4px; border-left-style: solid; }
table.breakdown tr.total-row td { background: #eef6ea;
                                   border-bottom: 2px solid #c3e1b0;
                                   padding-top: 10px; padding-bottom: 10px; }
section.method-note { background: #eef2f7; padding: 14px 18px;
                      border-left: 3px solid #0a2540; border-radius: 4px;
                      margin-bottom: 18px; }
.chart-section { background: #fff; padding: 18px 22px; border-radius: 6px;
  margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.chart-section h3 { margin: 0 0 4px; color: #0a2540; }
.chart-section .chart-desc { color: #586374; font-size: 13px; margin: 0 0 12px; }
.plot { width: 100%; min-height: 360px; }
a.nav-back { display: inline-block; color: #0a2540; text-decoration: none;
             font-weight: 600; margin-bottom: 10px; }
a.nav-back:hover { text-decoration: underline; }
"""

    method_note = """
<section class="method-note">
  <strong>Methodology.</strong> Each row shows the incremental and cumulative
  energy-per-MAC cost as the native operation descends the memory hierarchy,
  in steady state. The "ALU" layer is the bare-cell MAC energy at the SKU's
  manufacturing process. The "+Register" layer adds operand register-file
  access (3 byte-accesses per MAC: two operand reads + accumulator update)
  scaled by process node using Horowitz's ISSCC 2014 baseline. The "+L1"
  layer adds local scratchpad / shared-memory access for operand delivery,
  amortized by the architecture's natural reuse factor (PE array column
  dimension for output-stationary / systolic fabrics; matrix-fragment reuse
  for warp-level tensor cores). No workload is assumed - this is the
  theoretical energy floor of the native op itself.
</section>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Native-op energy breakdown</title>
  <script src="{_PLOTLY_CDN}"></script>
  <style>{_CSS}
{extra_css}
  </style>
</head>
<body>
{header}
<main>
  <p><a class="nav-back" href="index.html">&lt; Back to index</a></p>
  <section class="page-header">
    <h2>Native-op energy floor: ALU -&gt; +Register -&gt; +L1</h2>
    <div class="meta">Per-MAC energy breakdown for each architecture's
      native operation, no workload assumed.</div>
  </section>
  {method_note}
  <section class="chart-section">
    <h3>Stacked layer breakdown (incremental)</h3>
    <p class="chart-desc">Each segment is the incremental cost of adding
    one layer of the memory hierarchy. Total bar length is the total
    native-op energy per MAC.</p>
    <div id="chart_progression" class="plot"></div>
  </section>
  <section class="chart-section">
    <h3>Cumulative progression</h3>
    <p class="chart-desc">Same data viewed as the cumulative per-MAC cost
    as you descend from ALU through register file to L1.</p>
    <div id="chart_cumulative" class="plot"></div>
  </section>
  {_render_breakdown_table(report)}
</main>
{footer}
<script>
{_render_chart_js(report)}
</script>
</body>
</html>
"""

#!/usr/bin/env python
"""NVIDIA Jetson **industrial** roadmap visualization (embodied / industrial AI).

Targets the four-generation industrial-grade Jetson lineage. Verified
2026-05-13:

  | SKU                          | GPU arch  | Process     | GA date         | EOL                       |
  |------------------------------|-----------|-------------|-----------------|---------------------------|
  | Jetson TX2i                  | Pascal    | TSMC 16FF   | Apr 2018        | LTB Jul 2026 / LTS Jul 2027 |
  | Jetson AGX Xavier Industrial | Volta     | TSMC 12FFN  | Jul 2021        | LTB Jul 2026 / LTS Jul 2027 |
  | Jetson AGX Orin Industrial   | Ampere    | Samsung 8nm | Mid 2023        | Active (lifecycle through Jul 2033) |
  | IGX T5000 (Thor industrial)  | Blackwell | TSMC 4NP    | Dec 2025        | Active (lifecycle through Aug 2035) |

Sources:
  TX2i:                  https://forums.developer.nvidia.com/t/jetson-product-eol-updates/370052
  AGX Xavier Industrial: https://blogs.nvidia.com/blog/jetson-agx-xavier-industrial-use-ai/
  AGX Orin Industrial:   https://developer.nvidia.com/blog/step-into-the-future-of-industrial-grade-edge-ai-with-nvidia-jetson-agx-orin-industrial
  IGX T5000:             https://blogs.nvidia.com/blog/igx-thor-processor-physical-ai-industrial-medical-edge/
  Jetson T5000 (commercial Thor module that IGX T5000 derives from):
                         https://nvidianews.nvidia.com/news/nvidia-blackwell-powered-jetson-thor-now-available-accelerating-the-age-of-general-robotics

Caveats worth keeping in mind for any slide based on this output:

  1. **Thor industrial branding is split.** NVIDIA does not yet ship a
     SKU literally named "Jetson AGX Thor Industrial." The industrial /
     medical-grade Blackwell-Thor platform is **IGX Thor** (IGX T5000
     SoM, IGX T7000 board kit). The commercial **Jetson T5000** module
     (Aug 2025 GA) is the same silicon but is NOT industrial-grade
     (no extended-temp / ECC / functional-safety guarantees). DRIVE AGX
     Thor (automotive) is dev-kit-only.
  2. **TX2i / Xavier Industrial EOLs were both accelerated** in an April
     2026 PCN due to LPDDR4 manufacturer EOL. NVIDIA's lifecycle page
     may still show stale "available through 2031" entries -- treat the
     forum PCN as authoritative.
  3. **No resource models exist in the graphs/ catalog for TX2 or Xavier
     yet.** This script lists those SKUs as the slide's *intent* but
     skips them at analysis time with a console warning. Adding TX2
     (Pascal, 16nm) + Xavier (Volta, 12nm) resource models is tracked
     separately. The two we do have are mapped via the commercial-grade
     mapper as a proxy for the industrial sibling -- same silicon,
     different binning / temp grade / package; performance and energy
     per workload are essentially identical.

Workload: ``Linear(2048, 2048) + bias + atan`` at batch=1.
4M weights = 4 MB at INT8. At batch=1 this is solidly memory-bandwidth
bound on every Jetson (the matrix has to stream from LPDDR; on-chip L2
is 2-8 MB and the GPU L2 isn't modeled as a coherent weight pool --
per-launch L2 set-aside on Hopper+ isn't covered here). The operator
fuses into one subgraph so the 2K-element activation vectors stay in
registers / shared memory; only the weight matrix and the external
input/output activations cross the DRAM boundary.

Two metrics are plotted vs release date:

  * **Sustained throughput** (inferences / sec) -- 1 / latency
  * **Energy efficiency** (inferences / joule) -- 1 / energy_per_inference

Both are derived from the existing roofline + energy model
(``UnifiedAnalyzer``); no measurement here. Each chip is analyzed with
its mapper's default thermal profile (the closest thing to "stock"
behavior) and the per-chip TDP appears in the legend for context.

**Note on terminology**: "intelligence per Watt" in NVIDIA's marketing is
``throughput / TDP`` (industry convention). That's a different number
from ``inferences / joule`` because the chip rarely runs at its full TDP
on a small workload; the model here estimates average power as
``energy / latency``, which for this workload is well below TDP. The
CSV output includes both metrics so you can pick the framing that
matches your slide.

Usage:
  python cli/analyze_jetson_roadmap.py --output jetson_roadmap.png
  python cli/analyze_jetson_roadmap.py --output jetson_roadmap.csv
  python cli/analyze_jetson_roadmap.py --output jetson_roadmap.json
  python cli/analyze_jetson_roadmap.py --output jetson_roadmap.md
  python cli/analyze_jetson_roadmap.py --precision int8 --output roadmap_int8.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from graphs.estimation.unified_analyzer import UnifiedAnalyzer
from graphs.hardware.mappers.accelerators.kpu import KPUMapper
from graphs.hardware.models.accelerators.kpu_yaml_loader import (
    load_kpu_resource_model_from_yaml,
)
from graphs.hardware.architectural_energy import KPUTileEnergyAdapter


def create_kpu_t64_7nm_mapper() -> KPUMapper:
    """KPU-T64 on TSMC N7 HPC -- the closest available KPU process node
    to the Jetsons in this chart (Orin = Samsung 8nm, Thor = TSMC 4NP).

    The standard ``create_kpu_t64_mapper()`` factory loads the 16nm
    variant which biases the comparison toward older silicon. T64 ships
    in three process variants in embodied-schemas (16nm / 12nm / 7nm);
    we load the 7nm SKU directly here for a fairer roadmap-era
    comparison. A 5nm or 4nm KPU SKU does not yet exist in the catalog.
    """
    model = load_kpu_resource_model_from_yaml("kpu_t64_32x32_lp5x4_7nm_tsmc_hpc")
    model.architecture_energy_model = KPUTileEnergyAdapter(model.tile_energy_model)
    return KPUMapper(model)
from graphs.hardware.mappers.cpu import (
    create_ampere_ampereone_1core_reference_mapper,
)
from graphs.hardware.mappers.gpu import (
    create_jetson_orin_agx_64gb_mapper,
    create_jetson_thor_128gb_mapper,
)
from graphs.hardware.resource_model import Precision


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

class Atan(nn.Module):
    """Elementwise atan as a Module so torch.fx can trace it cleanly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan(x)


def build_workload(width: int) -> tuple[nn.Module, torch.Tensor, str]:
    """Linear(width, width) + bias + atan, batch=1.

    Default width=2048 -> 4,194,304 weights ~= 4 M, the spec's "medium
    sized linear operator" target. Returns the model, the input tensor,
    and a display name for the chart.
    """
    model = nn.Sequential(
        nn.Linear(width, width),
        Atan(),
    )
    model.eval()
    input_tensor = torch.randn(1, width)
    weights = width * width
    label = f"Linear({width}x{width})+bias+atan ({weights / 1e6:.1f}M weights, batch=1)"
    return model, input_tensor, label


# ---------------------------------------------------------------------------
# Product definitions
# ---------------------------------------------------------------------------

@dataclass
class RoadmapProduct:
    """One Jetson industrial SKU on the roadmap.

    ``factory`` is ``None`` for SKUs whose silicon doesn't have a
    resource model in the catalog yet (TX2 / Pascal, Xavier / Volta);
    those rows still document the slide's intent but are skipped at
    analysis time with a console warning. ``factory_proxy_note`` says
    which commercial-grade mapper stands in for the industrial sibling
    when the two share silicon."""

    name: str
    factory: Optional[Callable]
    release_date: date
    eol_date: date
    architecture: str
    process_node: str
    color: str
    marker: str
    factory_proxy_note: str = ""

    def short_name(self) -> str:
        return (
            self.name.replace("Jetson ", "")
            .replace(" Industrial", " Ind.")
        )


# The four-generation industrial-grade Jetson lineage. Dates / EOLs in
# the module docstring above. Chart x-axis is year-resolution; the day
# of the month is illustrative.
PRODUCTS: List[RoadmapProduct] = [
    RoadmapProduct(
        name="Jetson TX2i",
        factory=None,  # Pascal resource model not in catalog yet
        release_date=date(2018, 4, 24),
        eol_date=date(2027, 7, 1),  # LTS Jul 2027 (PCN-accelerated, see docstring)
        architecture="Pascal",
        process_node="TSMC 16FF",
        color="#9467bd",   # tab:purple
        marker="D",
    ),
    RoadmapProduct(
        name="Jetson AGX Xavier Industrial",
        factory=None,  # Volta resource model not in catalog yet
        release_date=date(2021, 7, 1),
        eol_date=date(2027, 7, 1),  # LTS Jul 2027 (PCN-accelerated, see docstring)
        architecture="Volta",
        process_node="TSMC 12FFN",
        color="#ff7f0e",   # tab:orange
        marker="s",
    ),
    RoadmapProduct(
        name="Jetson AGX Orin Industrial",
        factory=create_jetson_orin_agx_64gb_mapper,
        release_date=date(2023, 6, 1),  # COMPUTEX 2023 May 29; mid-2023 GA
        eol_date=date(2033, 7, 1),  # Lifecycle commitment Jul 2033
        architecture="Ampere",
        process_node="Samsung 8nm",
        color="#1f77b4",   # tab:blue
        marker="o",
        factory_proxy_note="Mapper proxy: AGX Orin 64GB commercial",
    ),
    RoadmapProduct(
        name="IGX T5000",
        factory=create_jetson_thor_128gb_mapper,
        release_date=date(2025, 12, 1),  # IGX T5000 SoM GA Dec 2025
        eol_date=date(2035, 8, 1),  # Lifecycle commitment Aug 2035
        architecture="Blackwell",
        process_node="TSMC 4NP",
        color="#2ca02c",   # tab:green
        marker="^",
        factory_proxy_note="Mapper proxy: Jetson T5000 commercial (same silicon)",
    ),
    # Reference data point: a 192-core ARM server CPU sized roughly the
    # same calendar window as the Orin / Thor industrial generation.
    # Included to sanity-check what the CPU mapper does with a single
    # batch=1 Linear -- naively splitting 4M weights across 192 cores
    # would leave each core with ~21K weights of work, well below any
    # reasonable per-core efficiency floor; the mapper should resolve
    # this as a single-core (or few-core) job, not a 192-way fanout.
    # Synthetic single-core ARM reference. Issue #175 demonstrated that
    # the multi-core AmpereOne SKUs report unreliable throughput on
    # batch=1 matvec (naive compute fanout across all cores), so the
    # AmpereOne A192 SKU is intentionally omitted from this chart until
    # the per-operator concurrency cap lands. The 1-core slice is kept
    # as a useful per-core ARM data point in its own right.
    RoadmapProduct(
        name="AmpereOne 1-core (synthetic ref)",
        factory=create_ampere_ampereone_1core_reference_mapper,
        release_date=date(2024, 5, 1),
        eol_date=date(2031, 5, 1),
        architecture="ARM v8.6+ (1c)",
        process_node="TSMC 5nm",
        color="#8c564b",   # tab:brown
        marker="x",
    ),
    # KPU reference SKU. T64 alone is enough to make the comparison
    # legible (it already beats Thor on this workload). T256 deferred
    # until the avg_power-vs-TDP modeling fix lands. Process node:
    # TSMC N7 HPC -- closest available KPU SKU to the chart's Orin
    # (Samsung 8nm) / Thor (TSMC 4NP) era. A 5nm / 4nm KPU is not yet
    # in the catalog.
    RoadmapProduct(
        name="Stillwater KPU-T64",
        factory=create_kpu_t64_7nm_mapper,
        release_date=date(2027, 1, 1),
        eol_date=date(2037, 1, 1),
        architecture="KPU domain-flow (64 tiles)",
        process_node="TSMC N7 HPC",
        color="#9467bd",   # tab:purple
        marker="P",
    ),
]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass
class ProductResult:
    product: RoadmapProduct
    latency_ms: float
    energy_mj: float
    tdp_w: float
    perf_inf_per_s: float
    intelligence_inf_per_j: float


def analyze(
    product: RoadmapProduct,
    model: nn.Module,
    input_tensor: torch.Tensor,
    precision: Precision,
    workload_name: str,
) -> ProductResult:
    """Run UnifiedAnalyzer on one Jetson; derive perf and intelligence/W."""
    mapper = product.factory()
    rm = mapper.resource_model
    default_profile_name = rm.default_thermal_profile
    tdp_w = rm.thermal_operating_points[default_profile_name].tdp_watts

    analyzer = UnifiedAnalyzer(verbose=False)
    result = analyzer.analyze_model_with_custom_hardware(
        model=model,
        input_tensor=input_tensor,
        model_name=workload_name,
        hardware_mapper=mapper,
        precision=precision,
    )

    latency_ms = result.total_latency_ms
    energy_mj = result.energy_per_inference_mj
    # batch=1 so 1 inference per call:
    perf = 1000.0 / latency_ms                  # inferences / sec
    intelligence = 1000.0 / max(energy_mj, 1e-9)  # inferences / joule

    return ProductResult(
        product=product,
        latency_ms=latency_ms,
        energy_mj=energy_mj,
        tdp_w=tdp_w,
        perf_inf_per_s=perf,
        intelligence_inf_per_j=intelligence,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _row_dict(r: ProductResult) -> dict:
    """Per-product dict used by csv/json/md/txt writers. One source of
    truth so all four formats see identical columns + rounding."""
    avg_power_w = r.energy_mj / r.latency_ms if r.latency_ms > 0 else 0.0
    throughput_per_tdp = r.perf_inf_per_s / r.tdp_w if r.tdp_w > 0 else 0.0
    return {
        "name": r.product.name,
        "release_date": r.product.release_date.isoformat(),
        "eol_date": r.product.eol_date.isoformat(),
        "architecture": r.product.architecture,
        "process_node": r.product.process_node,
        "tdp_w": r.tdp_w,
        "latency_ms": round(r.latency_ms, 3),
        "energy_per_inference_mj": round(r.energy_mj, 3),
        "avg_power_w_during_inference": round(avg_power_w, 2),
        "perf_inferences_per_sec": round(r.perf_inf_per_s, 1),
        "energy_efficiency_inferences_per_joule": round(r.intelligence_inf_per_j, 1),
        "throughput_per_tdp_inferences_per_sec_per_w": round(throughput_per_tdp, 1),
    }


_FIELDS = [
    "name", "release_date", "eol_date", "architecture", "process_node",
    "tdp_w", "latency_ms", "energy_per_inference_mj",
    "avg_power_w_during_inference",
    "perf_inferences_per_sec",
    "energy_efficiency_inferences_per_joule",
    "throughput_per_tdp_inferences_per_sec_per_w",
]


def write_csv(results: List[ProductResult], out: Path) -> None:
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(_row_dict(r))
    print(f"info: wrote {out}", file=sys.stderr)


def write_json(results: List[ProductResult], out: Path) -> None:
    import json
    payload = [_row_dict(r) for r in results]
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"info: wrote {out}", file=sys.stderr)


def write_markdown(results: List[ProductResult], out: Path) -> None:
    headers = _FIELDS
    rows = [_row_dict(r) for r in results]
    lines: list[str] = []
    lines.append("# Jetson industrial roadmap")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"info: wrote {out}", file=sys.stderr)


def write_text(results: List[ProductResult], out: Path) -> None:
    rows = [_row_dict(r) for r in results]
    # Column widths from the longer of header / value
    widths = {h: max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in _FIELDS}
    lines: list[str] = []
    lines.append("Jetson industrial roadmap")
    lines.append("")
    lines.append("  " + "  ".join(h.ljust(widths[h]) for h in _FIELDS))
    lines.append("  " + "  ".join("-" * widths[h] for h in _FIELDS))
    for row in rows:
        lines.append("  " + "  ".join(str(row[h]).ljust(widths[h]) for h in _FIELDS))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"info: wrote {out}", file=sys.stderr)


def write_plot(
    results: List[ProductResult],
    out: Path,
    workload_name: str,
    precision_label: str,
) -> None:
    """Render the roadmap chart with a broken-axis treatment per panel.

    The chart compares conventional silicon (Jetsons + ARM CPU) against
    a KPU. The KPU's metrics on this workload are >100x the conventional
    bars, which collapses the conventional band onto the axis line if
    plotted on a single linear y-axis. To keep both regimes visible
    *and* preserve linear semantics within each regime, each metric
    panel is split into two stacked sub-axes:

      * upper sub-axis: zoomed onto the high-band metric (KPU)
      * lower sub-axis: zoomed onto the low-band metric (others)

    Diagonal slash markers between the sub-axes indicate the y-axis
    discontinuity. Bars are drawn on whichever sub-axis their value
    falls in (bars in the wrong band sit outside ylim and aren't
    rendered).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    results = sorted(results, key=lambda r: r.product.release_date)

    # Partition the products into a high-band (>10x the next-largest
    # value) and the low-band on each metric. In practice the partition
    # is the same for both metrics here, but we compute it per metric
    # so the chart degrades gracefully if we add SKUs later.
    def _partition(values: List[float]) -> tuple[float, float, float]:
        """Return (low_band_max, high_band_min, gap_factor). Gap factor
        > 10 means a broken axis is worth it; otherwise we'd just use
        one axis. Caller decides what to do with the verdict."""
        s = sorted(values)
        # Find the largest gap between adjacent sorted values.
        gaps = [(s[i+1] / max(s[i], 1e-9), i) for i in range(len(s) - 1)]
        ratio, idx = max(gaps, key=lambda t: t[0])
        return s[idx], s[idx + 1], ratio

    perf_vals = [r.perf_inf_per_s for r in results]
    eff_vals = [r.intelligence_inf_per_j for r in results]
    perf_low_max, perf_high_min, perf_gap = _partition(perf_vals)
    eff_low_max, eff_high_min, eff_gap = _partition(eff_vals)

    # gridspec: 5 rows = perf-top / perf-bot / SPACER / eff-top /
    # eff-bot. The middle row is empty so the Performance and Efficiency
    # panels don't run into each other; without it the Efficiency title
    # sits flush against the Performance lower band. The high-band rows
    # get less vertical space because they carry one bar each.
    fig = plt.figure(figsize=(11, 10.5))
    gs = fig.add_gridspec(
        5, 1,
        height_ratios=[1.2, 2.6, 0.5, 1.2, 2.6],
        hspace=0.06,
    )
    ax_perf_top = fig.add_subplot(gs[0])
    ax_perf_bot = fig.add_subplot(gs[1], sharex=ax_perf_top)
    ax_eff_top = fig.add_subplot(gs[3], sharex=ax_perf_top)
    ax_eff_bot = fig.add_subplot(gs[4], sharex=ax_perf_top)

    fig.suptitle(
        f"Compute roadmap @ {precision_label}\n"
        f"Workload: {workload_name}",
        fontsize=12, y=0.995,
    )

    def _draw_lifecycle_bar(axes: tuple, r: ProductResult, y: float,
                            with_label: bool) -> None:
        """Draw the bar on BOTH sub-axes; ylim clips to the right band."""
        legend_label = (
            f"{r.product.name} -- "
            f"{r.product.architecture} on {r.product.process_node}, "
            f"{r.tdp_w:.0f}W"
        ) if with_label else None
        midpoint = (r.product.release_date
                    + (r.product.eol_date - r.product.release_date) / 2)
        for i, ax in enumerate(axes):
            ax.hlines(
                y=y,
                xmin=r.product.release_date,
                xmax=r.product.eol_date,
                color=r.product.color, linewidth=22, alpha=0.85, zorder=2,
                # Only register the legend entry on one sub-axis to
                # avoid duplicate entries.
                label=legend_label if i == 0 else None,
            )
            ax.annotate(
                r.product.name,
                xy=(midpoint, y),
                ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=4,
            )

    # Draw bars
    for r in results:
        _draw_lifecycle_bar((ax_perf_top, ax_perf_bot), r,
                            r.perf_inf_per_s, with_label=True)
        _draw_lifecycle_bar((ax_eff_top, ax_eff_bot), r,
                            r.intelligence_inf_per_j, with_label=False)

    # Y-axis bands. 18% margin around each cluster so bars float in
    # whitespace instead of touching the axis edges.
    def _band_ylim(low: float, high: float) -> tuple[float, float]:
        span = max(high - low, high * 0.1)
        return (low - 0.18 * span, high + 0.18 * span)

    # Performance: low band covers everyone <= perf_low_max, high band
    # covers KPU value(s) >= perf_high_min.
    perf_lows = [v for v in perf_vals if v <= perf_low_max]
    perf_highs = [v for v in perf_vals if v >= perf_high_min]
    ax_perf_bot.set_ylim(*_band_ylim(min(perf_lows), max(perf_lows)))
    ax_perf_top.set_ylim(*_band_ylim(min(perf_highs), max(perf_highs)))

    eff_lows = [v for v in eff_vals if v <= eff_low_max]
    eff_highs = [v for v in eff_vals if v >= eff_high_min]
    ax_eff_bot.set_ylim(*_band_ylim(min(eff_lows), max(eff_lows)))
    ax_eff_top.set_ylim(*_band_ylim(min(eff_highs), max(eff_highs)))

    # Hide spines / ticks at the broken-axis seam, then add diagonal
    # slash markers across the seam so the discontinuity reads visually.
    def _add_break(top_ax, bot_ax) -> None:
        top_ax.spines["bottom"].set_visible(False)
        bot_ax.spines["top"].set_visible(False)
        top_ax.tick_params(axis="x", which="both",
                           bottom=False, labelbottom=False)
        # Diagonal "wiggle" -- short slashes at the break boundary on
        # both sub-axes' left and right spines.
        d = 0.012
        kw = dict(color="black", clip_on=False, linewidth=1.2)
        top_ax.plot((-d, +d), (-d, +d),
                    transform=top_ax.transAxes, **kw)
        top_ax.plot((1 - d, 1 + d), (-d, +d),
                    transform=top_ax.transAxes, **kw)
        bot_ax.plot((-d, +d), (1 - d, 1 + d),
                    transform=bot_ax.transAxes, **kw)
        bot_ax.plot((1 - d, 1 + d), (1 - d, 1 + d),
                    transform=bot_ax.transAxes, **kw)

    _add_break(ax_perf_top, ax_perf_bot)
    _add_break(ax_eff_top, ax_eff_bot)

    # Hide the in-between xticks on the inner axes (perf_bot and
    # eff_top); only the bottom-most axis carries the year labels.
    ax_perf_bot.tick_params(axis="x", which="both",
                            bottom=False, labelbottom=False)
    ax_eff_top.tick_params(axis="x", which="both",
                           bottom=False, labelbottom=False)

    # Titles, labels, grid
    ax_perf_top.set_title("Performance over availability window")
    ax_eff_top.set_title("Efficiency over availability window: "
                         "process + architecture impact")
    # Y-labels span the broken pair: anchor on the bottom axis with
    # adjusted y position so they read as one label per panel.
    ax_perf_bot.set_ylabel("Sustained throughput (inferences / sec)",
                           y=0.7)
    ax_eff_bot.set_ylabel("Energy efficiency (inferences / joule)",
                          y=0.7)
    ax_eff_bot.set_xlabel("Calendar year")
    for ax in (ax_perf_top, ax_perf_bot, ax_eff_top, ax_eff_bot):
        ax.grid(True, alpha=0.3)
    # Place the legend BELOW the chart in a 2-column row so it never
    # collides with bars or the broken-axis seam. Anchored in figure
    # coordinates so it floats below ax_eff_bot regardless of layout.
    handles, labels = ax_perf_top.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        bbox_transform=fig.transFigure,
        fontsize=9, framealpha=0.95, ncol=2,
    )

    # X-axis: fixed 5-year majors so the axis reads as a calendar
    # timeline. Anchored to 2020 -> 2040.
    ax_eff_bot.xaxis.set_major_locator(
        mdates.YearLocator(base=5, month=1, day=1))
    ax_eff_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_eff_bot.xaxis.set_minor_locator(mdates.YearLocator(base=1))
    ax_eff_bot.set_xlim(date(2020, 1, 1), date(2040, 1, 1))

    # Replace the matplotlib "1e6" exponent badge on the top sub-axes
    # with an inline "M" suffix on each tick label -- reads more
    # naturally on a perf chart and avoids the exponent floating into
    # the chart area.
    from matplotlib.ticker import FuncFormatter

    def _abbrev(value: float, _pos) -> str:
        if abs(value) >= 1e6:
            return f"{value / 1e6:.2f}M"
        if abs(value) >= 100e3:
            return f"{value / 1e3:.0f}K"
        if abs(value) >= 1e3:
            # Keep one decimal under 100K so adjacent ticks like
            # 23.5K / 24.0K / 24.5K don't all collapse to "24K".
            return f"{value / 1e3:.1f}K"
        return f"{value:.0f}"

    for ax in (ax_perf_top, ax_perf_bot, ax_eff_top, ax_eff_bot):
        ax.yaxis.set_major_formatter(FuncFormatter(_abbrev))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"info: wrote {out}", file=sys.stderr)
    print(f"info: perf gap = {perf_gap:.1f}x, "
          f"eff gap = {eff_gap:.1f}x", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PRECISION_BY_NAME = {
    "fp32": Precision.FP32,
    "fp16": Precision.FP16,
    "int8": Precision.INT8,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path. PNG (or any matplotlib format) generates the chart; "
             ".csv writes the underlying numbers.",
    )
    parser.add_argument(
        "--precision", default="int8", choices=sorted(PRECISION_BY_NAME),
        help="Inference precision (default int8 -- the only precision all "
             "three Jetsons declare today; Thor's resource model is INT8-only "
             "in the current graphs/ catalog. fp16/fp32 work for Orin NX and "
             "AGX Orin but raise on Thor).",
    )
    parser.add_argument(
        "--width", type=int, default=2048,
        help="Linear width N (Linear(N, N) -> N^2 weights). "
             "Default 2048 -> 4M weights.",
    )
    args = parser.parse_args()

    precision = PRECISION_BY_NAME[args.precision]
    model, input_tensor, workload_name = build_workload(args.width)

    print(f"workload: {workload_name}", file=sys.stderr)
    print(f"precision: {args.precision}", file=sys.stderr)
    print(file=sys.stderr)

    results: List[ProductResult] = []
    skipped: List[RoadmapProduct] = []
    for p in PRODUCTS:
        if p.factory is None:
            skipped.append(p)
            print(
                f"  {p.short_name():28s} SKIPPED -- no resource model in catalog "
                f"({p.architecture}, {p.process_node})",
                file=sys.stderr,
            )
            continue
        r = analyze(p, model, input_tensor, precision, workload_name)
        results.append(r)
        print(
            f"  {p.short_name():28s} "
            f"{r.tdp_w:5.1f}W  "
            f"{r.latency_ms:7.3f} ms  "
            f"{r.energy_mj:7.3f} mJ  "
            f"-> {r.perf_inf_per_s:8.1f} inf/s, "
            f"{r.intelligence_inf_per_j:8.1f} inf/J",
            file=sys.stderr,
        )

    if skipped:
        print(
            f"\nnote: {len(skipped)} SKU(s) skipped because their silicon's "
            "resource model is not in the graphs/ catalog yet. The chart's "
            "data series only covers the analyzable products; the SKU table "
            "in this script's docstring documents the full intended roadmap.",
            file=sys.stderr,
        )

    out = Path(args.output)
    suffix = out.suffix.lower()
    if suffix == ".csv":
        write_csv(results, out)
    elif suffix == ".json":
        write_json(results, out)
    elif suffix in {".md", ".markdown"}:
        write_markdown(results, out)
    elif suffix in {".txt", ".text"}:
        write_text(results, out)
    else:
        # Default: render plot (covers .png, .pdf, .svg, anything matplotlib accepts)
        write_plot(results, out, workload_name, args.precision.upper())

    return 0


if __name__ == "__main__":
    sys.exit(main())

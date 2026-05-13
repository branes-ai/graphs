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
class JetsonProduct:
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
JETSONS: List[JetsonProduct] = [
    JetsonProduct(
        name="Jetson TX2i",
        factory=None,  # Pascal resource model not in catalog yet
        release_date=date(2018, 4, 24),
        eol_date=date(2027, 7, 1),  # LTS Jul 2027 (PCN-accelerated, see docstring)
        architecture="Pascal",
        process_node="TSMC 16FF",
        color="#9467bd",   # tab:purple
        marker="D",
    ),
    JetsonProduct(
        name="Jetson AGX Xavier Industrial",
        factory=None,  # Volta resource model not in catalog yet
        release_date=date(2021, 7, 1),
        eol_date=date(2027, 7, 1),  # LTS Jul 2027 (PCN-accelerated, see docstring)
        architecture="Volta",
        process_node="TSMC 12FFN",
        color="#ff7f0e",   # tab:orange
        marker="s",
    ),
    JetsonProduct(
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
    JetsonProduct(
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
]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass
class ProductResult:
    product: JetsonProduct
    latency_ms: float
    energy_mj: float
    tdp_w: float
    perf_inf_per_s: float
    intelligence_inf_per_j: float


def analyze(
    product: JetsonProduct,
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
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    results = sorted(results, key=lambda r: r.product.release_date)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)

    # Title
    fig.suptitle(
        f"NVIDIA Jetson roadmap @ {precision_label}\n"
        f"Workload: {workload_name}",
        fontsize=12, y=0.995,
    )

    def _draw_lifecycle_bar(ax, r: ProductResult, y: float, with_label: bool) -> None:
        """Horizontal bar from release_date to EOL at metric y-value.

        Bars communicate the *application window* of each SKU: the period
        where it can actually be designed in. Filled marker at the
        release-date end (start of availability), open marker at EOL."""
        legend_label = (
            f"{r.product.short_name()} -- "
            f"{r.product.architecture} on {r.product.process_node}, "
            f"{r.tdp_w:.0f}W "
            f"({r.product.release_date.year}-{r.product.eol_date.year})"
        ) if with_label else None
        ax.hlines(
            y=y,
            xmin=r.product.release_date,
            xmax=r.product.eol_date,
            color=r.product.color, linewidth=8, alpha=0.85, zorder=2,
            label=legend_label,
        )
        # Filled marker at release (in-market start)
        ax.scatter(
            [r.product.release_date], [y],
            c=r.product.color, marker=r.product.marker, s=160,
            edgecolors="black", linewidth=1.0, zorder=3,
        )
        # Open marker at EOL (end-of-availability)
        ax.scatter(
            [r.product.eol_date], [y],
            facecolors="white", edgecolors=r.product.color,
            marker=r.product.marker, s=120, linewidth=1.5, zorder=3,
        )
        ax.annotate(
            r.product.short_name(),
            xy=(r.product.release_date, y),
            xytext=(8, 8), textcoords="offset points", fontsize=9,
        )

    # Panel 1: throughput
    for r in results:
        _draw_lifecycle_bar(ax1, r, r.perf_inf_per_s, with_label=True)
    ax1.set_ylabel("Sustained throughput (inferences / sec)")
    ax1.set_title("Performance over availability window: release date -> EOL")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # Panel 2: energy efficiency
    for r in results:
        _draw_lifecycle_bar(ax2, r, r.intelligence_inf_per_j, with_label=False)
    ax2.set_ylabel("Energy efficiency (inferences / joule)")
    ax2.set_title("Efficiency over availability window: process + architecture impact")
    ax2.set_xlabel("Calendar year (filled = release / GA, open = EOL)")
    ax2.grid(True, alpha=0.3)

    # X-axis formatting -- yearly major ticks, quarterly minors
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

    # Pad the x-axis to cover the full release -> EOL span
    earliest = min(r.product.release_date for r in results)
    latest = max(r.product.eol_date for r in results)
    span_days = (latest - earliest).days
    pad = max(120, span_days // 30)
    from datetime import timedelta
    ax2.set_xlim(earliest - timedelta(days=pad), latest + timedelta(days=pad))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"info: wrote {out}", file=sys.stderr)


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
    skipped: List[JetsonProduct] = []
    for p in JETSONS:
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

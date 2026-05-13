#!/usr/bin/env python
"""NVIDIA Jetson roadmap visualization.

Plots sustained throughput and intelligence-per-Watt for a single
``Linear(D, D) + bias + atan(.)`` workload at batch=1, across the three
relevant Jetson products in chronological release order:

  * Jetson AGX Orin 64GB    (Mar 2022 announce, 2022-Q2 GA, Ampere on Samsung 8nm)
  * Jetson Orin NX 16GB     (Sep 2022 announce, 2023-Q1 GA, Ampere on Samsung 8nm)
  * Jetson AGX Thor 128GB   (GTC 2024 announce, 2025 GA, Blackwell on TSMC 4NP)

Workload: a ``Linear(2048, 2048)`` (4M weights = 4 MB at INT8) plus the
bias add and an elementwise atan activation. At batch=1 this is solidly
memory-bandwidth bound on every Jetson (the matrix has to stream from
LPDDR; on-chip L2 is 2-8 MB and the GPU L2 isn't modeled as a coherent
weight pool -- per-launch L2 set-aside on Hopper+ isn't covered yet). The
operator fuses into one subgraph so the 2K-element activation vectors
stay in registers / shared memory; only the weight matrix and the
external input/output activations cross the DRAM boundary.

Two metrics are plotted vs release date:

  * **Sustained throughput** (inferences / sec) -- 1 / latency
  * **Intelligence per Watt** (inferences / joule) -- 1 / energy_per_inference

Both are derived from the existing roofline + energy model
(``UnifiedAnalyzer``); no measurement here. Each chip is analyzed with
its mapper's default thermal profile (the closest thing to "stock"
behavior) and the per-chip TDP appears in the legend for context.

Usage:
  python cli/jetson_roadmap.py --output jetson_roadmap.png
  python cli/jetson_roadmap.py --output jetson_roadmap.csv
  python cli/jetson_roadmap.py --precision int8 --output roadmap_int8.png
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
    create_jetson_orin_nx_16gb_mapper,
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
    """One Jetson on the roadmap."""

    name: str
    factory: Callable
    release_date: date
    architecture: str
    process_node: str
    color: str
    marker: str

    def short_name(self) -> str:
        return self.name.replace("Jetson ", "").replace(" 64GB", "").replace(" 16GB", "").replace(" 128GB", "")


# Release dates: announced-at-GTC dates rather than discrete-product GA
# (which varies by region / form factor / partner). Chart x-axis is
# year-resolution-ish so a few months of slop doesn't matter.
JETSONS: List[JetsonProduct] = [
    JetsonProduct(
        name="Jetson AGX Orin 64GB",
        factory=create_jetson_orin_agx_64gb_mapper,
        release_date=date(2022, 4, 1),
        architecture="Ampere",
        process_node="Samsung 8nm",
        color="#1f77b4",   # tab:blue
        marker="o",
    ),
    JetsonProduct(
        name="Jetson Orin NX 16GB",
        factory=create_jetson_orin_nx_16gb_mapper,
        release_date=date(2023, 1, 1),
        architecture="Ampere",
        process_node="Samsung 8nm",
        color="#1f77b4",   # tab:blue (same architecture)
        marker="s",
    ),
    JetsonProduct(
        name="Jetson AGX Thor 128GB",
        factory=create_jetson_thor_128gb_mapper,
        release_date=date(2025, 9, 1),
        architecture="Blackwell",
        process_node="TSMC 4NP",
        color="#2ca02c",   # tab:green
        marker="^",
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

def write_csv(results: List[ProductResult], out: Path) -> None:
    fields = [
        "name", "release_date", "architecture", "process_node",
        "tdp_w", "latency_ms", "energy_per_inference_mj",
        "perf_inferences_per_sec", "intelligence_inferences_per_joule",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "name": r.product.name,
                "release_date": r.product.release_date.isoformat(),
                "architecture": r.product.architecture,
                "process_node": r.product.process_node,
                "tdp_w": r.tdp_w,
                "latency_ms": round(r.latency_ms, 3),
                "energy_per_inference_mj": round(r.energy_mj, 3),
                "perf_inferences_per_sec": round(r.perf_inf_per_s, 1),
                "intelligence_inferences_per_joule": round(r.intelligence_inf_per_j, 1),
            })
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

    # Panel 1: throughput
    for r in results:
        legend_label = (
            f"{r.product.short_name()} -- "
            f"{r.product.architecture} on {r.product.process_node}, "
            f"{r.tdp_w:.0f}W"
        )
        ax1.scatter(
            [r.product.release_date], [r.perf_inf_per_s],
            c=r.product.color, marker=r.product.marker, s=220,
            edgecolors="black", linewidth=1.2, zorder=3,
            label=legend_label,
        )
        ax1.annotate(
            r.product.short_name(),
            xy=(r.product.release_date, r.perf_inf_per_s),
            xytext=(10, -4), textcoords="offset points", fontsize=9,
        )
    # Connect points with a thin line so the trajectory reads
    ax1.plot(
        [r.product.release_date for r in results],
        [r.perf_inf_per_s for r in results],
        color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1,
    )
    ax1.set_ylabel("Sustained throughput (inferences / sec)")
    ax1.set_title("Performance: sustained inferences/sec")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # Panel 2: intelligence/Watt
    for r in results:
        ax2.scatter(
            [r.product.release_date], [r.intelligence_inf_per_j],
            c=r.product.color, marker=r.product.marker, s=220,
            edgecolors="black", linewidth=1.2, zorder=3,
        )
        ax2.annotate(
            r.product.short_name(),
            xy=(r.product.release_date, r.intelligence_inf_per_j),
            xytext=(10, -4), textcoords="offset points", fontsize=9,
        )
    ax2.plot(
        [r.product.release_date for r in results],
        [r.intelligence_inf_per_j for r in results],
        color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1,
    )
    ax2.set_ylabel("Intelligence per Watt (inferences / joule)")
    ax2.set_title("Efficiency: process + architecture impact on intelligence-per-Watt")
    ax2.set_xlabel("Release date")
    ax2.grid(True, alpha=0.3)

    # X-axis formatting -- yearly major ticks, quarterly minors
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

    # Pad the x-axis a bit on each side
    earliest = min(r.product.release_date for r in results)
    latest = max(r.product.release_date for r in results)
    span_days = (latest - earliest).days
    pad = max(120, span_days // 10)
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
    for p in JETSONS:
        r = analyze(p, model, input_tensor, precision, workload_name)
        results.append(r)
        print(
            f"  {p.short_name():22s} "
            f"{r.tdp_w:5.1f}W  "
            f"{r.latency_ms:7.3f} ms  "
            f"{r.energy_mj:7.3f} mJ  "
            f"-> {r.perf_inf_per_s:8.1f} inf/s, "
            f"{r.intelligence_inf_per_j:8.1f} inf/J",
            file=sys.stderr,
        )

    out = Path(args.output)
    if out.suffix.lower() == ".csv":
        write_csv(results, out)
    else:
        write_plot(results, out, workload_name, args.precision.upper())

    return 0


if __name__ == "__main__":
    sys.exit(main())

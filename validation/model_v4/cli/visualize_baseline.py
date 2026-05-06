#!/usr/bin/env python
"""Visualize the per-shape performance spread for one (hardware, dtype) target.

Renders a 2-panel figure from a committed V4 baseline CSV:

    Panel 1 -- Roofline plot
        x = arithmetic intensity (FLOPS / byte)
        y = achieved throughput (GFLOPS)
        Both axes log. The "roof" is min(peak_FLOPS, AI * peak_DRAM_BW)
        drawn from the hardware mapper. Each shape is one point;
        matmul = circle, linear = triangle; color = predicted regime
        (per the V4 classifier).

    Panel 2 -- Latency vs working-set
        x = working-set bytes (input + weights + output)
        y = measured latency (ms)
        Same marker / color encoding. Phase transitions at L1 / L2 /
        DRAM boundaries become visible as kinks in the cloud.

Together these answer "how does shape modulate performance on this
hardware?" -- the original question that motivated the V4 sweep design.
The roofline panel exposes whether each shape is compute-bound vs
memory-bound; the latency-vs-WS panel exposes the cache-hierarchy
phase transitions.

Usage:
    python -m validation.model_v4.cli.visualize_baseline --hw jetson_orin_nano_8gb
    python -m validation.model_v4.cli.visualize_baseline --hw i7_12700k --out /tmp/i7.png

Without --out the file lands at validation/model_v4/results/plots/<hw>_roofline.png.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

from graphs.hardware.mappers import get_mapper_by_name
from graphs.hardware.resource_model import HardwareResourceModel, Precision

from validation.model_v4.harness.runner import SWEEP_HW_TO_MAPPER
from validation.model_v4.sweeps.classify import (
    Regime,
    bytes_per_element,
    op_footprint,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = REPO_ROOT / "validation" / "model_v4" / "results" / "baselines"
SWEEP_DIR = REPO_ROOT / "validation" / "model_v4" / "sweeps"
DEFAULT_PLOT_DIR = REPO_ROOT / "validation" / "model_v4" / "results" / "plots"


# Color per regime. Chosen to match the v4 plan's heatmap convention:
# greens = ALU-side, blues = on-chip BW, oranges = off-chip BW, gray = launch.
_REGIME_COLORS = {
    Regime.ALU_BOUND.value:     "#2ca02c",   # green
    Regime.L1_BOUND.value:      "#17becf",   # teal
    Regime.L2_BOUND.value:      "#1f77b4",   # blue
    Regime.DRAM_BOUND.value:    "#ff7f0e",   # orange
    Regime.LAUNCH_BOUND.value:  "#7f7f7f",   # gray
    Regime.AMBIGUOUS.value:     "#bcbd22",   # olive (rare; sweep generator filters most)
    Regime.UNSUPPORTED.value:   "#d62728",   # red (shouldn't appear in committed CSV)
}

# Markers per op
_OP_MARKERS = {"matmul": "o", "linear": "^"}


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------


def _parse_shape_string(s: str) -> tuple[int, ...]:
    """Baseline CSV shape column is e.g. "1024x1024x1024"."""
    return tuple(int(x) for x in s.split("x"))


def _parse_legacy_tuple(s: str) -> tuple[int, ...]:
    """Some CSVs (early V4-3) used Python-tuple literal in the shape column."""
    return tuple(ast.literal_eval(s))


def _shape_from_csv(s: str) -> tuple[int, ...]:
    if "x" in s:
        return _parse_shape_string(s)
    return _parse_legacy_tuple(s)


def load_baseline_rows(hw_key: str, op: str) -> list[dict]:
    """Read the per-shape baseline CSV for (hw, op). Returns one dict per row,
    with parsed shape tuple, latency_s as float, energy_j as Optional[float]."""
    path = BASELINE_DIR / f"{hw_key}_{op}.csv"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                latency_s = float(r["latency_s"])
            except (KeyError, ValueError):
                continue
            energy_j_raw = r.get("energy_j", "")
            try:
                energy_j = float(energy_j_raw) if energy_j_raw else None
            except ValueError:
                energy_j = None
            rows.append({
                "op": op,
                "shape": _shape_from_csv(r["shape"]),
                "dtype": r["dtype"],
                "latency_s": latency_s,
                "energy_j": energy_j,
            })
    return rows


def load_sweep_regimes(hw_key: str, op: str) -> dict[tuple, str]:
    """Load the validation+calibration sweeps and return
    {(shape, dtype): regime} for ``hw_key``.

    Both calibration and validation entries are merged because the
    baseline CSV combines them by op (cache key is (hw, op, shape,
    dtype) without a purpose dimension)."""
    out: dict[tuple, str] = {}
    for purpose in ("calibration", "validation"):
        path = SWEEP_DIR / f"{op}_{purpose}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for entry in payload["shapes"]:
            regime = entry["regime_per_hw"].get(hw_key)
            if regime is None:
                continue
            key = (tuple(entry["shape"]), entry["dtype"])
            out[key] = regime
    return out


def _resolve_precision(dtype: str) -> Precision:
    table = {
        "fp64": Precision.FP64, "fp32": Precision.FP32, "tf32": Precision.TF32,
        "fp16": Precision.FP16, "bf16": Precision.BF16,
        "fp8": Precision.FP8, "fp8_e4m3": Precision.FP8_E4M3,
        "fp8_e5m2": Precision.FP8_E5M2,
        "int64": Precision.INT64, "int32": Precision.INT32,
        "int16": Precision.INT16, "int8": Precision.INT8,
        "int4": Precision.INT4, "fp4": Precision.FP4,
    }
    return table[dtype.lower()]


def _peak_flops_for_dtype(hw: HardwareResourceModel, dtype: str) -> float:
    """Return effective peak FLOPS for the dtype, or 0 if unsupported."""
    from graphs.estimation.roofline import RooflineAnalyzer
    try:
        return RooflineAnalyzer._get_effective_peak_ops(hw, _resolve_precision(dtype))
    except (ValueError, KeyError):
        return 0.0


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def render_baseline_plot(hw_key: str, out_path: Path) -> None:
    # Lazy import: keeps the harness importable on hosts without matplotlib.
    import matplotlib.pyplot as plt
    import numpy as np

    if hw_key not in SWEEP_HW_TO_MAPPER:
        raise ValueError(
            f"Hardware key {hw_key!r} has no mapper-registry translation. "
            f"Known: {sorted(SWEEP_HW_TO_MAPPER)}")
    mapper = get_mapper_by_name(SWEEP_HW_TO_MAPPER[hw_key])
    hw = mapper.resource_model

    # Gather rows for both ops
    rows: list[dict] = []
    for op in ("matmul", "linear"):
        baseline_rows = load_baseline_rows(hw_key, op)
        regimes = load_sweep_regimes(hw_key, op)
        for r in baseline_rows:
            key = (tuple(r["shape"]), r["dtype"])
            r["regime"] = regimes.get(key, Regime.AMBIGUOUS.value)
            try:
                fp = op_footprint(op, r["shape"], r["dtype"])
            except (ValueError, ZeroDivisionError):
                continue
            r["flops"] = fp.flops
            r["working_set_bytes"] = fp.working_set_bytes
            r["operational_intensity"] = fp.operational_intensity
            r["achieved_gflops"] = fp.flops / r["latency_s"] / 1e9 if r["latency_s"] > 0 else 0.0
            r["latency_ms"] = r["latency_s"] * 1e3
            rows.append(r)

    if not rows:
        raise SystemExit(
            f"No baseline data found for {hw_key!r}. Expected one or both of "
            f"{BASELINE_DIR}/{hw_key}_matmul.csv, "
            f"{BASELINE_DIR}/{hw_key}_linear.csv")

    # ----- Figure -----
    fig, (ax_roof, ax_lat) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"V4 baseline performance spread -- {hw.name}  "
        f"({sum(1 for r in rows if r['op']=='matmul')} matmul + "
        f"{sum(1 for r in rows if r['op']=='linear')} linear shapes)",
        fontsize=12, y=0.99)

    # Per-dtype peak lines for the roofline (one set of rooflines per
    # dtype present in the data; usually a single dtype dominates)
    dtypes_present = sorted({r["dtype"] for r in rows})

    # ----- Panel 1: roofline -----
    peak_bw = hw.peak_bandwidth   # bytes/sec, dtype-independent (DRAM)
    for dtype in dtypes_present:
        peak_fp = _peak_flops_for_dtype(hw, dtype)
        if peak_fp <= 0:
            continue
        # AI breakpoint: above it, compute is the ceiling; below, BW.
        ai_breakpoint = peak_fp / peak_bw   # FLOPS/byte
        # Plot range: 0.1 to 10000 FLOPS/byte covers all realistic shapes
        ai_grid = np.logspace(-1, 4, 256)
        achievable_gflops = np.minimum(peak_fp, ai_grid * peak_bw) / 1e9
        ax_roof.plot(ai_grid, achievable_gflops, linestyle="--",
                     linewidth=1.2, alpha=0.6,
                     label=f"roof ({dtype}, peak={peak_fp/1e9:.0f} GFLOPS, "
                           f"BW={peak_bw/1e9:.0f} GB/s, AI_break={ai_breakpoint:.1f})")

    # Scatter the per-shape points
    for op in ("matmul", "linear"):
        for regime in _REGIME_COLORS:
            xs, ys = [], []
            for r in rows:
                if r["op"] != op or r["regime"] != regime:
                    continue
                if r["operational_intensity"] <= 0 or r["achieved_gflops"] <= 0:
                    continue
                xs.append(r["operational_intensity"])
                ys.append(r["achieved_gflops"])
            if xs:
                ax_roof.scatter(
                    xs, ys, s=36, alpha=0.75,
                    marker=_OP_MARKERS[op],
                    color=_REGIME_COLORS[regime],
                    edgecolor="black", linewidth=0.4,
                    label=None,   # legend assembled separately below
                )

    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlabel("Arithmetic intensity (FLOPS / byte)")
    ax_roof.set_ylabel("Achieved throughput (GFLOPS)")
    ax_roof.set_title("Roofline: where each shape lands vs the architectural ceiling")
    ax_roof.grid(True, which="both", alpha=0.25)
    ax_roof.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # ----- Panel 2: latency vs working set -----
    for op in ("matmul", "linear"):
        for regime in _REGIME_COLORS:
            xs, ys = [], []
            for r in rows:
                if r["op"] != op or r["regime"] != regime:
                    continue
                if r["working_set_bytes"] <= 0 or r["latency_ms"] <= 0:
                    continue
                xs.append(r["working_set_bytes"])
                ys.append(r["latency_ms"])
            if xs:
                ax_lat.scatter(
                    xs, ys, s=36, alpha=0.75,
                    marker=_OP_MARKERS[op],
                    color=_REGIME_COLORS[regime],
                    edgecolor="black", linewidth=0.4,
                )

    # Cache-hierarchy reference lines
    l1_total = hw.compute_units * hw.l1_cache_per_unit
    l2_total = hw.l2_cache_total or 0
    if l1_total:
        ax_lat.axvline(l1_total, color="#2ca02c", linestyle=":", alpha=0.5,
                       label=f"L1 total ({l1_total/2**20:.1f} MiB)")
    if l2_total:
        ax_lat.axvline(l2_total, color="#1f77b4", linestyle=":", alpha=0.5,
                       label=f"L2/LLC total ({l2_total/2**20:.1f} MiB)")

    ax_lat.set_xscale("log")
    ax_lat.set_yscale("log")
    ax_lat.set_xlabel("Working-set bytes")
    ax_lat.set_ylabel("Measured latency (ms)")
    ax_lat.set_title("Latency vs working set: cache-hierarchy phase transitions")
    ax_lat.grid(True, which="both", alpha=0.25)
    ax_lat.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # ----- Shared regime/op legend (below the panels) -----
    from matplotlib.lines import Line2D
    handles = []
    for regime, color in _REGIME_COLORS.items():
        # Only include regimes actually present in the data
        if not any(r["regime"] == regime for r in rows):
            continue
        handles.append(Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=color,
            markersize=9, label=regime, markeredgecolor="black"))
    for op, marker in _OP_MARKERS.items():
        if not any(r["op"] == op for r in rows):
            continue
        handles.append(Line2D(
            [0], [0], marker=marker, color="black", markerfacecolor="white",
            markersize=9, label=op, linestyle="None"))
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 7), bbox_to_anchor=(0.5, -0.01),
               fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hw", required=True,
                   choices=sorted(SWEEP_HW_TO_MAPPER),
                   help="hardware key (must have a baseline CSV under "
                        "validation/model_v4/results/baselines/)")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG path (default: "
                        "validation/model_v4/results/plots/<hw>_roofline.png)")
    args = p.parse_args(argv)

    out = args.out or (DEFAULT_PLOT_DIR / f"{args.hw}_roofline.png")
    render_baseline_plot(args.hw, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

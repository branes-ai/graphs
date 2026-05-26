#!/usr/bin/env python3
"""KPU family scaling study: T64 -> T128 -> T256 -> T768 (issue #122).

Extends the 3-way comparison backbone (compare_hardware.py) into an N-way
study of how the Stillwater KPU family scales with tile count. Answers the
embodied-AI-architect's question: where does a bigger KPU pay off, and where
does it just sit underutilized?

For a sweep of matmul / vector_add shapes in BF16 (the KPU's native fabric),
it computes per-SKU:

  - burst + thermally-bound (sustained) latency
  - energy per inference (J)
  - energy efficiency (inferences / J == throughput-per-watt)
  - compute utilization (achieved FLOP/s / peak FLOP/s)

and renders a multi-panel figure plus a data table. The energy panels are
meaningful now that the KPU energy model is calibrated (#177/#121 clamp,
#154 leakage-Vdd, #81 node-sourced dynamic energy).

Usage:
    python -m validation.model_v4.cli.compare_kpu_scaling --output kpu_scaling.png
    python -m validation.model_v4.cli.compare_kpu_scaling --output kpu_scaling.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from graphs.hardware.mappers import get_mapper_by_name
from graphs.hardware.resource_model import Precision
from validation.model_v4.cli.compare_hardware import (
    _draw_roof_for_hw,
    _enrich_predictions,
)
from validation.model_v4.harness.runner import SWEEP_HW_TO_MAPPER

# KPU family in capability order. Drives legend / layer order.
KPU_FAMILY = ["kpu_t64", "kpu_t128", "kpu_t256", "kpu_t768"]

# Sequential single-hue ramp so "bigger KPU" reads as "darker".
_KPU_COLORS = {
    "kpu_t64": "#FDAE61",   # light orange
    "kpu_t128": "#F46D43",  # orange
    "kpu_t256": "#D73027",  # red
    "kpu_t768": "#A50026",  # dark red
}

DEFAULT_PRECISION = "bf16"

# Square matmul shapes (M=K=N) spanning the small->large regime, plus a
# vector_add memory-bound sweep. Working-set size is the x-axis.
_MATMUL_DIMS = [256, 512, 1024, 2048, 4096]
_VECTOR_ADD_N = [1 << 16, 1 << 20, 1 << 22, 1 << 24, 1 << 26]


def _precision_enum(dtype: str) -> Precision:
    return Precision[dtype.upper()]


def kpu_scaling_rows(
    family: Optional[List[str]] = None,
    dtype: str = DEFAULT_PRECISION,
) -> List[dict]:
    """Enrich every (SKU, shape) pair and annotate with the scaling metrics.

    Returns a flat list of dicts (one per SKU x shape) carrying the
    _enrich_predictions fields plus: hw_key, efficiency_inf_per_j,
    utilization. None-prediction rows are dropped.
    """
    family = family or KPU_FAMILY
    rows: List[dict] = []
    for hw_key in family:
        mapper = get_mapper_by_name(SWEEP_HW_TO_MAPPER[hw_key])
        hw = mapper.resource_model
        peak_flops = hw.get_peak_ops(_precision_enum(dtype))
        shapes = [("matmul", (d, d, d)) for d in _MATMUL_DIMS]
        shapes += [("vector_add", (n,)) for n in _VECTOR_ADD_N]
        for op, shape in shapes:
            e = _enrich_predictions(op, shape, dtype, hw)
            if e is None or not e.get("predicted_energy_j"):
                continue
            e["hw_key"] = hw_key
            # Efficiency == throughput-per-watt: (1/sustained_latency)/avg_power
            # == 1 / energy_per_inference. Both panels read from this.
            e["efficiency_inf_per_j"] = 1.0 / e["predicted_energy_j"]
            # Utilization: achieved FLOP/s (burst) vs peak FLOP/s.
            achieved_flops = e["predicted_gflops"] * 1e9
            e["utilization"] = (
                achieved_flops / peak_flops if peak_flops > 0 else 0.0
            )
            rows.append(e)
    return rows


def summarize_crossovers(rows: List[dict]) -> List[str]:
    """Surface the scaling questions from #122 as text findings:
    where a bigger KPU stops winning on latency, and where it is
    underutilized. Compares adjacent family members at each matmul shape."""
    findings: List[str] = []
    by_key = {k: {} for k in KPU_FAMILY}
    for r in rows:
        if r["op"] == "matmul":
            by_key[r["hw_key"]][r["shape"]] = r

    pairs = list(zip(KPU_FAMILY, KPU_FAMILY[1:]))
    for small, big in pairs:
        shapes = sorted(by_key[small].keys() & by_key[big].keys())
        # Smallest shape where the bigger SKU is meaningfully (>5%) faster.
        payoff = None
        for s in shapes:
            sl = by_key[small][s]["predicted_latency_ms"]
            bl = by_key[big][s]["predicted_latency_ms"]
            if bl < sl * 0.95:
                payoff = (s, sl, bl)
                break
        if payoff:
            s, sl, bl = payoff
            findings.append(
                f"{big} first beats {small} at matmul{s}: "
                f"{bl:.4f} ms vs {sl:.4f} ms ({sl/bl:.2f}x)."
            )
        else:
            findings.append(
                f"{big} never beats {small} by >5% across the matmul sweep "
                f"-- {big}'s extra fabric is wave-quantization-bound here."
            )
    # Underutilization: largest matmul utilization per SKU.
    for k in KPU_FAMILY:
        if not by_key[k]:
            continue
        biggest = max(by_key[k], key=lambda s: s[0] * s[1] * s[2])
        u = by_key[k][biggest]["utilization"]
        findings.append(
            f"{k} peak matmul utilization (at {biggest}): {u * 100:.0f}%."
        )
    return findings


def render_kpu_scaling(
    rows: List[dict], out_path: Path, dtype: str = DEFAULT_PRECISION
) -> None:
    """5-panel scaling figure: roofline, latency, energy/inf, efficiency,
    utilization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    ax_roof, ax_lat, ax_egy = axes[0]
    ax_eff, ax_util, ax_blank = axes[1]
    ax_blank.axis("off")

    def _rows(hw_key, op):
        return [r for r in rows if r["hw_key"] == hw_key and r["op"] == op]

    for hw_key in KPU_FAMILY:
        color = _KPU_COLORS[hw_key]
        hw = get_mapper_by_name(SWEEP_HW_TO_MAPPER[hw_key]).resource_model
        _draw_roof_for_hw(ax_roof, hw, dtype, color, hw_key.upper())
        for op in ("matmul", "vector_add"):
            rs = sorted(_rows(hw_key, op), key=lambda r: r["working_set_bytes"])
            if not rs:
                continue
            ws = [r["working_set_bytes"] for r in rs]
            style = "-o" if op == "matmul" else "--s"
            lbl = f"{hw_key.upper()} {op}"
            ax_lat.plot(ws, [r["predicted_latency_ms"] for r in rs], style,
                        color=color, label=lbl, markersize=5)
            ax_egy.plot(ws, [r["predicted_energy_j"] * 1e3 for r in rs], style,
                        color=color, label=lbl, markersize=5)
            ax_eff.plot(ws, [r["efficiency_inf_per_j"] for r in rs], style,
                        color=color, label=lbl, markersize=5)
            ax_util.plot(ws, [r["utilization"] * 100 for r in rs], style,
                         color=color, label=lbl, markersize=5)

    ax_roof.set_title(f"Roofline ({dtype})")
    ax_lat.set(xscale="log", yscale="log", xlabel="working set (bytes)",
               ylabel="latency (ms)", title="Latency vs working set")
    ax_egy.set(xscale="log", yscale="log", xlabel="working set (bytes)",
               ylabel="energy / inference (mJ)", title="Energy per inference")
    ax_eff.set(xscale="log", yscale="log", xlabel="working set (bytes)",
               ylabel="efficiency (inferences / J)",
               title="Energy efficiency (== throughput / W)")
    ax_util.set(xscale="log", xlabel="working set (bytes)",
                ylabel="compute utilization (%)",
                title="Utilization vs working set")
    for ax in (ax_lat, ax_egy, ax_eff, ax_util):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Stillwater KPU family scaling study (T64 -> T128 -> T256 -> T768)",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_table(rows: List[dict], out_path: Path) -> None:
    """Emit the scaling data as CSV / Markdown / JSON (by extension)."""
    cols = ["hw_key", "op", "shape", "dtype", "working_set_bytes",
            "predicted_latency_ms", "thermally_bound_latency_ms",
            "predicted_gflops", "predicted_energy_j", "predicted_avg_power_w",
            "efficiency_inf_per_j", "utilization"]
    ext = out_path.suffix.lower()
    if ext == ".json":
        import json
        out_path.write_text(json.dumps(
            [{c: r.get(c) for c in cols} for r in rows], indent=2, default=str))
        return
    sep = "," if ext == ".csv" else " | "
    lines = [sep.join(cols)]
    if ext in (".md", ".markdown"):
        lines.append(sep.join(["---"] * len(cols)))
    for r in rows:
        lines.append(sep.join(str(r.get(c, "")) for c in cols))
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", "-o", required=True,
                   help="Output path. .png/.pdf renders the figure; "
                        ".csv/.md/.json writes the data table.")
    p.add_argument("--dtype", default=DEFAULT_PRECISION,
                   help="Comparison precision (default: bf16, KPU-native).")
    args = p.parse_args(argv)

    rows = kpu_scaling_rows(dtype=args.dtype)
    if not rows:
        print("error: no KPU predictions produced", file=sys.stderr)
        return 1

    out = Path(args.output)
    if out.suffix.lower() in (".png", ".pdf", ".svg"):
        render_kpu_scaling(rows, out, dtype=args.dtype)
    else:
        _write_table(rows, out)
    print(f"info: wrote {out} ({len(rows)} rows across {len(KPU_FAMILY)} KPUs)")

    print("\nScaling findings:")
    for f in summarize_crossovers(rows):
        print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

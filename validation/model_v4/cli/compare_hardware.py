#!/usr/bin/env python
"""Side-by-side hardware comparison plot.

Renders a single 1x3 figure overlaying analyzer predictions for several
hardware targets on a shared shape pool (the matmul + linear + vector_add
sweep validation set):

    Panel 1 -- Roofline
        x = arithmetic intensity (FLOPS / byte), log scale
        y = achievable throughput (GFLOPS), log scale
        One dashed roof line per hardware (peak FLOPS / DRAM BW + AI
        breakpoint annotation). Predicted markers for each hardware
        color-coded; one marker per (shape, hardware) pair.

    Panel 2 -- Latency vs working-set
        x = working-set bytes, log scale
        y = predicted latency (ms), log scale
        Cache reference lines from the FIRST hardware listed (the
        comparison anchor). Each hardware has a colored marker series.

    Panel 3 -- Energy per inference vs working-set
        x = working-set bytes, log scale
        y = predicted energy (joules), log scale
        TDP-budget reference lines per hardware (energy = TDP * latency
        is the upper bound). Same color encoding as panel 2.

Each hardware uses its NATIVE precision (CPU -> fp32, GPU -> fp16,
KPU -> fp16) so the comparison is "what each chip would actually
deliver if you targeted its preferred precision," not "all run at the
same precision." Override per-hw via --dtype-for HARDWARE=DTYPE.

Usage:
    python -m validation.model_v4.cli.compare_hardware
    python -m validation.model_v4.cli.compare_hardware --hw i7_12700k jetson_orin_nano_8gb kpu_t64
    python -m validation.model_v4.cli.compare_hardware --out /tmp/compare.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from graphs.hardware.mappers import get_mapper_by_name
from graphs.hardware.resource_model import HardwareResourceModel

from validation.model_v4.cli.visualize_baseline import (
    DEFAULT_PLOT_DIR,
    _peak_flops_for_dtype,
    load_sweep_shapes_for_predictions,
    _resolve_precision,
)
from validation.model_v4.harness.runner import (
    SWEEP_HW_TO_MAPPER,
    _build_subgraph,
    _predict_energy_j,
    _predict_latency_s,
)
from validation.model_v4.sweeps.classify import op_footprint


# Default hardware to compare. Order = legend order = layer order in the plot.
DEFAULT_TARGETS = ["i7_12700k", "jetson_orin_nano_8gb", "kpu_t64"]

# Default dtype per hardware (native). Override via --dtype-for.
DEFAULT_DTYPES = {
    "i7_12700k": "fp32",
    "h100_sxm5_80gb": "fp16",
    "jetson_orin_nano_8gb": "fp16",
    "jetson_orin_agx_64gb": "fp16",
    "jetson_orin_nx_16gb": "fp16",
    "kpu_t64": "fp16",  # KPU's BF16 fabric also delivers fp16 at the same TFLOPS
}

# Distinct, colorblind-friendly per-hardware colors.
_HW_COLORS = {
    "i7_12700k":             "#4477AA",  # blue
    "jetson_orin_nano_8gb":  "#228833",  # green
    "kpu_t64":               "#EE6677",  # red
    "h100_sxm5_80gb":        "#CCBB44",  # yellow (datacenter GPU)
    "jetson_orin_agx_64gb":  "#66CCEE",  # cyan
    "jetson_orin_nx_16gb":   "#AA3377",  # purple
}

_OP_MARKERS = {"matmul": "o", "linear": "^", "vector_add": "s"}


def _enrich_predictions(
    op: str, shape: tuple, dtype: str, hw: HardwareResourceModel,
) -> Optional[dict]:
    """Run the analyzer in tier-aware mode (V5-3b production default).

    Returns a dict with operational_intensity, working_set_bytes,
    predicted_latency_ms, predicted_gflops, predicted_energy_j,
    predicted_avg_power_w. Returns None if the analyzer can't predict
    (e.g. dtype not supported by the mapper)."""
    try:
        fp = op_footprint(op, shape, dtype)
    except (ValueError, ZeroDivisionError):
        return None
    if fp.flops <= 0 or fp.working_set_bytes <= 0:
        return None
    try:
        precision = _resolve_precision(dtype)
        sg = _build_subgraph(op, shape, dtype)
        pred_lat, _ = _predict_latency_s(
            sg, hw, precision, use_tier_aware_memory=True,
        )
        pred_egy = _predict_energy_j(sg, hw, precision, pred_lat)
    except (ValueError, KeyError, AttributeError, ZeroDivisionError) as e:
        print(
            f"  warning: prediction failed for {op}{tuple(shape)} {dtype} "
            f"on {hw.name}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return None
    if pred_lat <= 0:
        return None
    return {
        "op": op,
        "shape": tuple(shape),
        "dtype": dtype,
        "operational_intensity": fp.operational_intensity,
        "working_set_bytes": fp.working_set_bytes,
        "predicted_latency_ms": pred_lat * 1e3,
        "predicted_gflops": fp.flops / pred_lat / 1e9,
        "predicted_energy_j": pred_egy if pred_egy is not None else None,
        "predicted_avg_power_w": (
            pred_egy / pred_lat if pred_egy and pred_lat > 0 else None
        ),
    }


def _draw_roof_for_hw(ax, hw, dtype, color, label_prefix):
    import numpy as np

    peak_fp = _peak_flops_for_dtype(hw, dtype)
    if peak_fp <= 0:
        return
    peak_bw = hw.peak_bandwidth
    ai_break = peak_fp / peak_bw if peak_bw > 0 else 0
    ai_grid = np.logspace(-1, 4, 256)
    achievable = np.minimum(peak_fp, ai_grid * peak_bw) / 1e9
    ax.plot(
        ai_grid, achievable, linestyle="--", linewidth=1.4,
        color=color, alpha=0.75,
        label=(
            f"{label_prefix} ({dtype}: peak={peak_fp/1e12:.2f} TFLOPS, "
            f"BW={peak_bw/1e9:.0f} GB/s, AI_break={ai_break:.0f})"
        ),
    )
    # Mark the AI breakpoint with a vertical guide.
    ax.axvline(ai_break, color=color, linestyle=":", linewidth=0.8,
               alpha=0.4)


def _draw_cache_lines(ax, hw, color):
    """Draw cache-tier guides for the comparison anchor hardware."""
    l1_total = hw.compute_units * hw.l1_cache_per_unit
    l2_total = hw.l2_cache_total or 0
    if l1_total:
        ax.axvline(
            l1_total, color=color, linestyle=":", alpha=0.35, linewidth=1.0,
            label=f"{hw.name} L1 ({l1_total/2**20:.1f} MiB)",
        )
    if l2_total:
        ax.axvline(
            l2_total, color=color, linestyle="-.", alpha=0.35, linewidth=1.0,
            label=f"{hw.name} L2 ({l2_total/2**20:.1f} MiB)",
        )


def render_comparison(
    hw_keys: List[str],
    out_path: Path,
    dtype_overrides: Optional[Dict[str, str]] = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dtype_overrides = dtype_overrides or {}

    # Resolve mappers + per-hw dtype + per-hw rows.
    targets = []
    for hw_key in hw_keys:
        if hw_key not in SWEEP_HW_TO_MAPPER:
            raise ValueError(
                f"Hardware key {hw_key!r} unknown. "
                f"Known: {sorted(SWEEP_HW_TO_MAPPER)}"
            )
        mapper = get_mapper_by_name(SWEEP_HW_TO_MAPPER[hw_key])
        hw = mapper.resource_model
        dtype = dtype_overrides.get(hw_key, DEFAULT_DTYPES.get(hw_key, "fp16"))
        targets.append((hw_key, hw, dtype))

    print("Generating predictions for each hardware:")
    rows_by_hw: Dict[str, List[dict]] = {}
    for hw_key, hw, dtype in targets:
        per_hw_rows: List[dict] = []
        for op in ("matmul", "linear", "vector_add"):
            for shape_row in load_sweep_shapes_for_predictions(op, dtype):
                e = _enrich_predictions(op, shape_row["shape"], dtype, hw)
                if e is not None:
                    per_hw_rows.append(e)
        rows_by_hw[hw_key] = per_hw_rows
        n_mm = sum(1 for r in per_hw_rows if r["op"] == "matmul")
        n_lin = sum(1 for r in per_hw_rows if r["op"] == "linear")
        n_va = sum(1 for r in per_hw_rows if r["op"] == "vector_add")
        print(
            f"  {hw_key:>22s} ({dtype}): "
            f"{n_mm} matmul + {n_lin} linear + {n_va} vector_add"
        )

    fig, (ax_roof, ax_lat, ax_egy) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Hardware comparison -- analyzer predictions, V5-3b tier-aware "
        f"(predictions-only; native dtype per hardware)",
        fontsize=12, y=0.995,
    )

    # ---- Panel 1: roofline ----
    for hw_key, hw, dtype in targets:
        color = _HW_COLORS.get(hw_key, "#444444")
        _draw_roof_for_hw(ax_roof, hw, dtype, color, hw.name)
        for op, marker in _OP_MARKERS.items():
            xs = [r["operational_intensity"] for r in rows_by_hw[hw_key] if r["op"] == op]
            ys = [r["predicted_gflops"] for r in rows_by_hw[hw_key] if r["op"] == op]
            if xs:
                ax_roof.scatter(
                    xs, ys, s=28, marker=marker, alpha=0.6,
                    facecolors="none", edgecolor=color, linewidth=1.0,
                )
    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlabel("Arithmetic intensity (FLOPS / byte)")
    ax_roof.set_ylabel("Achievable throughput (GFLOPS)")
    ax_roof.set_title("Roofline (peak roofs + predicted markers)")
    ax_roof.grid(True, which="both", alpha=0.25)
    ax_roof.legend(fontsize=7, loc="lower right", framealpha=0.85)

    # ---- Panel 2: latency vs WS ----
    # Cache-line reference from the first listed hardware (anchor).
    anchor_hw_key, anchor_hw, _ = targets[0]
    _draw_cache_lines(ax_lat, anchor_hw, _HW_COLORS.get(anchor_hw_key, "#444"))
    for hw_key, hw, dtype in targets:
        color = _HW_COLORS.get(hw_key, "#444444")
        for op, marker in _OP_MARKERS.items():
            xs = [r["working_set_bytes"] for r in rows_by_hw[hw_key] if r["op"] == op]
            ys = [r["predicted_latency_ms"] for r in rows_by_hw[hw_key] if r["op"] == op]
            if xs:
                ax_lat.scatter(
                    xs, ys, s=28, marker=marker, alpha=0.6,
                    facecolors="none", edgecolor=color, linewidth=1.0,
                )
    ax_lat.set_xscale("log")
    ax_lat.set_yscale("log")
    ax_lat.set_xlabel("Working-set bytes")
    ax_lat.set_ylabel("Predicted latency (ms)")
    ax_lat.set_title("Latency vs working-set")
    ax_lat.grid(True, which="both", alpha=0.25)
    ax_lat.legend(fontsize=7, loc="upper left", framealpha=0.85)

    # ---- Panel 3: energy per inference vs WS ----
    for hw_key, hw, dtype in targets:
        color = _HW_COLORS.get(hw_key, "#444444")
        for op, marker in _OP_MARKERS.items():
            xs, ys = [], []
            for r in rows_by_hw[hw_key]:
                if r["op"] != op or r["predicted_energy_j"] is None:
                    continue
                if r["working_set_bytes"] <= 0 or r["predicted_energy_j"] <= 0:
                    continue
                xs.append(r["working_set_bytes"])
                ys.append(r["predicted_energy_j"])
            if xs:
                ax_egy.scatter(
                    xs, ys, s=28, marker=marker, alpha=0.6,
                    facecolors="none", edgecolor=color, linewidth=1.0,
                )
    ax_egy.set_xscale("log")
    ax_egy.set_yscale("log")
    ax_egy.set_xlabel("Working-set bytes")
    ax_egy.set_ylabel("Predicted energy per inference (J)")
    ax_egy.set_title("Energy per inference vs working-set")
    ax_egy.grid(True, which="both", alpha=0.25)

    # Shared legend below all panels: hardware (color) + op (marker).
    handles = []
    for hw_key, _, dtype in targets:
        color = _HW_COLORS.get(hw_key, "#444444")
        handles.append(Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="none", markeredgecolor=color, markersize=10,
            markeredgewidth=1.5,
            label=f"{hw_key} ({dtype})",
        ))
    for op, marker in _OP_MARKERS.items():
        handles.append(Line2D(
            [0], [0], marker=marker, color="black",
            markerfacecolor="white", markersize=9,
            linestyle="None", label=op,
        ))
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(len(handles), 8), bbox_to_anchor=(0.5, -0.02),
        fontsize=9, framealpha=0.9,
    )

    import matplotlib.pyplot as plt
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--hw", nargs="+", default=DEFAULT_TARGETS,
        choices=sorted(SWEEP_HW_TO_MAPPER),
        help="Hardware keys to compare (default: i7 + Orin Nano + KPU T64)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG path (default: validation/model_v4/results/plots/"
             "compare_<hw1>_<hw2>_<...>.png)",
    )
    p.add_argument(
        "--dtype-for", action="append", default=[],
        metavar="HW=DTYPE",
        help="Override the dtype for one hardware "
             "(e.g. --dtype-for kpu_t64=bf16). May repeat.",
    )
    args = p.parse_args(argv)

    overrides: Dict[str, str] = {}
    for spec in args.dtype_for:
        if "=" not in spec:
            print(f"warning: ignoring malformed --dtype-for {spec!r}",
                  file=sys.stderr)
            continue
        k, v = spec.split("=", 1)
        overrides[k.strip()] = v.strip()

    out = args.out or (
        DEFAULT_PLOT_DIR / f"compare_{'_'.join(args.hw)}.png"
    )
    render_comparison(args.hw, out, dtype_overrides=overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main())

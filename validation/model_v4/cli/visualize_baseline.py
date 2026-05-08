#!/usr/bin/env python
"""Visualize the per-shape performance spread for one (hardware) target.

Renders an N-row by 3-column figure from a committed V4 baseline CSV.
Each row is one dtype present in the baseline; each row has three panels:

    Panel 1 -- Roofline plot
        x = arithmetic intensity (FLOPS / byte)
        y = achieved throughput (GFLOPS)
        Both axes log. The "roof" is min(peak_FLOPS, AI * peak_DRAM_BW)
        drawn from the hardware mapper. Each shape is one point;
        matmul = circle, linear = triangle; color = predicted regime.
        Filled marker = measured; hollow companion = analyzer-predicted.
        Thin gray segment connects each pred-meas pair (vertical because
        AI is shape-only). The vertical span IS the calibration gap.

    Panel 2 -- Latency vs working-set
        x = working-set bytes
        y = latency (ms), log axis
        Cache-hierarchy reference lines for L1 / L2 / LLC drawn from
        the mapper. Same encoding as panel 1.

    Panel 3 -- Avg power vs working-set
        x = working-set bytes
        y = avg_power = energy_j / latency_s (W)
        Reference line at hardware TDP (when known). Hollow markers for
        sub-1ms shapes (where NVML/RAPL/INA3221 noise floor per #71 makes
        single-shot energy unreliable).

Together the three panels expose how shape modulates performance AND
where the analyzer's predictions diverge from measurements (the
calibration debt being addressed in the per-mapper calibration issues
like #91 for Orin Nano).

Usage:
    python -m validation.model_v4.cli.visualize_baseline --hw jetson_orin_nano_8gb
    python -m validation.model_v4.cli.visualize_baseline --hw i7_12700k --out /tmp/i7.png
    python -m validation.model_v4.cli.visualize_baseline --hw <key> --no-overlay   # measurement-only

Without --out the file lands at validation/model_v4/results/plots/<hw>_roofline.png.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Optional

from graphs.hardware.mappers import get_mapper_by_name
from graphs.hardware.resource_model import HardwareResourceModel

from validation.model_v4.harness.runner import (
    SWEEP_HW_TO_MAPPER,
    _build_subgraph,
    _predict_energy_j,
    _predict_latency_s,
    _resolve_precision,
)
from validation.model_v4.sweeps.classify import (
    Regime,
    op_footprint,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = REPO_ROOT / "validation" / "model_v4" / "results" / "baselines"
SWEEP_DIR = REPO_ROOT / "validation" / "model_v4" / "sweeps"
DEFAULT_PLOT_DIR = REPO_ROOT / "validation" / "model_v4" / "results" / "plots"


# Color per regime. Greens = ALU-side, blues = on-chip BW, oranges =
# off-chip BW, gray = launch.
_REGIME_COLORS = {
    Regime.ALU_BOUND.value:     "#2ca02c",   # green
    Regime.L1_BOUND.value:      "#17becf",   # teal
    Regime.L2_BOUND.value:      "#1f77b4",   # blue
    Regime.DRAM_BOUND.value:    "#ff7f0e",   # orange
    Regime.LAUNCH_BOUND.value:  "#7f7f7f",   # gray
    Regime.AMBIGUOUS.value:     "#bcbd22",
    Regime.UNSUPPORTED.value:   "#d62728",
}

_OP_MARKERS = {"matmul": "o", "linear": "^", "vector_add": "s"}

# RAPL/NVML/INA3221 single-shot energy noise floor; same constant as
# validation/model_v4/harness/assertions.py::ENERGY_RELIABLE_LATENCY_S.
ENERGY_RELIABLE_S = 1e-3


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------


def _shape_from_csv(s: str) -> tuple[int, ...]:
    """CSV ships shape as one of:
    - '1024x1024x1024' (matmul / linear, 'x'-separated dims)
    - '1024' (vector_add, single dim)
    - '(1024, 1024)' (Python tuple literal, fallback)
    """
    if "x" in s:
        return tuple(int(x) for x in s.split("x"))
    # Single-int shapes (vector_add): "1024" -> (1024,)
    s_stripped = s.strip()
    if s_stripped.lstrip("-").isdigit():
        return (int(s_stripped),)
    # Tuple literal fallback
    parsed = ast.literal_eval(s)
    if isinstance(parsed, int):
        return (parsed,)
    return tuple(parsed)


def load_baseline_rows(hw_key: str, op: str) -> list[dict]:
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


def load_sweep_shapes_for_predictions(
    op: str, dtype: str,
) -> list[dict]:
    """Predictions-only path: read shapes (regardless of hardware) from
    the sweep validation JSON. Used by ``--predictions-only`` mode for
    targets without a baseline CSV (e.g. the KPU T64).

    The dtype filter exists because each target validates a fixed
    dtype (i7 -> fp32, GPU/KPU -> fp16/bf16); we only emit shapes
    whose dtype matches the target's calibration precision."""
    path = SWEEP_DIR / f"{op}_validation.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    rows: list[dict] = []
    for entry in payload["shapes"]:
        if entry["dtype"] != dtype:
            continue
        rows.append({
            "op": op,
            "shape": tuple(entry["shape"]),
            "dtype": entry["dtype"],
            # Predictions-only: no measured fields. _enrich_row will
            # tolerate None/0 here -- the measured-marker scatter
            # paths skip rows whose meas_y_key is None.
            "latency_s": 0.0,
            "energy_j": None,
        })
    return rows


def _classify_for_target(
    op: str, shape: tuple, dtype: str, hw: HardwareResourceModel,
) -> str:
    """Run the analytical classifier for predictions-only targets that
    don't appear in the sweep's regime_per_hw map."""
    from validation.model_v4.sweeps.classify import classify_regime

    try:
        return classify_regime(op, shape, dtype, hw).value
    except (ValueError, KeyError):
        return Regime.AMBIGUOUS.value


def _peak_flops_for_dtype(hw: HardwareResourceModel, dtype: str) -> float:
    from graphs.estimation.roofline import RooflineAnalyzer
    try:
        return RooflineAnalyzer._get_effective_peak_ops(hw, _resolve_precision(dtype))
    except (ValueError, KeyError):
        return 0.0


def _enrich_row(r: dict, hw: HardwareResourceModel, with_predictions: bool) -> Optional[dict]:
    """Compute derived fields + analyzer predictions for one row.

    Returns None if the row's shape can't be resolved (e.g., the dtype
    isn't supported by the mapper). The caller drops Nones.

    Predictions-only mode (no measured row): pass ``r["latency_s"] == 0``
    and ``r["energy_j"] is None``; the measured fields stay None and
    the scatter paths skip them naturally."""
    op = r["op"]
    try:
        fp = op_footprint(op, r["shape"], r["dtype"])
    except (ValueError, ZeroDivisionError):
        return None
    if fp.working_set_bytes <= 0 or fp.flops <= 0:
        return None
    has_measurement = r["latency_s"] > 0

    r = dict(r)
    r["flops"] = fp.flops
    r["working_set_bytes"] = fp.working_set_bytes
    r["operational_intensity"] = fp.operational_intensity
    if has_measurement:
        r["measured_gflops"] = fp.flops / r["latency_s"] / 1e9
        r["measured_latency_ms"] = r["latency_s"] * 1e3
        r["measured_avg_power_w"] = (r["energy_j"] / r["latency_s"]
                                     if r["energy_j"] and r["latency_s"] > 0 else None)
    else:
        # Predictions-only path -- no measured markers will render.
        r["measured_gflops"] = None
        r["measured_latency_ms"] = None
        r["measured_avg_power_w"] = None

    if with_predictions:
        # Narrow exception list -- catch the failures we expect from
        # _resolve_precision (unknown dtype string -> ValueError),
        # _build_subgraph (unsupported op -> ValueError), and the
        # analyzer paths (KeyError on missing precision profile,
        # AttributeError on legacy mappers, ZeroDivisionError on a
        # degenerate shape). Letting other exceptions propagate keeps
        # bugs visible. Failures emit a single warning so a developer
        # can see *why* a shape's predicted markers are missing instead
        # of silently dropping them.
        try:
            precision = _resolve_precision(r["dtype"])
            sg = _build_subgraph(op, r["shape"], r["dtype"])
            # use_tier_aware_memory=True matches the V5-3b production
            # default; predicted markers reflect what the orchestrator
            # actually sees at runtime.
            pred_lat, _ = _predict_latency_s(
                sg, hw, precision, use_tier_aware_memory=True
            )
            pred_egy = _predict_energy_j(sg, hw, precision, pred_lat)
            r["predicted_latency_ms"] = pred_lat * 1e3
            r["predicted_gflops"] = (fp.flops / pred_lat / 1e9
                                     if pred_lat > 0 else None)
            r["predicted_avg_power_w"] = (pred_egy / pred_lat
                                          if pred_egy and pred_lat > 0 else None)
        except (ValueError, KeyError, AttributeError, ZeroDivisionError) as e:
            print(
                f"warning: prediction failed for "
                f"{op}({','.join(str(d) for d in r['shape'])}) {r['dtype']}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            r["predicted_latency_ms"] = None
            r["predicted_gflops"] = None
            r["predicted_avg_power_w"] = None
    return r


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _scatter_pair(ax, rows, x_key: str, meas_y_key: str,
                  pred_y_key: Optional[str],
                  reliable_pred: callable = lambda r: True):
    """Common scatter logic: filled = measured, hollow = predicted (when
    requested), thin gray segment connecting each pair."""
    # Pred -> meas connecting segments first (drawn under markers)
    if pred_y_key is not None:
        for r in rows:
            x = r.get(x_key)
            y_m = r.get(meas_y_key)
            y_p = r.get(pred_y_key)
            if x is None or y_m is None or y_p is None or not reliable_pred(r):
                continue
            ax.plot([x, x], [y_p, y_m], color="#cccccc", linewidth=0.6,
                    alpha=0.5, zorder=1)

    # Measured (filled)
    for op in ("matmul", "linear", "vector_add"):
        for regime in _REGIME_COLORS:
            xs, ys = [], []
            for r in rows:
                if r["op"] != op or r["regime"] != regime:
                    continue
                x = r.get(x_key)
                y = r.get(meas_y_key)
                if x is None or y is None or x <= 0 or y <= 0:
                    continue
                xs.append(x)
                ys.append(y)
            if xs:
                ax.scatter(xs, ys, s=36, alpha=0.8,
                           marker=_OP_MARKERS[op],
                           color=_REGIME_COLORS[regime],
                           edgecolor="black", linewidth=0.4, zorder=3)

    # Predicted (hollow, same color/marker, no fill)
    if pred_y_key is not None:
        for op in ("matmul", "linear", "vector_add"):
            for regime in _REGIME_COLORS:
                xs, ys = [], []
                for r in rows:
                    if r["op"] != op or r["regime"] != regime:
                        continue
                    x = r.get(x_key)
                    y = r.get(pred_y_key)
                    if (x is None or y is None or x <= 0 or y <= 0
                            or not reliable_pred(r)):
                        continue
                    xs.append(x)
                    ys.append(y)
                if xs:
                    ax.scatter(xs, ys, s=36, alpha=0.6,
                               marker=_OP_MARKERS[op],
                               facecolors="none",
                               edgecolor=_REGIME_COLORS[regime], linewidth=1.0,
                               zorder=2)


def _draw_roof(ax, hw, dtype):
    import numpy as np
    peak_fp = _peak_flops_for_dtype(hw, dtype)
    if peak_fp <= 0:
        return
    peak_bw = hw.peak_bandwidth
    ai_break = peak_fp / peak_bw if peak_bw > 0 else 0
    ai_grid = np.logspace(-1, 4, 256)
    achievable = np.minimum(peak_fp, ai_grid * peak_bw) / 1e9
    ax.plot(ai_grid, achievable, linestyle="--", linewidth=1.2,
            color="#444444", alpha=0.6,
            label=(f"roof ({dtype}: peak={peak_fp/1e9:.0f} GFLOPS, "
                   f"BW={peak_bw/1e9:.0f} GB/s, AI_break={ai_break:.1f})"))


def _draw_cache_lines(ax, hw):
    l1_total = hw.compute_units * hw.l1_cache_per_unit
    l2_total = hw.l2_cache_total or 0
    if l1_total:
        ax.axvline(l1_total, color="#2ca02c", linestyle=":", alpha=0.5,
                   label=f"L1 total ({l1_total/2**20:.1f} MiB)")
    if l2_total:
        ax.axvline(l2_total, color="#1f77b4", linestyle=":", alpha=0.5,
                   label=f"L2/LLC total ({l2_total/2**20:.1f} MiB)")


def _draw_tdp_line(ax, hw):
    # Pull TDP via the same path EnergyAnalyzer uses. The TDP line is a
    # convenience reference, not core to the plot, so a failure here
    # (e.g., legacy mapper without thermal_operating_points, or a future
    # refactor of EnergyAnalyzer's __init__) should NOT crash the
    # render. Log and continue rather than swallow silently.
    try:
        from graphs.estimation.energy import EnergyAnalyzer
        analyzer = EnergyAnalyzer(hw)
        tdp = analyzer.tdp_watts
        if tdp and tdp > 0:
            ax.axhline(tdp, color="#d62728", linestyle=":", alpha=0.6,
                       label=f"TDP ({tdp:.0f} W)")
    except Exception as exc:   # noqa: BLE001 -- best-effort cosmetic line
        print(
            f"warning: TDP reference line skipped for {hw.name!r}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render_baseline_plot(hw_key: str, out_path: Path,
                         with_overlay: bool = True,
                         predictions_only: bool = False,
                         predictions_dtype: Optional[str] = None) -> None:
    """Render the 3-panel V4 visualization.

    ``predictions_only`` switches to a no-baseline path: shapes come
    from the sweep validation JSON, regimes come from the analytical
    classifier, and the measured-marker scatter paths skip cleanly
    (only hollow predicted markers render). Use this for hardware
    that doesn't have a baseline CSV yet (e.g. KPU T64).

    ``predictions_dtype`` filters the sweep shapes to one precision
    (defaults to fp16 for non-CPU targets, fp32 for CPU).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if hw_key not in SWEEP_HW_TO_MAPPER:
        raise ValueError(
            f"Hardware key {hw_key!r} has no mapper-registry translation. "
            f"Known: {sorted(SWEEP_HW_TO_MAPPER)}")
    mapper = get_mapper_by_name(SWEEP_HW_TO_MAPPER[hw_key])
    hw = mapper.resource_model

    # Gather + enrich rows
    raw_rows: list[dict] = []
    if predictions_only:
        # No baselines; pull shapes from the sweep validation JSON
        # and classify on the fly via the analytical classifier (the
        # sweep entry's regime_per_hw map may not include this target).
        if predictions_dtype is None:
            predictions_dtype = (
                "fp32" if hw.hardware_type.name == "CPU" else "fp16"
            )
        # KPU's native bf16 is the right comparison precision; allow
        # an explicit override via predictions_dtype.
        for op in ("matmul", "linear", "vector_add"):
            for r in load_sweep_shapes_for_predictions(op, predictions_dtype):
                r["regime"] = _classify_for_target(
                    op, r["shape"], r["dtype"], hw,
                )
                raw_rows.append(r)
        # Predictions are mandatory in this mode; force overlay on so
        # the predicted-marker scatter actually fires.
        with_overlay = True
    else:
        for op in ("matmul", "linear", "vector_add"):
            baseline_rows = load_baseline_rows(hw_key, op)
            regimes = load_sweep_regimes(hw_key, op)
            for r in baseline_rows:
                key = (tuple(r["shape"]), r["dtype"])
                r["regime"] = regimes.get(key, Regime.AMBIGUOUS.value)
                raw_rows.append(r)
    enriched = [_enrich_row(r, hw, with_overlay) for r in raw_rows]
    rows = [r for r in enriched if r is not None]
    if not rows:
        if predictions_only:
            raise SystemExit(
                f"No sweep shapes for {hw_key!r} at "
                f"dtype={predictions_dtype!r}; check the sweep JSONs."
            )
        raise SystemExit(
            f"No baseline data found for {hw_key!r}. Expected one or both of "
            f"{BASELINE_DIR}/{hw_key}_matmul.csv, "
            f"{BASELINE_DIR}/{hw_key}_linear.csv "
            f"(or pass --predictions-only)")

    dtypes_present = sorted({r["dtype"] for r in rows})
    n_rows = len(dtypes_present)
    fig, axes_grid = plt.subplots(n_rows, 3,
                                  figsize=(18, 5.5 * max(n_rows, 1)),
                                  squeeze=False)

    n_mm = sum(1 for r in rows if r['op'] == 'matmul')
    n_lin = sum(1 for r in rows if r['op'] == 'linear')
    n_va = sum(1 for r in rows if r['op'] == 'vector_add')
    title_op_count = f"{n_mm} matmul + {n_lin} linear + {n_va} vector_add"
    if predictions_only:
        overlay_note = "  (predictions-only -- no V4 baseline)"
        title_prefix = "Roofline predictions"
    else:
        overlay_note = (
            "" if not with_overlay else "  (filled=measured, hollow=predicted)"
        )
        title_prefix = "V4 baseline performance spread"
    fig.suptitle(
        f"{title_prefix} -- {hw.name}  ({title_op_count} shapes){overlay_note}",
        fontsize=12, y=0.995)

    for row_idx, dtype in enumerate(dtypes_present):
        dtype_rows = [r for r in rows if r["dtype"] == dtype]
        ax_roof, ax_lat, ax_pwr = axes_grid[row_idx]

        # ----- Panel 1: roofline -----
        _draw_roof(ax_roof, hw, dtype)
        _scatter_pair(ax_roof, dtype_rows,
                      x_key="operational_intensity",
                      meas_y_key="measured_gflops",
                      pred_y_key="predicted_gflops" if with_overlay else None)
        ax_roof.set_xscale("log"); ax_roof.set_yscale("log")
        ax_roof.set_xlabel("Arithmetic intensity (FLOPS / byte)")
        ax_roof.set_ylabel("Achieved throughput (GFLOPS)")
        ax_roof.set_title(f"Roofline ({dtype})")
        ax_roof.grid(True, which="both", alpha=0.25)
        ax_roof.legend(fontsize=7, loc="lower right", framealpha=0.85)

        # ----- Panel 2: latency vs WS -----
        _draw_cache_lines(ax_lat, hw)
        _scatter_pair(ax_lat, dtype_rows,
                      x_key="working_set_bytes",
                      meas_y_key="measured_latency_ms",
                      pred_y_key="predicted_latency_ms" if with_overlay else None)
        ax_lat.set_xscale("log"); ax_lat.set_yscale("log")
        ax_lat.set_xlabel("Working-set bytes")
        ax_lat.set_ylabel("Latency (ms)")
        ax_lat.set_title(f"Latency vs working-set ({dtype})")
        ax_lat.grid(True, which="both", alpha=0.25)
        ax_lat.legend(fontsize=7, loc="upper left", framealpha=0.85)

        # ----- Panel 3: avg power vs WS -----
        _draw_tdp_line(ax_pwr, hw)
        # For power, mark sub-1ms points as hollow regardless of overlay
        # mode: their measured energy is per-#71 RAPL/NVML noise. The
        # noise floor is a measurement artifact, not a model artifact;
        # in predictions-only mode all rows count as reliable because
        # there's no RAPL/NVML reading at all.
        def _reliable(r):
            if predictions_only:
                return True
            return r["latency_s"] >= ENERGY_RELIABLE_S
        # Filled measured for reliable shapes
        rel_rows = [r for r in dtype_rows if _reliable(r)]
        unrel_rows = [r for r in dtype_rows if not _reliable(r)]
        _scatter_pair(ax_pwr, rel_rows,
                      x_key="working_set_bytes",
                      meas_y_key="measured_avg_power_w",
                      pred_y_key="predicted_avg_power_w" if with_overlay else None)
        # Sub-1ms shapes: open markers (faint), no pred connector
        for op in ("matmul", "linear", "vector_add"):
            for regime in _REGIME_COLORS:
                xs, ys = [], []
                for r in unrel_rows:
                    if r["op"] != op or r["regime"] != regime:
                        continue
                    if r.get("measured_avg_power_w") in (None, 0):
                        continue
                    if r["working_set_bytes"] <= 0:
                        continue
                    xs.append(r["working_set_bytes"])
                    ys.append(r["measured_avg_power_w"])
                if xs:
                    ax_pwr.scatter(xs, ys, s=20, alpha=0.35,
                                   marker=_OP_MARKERS[op],
                                   facecolors="none",
                                   edgecolor=_REGIME_COLORS[regime],
                                   linewidth=0.6, zorder=2)
        ax_pwr.set_xscale("log")
        ax_pwr.set_xlabel("Working-set bytes")
        ax_pwr.set_ylabel("Avg power (W)")
        ax_pwr.set_title(f"Avg power vs working-set ({dtype})")
        ax_pwr.grid(True, which="both", alpha=0.25)
        ax_pwr.legend(fontsize=7, loc="upper left", framealpha=0.85)
        # Power note: sub-1ms shapes have unreliable energy
        if unrel_rows:
            ax_pwr.text(
                0.98, 0.02,
                f"{len(unrel_rows)} shapes < 1 ms (faint)\nare RAPL/NVML noisy (#71)",
                transform=ax_pwr.transAxes, fontsize=7, alpha=0.6,
                ha="right", va="bottom")

    # Shared regime/op legend below all rows
    handles = []
    for regime, color in _REGIME_COLORS.items():
        if not any(r["regime"] == regime for r in rows):
            continue
        handles.append(Line2D([0], [0], marker="s", color="w",
                              markerfacecolor=color, markersize=9,
                              markeredgecolor="black", label=regime))
    for op, marker in _OP_MARKERS.items():
        if not any(r["op"] == op for r in rows):
            continue
        handles.append(Line2D([0], [0], marker=marker, color="black",
                              markerfacecolor="white", markersize=9,
                              linestyle="None", label=op))
    if with_overlay:
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="black", markersize=9,
                              label="measured"))
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="none", markeredgecolor="black",
                              markersize=9, markeredgewidth=1.2,
                              label="predicted"))
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 9), bbox_to_anchor=(0.5, -0.005),
               fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    try:
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    except ValueError:
        # out_path lives outside the repo (e.g. /tmp/...); print as-is.
        print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hw", required=True,
                   choices=sorted(SWEEP_HW_TO_MAPPER),
                   help="hardware key (must have a baseline CSV unless "
                        "--predictions-only)")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG path (default: "
                        "validation/model_v4/results/plots/<hw>_roofline.png)")
    p.add_argument("--no-overlay", action="store_true",
                   help="skip the predicted (analyzer) overlay; "
                        "only render measured points")
    p.add_argument("--predictions-only", action="store_true",
                   help="skip baseline lookup; render predictions for "
                        "shapes from the sweep validation JSON. Use for "
                        "hardware that doesn't have a baseline yet "
                        "(e.g. KPU T64, AGX, NX, H100).")
    p.add_argument("--predictions-dtype", default=None,
                   help="precision to use in --predictions-only mode "
                        "(default: fp32 for CPU, fp16 for GPU/KPU). "
                        "KPU's native bf16 is a sensible explicit choice "
                        "for compute comparisons.")
    args = p.parse_args(argv)

    out = args.out or (DEFAULT_PLOT_DIR / f"{args.hw}_roofline.png")
    render_baseline_plot(
        args.hw, out,
        with_overlay=not args.no_overlay,
        predictions_only=args.predictions_only,
        predictions_dtype=args.predictions_dtype,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

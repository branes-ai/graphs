"""Re-augment sweep regime labels from baseline measurements.

Why this exists, separately from ``_augment.py``:

The default ``_augment.py`` re-runs the *analytical* classifier
(``classify_regime``) against a hardware mapper -- a pure function of
the resource model (peak FLOPS, peak BW, cache capacities). For shapes
where the analytical model and the silicon agree, this works fine. But
for tilable workloads on GPUs (matmul, linear), cuBLAS hides the
working set via K-tiling, so a shape with ``WS > L2`` and high OI
runs ALU-bound on real silicon even though the WS-bucketing classifier
says DRAM_BOUND.

This script is the *measurement-priority* path: when a baseline CSV
exists for a (hardware, op) pair, every shape in the sweep that also
has a baseline row gets its ``regime_per_hw[hw_key]`` rewritten to
match the regime ``infer_regime_measured`` reports for the silicon's
latency. Sweep label = silicon truth, by construction.

Two safety rails:

1. **Concrete only.** ``infer_regime_measured`` returns AMBIGUOUS for
   shapes where neither ``flops_util >= 0.70`` nor ``bw_util >= 0.70``
   fires. Those records keep their analytical label rather than become
   AMBIGUOUS (the runner skips AMBIGUOUS regimes; we do not want to
   shrink the validation pool just because a shape lands in the
   no-mans-land between compute and bandwidth bounds). The analytical
   classifier still gets to claim ALU / L2 / DRAM_BOUND for those
   shapes -- the latency-band / energy-band tolerance checks are the
   final gate either way.

2. **Baseline only.** Shapes without a baseline row are not touched.
   This script only refines labels for hardware that have committed
   baselines (currently i7-12700K + Jetson Orin Nano 8GB; expand as
   more baselines are captured).

Idempotent: re-running with the same baselines produces byte-identical
JSON.

Usage:

    PYTHONPATH=src:. python -m validation.model_v4.sweeps._augment_from_baseline

    # Restrict to one op:
    PYTHONPATH=src:. python -m validation.model_v4.sweeps._augment_from_baseline --op matmul

    # Restrict to one hardware:
    PYTHONPATH=src:. python -m validation.model_v4.sweeps._augment_from_baseline --hw jetson_orin_nano_8gb

The script overwrites the sweep JSONs in place. Commit the diffs to
keep the sweep regime labels in lockstep with the committed baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from graphs.hardware.resource_model import Precision

from validation.model_v4.ground_truth.cache import (
    CacheKey,
    DEFAULT_BASELINE_DIR,
    load_baseline,
)
from validation.model_v4.harness.assertions import (
    MeasurementContext,
    infer_regime_measured,
)
from validation.model_v4.sweeps._augment import KNOWN_TARGETS
from validation.model_v4.sweeps.classify import (
    DEFAULT_LAUNCH_OVERHEAD_S,
    Regime,
    op_footprint,
)


SWEEP_DIR = Path(__file__).resolve().parent

OPS = ("matmul", "linear", "vector_add")
PURPOSES = ("calibration", "validation")


_PRECISION_BY_DTYPE: dict[str, Precision] = {
    "fp64": Precision.FP64,
    "fp32": Precision.FP32,
    "tf32": Precision.TF32,
    "fp16": Precision.FP16,
    "bf16": Precision.BF16,
    "int8": Precision.INT8,
}


def augment_sweep_from_baseline(
    sweep_path: Path,
    hw_key: str,
    *,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
) -> tuple[int, int, int]:
    """Re-label entries in ``sweep_path`` for ``hw_key`` from baseline.

    Returns ``(rewritten, kept_ambiguous, no_baseline)``:
      * ``rewritten``       -- entries whose label changed to match the
                               measured regime
      * ``kept_ambiguous``  -- entries where measurement was AMBIGUOUS;
                               left at the analytical label
      * ``no_baseline``     -- entries with no baseline row; untouched

    Idempotent: re-running with identical inputs makes no further
    changes.
    """
    target = KNOWN_TARGETS[hw_key]
    hw = target.hw
    sweep = json.loads(sweep_path.read_text())
    op = sweep["op"]

    cache = load_baseline(hw_key, op, baseline_dir=baseline_dir)
    if not cache:
        # No baseline -> nothing to do for this (hw, op).
        return (0, 0, 0)

    precision = _PRECISION_BY_DTYPE[target.dtype]
    ctx = MeasurementContext(
        peak_flops=hw.get_peak_ops(precision),
        peak_dram_bandwidth_bps=hw.peak_bandwidth,
        launch_overhead_s=DEFAULT_LAUNCH_OVERHEAD_S,
    )

    rewritten = 0
    kept_amb = 0
    no_base = 0
    for entry in sweep["shapes"]:
        if entry["dtype"] != target.dtype:
            continue
        shape = tuple(entry["shape"])
        meas = cache.get(CacheKey(hw_key, op, shape, entry["dtype"]))
        if meas is None:
            no_base += 1
            continue
        fp = op_footprint(op, shape, entry["dtype"])
        regime_meas = infer_regime_measured(
            fp.flops, fp.working_set_bytes, meas.latency_s, ctx
        )
        if regime_meas == Regime.AMBIGUOUS:
            kept_amb += 1
            continue
        old = entry.get("regime_per_hw", {}).get(hw_key)
        new = regime_meas.value
        if old != new:
            rewritten += 1
        entry.setdefault("regime_per_hw", {})[hw_key] = new

    # Surface that this (hw, op) was re-augmented from baseline.
    augmented = sweep.setdefault("augmented_with_hardware_from_baseline", [])
    if hw_key not in augmented:
        augmented.append(hw_key)
        augmented.sort()

    sweep_path.write_text(json.dumps(sweep, indent=2))
    return (rewritten, kept_amb, no_base)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hw", choices=sorted(KNOWN_TARGETS), default=None,
        help="Restrict to one hardware key (default: all known)",
    )
    parser.add_argument(
        "--op", choices=OPS, default=None,
        help="Restrict to one op (default: all)",
    )
    parser.add_argument(
        "--purpose", choices=PURPOSES, default=None,
        help="Restrict to one purpose (default: both)",
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR,
        help="Where the cached baselines live",
    )
    args = parser.parse_args(argv)

    hw_keys = [args.hw] if args.hw else sorted(KNOWN_TARGETS)
    ops = [args.op] if args.op else OPS
    purposes = [args.purpose] if args.purpose else PURPOSES

    total_rewritten = 0
    total_kept_amb = 0
    for op in ops:
        for purpose in purposes:
            sweep_path = SWEEP_DIR / f"{op}_{purpose}.json"
            if not sweep_path.exists():
                continue
            for hw_key in hw_keys:
                r, k, n = augment_sweep_from_baseline(
                    sweep_path, hw_key, baseline_dir=args.baseline_dir
                )
                if r or k:
                    print(
                        f"{sweep_path.name} [{hw_key}]: "
                        f"{r} rewritten, {k} kept-ambiguous, {n} no-baseline"
                    )
                    total_rewritten += r
                    total_kept_amb += k

    print(f"\nTotal: {total_rewritten} rewritten, {total_kept_amb} kept-ambiguous")
    return 0


if __name__ == "__main__":
    sys.exit(main())

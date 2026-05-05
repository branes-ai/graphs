#!/usr/bin/env python
"""Refresh cached ground-truth measurements for a (hardware, op) pair.

Usage:
    python -m validation.model_v4.cli.capture_ground_truth \\
        --hw i7_12700k --op matmul --purpose calibration
    python -m validation.model_v4.cli.capture_ground_truth \\
        --hw i7_12700k --op linear --purpose validation --warmup 5 --trials 21

The slow path: walks every shape in the sweep, runs it on real
silicon, writes results to validation/model_v4/results/baselines/.

Cache invalidation is **explicit** (per the v4 plan): this CLI is the
only writer. The validate CLI is read-only -- it never silently moves
the validation reference.

Currently the only ground-truth backend is ``PyTorchCPUMeasurer``
(wall-clock + Intel RAPL). GPU (V4-4) and KPU simulator (V4-5)
backends will dispatch via ``--hw``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from validation.model_v4.ground_truth.cache import CacheKey, store
from validation.model_v4.ground_truth.pytorch_cpu import (
    DEFAULT_TIMED_TRIALS,
    DEFAULT_WARMUP_TRIALS,
    PyTorchCPUMeasurer,
)
from validation.model_v4.sweeps.classify import Regime
from validation.model_v4.workloads.linear import build_linear
from validation.model_v4.workloads.matmul import build_matmul


SWEEP_DIR = Path(__file__).resolve().parents[1] / "sweeps"
BASELINE_DIR = Path(__file__).resolve().parents[1] / "results" / "baselines"


# Hardware -> measurer factory. Add cuda / simulator here when V4-4 / V4-5 land.
_MEASURER_FACTORY = {
    "i7_12700k": "pytorch_cpu",
    # "h100_sxm5_80gb": "pytorch_cuda",
}


def _make_measurer(hw_key: str, *, warmup_trials: int, timed_trials: int):
    backend = _MEASURER_FACTORY.get(hw_key)
    if backend == "pytorch_cpu":
        return PyTorchCPUMeasurer(
            hardware=hw_key,
            warmup_trials=warmup_trials,
            timed_trials=timed_trials,
        )
    raise ValueError(
        f"No ground-truth measurer registered for {hw_key!r}. "
        f"Known: {sorted(k for k, v in _MEASURER_FACTORY.items() if v)}"
    )


def _build_workload(op: str, shape: tuple, dtype: str):
    if op == "matmul":
        M, K, N = shape
        return build_matmul(M, K, N, dtype)
    if op == "linear":
        B, IN, OUT = shape
        return build_linear(B, IN, OUT, dtype)
    raise ValueError(f"Unsupported op {op!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Constrain to keys that have a registered measurer factory; otherwise
    # a valid parse would crash in _make_measurer() with a traceback.
    # SWEEP_HW_TO_MAPPER is the broader set used by the validate CLI;
    # capture is narrower because not every target has a measurer yet
    # (e.g., GPU lands in V4-4, KPU in V4-5).
    p.add_argument("--hw", required=True,
                   choices=sorted(k for k, v in _MEASURER_FACTORY.items() if v))
    p.add_argument("--op", required=True, choices=["matmul", "linear"])
    p.add_argument("--purpose", default="calibration",
                   choices=["calibration", "validation"])
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_TRIALS)
    p.add_argument("--trials", type=int, default=DEFAULT_TIMED_TRIALS)
    p.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of shapes processed (for smoke testing)")
    args = p.parse_args(argv)

    sweep_path = SWEEP_DIR / f"{args.op}_{args.purpose}.json"
    if not sweep_path.exists():
        print(f"sweep file not found: {sweep_path}", file=sys.stderr)
        return 2

    sweep = json.loads(sweep_path.read_text())
    op = sweep["op"]

    measurer = _make_measurer(args.hw, warmup_trials=args.warmup,
                              timed_trials=args.trials)

    # Filter to shapes labeled for this hardware, excluding UNSUPPORTED /
    # AMBIGUOUS. The runner does the same filtering at validate time; we
    # do it here to avoid wasting wall time on shapes the harness will
    # skip anyway.
    candidates = []
    for entry in sweep["shapes"]:
        regime = entry["regime_per_hw"].get(args.hw)
        if regime is None:
            continue
        if regime in (Regime.UNSUPPORTED.value, Regime.AMBIGUOUS.value):
            continue
        candidates.append(entry)
    if args.limit is not None:
        candidates = candidates[:args.limit]

    print(f"Capturing {len(candidates)} shapes for ({args.hw}, {op}, {args.purpose})")
    print(f"  measurer: {measurer.name}, warmup={args.warmup}, trials={args.trials}")
    print(f"  baseline_dir: {args.baseline_dir}")
    print()

    t_start = time.time()
    for i, entry in enumerate(candidates, 1):
        shape = tuple(entry["shape"])
        dtype = entry["dtype"]
        workload = _build_workload(op, shape, dtype)
        meas = measurer.measure(workload.model, workload.inputs)
        store(
            CacheKey(args.hw, op, shape, dtype), meas,
            baseline_dir=args.baseline_dir,
        )
        energy_str = (f"{meas.energy_j*1e3:.2f}mJ" if meas.energy_j is not None else "n/a")
        print(f"  [{i:>3d}/{len(candidates)}] {workload.name:42s}  "
              f"{meas.latency_s*1e3:>9.3f}ms  {energy_str}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s. Baseline written to "
          f"{args.baseline_dir / f'{args.hw}_{op}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Run V4 validation against a cached baseline (prediction-only).

Usage:
    python -m validation.model_v4.cli.validate \\
        --hw i7_12700k --op matmul --purpose validation
    python -m validation.model_v4.cli.validate \\
        --hw h100_sxm5_80gb --op linear --format markdown

The plan calls for separate ``validate_matmul`` / ``validate_linear``
commands; this single CLI exposes them via ``--op`` so the dispatch is
in one place. The shell aliases in cli/validate_matmul and
cli/validate_linear are 4-line wrappers if/when they're added.

This is the **prediction-only** path: the harness reads cached ground
truth and asserts predictions against it. To capture fresh ground
truth (the slow path that exercises real silicon), use
``cli/capture_ground_truth.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from validation.model_v4.harness.report import (
    format_json,
    format_markdown,
    format_text,
)
from validation.model_v4.harness.runner import (
    SWEEP_HW_TO_MAPPER,
    RunnerConfig,
    run_sweep,
)


SWEEP_DIR = Path(__file__).resolve().parents[1] / "sweeps"
BASELINE_DIR = Path(__file__).resolve().parents[1] / "results" / "baselines"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hw", required=True,
                   choices=sorted(SWEEP_HW_TO_MAPPER.keys()),
                   help="Hardware key (matches sweep JSON 'regime_per_hw')")
    p.add_argument("--op", required=True, choices=["matmul", "linear"],
                   help="Operator to validate")
    p.add_argument("--purpose", default="validation",
                   choices=["validation", "calibration"],
                   help="Which sweep file to use (default: validation)")
    p.add_argument("--format", default="text",
                   choices=["text", "markdown", "json"],
                   help="Output format")
    p.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR,
                   help=f"Where the cached baselines live (default: {BASELINE_DIR})")
    p.add_argument("--output", type=Path, default=None,
                   help="Write the report to a file instead of stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    sweep_path = SWEEP_DIR / f"{args.op}_{args.purpose}.json"
    if not sweep_path.exists():
        print(f"sweep file not found: {sweep_path}", file=sys.stderr)
        return 2

    cfg = RunnerConfig(
        sweep_path=sweep_path,
        hardware_key=args.hw,
        baseline_dir=args.baseline_dir,
        # Prediction-only: do NOT auto-refresh missing measurements.
        refresh_measurements=False,
        measurer=None,
    )
    result = run_sweep(cfg)

    if args.format == "json":
        rendered = format_json(result)
    elif args.format == "markdown":
        rendered = format_markdown(result, op=args.op)
    else:
        rendered = format_text(result, op=args.op)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered)

    # Exit code: non-zero if any record failed, so CI can gate on this.
    failures = sum(1 for r in result.records if not r.all_pass())
    return 0 if failures == 0 and result.records else (1 if failures else 0)


if __name__ == "__main__":
    sys.exit(main())

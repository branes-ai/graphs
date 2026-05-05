"""Augment existing sweep JSONs with regimes for additional hardware.

Why this exists, separately from ``_generate.py``:

The original generator (``_generate.py``) picks shapes that exhibit
specific regimes on a fixed list of TARGETS (currently i7-12700K and
H100). Adding a new target there would change which shapes get chosen
because the sampling iterates over (target, regime) buckets -- so
re-running ``_generate.py`` with extra targets churns the committed
sweeps and invalidates the i7 baseline that's already in the repo.

This script is the *additive* path: it reads the existing sweep, runs
the regime classifier against new hardware mappers, and inserts the
results into ``regime_per_hw`` for each entry. Existing entries (their
shapes, dtypes, and original target regimes) are left untouched. Run
this when adding a new hardware target without re-sampling.

Usage:

    python -m validation.model_v4.sweeps._augment \\
        --hw jetson_orin_nano_8gb \\
        --dtype fp16 \\
        --op matmul \\
        --purpose validation

    # Re-augment everything with all currently-supported targets:
    python -m validation.model_v4.sweeps._augment --all

The script is idempotent: re-running with the same (hw, dtype) is a
no-op (existing keys in ``regime_per_hw`` are overwritten with the
same value).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import create_h100_sxm5_80gb_mapper
from graphs.hardware.mappers.gpu import (
    create_jetson_orin_nano_8gb_mapper,
    create_jetson_orin_agx_64gb_mapper,
    create_jetson_orin_nx_16gb_mapper,
)

from validation.model_v4.sweeps.classify import Regime, classify_regime


SWEEP_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class _Target:
    hw_key: str           # the key written into regime_per_hw
    dtype: str            # the precision to classify against
    factory: Callable     # callable returning a mapper

    @property
    def hw(self):
        return self.factory().resource_model


# Registry of targets the augmenter knows how to classify against.
# Adding a new entry here is the single point of change for V4-4 hardware
# expansion. The CPU / H100 entries are duplicated from _generate.py
# because the augmenter must be able to *re-augment* existing entries
# (idempotently) without depending on the generator's specific dtype
# choices.
KNOWN_TARGETS: dict[str, _Target] = {
    "i7_12700k": _Target(
        hw_key="i7_12700k", dtype="fp32", factory=create_i7_12700k_mapper),
    "h100_sxm5_80gb": _Target(
        hw_key="h100_sxm5_80gb", dtype="fp16", factory=create_h100_sxm5_80gb_mapper),
    "jetson_orin_nano_8gb": _Target(
        hw_key="jetson_orin_nano_8gb", dtype="fp16",
        factory=create_jetson_orin_nano_8gb_mapper),
    "jetson_orin_agx_64gb": _Target(
        hw_key="jetson_orin_agx_64gb", dtype="fp16",
        factory=create_jetson_orin_agx_64gb_mapper),
    "jetson_orin_nx_16gb": _Target(
        hw_key="jetson_orin_nx_16gb", dtype="fp16",
        factory=create_jetson_orin_nx_16gb_mapper),
}


def augment_sweep(sweep_path: Path, target: _Target) -> tuple[int, int]:
    """Insert ``target.hw_key`` regimes into the sweep JSON at ``sweep_path``.

    Only entries whose existing dtype matches ``target.dtype`` are
    classified; entries with non-matching dtype get a clean skip rather
    than a forced UNSUPPORTED label (the entry just won't have a
    regime for this target, which is exactly the runner's expectation
    -- no regime => skip in run_sweep).

    Returns: (entries_classified, entries_skipped_dtype).
    """
    sweep = json.loads(sweep_path.read_text())
    op = sweep["op"]
    hw = target.hw

    classified = 0
    skipped_dtype = 0
    for entry in sweep["shapes"]:
        if entry["dtype"] != target.dtype:
            skipped_dtype += 1
            continue
        regime = classify_regime(op, tuple(entry["shape"]),
                                 entry["dtype"], hw)
        entry.setdefault("regime_per_hw", {})[target.hw_key] = regime.value
        classified += 1

    # Surface in the JSON metadata that this hw was augmented in.
    augmented = sweep.setdefault("augmented_with_hardware", [])
    if target.hw_key not in augmented:
        augmented.append(target.hw_key)
        augmented.sort()

    sweep_path.write_text(json.dumps(sweep, indent=2))
    return classified, skipped_dtype


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hw", choices=sorted(KNOWN_TARGETS),
                   help="Augment one specific hardware key")
    g.add_argument("--all", action="store_true",
                   help="Augment with every known target")
    p.add_argument("--op", choices=["matmul", "linear"], default=None,
                   help="Restrict to one op (default: both)")
    p.add_argument("--purpose", choices=["calibration", "validation"], default=None,
                   help="Restrict to one purpose (default: both)")
    args = p.parse_args(argv)

    targets = (list(KNOWN_TARGETS.values()) if args.all
               else [KNOWN_TARGETS[args.hw]])

    ops = [args.op] if args.op else ["matmul", "linear"]
    purposes = [args.purpose] if args.purpose else ["calibration", "validation"]

    for op in ops:
        for purpose in purposes:
            sweep_path = SWEEP_DIR / f"{op}_{purpose}.json"
            if not sweep_path.exists():
                print(f"  skip (no file): {sweep_path.name}")
                continue
            for target in targets:
                classified, skipped = augment_sweep(sweep_path, target)
                print(f"{sweep_path.name} [{target.hw_key}]: "
                      f"{classified} classified, {skipped} dtype-mismatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())

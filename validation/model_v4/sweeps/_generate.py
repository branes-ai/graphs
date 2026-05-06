"""Generator for the v4 calibration / validation sweep JSON files.

Run with: ``PYTHONPATH=src:. python -m validation.model_v4.sweeps._generate``

Emits four files into ``validation/model_v4/sweeps/``:

* ``matmul_calibration.json``   ~30 shapes for fitting constants
* ``matmul_validation.json``    ~80 shapes the model has not seen during fit
* ``linear_calibration.json``
* ``linear_validation.json``

Each entry records the per-hardware regime label so the harness does not
re-classify on every run (and so regression tests can check the JSON
without importing the classifier).

Strategy:
1. Build a dense candidate pool from log-spaced cube and rectangular shapes.
2. Classify every candidate against (i7-12700K, fp32) and (H100, fp16).
3. Drop any candidate that is AMBIGUOUS on either target (Principle 1).
4. Bucket by `(regime_i7, regime_h100, dtype)`.
5. Sample target counts per regime, alternating sample and skip across the
   sorted list to maximize separation between calibration and validation.

Reproducible: a fixed random seed and deterministic ordering mean re-running
the generator on the same inputs produces byte-identical JSON.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple

from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import create_h100_sxm5_80gb_mapper
from graphs.hardware.resource_model import HardwareResourceModel

from validation.model_v4.sweeps.classify import (
    Regime,
    classify_regime,
)


SWEEP_DIR = Path(__file__).resolve().parent
SEED = 20260505
random.seed(SEED)


# Targets the harness initially supports
TARGETS: list[tuple[str, str, HardwareResourceModel]] = [
    ("i7_12700k", "fp32", create_i7_12700k_mapper().resource_model),
    ("h100_sxm5_80gb", "fp16", create_h100_sxm5_80gb_mapper().resource_model),
]


# Distribution targets per regime per sweep purpose. Calibration sweep is
# small and balanced; validation sweep is large and weighted toward DRAM
# (the regime that bites hardest in real workloads).
CALIBRATION_PER_REGIME = {
    Regime.LAUNCH_BOUND: 4,
    Regime.ALU_BOUND:    6,
    Regime.L2_BOUND:     6,
    Regime.DRAM_BOUND:   8,
}
VALIDATION_PER_REGIME = {
    Regime.LAUNCH_BOUND: 10,
    Regime.ALU_BOUND:    18,
    Regime.L2_BOUND:     20,
    Regime.DRAM_BOUND:   30,
}


@dataclass(frozen=True)
class SweepEntry:
    shape: Tuple[int, ...]
    dtype: str
    regime_per_hw: dict[str, str]  # {target_key: regime_value}


# ---------------------------------------------------------------------------
# Candidate-pool generators
# ---------------------------------------------------------------------------


def _matmul_candidates() -> Iterable[Tuple[int, int, int]]:
    """Exhaustive matmul shapes from a curated dim pool.

    Deterministic (no shuffle) so re-running the generator produces the same
    JSON. Pool size: 17 dims -> 4913 shapes, manageable for classification.
    """
    sizes = [64, 96, 128, 192, 256, 384, 512, 768, 1024,
             1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
    for m in sizes:
        for k in sizes:
            for n in sizes:
                yield (m, k, n)


def _vector_add_candidates() -> Iterable[Tuple[int]]:
    """Vector-add candidate pool spanning the cache hierarchy.

    Vector add is the V5 plan's zero-reuse ground truth: bytes-per-second
    measured at each tier. The classifier will collapse most of these to
    LAUNCH_BOUND (small N) or DRAM_BOUND (large N) because vector_add's
    OI = 1/(3*bpe) is well below any hardware's AI breakpoint. The
    *tier* information comes from the working-set bucket each N falls
    into (visualized in the lat-vs-WS panel), not from the V4 regime
    classification. V5-3's tier_picker will reinterpret these same N
    values via tier-hit decomposition and emit L1_BOUND / L2_BOUND /
    DRAM_BOUND directly.

    N values log-spaced from 1 KB (256 fp32 elements) to 1 GB
    (256 M fp32 elements). Covers L1 / L2 / L3 / DRAM on every V4 target.
    """
    # Powers of 4 from 256 to 268M. Eleven values; one log decade between
    # neighbors gives clean WS-bucket separation in the visualizer.
    sizes = [256, 1024, 4096, 16384, 65536, 262144, 1048576,
             4194304, 16777216, 67108864, 268435456]
    for n in sizes:
        yield (n,)


def _linear_candidates() -> Iterable[Tuple[int, int, int]]:
    """Exhaustive Linear shapes from realistic batch + feature dim pools.

    Includes large-B / modest-feats tall-skinny shapes (e.g., B=8192,
    IN=OUT=768) because those are structurally the only Linear shapes
    that can hit ALU_BOUND on a high-AI-breakpoint accelerator like H100
    fp16: enough compute to clear the launch threshold while keeping
    WS = (B*IN + IN*OUT + B*OUT)*bpe under the L1 budget.
    """
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1024, 2048, 4096, 8192, 16384]
    feats = [128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536,
             2048, 3072, 4096, 8192, 12288, 16384]
    for b in batches:
        for i in feats:
            for o in feats:
                yield (b, i, o)


# ---------------------------------------------------------------------------
# Bucketing + sampling
# ---------------------------------------------------------------------------


def _classify_all_targets(op: str, shape: Tuple[int, ...], dtype: str) -> dict[str, Regime] | None:
    """Classify the shape against every target. Returns None if any target
    classifies it as AMBIGUOUS (Principle 1: failures must attribute to one
    layer)."""
    out: dict[str, Regime] = {}
    for key, _dtype, hw in TARGETS:
        if dtype != _dtype:
            # Each target has a fixed dtype in this generator. A shape is
            # only included if its dtype matches exactly one target.
            continue
        r = classify_regime(op, shape, dtype, hw)
        if r is Regime.AMBIGUOUS:
            return None
        if r is Regime.UNSUPPORTED:
            return None
        out[key] = r
    return out if out else None


def _bucket(op: str, candidates: Iterable[Tuple[int, ...]],
            dtypes: list[str]) -> dict[tuple[str, Regime], list[SweepEntry]]:
    """Group candidates by (target_key, regime) so we can sample per bucket."""
    buckets: dict[tuple[str, Regime], list[SweepEntry]] = defaultdict(list)
    for shape in candidates:
        for dtype in dtypes:
            classified = _classify_all_targets(op, shape, dtype)
            if classified is None:
                continue
            entry = SweepEntry(
                shape=tuple(shape),
                dtype=dtype,
                regime_per_hw={k: r.value for k, r in classified.items()},
            )
            for target_key, regime in classified.items():
                buckets[(target_key, regime)].append(entry)
    return buckets


def _sample_split(
    buckets: dict[tuple[str, Regime], list[SweepEntry]],
    cal_target: dict[Regime, int],
    val_target: dict[Regime, int],
) -> tuple[list[SweepEntry], list[SweepEntry]]:
    """Pick disjoint calibration and validation entries per regime.

    Strategy: for each (target, regime) bucket, sort entries by working-set
    size (deterministic), then take alternating entries -- evens to
    calibration, odds to validation -- so the two sets cover similar ranges
    without overlap. Trim to target counts.
    """
    cal_entries: dict[tuple, SweepEntry] = {}
    val_entries: dict[tuple, SweepEntry] = {}

    def _sort_key(e: SweepEntry) -> tuple:
        # Order by total dim product so sweeps span size range evenly.
        prod = 1
        for d in e.shape:
            prod *= d
        return (prod, e.shape, e.dtype)

    for (target_key, regime), entries in buckets.items():
        unique = sorted({(e.shape, e.dtype): e for e in entries}.values(), key=_sort_key)
        cal_quota = cal_target.get(regime, 0)
        val_quota = val_target.get(regime, 0)
        cal_picked = 0
        val_picked = 0
        for i, e in enumerate(unique):
            key = (e.shape, e.dtype)
            if i % 2 == 0 and cal_picked < cal_quota and key not in val_entries:
                cal_entries[key] = e
                cal_picked += 1
            elif val_picked < val_quota and key not in cal_entries:
                val_entries[key] = e
                val_picked += 1
            if cal_picked >= cal_quota and val_picked >= val_quota:
                break

    return (sorted(cal_entries.values(), key=_sort_key),
            sorted(val_entries.values(), key=_sort_key))


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _write_json(path: Path, op: str, purpose: str, entries: list[SweepEntry]) -> None:
    payload = {
        "op": op,
        "purpose": purpose,
        "generator_seed": SEED,
        "generated_against_hardware": [t[0] for t in TARGETS],
        "shapes": [
            {"shape": list(e.shape), "dtype": e.dtype, "regime_per_hw": dict(e.regime_per_hw)}
            for e in entries
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path.relative_to(Path.cwd())}: {len(entries)} entries")


def _summarize(entries: list[SweepEntry], label: str) -> None:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for e in entries:
        for hw, regime in e.regime_per_hw.items():
            counts[(hw, regime)] += 1
    print(f"  {label} distribution:")
    for key in sorted(counts):
        print(f"    {key[0]:>20s}  {key[1]:<14s}  {counts[key]:>3d}")


def main() -> None:
    print(f"Generator seed: {SEED}")
    print(f"Targets: {[t[0] for t in TARGETS]}")
    print()

    # matmul: i7-12700k expects fp32, H100 expects fp16. Generate both
    # halves and write a single combined file (the harness filters by hw).
    for op_name, gen, dtypes in [
        ("matmul",     _matmul_candidates,     ["fp32", "fp16"]),
        ("linear",     _linear_candidates,     ["fp32", "fp16"]),
        ("vector_add", _vector_add_candidates, ["fp32", "fp16"]),
    ]:
        print(f"=== {op_name} ===")
        candidates = list(gen())
        print(f"  candidate pool: {len(candidates)} shapes")
        buckets = _bucket(op_name, candidates, dtypes)
        print(f"  populated buckets: {len(buckets)}")

        cal, val = _sample_split(buckets, CALIBRATION_PER_REGIME, VALIDATION_PER_REGIME)
        # Sanity: no overlap
        cal_keys = {(tuple(e.shape), e.dtype) for e in cal}
        val_keys = {(tuple(e.shape), e.dtype) for e in val}
        assert cal_keys.isdisjoint(val_keys), "calibration / validation must be disjoint"

        _write_json(SWEEP_DIR / f"{op_name}_calibration.json",
                    op_name, "calibration", cal)
        _summarize(cal, "calibration")
        _write_json(SWEEP_DIR / f"{op_name}_validation.json",
                    op_name, "validation", val)
        _summarize(val, "validation")
        print()


if __name__ == "__main__":
    main()

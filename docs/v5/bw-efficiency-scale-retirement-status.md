# V5-6: `bw_efficiency_scale` Retirement Status

**Phase:** A (deprecation) shipped. Phase B (final removal) pending.

The V5-6 phase of the V5 memory-hierarchy rewrite is to retire
`RooflineAnalyzer._get_bandwidth_efficiency_scale` -- the 515-line
hand-tuned scalar bandwidth-efficiency function -- once every active
mapper has populated `memory_hierarchy` + tier-specific calibrations.
This file tracks the calibration coverage gap blocking final removal.

## Phase A (this doc's PR)

- `_get_bandwidth_efficiency_scale` is now a clearly-labeled
  fallback-only path. The V5-3b tier-aware roofline is the default
  (per PR #113).
- The function emits a one-time `DeprecationWarning` per
  `RooflineAnalyzer` instance on first invocation, so callers can
  spot which workflows still depend on the legacy path.
- The function body is unchanged; no behavior change for callers.
- `_analyze_subgraph` was restructured so the legacy function is
  only invoked when the tier-aware path declines (no wasted work
  when the override is going to win).

The fallback fires when ANY of:

  (a) `use_tier_aware_memory=False` -- caller opted OUT
  (b) Subgraph isn't tier-aware-eligible:
      - multi-op (`num_operators != 1`)
      - unsupported op type (no `ReuseModel` registered)
      - 3D-batched matmul / broadcast elementwise / mismatched dtypes
  (c) Hardware's `memory_hierarchy` has < 2 tiers OR no
      `tier_achievable_fractions` calibration

## Phase B (future PRs, gated on per-mapper calibration)

Once every active mapper has populated `memory_hierarchy` AND
`tier_achievable_fractions`, AND the V4 floor parity is verified for
each:

1. Delete `_get_bandwidth_efficiency_scale` and its tests
2. Make the tier-aware path mandatory (remove the `use_tier_aware_memory`
   flag, or keep it but raise on `False`)
3. Update the roofline docstring to drop legacy references

## Calibration coverage matrix

To unblock Phase B, every row needs ✓ in both columns AND a verified
V4 floor that's equal-or-better under the tier-aware path.

| mapper | memory_hierarchy populated | tier_achievable_fractions calibrated | V4 floor parity verified |
|---|---|---|---|
| **i7-12700K (tiny)** | ✓ L1 + L2 + L3 + DRAM | ✓ {L1, L2, L3, DRAM} + per-op (matmul/linear) | ✓ |
| **i7-12700K (large)** | ✓ same | ✓ same (sibling) | ✓ (same hardware) |
| Jetson Orin Nano 8GB | ✓ L1 + L2 + DRAM | ⚠️ DRAM only | ⚠️ no matmul/linear baseline |
| Jetson Orin AGX 64GB | ⚠️ DRAM only | ✗ none | ⚠️ no baseline |
| Jetson Orin NX 16GB | ⚠️ DRAM only | ✗ none | ⚠️ no baseline |
| H100 SXM5 80GB | ✓ L1 + L2 + DRAM (#61) | ✗ none | ⚠️ no baseline |
| A100 SXM4 80GB | ⚠️ DRAM only | ✗ none | ⚠️ no baseline |
| V100 / T4 / RTX | ⚠️ DRAM only | ✗ none | ⚠️ no baseline |
| KPU mappers (T64/T128) | ⚠️ DRAM only | ✗ none | ⚠️ no baseline |
| TPU mappers | ⚠️ DRAM only | ✗ none | ⚠️ no baseline |
| ARM Mali GPU | ⚠️ DRAM only | ✗ none | ⚠️ no baseline |
| Other accelerators | ⚠️ DRAM only | ✗ none | ⚠️ no baseline |

Only i7-12700K is fully calibrated. Phase B blocked.

## How to advance a mapper from ⚠️ to ✓

For each uncalibrated mapper:

1. **Capture V4 baselines on the hardware** (vector_add at minimum;
   matmul/linear if the hardware supports them and a measurer exists)
2. **Derive the DRAM achievable_fraction** from the plateau rows
   (largest N values, working set >> outermost cache)
3. **Optionally derive per-tier and per-op fractions** if the
   hardware has multi-tier `memory_hierarchy` and the V5-2b workload
   covers L1/L2/L3 boundaries
4. **Set values** in the mapper factory's `tier_achievable_fractions`
   (and `tier_achievable_fractions_by_op` if matmul/linear differ
   from vector_add)
5. **Verify V4 floor parity**: run the V4 sweep with both
   `--use-tier-aware-memory` and the explicit opt-out
   (`use_tier_aware_memory=False`); the tier-aware path's PASS rate
   must be equal or better
6. **Update this file** moving the row from ⚠️ to ✓
7. Commit

Once all rows are ✓, ship Phase B.

## Why phased

The 515-line `_get_bandwidth_efficiency_scale` has hand-tuned
heuristics for kernel-size scaling, depthwise conv penalties, GPU
small-kernel curves, and CPU bw_efficiency derate. Stripping it
without per-mapper calibration in place would over-predict throughput
for ~40 uncalibrated targets and break their V4 floors.

Phase A is non-disruptive: same behavior, clearly-labeled deprecated,
warning surfaces who's still using it. Phase B can land
incrementally per-mapper without surprise.

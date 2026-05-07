# i7-12700K L3 Calibration Analysis (V5-5 follow-up)

**Status:** L3 `achievable_fraction = 0.84` on
`create_i7_12700k_mapper` and `create_i7_12700k_large_mapper`,
derived by 2-point regression on dispatch-corrected L3-bound rows
of `i7_12700k_vector_add.csv`. Previous value 0.82 came from a
single point (N=1M); the regression confirms the value is right
to within 0.02 but documents the methodology for future updates.

## TL;DR

L3 effective bandwidth on i7-12700K for L3-bound shapes
(WS in `(L1_aggregate, L3_capacity]`) is **167 ± 1 GB/s** under
single-thread `torch.add(a, b)`. As a fraction of the i7 mapper's
L3 peak (200 GB/s), that's **0.836**, rounded to **0.84**.

This calibration value is the BEST single-point fit available from
V5-2b vector_add data, but it cannot fix the V4 N=262K floor
failure (current +75% over-prediction). That failure is a
structural model gap -- the i7 mapper conflates per-core L2
(1.25 MB) and chip-wide L3 (25 MB) into a single L3 tier, so
shapes whose data benefits from per-core L2 hits (like
N=262K, WS=3MB) appear to exceed L3 peak (1.71x) and the binary
tier picker can't place them correctly. Fix needs an L2 tier in
the mapper, not a calibration knob.

## Methodology

For each L3-bound vector_add baseline shape, derive the
dispatch-corrected effective bandwidth:

```
BW = working_set_bytes / (latency - dispatch_overhead)
```

`dispatch_overhead = 1.79 us`, derived in
`docs/calibration/i7-12700k-l1-calibration-analysis.md` from the
two L1-resident points (N=256, N=1024). Same physical dispatch chain
applies to all vector_add invocations, so the constant is reusable.

Then compute the achievable fraction:

```
fraction = BW / L3_peak     where L3_peak = 200 GB/s
```

## Data

i7 hierarchy: `L1_aggregate = 512 KB` (16 cores * 32 KB),
`L3 = 25 MB`. Operand-aware tier picker assigns L3-binding when
`L1_agg < 3*N*bpe <= L3_cap`, i.e. `42K < N <= 2M` for fp32
vector_add.

Three baseline rows fall in this range:

| N | WS | latency | dispatch-corrected | BW | fraction |
|---|---|---|---|---|---|
| 65K | 768 KB | 6.48 us | 4.69 us | 167.6 GB/s | **0.838** |
| 262K | 3 MB | 10.96 us | 9.17 us | 343.0 GB/s | **1.715** |
| 1M | 12 MB | 77.26 us | 75.47 us | 166.7 GB/s | **0.834** |

## Analysis

The N=65K and N=1M points are remarkably consistent (0.838 vs 0.834,
delta 0.5%). They span 16x in working set (768 KB to 12 MB) yet
yield essentially the same effective bandwidth. **This is the L3
streaming bandwidth signal**.

The N=262K point shows 1.715x the L3 peak -- non-physical. The
data is partially hitting faster caches:

* On Alder Lake, each P-core has 1.25 MB of L2 cache
* For a 3 MB working set on a single thread, ~40% of the data
  can be resident in the active core's L2 (BW ~200+ GB/s/core)
  and the remainder streams from L3
* The mixed BW (per-core L2 + chip-wide L3) exceeds L3 peak
  alone

The i7 mapper doesn't currently model per-core L2 -- it stores the
LLC value (25 MB) in `l2_cache_total` per the M1 schema convention
and emits a single L3 tier in `memory_hierarchy` (no distinct L2
hop). So the V5-3b tier picker correctly identifies N=262K as
"binding at L3" but the bandwidth assumption (streaming from L3
only) misses the L2 boost.

**Calibration implication:** the N=65K + N=1M average (0.836) is
the true L3 streaming fraction. The N=262K outlier is not a
calibration anomaly but a SHAPE-DEPENDENT MIXED-CACHE behavior the
binary tier model cannot capture. Including it in the regression
would inflate the calibration above 1.0, breaking other shapes
where the data really is purely L3-resident.

## Why this can't fix V4 N=262K floor failure

The V4 vector_add validation harness fails on N=262K with current
calibration (0.82 -> +75% over-prediction). The structural reason:

```
Shape:         N=262K fp32, WS=3 MB
Measured:      11.0 us (incl. ~2 us dispatch -> ~9.2 us at-the-data)
Required BW:   3 MB / 9.2 us = 326 GB/s effective
L3 peak:       200 GB/s

326 / 200 = 1.63x -- exceeds L3 peak by 63%.
```

No L3 `achievable_fraction` in `[0.0, 1.0]` (the dataclass invariant
range) produces a prediction matching the measurement. Even at the
non-physical 1.0 (perfect L3 streaming) the prediction would be
3 MB / 200 GB/s = 15 us, vs measured 11 us -- still +36% over,
outside the 30% LAUNCH band.

The only fixes are:

1. **Add per-core L2 tier to the i7 mapper.** L2 = 1.25 MB / core,
   12 effective cores, BW ~150 GB/s / core (TBD via microbench).
   The operand-aware picker would route N=262K to L2 instead of
   L3, modeling the cache hits correctly.

2. **Weighted multi-tier model.** When WS spans multiple tiers,
   `effective_bw = alpha * BW_inner + (1-alpha) * BW_outer` where
   `alpha = (tier_capacity - WS) / tier_capacity` (or similar).
   Allows partial-L2 + partial-L3 mixing without adding tiers.

3. **Cache-line-aware partial residency.** Even more granular,
   models specific access patterns. Significant complexity.

Until any of these land, the N=262K failure is the model gap, not
a calibration gap. The L3 = 0.84 value is the right single-point
calibration regardless.

## When to revisit

Revisit this calibration when ANY of:

1. **Per-core L2 tier added to the i7 mapper.** Once L2 is its own
   tier, the picker may route some currently-L3-binding shapes to
   L2 instead. The L3 calibration would then apply only to truly
   L3-streaming shapes and may need re-derivation from a
   different (smaller) set of baseline rows.

2. **Multi-thread vector_add benchmark captured.** Single-thread
   data only sees one core's slice of the L3 BW (200 GB/s shared
   across all 12-16 cores). A multi-thread benchmark would let us
   measure aggregate L3 BW directly and may shift the calibration
   significantly.

3. **L3 tier added with separate `l2_bandwidth_bps`.** The current
   i7 mapper has `l2_bandwidth_bps = None` (only L3). A future
   refinement that distinguishes per-core L2 from chip-wide L3
   would invalidate the current single-tier assumption.

The 0.84 value is locked by
`tests/hardware/test_tier_achievable_fractions.py::test_i7_12700k_l3_calibration`
and the structural limit by
`tests/hardware/test_i7_l3_calibration_cannot_fix_partial_l2_shapes.py`.

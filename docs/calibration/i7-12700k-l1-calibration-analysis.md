# i7-12700K L1 Calibration Analysis (V5-5 follow-up)

**Status:** L1 `achievable_fraction = 0.02` on
`create_i7_12700k_mapper` and `create_i7_12700k_large_mapper`,
matching the 2-point regression below.

**Update history:**
* PR #105: L1 left at default 1.0 with the rationale "dispatch
  floor dominates for every L1-binding matmul shape" -- correct
  for matmul but missed the vector_add medium-N regime.
* PR (this one): L1 set to 0.02 because vector_add at N=16K (a
  shape the V4 sweep validates) DOES exceed the 2 us op-aware
  dispatch floor with the proper L1 fraction, making the
  calibration value the binding constraint there.

## TL;DR

The 2-point regression on dispatch-corrected L1-resident vector_add
rows (N=256 and N=1024) yields **BW = 69 GB/s, dispatch = 1.79 us**.
As a fraction of the L1 aggregate peak (3500 GB/s), that's
**0.020**. Originally rejected as "structurally unobservable for
matmul" -- which is true -- but vector_add medium-N (N=4K to 16K)
sits in the regime where the L1 fraction directly drives the
prediction and 0.020 lands V4 floor passes within tolerance.

## Why vector_add can't isolate L1 BW directly

The V5-2b vector_add baseline at `i7_12700k_vector_add.csv` measures
single-threaded latency. For shapes that fit a single P-core's L1
(32 KB, so N <= 2,720 fp32 elements):

| N    | WS (KB) | latency (us) | derived BW (GB/s) |
|------|---------|--------------|-------------------|
| 256  | 3       | 1.84         | 1.7               |
| 1024 | 12      | 1.97         | 6.2               |

A two-point regression on these L1-resident shapes:

```
delta_lat = 1.97 us - 1.84 us = 130 ns
delta_ws  = (1024 - 256) * 12 = 9,216 bytes
BW = 9,216 / 130e-9 = 69.3 GB/s
dispatch overhead = 1.84 us - (3,072 / 69.3e9) = 1.79 us
```

Two issues with using this 69 GB/s as the L1 calibration anchor:

1. **Single-thread vs aggregate.** The i7 mapper exposes
   `peak_bandwidth_bps = 3500 GB/s` for L1 (16 cores * weighted
   per-unit). Single-threaded vector_add only uses one core, so
   it sees roughly `3500 / 16 = 219 GB/s` peak. The 69 GB/s
   observation is `0.32` of single-core peak (or `0.020` of the
   aggregate). The MemoryTier model has a single
   `achievable_fraction` per tier; it doesn't distinguish
   single-thread from all-thread workloads. Setting
   `achievable_fraction = 0.020` would correctly model a
   single-thread vector_add but actively under-predict any
   multi-threaded workload that genuinely uses all 16 cores'
   L1 BW concurrently. There's no calibration value that's
   right for both regimes.

2. **Dispatch dominates the regression.** The dispatch overhead
   (1.79 us) is more than 14x the BW-limited delta (130 ns).
   Small measurement perturbations propagate dramatically.
   Even a 50 ns increase on either point (~3% relative noise on
   1.84 us) would shift the BW estimate by 30%.

## Where L1 calibration matters: vector_add medium-N

The dispatch floor argument (below) is correct for **matmul**:
small matmul shapes have tiny C-tiles (clamped to min(M,N)) and
correspondingly tiny `bytes_loaded`, so memory_time stays in
tens-of-nanoseconds and the dispatch floor wins regardless of L1
fraction. But **vector_add at medium N** (4K-16K elements) has
working set 48-192 KB, all of which streams through L1, giving
memory_time on the order of microseconds when L1 fraction is
calibrated correctly:

| N | WS | math at L1=0.020 (= 70 GB/s) | floor (vector_add 2us) | binding |
|---|---|---|---|---|
| 1024 | 12 KB | 0.17 us | 2 us | floor |
| 4096 | 48 KB | 0.69 us | 2 us | floor |
| **16K** | **192 KB** | **2.74 us** | 2 us | **math wins** |

For N=16K the math part exceeds the floor, and L1 fraction directly
drives the prediction. With the current L1 = 0.020 the prediction
lands at 2.74 us vs measured 3.09 us (-9%, well inside the 30%
LAUNCH band). For comparison, **before** this calibration was set
(when L1 defaulted to 1.0 in the V5-1 / pre-PR-#110 era), the math
part was 0.055 us, the floor won, prediction was 2 us, and the
shape FAILED at -35% off measured.

This is what the prior conclusion missed: while L1 calibration is
structurally moot for matmul, it's load-bearing for vector_add
medium-N. Setting L1 = 0.020 is harmless for matmul (floor still
wins there) but materially improves vector_add V4 floors (3/5 ->
4/5 pass under tier-aware on i7).

## Why L1 calibration was originally judged moot (matmul-only argument)

Even if we settled on a defensible L1 value, it wouldn't change any
**matmul** predictions. The op-aware CPU dispatch floor in
`graphs/estimation/roofline.py::_analyze_subgraph` is 6 us for
matmul. For L1-binding matmul shapes, the tier-aware memory_time
is in tens of nanoseconds -- the floor is two orders of magnitude
larger.

Sweep across the V4 matmul shape range (M, K, N <= 128, the
launch_bound regime that generates L1-binding shapes):

```
$ python -c '
from graphs.estimation.reuse_models import REUSE_MODELS
from graphs.estimation.tier_picker import pick_binding_tier
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
hw = create_i7_12700k_mapper().resource_model
mm = REUSE_MODELS["matmul"]
shapes = [(M, K, N) for M in [32, 64, 96, 128]
                     for K in [32, 64, 96]
                     for N in [32, 64, 96, 128]]
for s in shapes:
    r = pick_binding_tier(mm, s, "fp32", hw.memory_hierarchy)
    if r and r.binding_tier.name == "L1":
        mem_us = r.bytes_loaded / r.binding_tier.effective_bandwidth_bps * 1e6
        # mem_us in [4.7e-3, 65.5e-3] us across all 48 L1-binding shapes
'
```

All 48 L1-binding shapes have memory_time in `[4.7 ns, 65.5 ns]`.
The 6 us matmul dispatch floor wins by 90x to 1300x. Setting
`L1.achievable_fraction = 0.020` changes memory_time to
`[235 ns, 3275 ns]` -- still all below the matmul floor. **For
matmul, no prediction changes.** The vector_add medium-N regime
above is the new reason L1 = 0.020 is set anyway.

## When this needs to be revisited (further)

L1 calibration is now set, but several open questions remain:

1. **A multi-threaded vector_add benchmark.** Single-thread data
   underestimates aggregate L1 BW. A multi-thread benchmark would
   let us measure the aggregate utilization directly and might
   shift the calibration significantly.

2. **The MemoryTier model gains thread-count awareness.** A more
   honest model would have `effective_bandwidth_bps(thread_count)`
   that interpolates between single-thread (`peak / num_units`)
   and aggregate (`peak`). Then the existing single-thread
   vector_add data calibrates the single-thread end of that curve
   without forcing a single low fraction across all workloads.

3. **The dispatch floor architecture changes.** If a future change
   tightens or removes the floor, more shape classes become
   memory-time-dominated and the L1 = 0.02 value may start
   affecting matmul/linear too.

The L1 = 0.02 value is locked by
`tests/hardware/test_tier_achievable_fractions.py::test_i7_12700k_l1_calibration`
and the matmul-specific dispatch-floor argument (which still holds)
by `tests/hardware/test_i7_l1_calibration_is_dispatch_floor_dominated.py`.

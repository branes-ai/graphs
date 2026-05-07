# i7-12700K L1 Calibration Analysis (V5-5 follow-up)

**Status:** L1 `achievable_fraction` is intentionally **left unset**
(default 1.0) on `create_i7_12700k_mapper` and
`create_i7_12700k_large_mapper`. This document captures the analysis
behind that decision so a future contributor doesn't think it's a
forgotten todo.

## TL;DR

For matmul on i7-12700K, the V5-3b tier-aware analyzer is
**dispatch-floor-bound for every L1-binding shape**. The CPU dispatch
floor (`5 us` in `RooflineAnalyzer._analyze_subgraph`) supersedes any
L1-derived memory_time, regardless of `L1.achievable_fraction`.
Setting L1 to 0.05 vs 1.0 produces identical predictions across all
48 L1-binding matmul shapes the V4 sweep generates. So the
calibration value is a no-op until either (a) the dispatch floor
moves or (b) a workload arrives whose L1-bound memory_time can
exceed 5 microseconds.

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

## Why L1 calibration is moot on the analyzer side

Even if we settled on a defensible L1 value, it wouldn't change any
predictions. The CPU dispatch floor in
`graphs/estimation/roofline.py::_analyze_subgraph`:

```python
if self.resource_model.hardware_type.name == 'CPU':
    actual_latency = max(actual_latency, 5e-6)
```

This 5 us floor came from #69's analysis: real PyTorch / nn.Module
calls have ~5 us of dispatch + parameter access + bias-add overhead
per forward call, regardless of kernel size. For L1-binding shapes,
the tier-aware memory_time is in tens of nanoseconds — the floor is
two orders of magnitude larger.

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
The 5 us floor wins by 75x to 1000x. Setting
`L1.achievable_fraction = 0.020` would change memory_time to
`[235 ns, 3275 ns]` -- still all below the floor. **No prediction
changes.**

## When this needs to be revisited

The L1 calibration becomes load-bearing when ANY of the following
land:

1. **The dispatch floor drops or becomes shape-dependent.** If a
   future change tightens the floor below ~50 ns for some shape
   class, L1-binding predictions could become memory-time-dominated
   and `L1.achievable_fraction` starts mattering.

2. **A multi-threaded vector_add benchmark gets captured.** Running
   the V5-2b workload across all 16 cores would let us measure the
   aggregate L1 BW directly. The natural place is to extend the V4
   capture path with a `--threads` knob.

3. **The MemoryTier model gains thread-count awareness.** A more
   honest model would have `effective_bandwidth_bps(thread_count)`
   that interpolates between single-thread (`peak / num_units`) and
   aggregate (`peak`). Then the existing single-thread vector_add
   data calibrates the single-thread end of that curve.

Until then, L1 stays absent from
`tier_achievable_fractions` on the i7 mappers, defaulting to 1.0.
The decision is locked by `tests/hardware/test_tier_achievable_fractions.py::test_i7_12700k_l1_uncalibrated`
and the analytical claim above by
`tests/hardware/test_i7_l1_calibration_is_dispatch_floor_dominated.py`.

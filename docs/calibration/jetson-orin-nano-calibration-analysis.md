# Jetson Orin Nano 8GB Calibration Analysis (V5 follow-up)

**Status:** DRAM `achievable_fraction = 0.55` is **verified against
matmul + linear baselines** (not just vector_add). L1 / L2 stay
default 1.0; insufficient data to calibrate. The remaining V4 floor
gap on Orin Nano is dominated by **model gaps**, not calibration
values.

## TL;DR

| op | V4 floor | failure modes |
|---|---|---|
| vector_add | 4/7 PASS | GPU dispatch floor |
| matmul | 22/48 PASS | dispatch floor + skinny tile + compute model |
| linear | 18/46 PASS | same |

Improving the floor requires three model fixes (in priority order):

1. **GPU dispatch floor** -- 16 L1-binding shapes currently predict
   5-13 µs but measure 120-170 µs (CUDA kernel launch overhead).
   *Addressed by PR #116.*
2. **~~Hardware-aware skinny matmul threshold~~** Aspect-ratio-based
   skinny detection -- the failing shapes have `min(M, N) = 64-256`
   (well above 16) but aspect ratios of 100+. Hypothesis was
   hardware-dependent threshold; empirical fix is shape-intrinsic
   aspect-ratio cutoff. *Addressed by the V5 follow-up
   aspect-ratio PR; see
   [`docs/v5/skinny-matmul-aspect-ratio-detection.md`](../v5/skinny-matmul-aspect-ratio-detection.md).
   Note: this fixes memory-physics latency precision but does NOT
   move V4 PASS counts because the dominant failure on these shapes
   is the regime classifier (compute model, item 3).*
3. **Compute efficiency model refinement** -- 19 -side
   under-predictions where compute_time is too fast vs reality;
   separate from the memory model. *Addressed by the V5 follow-up
   compute-derate PR; see
   [`docs/v5/jetson-compute-efficiency-derate.md`](../v5/jetson-compute-efficiency-derate.md).
   Per-mapper `compute_efficiency_overrides_by_op` field replaces the
   AGX-tuned legacy curve for matmul (0.70) and linear (0.94) on
   Orin Nano FP16. Latency-pass gains: matmul +8, linear +7-12.
   V4 PASS still blocked by the sweep-file regime classifier
   (44/48 matmul, 45/46 linear records have stale sweep regimes;
   regenerating the sweep is a separate task).*

Setting calibration fractions any differently won't close these
gaps -- the underlying physics (or model assumptions) is what's off.

## Methodology

For each baseline shape, derive observed effective bandwidth two
ways:

1. `BW_operand = operand_bytes / latency` -- assumes no reuse, just
   sums input + output + weight tensors
2. `BW_bytes_loaded = bytes_loaded / latency` -- uses the V5-3a reuse
   model's tile-streaming reload counts

Then `fraction = BW / peak_for_binding_tier`. Different ops should
converge if calibration is correct.

## Data summary

### DRAM tier (vector_add baseline)

Plateau rows (N >= 16M, WS >> 2 MB L2):

| N | WS | latency | BW | fraction (peak 102 GB/s) |
|---|---|---|---|---|
| 16M | 100 MB | 1634 us | 61.6 GB/s | 0.604 |
| 67M | 400 MB | 7135 us | 56.4 GB/s | 0.553 |
| 268M | 1.6 GB | 28460 us | 56.6 GB/s | 0.555 |

Median **0.55** -- matches the calibrated value.

### DRAM tier (matmul baseline, top 5 shapes)

Using `bytes_loaded` from the V5-3a reuse model:

| shape | operand | bytes_loaded | latency | BW | fraction |
|---|---|---|---|---|---|
| (64, 16384, 8192) | 272 MB | 539 MB | 4890 us | **110 GB/s** | **1.081** |
| (96, 12288, 6144) | 155 MB | 304 MB | 3113 us | 97.8 GB/s | 0.958 |
| (64, 8192, 16384) | 272 MB | 541 MB | 5599 us | 96.6 GB/s | 0.947 |
| (96, 6144, 12288) | 155 MB | 307 MB | 3199 us | 95.9 GB/s | 0.940 |
| (4096, 16384, 128) | 139 MB | 271 MB | 2840 us | 95.3 GB/s | 0.934 |

The `bytes_loaded` fractions cluster near and exceed peak (1.08x for
the top shape -- non-physical). This says the V5-3a reuse model is
**inflating** `bytes_loaded` for these Jetson shapes via the
ceil(N/Nt) reload count, which doesn't match how Jetson's cuBLAS
actually streams matmul.

Operand-based BW (`operand / latency`) on the same shapes converges
to the **0.55 plateau** matching vector_add -- no different per-op
fraction needed at DRAM. The cache-reuse benefit that bumped i7
matmul to 0.85 doesn't materialize on Jetson because the
cache hierarchy (L1=128 KB/SM, L2=2 MB) is too small to capture
significant matmul working sets above L2.

### L1 / L2 tiers

L1-binding shapes (operand <= L1 aggregate 1 MB):

* All measure ~120 us regardless of N. The Jetson kernel-launch
  overhead (~100 us) dominates the BW-limited part.
* Two-point regression to back out dispatch + BW yields **negative
  slopes** at small N -- noise floor exceeds signal.
* Conclusion: V5-2b vector_add data **cannot calibrate L1 BW** on
  Orin Nano.

L2-binding shapes (operand in (1 MB, 2 MB]):

* Only 1 vector_add data point in this range (N=262144, WS=1.5 MB).
  Latency 112 us still dispatch-dominated. Single point, no
  regression.
* No matmul / linear shapes in V4 sweep land here (the operand
  footprint either fits L1 aggregate or overflows L2 to DRAM).
* Conclusion: **cannot calibrate L2** from current baselines either.

L1 and L2 staying at default 1.0 is the honest answer until either:

* A multi-thread vector_add benchmark amortizes dispatch and exposes
  the BW-limited regime
* A different microbench (e.g., a timing loop within a single CUDA
  kernel) isolates per-tier BW from kernel-launch overhead

## V4 floor failure breakdown

26 matmul + 28 linear failures categorized:

### Category 1: GPU dispatch floor missing (10 + 6 = 16 shapes)

L1-binding small shapes:

| shape | predicted | measured | error |
|---|---|---|---|
| (64, 128, 64) matmul | 12 us | 119 us | -90% |
| (96, 64, 96) matmul | 12 us | 116 us | -89% |
| (1, 128, 256) linear | 6 us | 173 us | -97% |
| (2, 128, 128) linear | 6 us | 169 us | -97% |

The op-aware CPU dispatch floor (#108: vector_add 2 us / matmul 6 us
/ linear 9 us) doesn't apply to GPU. `_estimate_overhead` returns
5 us base for GPU, which is two orders of magnitude below the
empirical Jetson kernel-launch overhead.

**Fix:** add op-aware GPU dispatch floor mirroring the CPU one,
calibrated against Jetson's CUDA kernel-launch overhead. Expected
to fix all 16 of these failures.

### Category 2: Skinny matmul reload inflation (9 + 10 = 19 shapes)

DRAM-binding +side over-predictions:

| shape | bytes_loaded | predicted | measured | error |
|---|---|---|---|---|
| (64, 16384, 8192) matmul | 539 MB | 9612 us | 4890 us | +97% |
| (96, 12288, 6144) matmul | 304 MB | 5430 us | 3113 us | +74% |
| (64, 8192, 16384) linear | 541 MB | 9650 us | 5001 us | +93% |

`min(M, N)` for these shapes is 64-96 -- above the SKINNY_THRESHOLD
of 16 in `MatmulReuseModel`, so the optimal-square branch fires
with tile = (min, min) clamped. The reload count `ceil(N / tile)`
explodes when N >> tile, inflating `bytes_loaded` ~2x.

For comparison, on i7 the same shapes have operand fitting LLC (25
MB L3) so they bind at L3 with the calibrated L3 BW; on Jetson
they overflow the 2 MB L2 to DRAM, and the inflated `bytes_loaded`
divides by DRAM peak directly.

**Fix:** make the skinny threshold hardware-aware (or
shape-relative-to-L1-cap-aware). For Jetson, threshold ~64-96
would be more appropriate. Expected to fix 19 +side
over-predictions.

### Category 3: Compute model under-derate (11 + 8 = 19 shapes)

DRAM-binding -side under-predictions:

| shape | predicted | measured | error |
|---|---|---|---|
| (2048, 2048, 2048) matmul | 1799 us | 4636 us | -61% |
| (1536, 3072, 1536) matmul | 1519 us | 3647 us | -58% |
| (256, 12288, 4096) linear | 3669 us | 8104 us | -55% |

These are compute-bound shapes where `compute_time` per the model
exceeds `memory_time`. The model says faster than reality by 2-2.5x.

`_get_compute_efficiency_scale` for GPU returns a curve in
[0.3, 0.7] based on op size; the empirical compute throughput on
Jetson Orin Nano in 15W mode is harder to hit than the curve
assumes (thermal throttling + DVFS + small-cluster scheduling).

**Fix:** separate concern from memory calibration; would need a
compute-side calibration sweep. Out of V5 memory scope.

## When to revisit

* **GPU dispatch floor lands** (next PR) -> rerun this analysis;
  Category 1 gone, Category 2 + 3 remain.
* **Hardware-aware skinny threshold lands** -> Category 2 gone.
* **Multi-thread vector_add or L1/L2 microbench available** ->
  retry L1 / L2 calibration.
* **Compute model refinement** -> Category 3 addressable.

The Orin Nano calibration matrix in
`docs/v5/bw-efficiency-scale-retirement-status.md` will flip to ✓
in all three columns once Categories 1 and 2 are resolved AND V4
floor parity is verified -- Category 3 is non-blocking for
bw_efficiency_scale retirement (Phase B) since it's a
compute-side issue.

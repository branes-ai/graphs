# V5 Follow-up: 3-Way Hardware Comparison Plots

## What this PR delivers

1. **Regenerated baseline plots for i7 and Orin Nano** with the V5
   analyzer (compute derate, aspect-ratio skinny, GPU dispatch
   overhead, tier-aware memory). Predicted markers in the existing
   plots were stale -- they used the pre-V5 analyzer and were off by
   30%-2x for many shapes.
2. **`--predictions-only` mode for `visualize_baseline.py`**, so
   hardware without committed V4 baselines (KPU T64, AGX, NX, H100)
   can still be visualized. Shapes come from the sweep validation
   JSON; no measurements required.
3. **Standalone KPU T64 plot**: same 3-panel layout as the i7 and
   Orin Nano plots, predictions-only at fp16.
4. **`compare_hardware.py` -- a 3-way comparison CLI** that overlays
   roofline, latency, and energy predictions for several hardware
   targets on the same shape pool. Each hardware uses its native
   precision (CPU -> fp32, GPU -> fp16, KPU -> fp16; KPU's bf16
   fabric delivers the same TFLOPS as fp16).

## Bonus fixes shipped along the way

* **vector_add was missing from the visualizer** -- the loops only
  walked matmul and linear. Surfaced when the user asked
  "the plots do not show vector_add results: they should all be clearly
  memory bound." Added `vector_add` to the op iteration and a
  square (`s`) marker. Now all three op classes render.
* **Scalar shape parsing** (`_shape_from_csv`) didn't handle
  vector_add's single-int CSV format like `"1024"`. Fixed.
* **`_predict_latency_s` return-type drift**: the harness function now
  returns `(latency_s, binding_tier_name)` per V5-3b binding-tier
  surfacing, but the visualizer still treated the result as a bare
  float. Updated to unpack and pass `use_tier_aware_memory=True` to
  match the V5-3b production default.
* **Sweep classifier launch-bound bug** for memory-bound ops. The
  pre-fix check was:
  ```python
  if flops / peak_flops < launch_overhead * 5:
      return LAUNCH_BOUND
  ```
  This used the optimistic compute-time floor. For vector_add, FLOPS
  is tiny (one add per element), so even at N=16M the compute-time
  estimate is ~12 us, well below the 25 us launch threshold ->
  classified LAUNCH_BOUND despite memory time being 980 us
  (clearly DRAM_BOUND). Fixed by using the roofline floor:
  ```python
  if max(flops/peak_flops, ws/peak_bw) < launch_overhead * 5:
      return LAUNCH_BOUND
  ```
  Re-augmented all sweeps with this corrected classifier; six Jetson
  vector_add shapes flipped from launch_bound to dram_bound.

## Headline observations from the comparison plot

Default targets: `i7_12700k (fp32)`, `jetson_orin_nano_8gb (fp16)`,
`kpu_t64 (fp16)`.

### Roofline panel

Three peak roofs differ dramatically:

| target | peak FLOPS | peak BW | AI breakpoint |
|---|---|---|---|
| i7-12700K (fp32) | 1.44 TFLOPS | 75 GB/s | 19 |
| Jetson Orin Nano (fp16) | 1.33 TFLOPS | 102 GB/s | 13 |
| **KPU T64 (fp16)** | **33.20 TFLOPS** | **64 GB/s** | **519** |

KPU T64 has **25x the fp16 throughput** of Orin Nano at 63% of its
DRAM bandwidth. The AI breakpoint moves from 13 (Orin Nano) to 519
(KPU T64), meaning shapes need much higher arithmetic intensity to
saturate KPU's compute. For typical matmul shapes (OI = 100-500),
KPU is still **DRAM-bound**, not compute-bound.

### Latency panel

For matmul / linear shapes that fit in i7's L3 (25 MB) but exceed
Orin Nano / KPU's L2 (1.25 MB / 16 MB): i7 wins because it has more
shared cache. For shapes much larger than L3, all three become
DRAM-bound and the relative speedup is BW-ratio:
1.6x faster on Orin Nano, 0.85x as fast on KPU (lower BW).

For vector_add: latency tracks `WS / (peak_BW * achievable_fraction)`.
Orin Nano leads (102 GB/s * 0.55), then i7 (75 * achievable), then
KPU (64 * achievable).

### Energy panel

KPU T64 has the **lowest predicted energy per inference for compute-
heavy shapes**. The 25x compute throughput at lower TDP (KPU T64 ~6W
vs Orin Nano 15W) means significantly lower energy * latency product
on shapes with OI > 50. For pure-memory shapes (vector_add), Orin
Nano wins on energy because its DRAM BW is higher per watt.

## How to regenerate

```bash
# Regenerate per-hw plots after analyzer changes
python -m validation.model_v4.cli.visualize_baseline --hw i7_12700k
python -m validation.model_v4.cli.visualize_baseline --hw jetson_orin_nano_8gb

# Generate predictions-only plot for hardware without baselines
python -m validation.model_v4.cli.visualize_baseline --hw kpu_t64 --predictions-only

# 3-way comparison
python -m validation.model_v4.cli.compare_hardware
# or with custom targets:
python -m validation.model_v4.cli.compare_hardware \
    --hw i7_12700k jetson_orin_nano_8gb kpu_t64 h100_sxm5_80gb \
    --dtype-for kpu_t64=bf16
```

## Output paths

* `validation/model_v4/results/plots/i7_12700k_roofline.png`
* `validation/model_v4/results/plots/jetson_orin_nano_8gb_roofline.png`
* `validation/model_v4/results/plots/kpu_t64_roofline.png`
* `validation/model_v4/results/plots/compare_<hw1>_<hw2>_<...>.png`

## Cross-link

* PR #115 -- analysis flagging stale predictions
* PR #116 / #117 / #118 -- the three model fixes whose latency
  precision is reflected in the regenerated plots
* PR #119 -- measurement-priority sweep regime labels (visible in
  the i7 / Orin Nano plot regime coloring)

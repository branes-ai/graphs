# V5 Follow-up: Jetson Compute Efficiency Derate

## Context

The V5 follow-up Jetson Orin Nano calibration analysis (PR #115) flagged
"compute efficiency model refinement" as one of three model gaps blocking
V4 floor improvement. This PR closes that gap for matmul / linear at FP16
on Jetson Orin Nano via a new per-mapper calibration field.

## What was wrong

`RooflineAnalyzer._get_compute_efficiency_scale` is a single 240-line
function that returns a multiplicative scale applied to `peak_flops`:

```python
effective_peak_flops = self.peak_flops * compute_efficiency_scale
compute_time = sg.flops / effective_peak_flops
```

The function was calibrated on **Jetson Orin AGX (50W, FP32, conv2d-heavy
workloads with cuDNN+TF32)**. For large matmul (>5 GFLOPS) it returns
**1.4-1.5** -- meaning the model predicts compute throughput **1.4-1.5x
above peak**. This makes sense on AGX where:

* `peak_flops = peak_ops_per_sec` is the SPEC peak (no efficiency
  baked in)
* cuDNN secretly upgrades FP32 to TF32, achieving ~5x of FP32 spec
  on large convs
* The scale > 1 compensates for the spec-vs-effective gap

But Jetson Orin Nano works differently:

* `peak_ops_per_sec = 8 SMs * 1024 ops/clock * 1.02 GHz = 8.36 TFLOPS`
* `efficiency_factor = 0.85` (cuBLAS Tensor Core matmul fraction)
* `peak_flops = 8.36 * 0.85 = 7.10 TFLOPS` -- already the ACHIEVABLE peak

Multiplying 7.10 TFLOPS by the AGX-tuned 1.48 scale yields a predicted
compute throughput of **10.5 TFLOPS** -- 25% above the theoretical
spec peak, and **2.4x above the cuBLAS-achievable peak**.

## Per-shape data

Probe of `peak_flops`, `compute_efficiency_scale`, and measured
efficiency across the 20 large-matmul shapes from the Jetson Orin
Nano V4 baseline (`jetson_orin_nano_8gb_matmul.csv`):

| shape | flops (G) | cm_scale | pred compute_t (ms) | meas (ms) | meas eff |
|---|---|---|---|---|---|
| (96, 12288, 6144) | 14.50 | 1.477 | 1.382 | 3.11 | 0.656 |
| (192, 6144, 6144) | 14.50 | 1.477 | 1.382 | 2.98 | 0.685 |
| (1536, 3072, 1536) | 14.50 | 1.477 | 1.382 | 3.65 | 0.559 |
| (3072, 192, 12288) | 14.50 | 1.477 | 1.382 | 2.96 | 0.689 |
| (6144, 96, 12288) | 14.50 | 1.477 | 1.382 | 5.72 | 0.357 |
| (6144, 6144, 192) | 14.50 | 1.477 | 1.382 | 2.89 | 0.706 |
| (12288, 96, 6144) | 14.50 | 1.477 | 1.382 | 5.66 | 0.361 |
| (12288, 3072, 192) | 14.50 | 1.477 | 1.382 | 2.65 | 0.770 |
| (64, 8192, 16384) | 17.18 | 1.489 | 1.624 | 5.60 | 0.432 |
| (64, 16384, 8192) | 17.18 | 1.489 | 1.624 | 4.89 | 0.495 |
| (128, 4096, 16384) | 17.18 | 1.489 | 1.624 | 4.46 | 0.542 |
| (128, 8192, 8192) | 17.18 | 1.489 | 1.624 | 3.35 | 0.722 |
| (128, 16384, 4096) | 17.18 | 1.489 | 1.624 | 2.89 | 0.837 |
| (256, 2048, 16384) | 17.18 | 1.489 | 1.624 | 3.33 | 0.726 |
| (256, 4096, 8192) | 17.18 | 1.489 | 1.624 | 4.44 | 0.545 |
| (256, 8192, 4096) | 17.18 | 1.489 | 1.624 | 2.70 | 0.896 |
| (256, 16384, 2048) | 17.18 | 1.489 | 1.624 | 3.78 | 0.640 |
| (2048, 256, 16384) | 17.18 | 1.489 | 1.624 | 3.32 | 0.729 |
| (2048, 2048, 2048) | 17.18 | 1.489 | 1.624 | 4.64 | 0.521 |
| (2048, 16384, 256) | 17.18 | 1.489 | 1.624 | 3.64 | 0.665 |

`meas_eff` = `flops / (meas_ms * 1e-3 * 7.10 TFLOPS)`. Range:
0.36-0.90, median 0.66, mean 0.62.

The model says scale = 1.48 (compute is 48% FASTER than achievable
peak); reality says scale ≈ 0.62 (compute is 62% of achievable peak).
The model is **2.4x too optimistic**.

## What changed

### New field on `HardwareResourceModel`

```python
compute_efficiency_overrides_by_op: Dict[str, Dict[str, float]] = field(
    default_factory=dict
)
# Outer key: precision name ("fp16", "fp32", ...)
# Inner key: op kind ("matmul", "linear", "vector_add")
# Value: scale to use instead of _get_compute_efficiency_scale (range (0, 2.0])
```

### New helper on `RooflineAnalyzer`

```python
def _get_compute_efficiency_override(self, sg) -> Optional[float]:
    """Look up the per-(precision, op_kind) override; return None to fall
    through to the legacy curve."""
```

Called at the top of `_get_compute_efficiency_scale`. When the override
fires, the legacy 240-line curve is bypassed entirely.

### Calibrated values for Jetson Orin Nano FP16

```python
compute_efficiency_overrides_by_op={
    "fp16": {"matmul": 0.70, "linear": 0.94},
}
```

V4-baseline-fit; the choice is constrained by the legacy memory path's
energy-band test floor (`test_jetson_orin_nano_linear_pass_energy_floor`
requires >= 27/46 energy passes).

## V4 floor impact

| metric | path | baseline | this PR | delta |
|---|---|---|---|---|
| jetson matmul lat-pass | tier-aware | 22 | 30 | **+8** |
| jetson matmul lat-pass | legacy | 18 | 26 | **+8** |
| jetson matmul egy-pass | tier-aware | 29 | 29 | 0 |
| jetson matmul egy-pass | legacy | 24 | 24 | 0 |
| jetson linear lat-pass | tier-aware | 18 | 30 | **+12** |
| jetson linear lat-pass | legacy | 18 | 25 | **+7** |
| jetson linear egy-pass | tier-aware | 29 | 30 | +1 |
| jetson linear egy-pass | legacy | 29 | 28 | -1 (above 27 floor) |
| i7 matmul (all) | both | unchanged | unchanged | 0 |
| i7 linear (all) | both | unchanged | unchanged | 0 |
| jetson vector_add | both | unchanged | unchanged | 0 |

## V4 PASS counts: still unchanged for Jetson

`pass_regime AND pass_latency` is the V4 PASS gate. Jetson is bottlenecked
by `pass_regime` because the **sweep file's pre-computed regime
classifications** (in `regimes_validation_*.json`'s `regime_per_hw`) come
from an older model and disagree with the measured regime on most shapes
(only 4/48 matmul shapes and 1/46 linear shapes have correct sweep-side
regimes). The compute model can't lift PASS while the sweep classifier
is the gate.

This is a separate task: regenerate the sweep regime classifications
with the V5 model. Tracked but out of scope for this PR.

## Why the linear scale is 0.94 not 0.85

Pure latency-pass maximization on the tier-aware memory path picks
linear scale ~0.85. But the V4 floor tests run on the LEGACY memory
path (`RunnerConfig.use_tier_aware_memory` defaults to False), and on
that path 0.85 regresses the energy-pass band:

```text
LEGACY memory path, linear scale sweep:
   scale  lat-pass  energy-pass
   0.85   28        24      ← peak lat, energy fails (floor 27)
   0.88   28        25      ← still fails
   0.93   25        27      ← matches floor exactly
   0.94   25        28      ← +1 headroom; ships here
   1.00   22        28
```

Reason: on legacy, predicted memory-time is wider, so compute_time
matters less. Lowering compute scale grows predicted latency more
than it should, which inflates predicted `static_energy = avg_power *
latency` past the actual silicon energy.

When `RunnerConfig.use_tier_aware_memory` is set to the production
default (True), the optimal point shifts -- 0.85-0.88 becomes
Pareto-optimal again with energy-pass holding above 27. The right
follow-up is to align `RunnerConfig` defaults with the V5-3b
production default; then re-tune linear toward 0.85. For now, 0.94
is the safe Pareto value that respects the existing test floors.

## i7 unaffected

i7 mappers don't carry `compute_efficiency_overrides_by_op` (empty
dict default), so the override helper returns None and the legacy
curve fires unchanged. Verified: i7 matmul / linear V4 floors are
byte-identical before and after this PR.

## What this does NOT fix

* **Sweep regime classifier** -- 44/48 Jetson matmul shapes have wrong
  regime predictions baked into the sweep JSON. The compute fix lifts
  predicted regime classification accuracy on the analyzer side, but
  V4 PASS still gates on sweep-side regime.
* **Per-shape compute efficiency** -- a single scalar fits a 0.36-0.90
  measured-efficiency range. Per-shape variance (related to Tensor
  Core utilization patterns: aspect ratio, alignment, K size) is
  still uncaptured. A future per-shape model (analog of the
  aspect-ratio skinny detection in PR #117) could close more of the
  remaining gap.
* **Other Jetson SKUs** -- AGX 64GB and NX 16GB do not yet have
  V4 baselines, so their override values are absent. Set them as
  baselines land.

## Cross-link

* PR #115 -- analysis that flagged compute model derate as Category 3
* PR #116 -- Category 1 (GPU dispatch overhead bridge)
* PR #117 -- Category 2 (aspect-ratio skinny matmul detection)
* `src/graphs/hardware/resource_model.py` -- `compute_efficiency_overrides_by_op`
* `src/graphs/estimation/roofline.py` -- `_get_compute_efficiency_override`
* `docs/calibration/jetson-orin-nano-calibration-analysis.md`

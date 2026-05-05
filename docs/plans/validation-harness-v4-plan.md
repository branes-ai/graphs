# Plan: Model Validation Harness v4 (matmul + linear, layer-driven)

**Status:** Draft for review
**Date:** 2026-05-05
**Attempt number:** 4 (prior attempts archived in `bottom-up-microbenchmark-plan.md` and `model-based-validation-plan.md`)
**Scope of v4 first cut:** matmul and linear only, on CPU and GPU; KPU and TPU consistency-only

## Why a fourth attempt

Three prior validation efforts produced infrastructure but did not produce
defensible per-shape correctness signal for the analytical model. Best-guess
failure modes inferred from the existing code under `validation/`,
`tests/transform/partitioning/test_arithmetic_intensity.py`, and the prior
plan docs:

1. **Self-validation.** Comparing the analyzer's output to a previous
   version of the analyzer with no out-of-band ground truth.
2. **No regime stratification.** Sweeping by size without controlling
   which architectural layer a shape was *supposed* to bottleneck on.
   A passing test then just meant the model was internally consistent
   at one point on the roofline.
3. **No layer-specific failure attribution.** When a number was off
   by 20%, no way to tell whether the ALU peak was wrong, the L2 BW
   assumption drifted, or the kernel-launch overhead was miscalibrated.
4. **Calibration ≡ validation.** Same shapes used to fit constants and
   to "validate," producing 100% pass rates that meant nothing.

This plan addresses all four by making the harness **hardware-layer-centric**
(per the standing memory `feedback_validation_hardware_centric.md`), with
out-of-band ground truth, stratified shape sweeps, per-layer assertions,
and frozen out-of-sample validation shapes.

## Principles (review-and-correct surface)

These five principles are the contract for what the harness does. Everything
downstream (file layout, test code, CI gate) is mechanical given the
principles. **Read these first; if any are wrong the rest is wrong.**

### Principle 1 — Stratified shape sweep

Every sweep shape lands in exactly one known roofline regime on the target
hardware. The harness rejects shapes within ±20% of a regime boundary so
failures point at one architectural layer, not two.

| Regime | Definition | Layer being validated |
|---|---|---|
| `alu_bound` | Working set fits in L1 AND OI > peak FLOPS / peak L1 BW | ALU peak FLOPS |
| `l1_bound` | Working set ∈ (L1, L2] | L1 BW (per-tile scratchpad on KPU) |
| `l2_bound` | Working set ∈ (L2, L3 or on-chip SRAM] | L2 BW; aggregate SRAM BW on KPU/TPU |
| `dram_bound` | Working set > on-chip total | DRAM BW |
| `launch_bound` | Total ops < launch_overhead × peak FLOPS | Kernel-launch overhead constant |

Working set for matmul `(M, K, N, dtype)` = `(M*K + K*N + M*N) × bytes_per_element(dtype)`.
Working set for linear `(B, IN, OUT, dtype)` = `(B*IN + IN*OUT + B*OUT) × bytes_per_element(dtype)`.

### Principle 2 — Ground truth comes from outside this repo

| Hardware | Source of truth | What we capture |
|---|---|---|
| Intel i7-12700K (local) | PyTorch CPU + `perf stat` + Intel RAPL | latency_ms, energy_j |
| Jetson Orin Nano / AGX (available) | PyTorch CUDA + `cudaEvent` + tegrastats | latency_ms, power_w → energy_j |
| H100 / A100 (rentable) | PyTorch CUDA + `cudaEvent` + NVML | latency_ms, energy_j |
| Coral Edge TPU | TFLite delegate + on-board cycle counter | latency_ms only |
| KPU (no silicon) | Vendor cycle-accurate simulator output, JSON checked in | latency_cycles → latency_ms via clock |

**Rule:** if no ground-truth source exists, that hardware is excluded from
per-shape validation, not "validated" against another estimator. KPU and TPU
without simulator output are demoted to consistency-only checks
(monotonicity, dimensional analysis, no per-shape correctness assertion).

### Principle 3 — Per-layer assertions, not per-workload

For each shape, three assertions, all must hold:

```python
assert predicted_regime == measured_regime              # bottleneck classification
assert abs(predicted_latency - measured_latency) / measured_latency < TOL_LATENCY[regime]
assert abs(predicted_energy  - measured_energy)  / measured_energy  < TOL_ENERGY[regime]
```

Tolerances are per-regime, not global. Initial defaults (review and adjust):

| Regime | TOL_LATENCY | TOL_ENERGY | Rationale |
|---|---|---|---|
| `alu_bound` | 0.10 | 0.15 | Peak FLOPS is well-defined; ALU energy varies a bit by op mix |
| `l1_bound` | 0.15 | 0.20 | L1 BW spec is firm but pattern-dependent |
| `l2_bound` | 0.20 | 0.25 | L2 contention with other ops |
| `dram_bound` | 0.25 | 0.30 | Real DRAM BW varies with access pattern, prefetch state |
| `launch_bound` | 0.30 | 0.40 | Kernel launch is high-variance |

### Principle 4 — Calibration vs validation are separate, frozen, committed shape sets

```
validation/model_v4/sweeps/
  matmul_calibration.json   ~30 shapes used to fit efficiency constants
  matmul_validation.json    ~80 shapes the model has never seen during fit
  linear_calibration.json
  linear_validation.json
```

Calibration shapes are touched only during deliberate recalibration (rare).
Validation shapes are frozen. CI runs validation only. **Drift in
validation = real model drift; drift in calibration = re-fit, not a commit.**

### Principle 5 — Drift attribution is the output, not just pass/fail

The per-run report aggregates by `(hardware, op, regime)` and surfaces a
heatmap:

```
                   alu  l1   l2   dram  launch
i7-12700k matmul    OK  OK   FAIL OK    OK
i7-12700k linear    OK  OK   FAIL OK    OK
h100      matmul    OK  OK   OK   OK    OK
h100      linear    OK  FAIL OK   OK    OK
kpu-t64   matmul    OK  OK   OK   OK    -    (consistency-only)
```

Each `FAIL` cell points at exactly one architectural-model assumption that
drifted. Diagnosis is mechanical: read the per-shape table, find the drift
direction, identify the constant in the architectural-energy or roofline
model that needs adjustment.

## Per-layer assertion model

For each `(hardware, op, shape)` triple the harness produces a record:

```python
@dataclass
class ValidationRecord:
    hardware: str
    op: str                    # "matmul" or "linear"
    shape: tuple               # (M, K, N) or (B, IN, OUT)
    dtype: str                 # "fp32", "fp16", "int8", ...
    regime_predicted: str      # one of alu/l1/l2/dram/launch
    regime_measured: str       # inferred from achieved BW or FLOPS utilization
    latency_predicted_ms: float
    latency_measured_ms: float
    energy_predicted_j: float
    energy_measured_j: float
    bottleneck_layer: str      # which layer's spec this shape validates
    pass_regime: bool
    pass_latency: bool
    pass_energy: bool
    tolerance_latency: float   # the band that was applied
    tolerance_energy: float
```

Inference of `regime_measured` from raw measurements:

```
achieved_flops_util  = (flops / measured_latency) / peak_flops
achieved_bw_util     = (working_set_bytes / measured_latency) / peak_bandwidth

if achieved_flops_util > 0.7:                           regime_measured = alu_bound
elif achieved_bw_util > 0.7 (against L1 peak):          regime_measured = l1_bound
elif achieved_bw_util > 0.7 (against L2 peak):          regime_measured = l2_bound
elif achieved_bw_util > 0.7 (against DRAM peak):        regime_measured = dram_bound
elif measured_latency / launch_overhead < 5:            regime_measured = launch_bound
else:                                                   regime_measured = ambiguous
```

`ambiguous` records are excluded from pass/fail tallies and surfaced in a
separate "rejected by classifier" list — they indicate either a poorly
chosen sweep shape or a missing peak number for that hardware layer.

## Repo layout

```
validation/model_v4/
  workloads/
    __init__.py
    matmul.py             # build_matmul(M, K, N, dtype) -> nn.Module + input
    linear.py             # build_linear(B, IN, OUT, dtype) -> nn.Module + input
  sweeps/
    __init__.py
    classify.py           # given (shape, op, hardware) -> regime bucket
    matmul_calibration.json
    matmul_validation.json
    linear_calibration.json
    linear_validation.json
  ground_truth/
    __init__.py
    base.py               # protocol Measurer.measure(model, input) -> Measurement
    pytorch_cuda.py       # cudaEvent + NVML
    pytorch_cpu.py        # perf_event + RAPL
    simulator_kpu.py      # reads cycle-accurate sim CSV
    cache.py              # one canonical (hw, shape, dtype) -> measurement, on disk
  harness/
    __init__.py
    runner.py             # orchestrates predict + measure + assert per shape
    assertions.py         # tolerance bands per regime; ValidationRecord dataclass
    report.py             # heatmap + per-shape table; markdown + JSON output
  results/
    baselines/            # frozen reference: <hw>_<op>.csv per target (committed)
    runs/                 # one dir per run, gitignored
  cli/
    validate_matmul.py    # python -m validation.model_v4.cli.validate_matmul --hw h100
    validate_linear.py
    capture_ground_truth.py
```

## End-to-end flow (matmul example)

For a single matmul shape `(M=4096, K=4096, N=4096, dtype=fp16)` on H100:

1. **Bucket assignment** (`sweeps/classify.py`):
   - working_set = `(M*K + K*N + M*N) * 2 = 96 MB`
   - H100 L2 = 80 MB → working_set > L2 → bucket = `dram_bound`
   - flops = `2*M*K*N = 137 GFLOPs`
   - confirm: `flops / working_set = 1.4 KFLOPs/byte` < `H100 peak FLOPS / peak DRAM BW` ≈ `989 TFLOPS / 3 TB/s` = `330 FLOPS/byte` → memory-bound side, consistent with bucket.

2. **Predict** (`harness/runner.py`):
   - Build `nn.Module(matmul)` from `workloads/matmul.py`
   - Trace via `frontend.trace_and_partition(precision=fp16)`
   - Call `UnifiedAnalyzer.analyze_model_with_custom_hardware(...)`
   - Capture `predicted_latency_ms`, `predicted_bottleneck`, `predicted_energy_j`

3. **Measure** (`ground_truth/pytorch_cuda.py`):
   - Same `nn.Module` runs on real H100, 100 trials, median wall time via `cudaEvent`
   - NVML samples power; integrate over the run for energy
   - Capture `measured_latency_ms`, `measured_energy_j`
   - Infer `regime_measured` from achieved BW utilization

4. **Assert** (`harness/assertions.py`):
   - `predicted_bottleneck == measured_regime` → DRAM-bound
   - `|pred - meas| / meas < TOL_LATENCY[dram_bound] = 0.25`
   - `|pred - meas| / meas < TOL_ENERGY[dram_bound] = 0.30`

5. **Record** (`harness/report.py`):
   - Append a `ValidationRecord` to `results/runs/<timestamp>/h100_matmul.csv`
   - Update rolling baseline `results/baselines/h100_matmul.csv` (separate explicit step, not on every run)

## How developers use it

**Day-to-day during model changes:**

```bash
# Edit src/graphs/estimation/roofline.py, then:
python -m validation.model_v4.cli.validate_matmul --hw i7-12700k --regime all
# - reads cached ground truth from results/baselines/i7-12700k_matmul.csv
# - 80 shapes, ~5 seconds (no real measurement, just predict + compare)
# - prints heatmap; FAIL cells link to per-shape diagnostic
```

The dev loop is **prediction-only**: the framework reuses cached ground
truth so iteration is seconds, not minutes. Ground truth is captured once
per hardware (or whenever explicitly recalibrated).

**Capturing fresh ground truth (rare):**

```bash
python -m validation.model_v4.cli.capture_ground_truth --hw h100 --op matmul \
  --sweep validation
# Runs the 80 validation shapes on real H100, writes
# results/baselines/h100_matmul.csv. Diff is the audit trail.
```

**CI gate:**

CI runs prediction-only against checked-in baselines on every PR that
touches `estimation/`, `hardware/`, or `transform/`. A regression shows up
as a `FAIL` cell — the PR cannot merge until the cell is fixed or the
baseline is intentionally updated (separate commit, separate review).

**Adding a new hardware target:**

1. Add `ground_truth/<hw>.py` adapter (10–30 lines).
2. `capture_ground_truth --hw <hw> --op matmul --sweep calibration` to fit constants.
3. `capture_ground_truth --hw <hw> --op matmul --sweep validation` and commit baseline.
4. From then on, CI-gated.

## Open question — KPU and TPU ground truth

We have no KPU silicon. Coral Edge TPU is limited to int8 and the validation
surface is small. Three options for KPU:

- **(a)** Cycle-accurate simulator output, checked in as JSON, treated as ground
  truth. Cheap if a simulator already exists somewhere in the org.
- **(b)** Vendor-published microbenchmarks. TPU-v4 has these; KPU does not yet.
- **(c)** Mark KPU validation as **consistency-only**: no per-shape ground truth,
  only invariants like "compute_time monotonically decreases as ops grow at
  fixed bandwidth," "memory_time scales linearly with bytes_transferred,"
  "weight-stationary regime activates when weights ≤ 0.8 × on_chip."

**Recommended default for v4:** (c) for KPU until silicon or simulator output
is available. Be explicit in the report that KPU validation is weaker than
CPU/GPU validation. Pretending otherwise is what likely sank attempt 3.

## Phasing

| Phase | Scope | Exit criterion |
|---|---|---|
| **V4-0** | This plan + principles approved | User signs off on principles & assertion model |
| **V4-1** | `workloads/matmul.py`, `sweeps/classify.py`, regime bucketing tested on i7-12700K and H100 spec sheets | `python -m validation.model_v4.sweeps.classify` correctly buckets a curated handful of shapes |
| **V4-2** | `workloads/linear.py`, calibration + validation JSON for matmul + linear on i7-12700K | Two committed JSON files per op; ~30 calibration + ~80 validation shapes each |
| **V4-3** | `harness/runner.py`, `harness/assertions.py`, `harness/report.py` end-to-end on i7-12700K, ground truth from PyTorch CPU + RAPL | First `validate_matmul --hw i7-12700k` heatmap printed; baseline checked in |
| **V4-4** | GPU ground-truth path (`ground_truth/pytorch_cuda.py`) on Jetson Orin or rented H100 | Baseline checked in for at least one GPU |
| **V4-5** | KPU consistency-only checks | KPU monotonicity invariants in CI |
| **V4-6** | CI gate wired to PRs touching `estimation/`, `hardware/`, `transform/` | First PR blocked by a real drift |

## Definition of done (v4)

- Two ops (matmul, linear) at three precisions (fp32, fp16, int8) covered on
  at least i7-12700K and one GPU (Jetson Orin or H100).
- Calibration and validation shape sets are frozen JSON committed to the repo.
- CI runs prediction-only validation on every PR; fails on per-shape regime
  mismatch or out-of-band latency/energy.
- A drift-attribution heatmap is the standard output of any failure.
- KPU has at least consistency-only checks running.

## Cross-references

- `docs/plans/model-based-validation-plan.md` — prior attempt; this plan
  builds on its architecture-coverage matrix and supersedes its phases M1-M5.
- `docs/plans/bottom-up-microbenchmark-plan.md` — empirical-microbenchmark
  approach the prior plan also superseded.
- `~/.claude/projects/-home-stillwater-dev-branes-clones-graphs/memory/feedback_validation_hardware_centric.md` —
  standing instruction that drives Principle 1.

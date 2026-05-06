# V4 Model Validation Harness

The V4 harness is the analytical-model validation layer described in
[`docs/plans/validation-harness-v4-plan.md`](../../docs/plans/validation-harness-v4-plan.md).
Five principles drive its design (see plan for the full rationale):

1. **Stratified sweep** — every shape is classified into a bottleneck regime
   *before* measurement, so per-shape failures attribute to one architectural-
   model assumption.
2. **Out-of-band ground truth** — RAPL on CPU, NVML on desktop GPU, INA3221
   on Jetson, simulator output (eventually) on KPU. No self-validation against
   the analyzer's own previous output.
3. **Per-layer assertions** — three independent checks per shape (regime,
   latency, energy) so a failure points at one specific drift.
4. **Calibration / validation separation** — disjoint shape sets per op.
5. **Drift attribution** — heatmap output naming the offending hardware ×
   regime cell.

## Layout

```
validation/model_v4/
├── workloads/           # PyTorch nn.Module factories per op (matmul, linear)
├── sweeps/              # Frozen JSON shape sets + classifier + augmenter
│   ├── classify.py      # Pure-function regime classifier
│   ├── _generate.py     # Sampling generator (run once; output committed)
│   └── _augment.py      # Additive re-classification when adding hardware
├── ground_truth/        # Per-target measurer backends (V4-3, V4-4)
│   ├── pytorch_cpu.py     # wall-clock + Intel RAPL
│   ├── pytorch_cuda.py    # cudaEvent + NVML (H100, A100, ...)
│   └── pytorch_jetson.py  # cudaEvent + INA3221 (Orin, Thor)
├── invariants/          # Consistency-only checks (V4-5; KPU/TPU)
│   └── kpu.py             # roofline self-consistency, monotonicity, etc.
├── harness/             # predict + measure + assert orchestration
│   ├── runner.py
│   ├── assertions.py
│   └── report.py
├── cli/                 # capture_ground_truth.py, validate.py
└── results/baselines/   # Committed CSVs per (hardware, op)
```

## The CI gate (V4-6)

`.github/workflows/v4-validation.yml` runs the V4 contract on every PR
that touches:

- `src/graphs/{estimation,hardware,transform,core}/**`
- `validation/model_v4/**`
- `tests/validation_model_v4/**`
- The four V4-anchored CPU regression test files
  (`tests/analysis/test_roofline_cpu_*.py`, `test_energy_cpu_active_power.py`)
- `validation/model_v4/results/baselines/**`

The gate fails the PR check named **"V4 Model Validation"** if:

- any `tests/validation_model_v4/` test fails (regime classification math,
  V4 floor pass-rate regression, KPU invariant violation, sweep schema
  drift, ground-truth contract drift);
- any of the V4-calibrated CPU floor tests regress
  ([#67](https://github.com/branes-ai/graphs/issues/67),
  [#69](https://github.com/branes-ai/graphs/issues/69),
  [#71](https://github.com/branes-ai/graphs/issues/71),
  [#74](https://github.com/branes-ai/graphs/issues/74)).

To make the check **required for merge**, set it in
*Repo Settings → Branches → Branch protection rules → main → Required
status checks*.

The meta-test
[`tests/validation_model_v4/test_v4_ci_gate_workflow.py`](../../tests/validation_model_v4/test_v4_ci_gate_workflow.py)
asserts the workflow file is well-formed and continues to gate the
right paths and tests — so a future contributor narrowing the scope
trips a unit-test failure with a clear message.

## Adding a new hardware target

See
[`docs/plans/v4-capture-on-target.md`](../../docs/plans/v4-capture-on-target.md)
and the runbooks in [`bin/`](../../bin/) (`v4-capture-jetson-orin-nano.sh`,
`v4-capture-h100.sh`).

In short:

1. Add a `_Target(...)` entry to `KNOWN_TARGETS` in `sweeps/_augment.py`.
2. Run `python -m validation.model_v4.sweeps._augment --hw <new_key>` to
   add the new hardware's regimes to the existing sweep entries
   *additively* (existing baselines stay valid).
3. Add the same key to `_MEASURER_FACTORY` in `cli/capture_ground_truth.py`.
4. Add the same key to `SWEEP_HW_TO_MAPPER` in `harness/runner.py`.
5. Add the key to the `hw` fixture in `tests/validation_model_v4/test_sweeps.py`.
6. Run `bin/v4-capture-<hw>.sh` on the target box to capture the baseline CSV.
7. Open a PR with the CSV and add a `test_<hw>_pass_*_floor` case to
   `tests/validation_model_v4/test_v4_against_baseline.py`.

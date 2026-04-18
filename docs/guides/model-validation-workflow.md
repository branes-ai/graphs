# Model Validation Workflow

This guide describes how to validate the energy, memory, and
performance attributes of the micro-architecture models in the
`graphs` repository. It covers running the validation tests,
interpreting results, adding new models, and maintaining the
regression baseline.

---

## 1. Overview

The validation suite lives under `validation/model_validation/` and
tests the modeling infrastructure -- not physical hardware. It
exercises every mapper's `_calculate_energy`, `_calculate_latency`,
and `_calculate_energy_with_architecture` methods against five
consistency categories:

| Category | What it catches |
|----------|----------------|
| **Layer decomposition** | Energy components don't sum correctly; arch overhead missing |
| **Monotonicity** | More work doesn't increase energy; negative overhead where additive expected |
| **Precision scaling** | INT8 more expensive than FP32 (coefficient sign error) |
| **Coefficient sensitivity** | Dead coefficient (0% response); coupled coefficient (50%+ response) |
| **Cross-architecture ranking** | CPU modeled more efficient than TPU; edge faster than datacenter |

Plus two infrastructure tests:

- **Regression snapshots**: detects silent coefficient drift across PRs
- **Provenance audit**: verifies every mapper has essential fields populated

---

## 2. Running the Full Validation Suite

### Quick check (< 1 second)

```bash
python -m pytest validation/model_validation/ -q
```

This runs all 125+ model validation tests across 7 architecture
classes, including consistency, sensitivity, regression snapshots,
and provenance audit.

### Verbose with per-test detail

```bash
python -m pytest validation/model_validation/ -v
```

### Specific architecture class

```bash
# CPU only
python -m pytest validation/model_validation/test_stored_program_consistency.py -v

# GPU only
python -m pytest validation/model_validation/test_data_parallel_consistency.py -v

# TPU only
python -m pytest validation/model_validation/test_systolic_array_consistency.py -v

# KPU only
python -m pytest validation/model_validation/test_domain_flow_consistency.py -v

# DSP + DPU + CGRA + Hailo
python -m pytest validation/model_validation/test_accelerator_consistency.py -v

# Cross-architecture ranking
python -m pytest validation/model_validation/test_cross_architecture_ranking.py -v

# Coefficient sensitivity
python -m pytest validation/model_validation/test_coefficient_sensitivity.py -v

# Regression snapshots
python -m pytest validation/model_validation/test_regression_snapshot.py -v
```

### CI integration

The CI "Composition Check" job automatically runs:

```bash
python validation/composition/test_layer_composition.py
```

The model validation tests run as part of the broader test suite and
are discovered by pytest automatically.

---

## 3. Interpreting Results

### All tests pass

The model is internally consistent. No coefficient produces a
physically implausible result, and outputs match the regression
snapshot within 1%.

### Precision scaling failure

```text
FAILED test_stored_program_consistency.py::TestPrecisionScaling::test_int8_cheaper_than_fp32
  Intel-i7-12700K: INT8 compute energy 0.015 >= FP32 0.012
```

This means the energy_scaling table for INT8 is wrong -- INT8 should
have a scaling factor lower than FP32's 1.0. Fix by checking
`resource_model.energy_scaling[Precision.INT8]`.

### Coefficient sensitivity failure

```text
FAILED test_coefficient_sensitivity.py::TestEnergyPerFlopSensitivity::test_cpu
  Intel-i7-12700K: energy_per_flop +10% -> 0.0% compute energy change
```

The coefficient is dead -- probably being overridden by a tile energy
model or thermal operating point. Check whether the mapper uses
`_calculate_energy` from the base class or overrides it.

### Regression snapshot failure

```text
FAILED test_regression_snapshot.py::TestRegressionSnapshot::test_energy_matches_snapshot
  3 energy regression(s) detected (tolerance=1%):
    H100-SXM5-80GB/medium_matmul_fp32/compute_energy: expected=2.15e-03, got=2.37e-03, delta=10.2%
```

A coefficient changed since the last snapshot. If intentional,
regenerate:

```bash
python -c "
from validation.model_validation.test_regression_snapshot import (
    _generate_snapshot, _save_snapshot
)
_save_snapshot(_generate_snapshot())
"
```

Then commit the updated `snapshot.json` alongside your coefficient
change.

### Cross-architecture ranking failure

```text
FAILED test_cross_architecture_ranking.py::TestLatencyRanking::test_h100_faster_than_orin_agx_gpu
  H100 should be faster than Orin AGX
```

The datacenter GPU model is slower than the edge GPU for the same
workload. This usually indicates a thermal operating point or
efficiency factor issue. Check `_calculate_latency`'s thermal path
for the affected SKU.

---

## 4. Adding a New Hardware Mapper

When you add a new mapper (e.g., a new GPU or accelerator), follow
this checklist to ensure it passes the validation suite:

### Step 1: Create the resource model

Add a factory function in `hardware/models/<category>/<device>.py`
with cited datasheet parameters.

### Step 2: Register the mapper

Add it to `hardware/mappers/__init__.py` in the appropriate category
dict.

### Step 3: Run the validation suite

```bash
python -m pytest validation/model_validation/ -v
```

Your new mapper will automatically be picked up by the conftest
fixtures (they discover mappers by hardware type). Check that:

- Baseline energy is positive
- INT8 < FP32 compute energy
- Energy scaling factors are in (0, 4.0]
- Latency is non-negative
- Peak bandwidth is non-zero

### Step 4: Update the regression snapshot

```bash
python -c "
from validation.model_validation.test_regression_snapshot import (
    _generate_snapshot, _save_snapshot
)
_save_snapshot(_generate_snapshot())
"
```

Add your SKU to `REFERENCE_SKUS` in `test_regression_snapshot.py` if
it should be tracked in the snapshot.

### Step 5: Add cross-architecture checks (if applicable)

If your mapper represents a new performance tier (e.g., faster than
H100), add an ordering assertion in
`test_cross_architecture_ranking.py`.

---

## 5. Modifying Coefficients

When you change an energy, latency, or memory coefficient:

### Step 1: Run the sensitivity test for the affected class

```bash
python -m pytest validation/model_validation/test_coefficient_sensitivity.py -v -k "cpu"
```

### Step 2: Run the regression snapshot

```bash
python -m pytest validation/model_validation/test_regression_snapshot.py -v
```

If it fails, review the delta. Is it expected? If yes:

### Step 3: Regenerate the snapshot

```bash
python -c "
from validation.model_validation.test_regression_snapshot import (
    _generate_snapshot, _save_snapshot
)
_save_snapshot(_generate_snapshot())
"
git add validation/model_validation/snapshot.json
```

### Step 4: Run the full suite to check for cascading effects

```bash
python -m pytest validation/model_validation/ -v
```

---

## 6. Understanding the Test Architecture

### File layout

```text
validation/model_validation/
    __init__.py
    conftest.py                             # Shared fixtures
    snapshot.json                            # Regression baseline
    test_stored_program_consistency.py       # CPU (10 mappers, 14 tests)
    test_data_parallel_consistency.py        # GPU (10 mappers, 16 tests)
    test_systolic_array_consistency.py       # TPU (6 mappers, 10 tests)
    test_domain_flow_consistency.py          # KPU (3 mappers, 12 tests)
    test_accelerator_consistency.py          # DSP/DPU/CGRA/Hailo (15 mappers, 18 tests)
    test_cross_architecture_ranking.py       # Cross-class (12 tests)
    test_coefficient_sensitivity.py          # Sensitivity (22 tests)
    test_regression_snapshot.py              # Snapshot (4 tests)
    test_provenance_audit.py                 # Provenance (17 tests)
```

### Workload fixtures (conftest.py)

| Fixture | Shape | Purpose |
|---------|-------|---------|
| `tiny_matmul` | 64x64x64 | Tests small-op overhead |
| `medium_matmul` | 1024x1024x1024 | Typical DNN layer |
| `large_matmul` | 4096x4096x4096 | Compute-bound stress test |
| `depthwise_conv` | 3x3 56x56x64 | Memory-bound (low AI) |
| `elementwise_relu` | 1M elements | Zero compute, pure bandwidth |

### Mapper fixtures (conftest.py)

| Fixture | Hardware type | Count |
|---------|--------------|-------|
| `cpu_mappers` | CPU | 10 |
| `gpu_mappers` | GPU | 10 |
| `tpu_mappers` | TPU | 6 |
| `kpu_mappers` | KPU (Stillwater) | 3 |
| `hailo_mappers` | KPU (Hailo) | 2 |
| `dsp_mappers` | DSP | 10 |
| `dpu_cgra_mappers` | DPU + CGRA | 2 |

The DFM (Data Flow Machine) reference architecture is excluded from
fixtures because it intentionally models energy savings (negative
overheads) relative to the stored-program baseline.

---

## 7. Known Limitations

1. **Datacenter GPU thermal specs**: H100, A100, V100 have empty
   `performance_specs` in their `ThermalOperatingPoint`. This causes
   `_calculate_latency` to use a 0.01x penalty path. Cross-arch
   latency tests work around this by computing from `peak_ops`
   directly.

2. **TPU v1 FP32 overreporting**: TPU v1 is INT8-only hardware; its
   FP32/BF16 peak falls back to the INT8 profile, making it appear
   faster than TPU v4 at FP32. Cross-generation tests use BF16
   (native TPU precision).

3. **Hailo-10H vs Hailo-8**: The model data shows Hailo-8 (26 TOPS)
   faster than Hailo-10H (20 TOPS). No ordering assertion is made
   pending data review.

4. **Tile energy model bypass**: Coral Edge TPU and KPU mappers with
   `TPUTileEnergyModel` / `KPUTileEnergyModel` override
   `_calculate_energy` with their own coefficients. Perturbing
   `energy_per_flop_fp32` on the resource model has no effect.
   Sensitivity tests skip these mappers.

---

## 8. Quick Reference

```bash
# Full validation (< 1s)
python -m pytest validation/model_validation/ -q

# After changing a coefficient
python -m pytest validation/model_validation/ -v
# If snapshot fails:
python -c "from validation.model_validation.test_regression_snapshot import _generate_snapshot, _save_snapshot; _save_snapshot(_generate_snapshot())"
git add validation/model_validation/snapshot.json

# After adding a mapper
python -m pytest validation/model_validation/ -v
# Update snapshot if needed

# After modifying architectural_energy.py
python -m pytest validation/model_validation/test_coefficient_sensitivity.py -v
python -m pytest validation/model_validation/test_cross_architecture_ranking.py -v
```

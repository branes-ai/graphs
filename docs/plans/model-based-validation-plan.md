# Plan: Model-Based Validation of the Micro-Architecture Modeling Infrastructure

**Status:** Draft
**Date:** 2026-04-17
**Supersedes:** Phases 3-6 of `docs/plans/bottom-up-microbenchmark-plan.md`
**Companion:** `docs/assessments/empirical-vs-model-based-validation.md`

## Motivation

The original bottom-up plan called for empirical execution benchmarks
on real hardware at each of six physical layers. Phases 0-2 revealed
that:

1. PyTorch-level benchmarks cannot isolate micro-architectural ALU
   behavior (dispatch overhead is ~1000x larger than the signal).
2. True empirical Layer 1-2 characterization requires C/C++
   microbenchmarks -- a significant effort orthogonal to validating
   the modeling code.
3. The actual goal is to validate the **modeling infrastructure**
   (CPU/GPU/TPU/KPU energy and latency models), not to characterize
   real silicon.
4. We have 45 hardware mappers across 8 architecture classes. Only
   2-3 SKUs have physical hardware available. Model-based validation
   covers all 45 without hardware access.

## What Model-Based Validation Means

Each architecture energy model
(`StoredProgramEnergyModel`, `DataParallelEnergyModel`,
`SystolicArrayEnergyModel`, `DomainFlowEnergyModel`, etc.)
makes layered predictions:

```
Layer 1: ALU energy = ops x energy_per_op(precision)
Layer 2: Register/SIMD = Layer 1 + instruction_fetch + register_rw + simd_overhead
Layer 3: Tile/scratchpad = Layer 2 + L1/L2 cache or scratchpad energy
Layer 4: On-chip = Layer 3 + L3/LLC/NoC energy
Layer 5: DRAM = Layer 4 + off-chip memory energy
Layer 6: Cluster = Layer 5 + inter-chip communication
```

A model-based test extracts each layer's contribution independently,
composes them, and verifies:

1. **Internal consistency:** Layer 1 + Layer 2 + Layer 3 + ... = the
   model's own total. If not, the model has an accounting bug.
2. **Cross-layer monotonicity:** Adding a layer should only increase
   energy/latency. A negative contribution signals a coefficient
   sign error.
3. **Cross-architecture plausibility:** For the same workload, the
   models should rank architectures consistently with known physics:
   TPU < KPU < GPU < CPU for matrix energy at large batch.
4. **Coefficient sensitivity:** Perturbing one coefficient by +/-10%
   should produce a proportional change in the affected layer's
   output, not a disproportionate swing (which indicates coupling).
5. **Composition accuracy:** The layered prediction should match the
   model's end-to-end `_calculate_energy` / `_calculate_latency`
   within numerical tolerance (< 0.1%).

## Architecture Coverage Matrix

| Architecture class | Energy model | Tile model | Mapper count | Representative SKUs |
|--------------------|-------------|------------|-------------|---------------------|
| STORED_PROGRAM | `StoredProgramEnergyModel` | -- | 11 | i7-12700K, Xeon 8490H, EPYC 9654, AmpereOne, Orin CPU |
| DATA_PARALLEL | `DataParallelEnergyModel` | -- | 10 | H100, A100, V100, B100, Jetson Orin AGX/NX/Nano |
| SYSTOLIC_ARRAY | `SystolicArrayEnergyModel` | `TPUTileEnergyModel` | 6 | TPU v1/v3/v4/v5p, Coral, TPU Edge Pro |
| DOMAIN_FLOW | `DomainFlowEnergyModel` | `KPUTileEnergyModel` | 3 | KPU T64/T256/T768 |
| SPATIAL_PARTITION | `SpatialPartitionEnergyModel` | -- | 2 | Hailo-8, Hailo-10H |
| ADAPTIVE_DATAPATH | `AdaptiveDatapathEnergyModel` | -- | 2 | Xilinx Vitis AI DPU, Stanford Plasticine CGRA |
| DATA_FLOW_MACHINE | `DataFlowMachineEnergyModel` | -- | 1 | DFM (reference) |

**Total: 7 architecture classes, 10 energy/tile models, 45 mappers.**

## Test Structure

### Per-Architecture-Class Test Module

One test module per architecture class under
`validation/model_validation/`:

```
validation/model_validation/
    __init__.py
    test_stored_program_consistency.py      # CPU / DSP
    test_data_parallel_consistency.py       # GPU
    test_systolic_array_consistency.py      # TPU
    test_domain_flow_consistency.py         # KPU
    test_spatial_partition_consistency.py   # Hailo
    test_adaptive_datapath_consistency.py   # FPGA / CGRA / DPU
    test_cross_architecture_ranking.py     # Cross-class comparisons
    conftest.py                            # Shared fixtures (workloads, mappers)
```

### Five Check Categories Per Architecture

Each test module exercises these five categories:

**1. Layer decomposition (internal consistency)**

```python
def test_energy_layers_sum_to_total(mapper, workload):
    """
    Decompose energy into layers (ALU, register, cache, DRAM, etc.)
    via the architectural energy model, then verify they sum to the
    model's total within 0.1%.
    """
    total = mapper._calculate_energy(ops, bytes, precision)
    arch = mapper._calculate_energy_with_architecture(ops, bytes, precision, ctx)
    layered_sum = arch.compute_overhead + arch.data_movement_overhead + arch.control_overhead + base_compute + base_memory
    assert abs(layered_sum - total_with_arch) / total_with_arch < 0.001
```

**2. Monotonicity**

```python
def test_energy_increases_with_layers(mapper, workload):
    """Adding architectural overhead should not decrease energy."""
    base_compute, base_memory = mapper._calculate_energy(ops, bytes, prec)
    with_arch_compute, with_arch_memory, breakdown = mapper._calculate_energy_with_architecture(ops, bytes, prec)
    assert with_arch_compute >= base_compute
    assert with_arch_memory >= base_memory
```

**3. Precision scaling**

```python
def test_lower_precision_uses_less_energy(mapper, workload):
    """INT8 should use less energy than FP32 for the same op count."""
    e_fp32 = mapper._calculate_energy(ops, bytes, Precision.FP32)
    e_int8 = mapper._calculate_energy(ops, bytes, Precision.INT8)
    assert sum(e_int8) < sum(e_fp32)
```

**4. Coefficient perturbation (sensitivity)**

```python
def test_alu_energy_scales_with_coefficient(energy_model, workload):
    """10% increase in energy_per_flop should produce ~10% increase in
    compute energy, not a 50% swing."""
    baseline = energy_model.compute_architectural_energy(ops, bytes, ce, de, ctx)
    # Perturb
    original = energy_model.alu_energy_per_op
    energy_model.alu_energy_per_op *= 1.10
    perturbed = energy_model.compute_architectural_energy(ops, bytes, ce, de, ctx)
    energy_model.alu_energy_per_op = original
    # Check proportionality
    delta = abs(perturbed.compute_overhead - baseline.compute_overhead)
    expected_delta = baseline.compute_overhead * 0.10
    assert abs(delta - expected_delta) / expected_delta < 0.50  # within 50% of expected
```

**5. Cross-mapper consistency (within same architecture class)**

```python
def test_h100_faster_than_v100(workload):
    """H100 should produce lower latency than V100 for same workload."""
    h100 = get_mapper_by_name('H100-SXM5-80GB')
    v100 = get_mapper_by_name('V100-SXM3-32GB')
    lat_h100 = h100.map_subgraph(workload, ...).estimated_latency
    lat_v100 = v100.map_subgraph(workload, ...).estimated_latency
    assert lat_h100 < lat_v100
```

### Cross-Architecture Ranking Test

`test_cross_architecture_ranking.py` verifies the taxonomy ordering
for a reference workload (e.g., 1024x1024 GEMM at INT8):

```python
def test_energy_ranking_for_large_matmul():
    """
    For a large INT8 matmul, energy should follow:
    TPU < KPU < GPU < CPU
    (within architecture-class representatives at comparable TDP)
    """
    ...
```

### Workload Fixtures

`conftest.py` provides standardized `SubgraphDescriptor` fixtures:

- **tiny_matmul**: M=N=K=64, 524K FLOPs (tests small-op overhead)
- **medium_matmul**: M=N=K=1024, 2.1G FLOPs (typical layer)
- **large_matmul**: M=N=K=4096, 137G FLOPs (compute-bound)
- **depthwise_conv**: 3x3 depthwise, 56x56x64 (memory-bound)
- **elementwise**: 1M-element ReLU (zero-compute, pure bandwidth)

## Integration with Existing Infrastructure

### Reuse from Phases 0-2

- **`LayerTag`** — model-based checks use `LayerTag.COMPOSITE` since
  they test the full model, not a single physical layer.
- **`field_provenance`** — model-based tests that pass write
  `THEORETICAL` (confirmed-consistent) provenance; failing tests
  flag the field as needing investigation.
- **Composition registry** — model-based checks register via
  `register_layer_check()` and run in the CI composition gate.
- **`graphs.calibration.fitters`** — model-based "fitters" verify
  analytical coefficients rather than fitting from measurements.
  Same `apply_to_model` pattern.

### CI Integration

All model-based tests run in the existing "Composition Check" CI job
(added in Phase 0.5). No hardware needed. Tests that touch
architectural energy models are automatically gated.

## Phasing

### Phase M1: Core consistency tests (1-2 weeks)

Stand up `validation/model_validation/` with:
- `conftest.py` (workload fixtures, mapper fixtures)
- `test_stored_program_consistency.py` (CPU: 11 mappers)
- `test_data_parallel_consistency.py` (GPU: 10 mappers)

Five check categories for each. This alone covers 21 of 45 mappers
and the two most-used architecture classes.

### Phase M2: Accelerator consistency tests (1 week)

- `test_systolic_array_consistency.py` (TPU: 6 mappers)
- `test_domain_flow_consistency.py` (KPU: 3 mappers)
- `test_spatial_partition_consistency.py` (Hailo: 2 mappers)
- `test_adaptive_datapath_consistency.py` (DPU/CGRA: 2 mappers)

Covers remaining 14 mappers + the DFM reference.

### Phase M3: Cross-architecture ranking (1 week)

- `test_cross_architecture_ranking.py`
- Taxonomy ordering checks for energy and latency
- TDP-normalized comparisons (energy per inference per watt)
- Known-physics invariants (e.g., systolic array should beat stored
  program for large regular matmuls)

### Phase M4: Sensitivity and regression (1 week)

- Coefficient perturbation tests (10% bump -> proportional response)
- Snapshot-based regression: record model outputs for a reference
  workload set, fail if any output changes by > 1% without a
  corresponding coefficient update
- Provenance audit: every mapper's field_provenance map should have
  entries for at least {peak_bandwidth, energy_per_flop_fp32,
  energy_per_byte}

### Phase M5 (optional): Empirical overlay

When C/C++ microbenchmarks or hardware rental becomes available:
- Run empirical benchmarks on i7-12700K, Orin Nano
- Compare against the model-based predictions
- Upgrade provenance from THEORETICAL to CALIBRATED for fields
  where measurement and model agree within tolerance
- File issues for fields where they disagree

## Effort Estimate

- Phase M1: 1-2 weeks (CPU + GPU, most mappers)
- Phase M2: 1 week (accelerators, fewer mappers but more exotic)
- Phase M3: 1 week (cross-architecture, mostly fixture wiring)
- Phase M4: 1 week (sensitivity, snapshot infrastructure)
- **Total: 4-5 weeks** (vs 13 weeks for the original empirical plan)

The effort is lower because:
1. No hardware access needed (runs anywhere, including CI)
2. No C/C++ microbenchmark development
3. No power-measurement plumbing per SKU
4. Test patterns are reusable across architecture classes (same 5
   check categories, different mapper fixtures)

## Definition of Done

1. Every architecture class has a consistency test module exercising
   all 5 check categories.
2. All 45 mappers are covered by at least one test.
3. `test_cross_architecture_ranking.py` verifies the taxonomy
   ordering for at least 3 workload types.
4. CI composition gate runs all model-based tests and fails on
   regression.
5. No mapper produces negative energy, latency, or utilization.
6. `field_provenance` is populated for every mapper's core fields
   after the test suite runs.

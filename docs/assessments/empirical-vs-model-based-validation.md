# Assessment: Empirical vs Model-Based Validation

**Date:** 2026-04-17
**Context:** Phases 0-2 of the bottom-up benchmark plan (M4.5) have
been implemented as empirical execution benchmarks. This assessment
captures what we learned and motivates the pivot to model-based
validation for the remaining phases.

---

## What We Built (Phases 0-2 Empirical)

| Phase | What it does | Status |
|-------|-------------|--------|
| 0 | LayerTag, field_provenance, fitters subpackage, composition scaffold | Merged (PR #1) |
| 0.5 | PowerMeter (RAPL, tegrastats, NVML, NoOp), CI composition gate | Merged (PR #13) |
| 1 | FMA-rate benchmark, precision sweep, Layer1ALUFitter, Layer 1 -> GEMM composition check | Merged (PR #14) |
| 2 | SIMD width sweep, register pressure, Layer2RegisterFitter, Layer 1+2 -> GEMM composition check, CLI runner | PR #15 (open) |

## What We Learned

### PyTorch is the wrong measurement layer for micro-architecture

Running `cli/run_layer_benchmarks.py` on the i7-12700K (the target
SKU) produced:

| Metric | Measured | Analytical | Ratio | Root cause |
|--------|----------|-----------|-------|------------|
| FP32 GFLOPS | 6.8 | 720 (peak) / 360 (50% eff) | 0.01-0.02x | PyTorch dispatch overhead (~5-50 us per `addcmul` call) dominates the ~1 ns FMA |
| pJ/op | 6040 | 2.1 | 2876x | RAPL measures entire socket (idle cores, memory controller, PCH) during a mostly-idle single-core workload |
| SIMD efficiency | 0.698 | 0.70 (model constant) | ~1.0x | Coincidence: PyTorch's internal SIMD at large vectors is ~70%, not the per-lane efficiency we intended |
| ILP ratio | 0.98 | >1.5 expected | ~0.65x | PyTorch collapses the dependency chain into one kernel per call; Python-level ILP is invisible |

### What would fix empirical Layer 1-2

True micro-architectural isolation requires:
- **C/C++ microbenchmarks** with inline ASM or intrinsics
- **rdtsc** timing (sub-nanosecond) instead of `time.perf_counter`
- **Core affinity** (`sched_setaffinity`) to pin to one core
- **Power gating** of idle cores before RAPL measurement
- This is ~2-4 weeks of C development per SKU family, not a Python
  script change

### The actual goal is different

The bottom-up plan's purpose is to validate the *modeling
infrastructure* (CPU/GPU/TPU/KPU energy and latency models), not to
characterize *real silicon*. The four architecture models
(`StoredProgramEnergyModel`, `DataParallelEnergyModel`,
`SystolicArrayEnergyModel`, `DomainFlowEnergyModel`) make layered
predictions that should compose consistently. A bug in one layer's
coefficients can cancel against a bug in another layer and still
produce a reasonable composite -- that's the mis-cancellation risk.

Model-based validation catches this without any physical hardware:
take each model's Layer-N prediction in isolation, compose upward,
and verify the result matches the model's own composite prediction.
If it doesn't, the coefficients are internally inconsistent.

---

## What Stays Valuable

The empirical infrastructure we built is NOT wasted:

1. **Phase 0 scaffolding** (LayerTag, field_provenance, composition
   registry, fitters subpackage) is exactly what model-based tests
   will use. No changes needed.
2. **Phase 0.5 PowerMeter** will be used when we eventually
   characterize real hardware (Jetson Orin Nano, rental AGX). It's
   just not the bottleneck.
3. **Composition check pattern** (predict from layers 1..N, diff
   against layer N+1) is the same pattern for model-based checks --
   just replace "measured GFLOPS" with "model's composite prediction."
4. **CLI runner** (`cli/run_layer_benchmarks.py`) is useful for
   end-to-end smoke tests on real hardware.
5. **Fitter pattern** (consume results, write provenance) carries
   over; model-based fitters validate analytical coefficients rather
   than fitting from measurements.

## What Changes

| Aspect | Empirical (Phases 1-2) | Model-based (Phases 1-6 retrofit) |
|--------|----------------------|-----------------------------------|
| Data source | Real hardware (RAPL, CUDA events) | Analytical model predictions |
| Coverage | Only SKUs we can physically access | All 45 hardware mappers |
| What it validates | Model vs silicon | Model vs itself (internal consistency) |
| Hardware needed | i7-12700K, Orin Nano, etc. | None (runs in CI) |
| TPU/KPU coverage | None (no silicon) | Full (exercises the model code paths) |
| Mis-cancellation detection | Only at boundaries with measurement | At every layer independently |
| Effort to add a new SKU | Run benchmarks on hardware | Write no code (SKU already has a model) |

---

## Recommendation

Pivot Phases 3-6 to model-based validation. Retroactively add
model-based checks for Layers 1-2 so every architecture class has
coverage. Keep the empirical benchmarks as an optional overlay for
the SKUs where we have hardware access.

See [`docs/plans/model-based-validation-plan.md`](../plans/model-based-validation-plan.md).

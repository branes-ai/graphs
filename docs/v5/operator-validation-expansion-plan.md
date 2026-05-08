# Plan: Validation Expansion -- Activations, Element-wise, Softmax, LayerNorm, Attention Head

**Status:** Plan, not yet implemented. Each phase below has a tracking GitHub
issue (filed 2026-05-08):

| phase | operator(s) | issue |
|---|---|---|
| 1 | activations (atan, sigmoid, tanh) + SFU modeling | #123 |
| 2 | element-wise matrix (matrix_add, matrix_add_transposed) | #124 |
| 3 | softmax (naive + online) + composite-op infrastructure | #125 |
| 4 | LayerNorm + RMSNorm | #126 |
| 5 | attention head (decomposed graph baseline) | #127 |

## Why this matters

The current V4 methodology characterizes BLAS L1/L2/L3 (`vector_add`, `linear`,
`matmul`) -- necessary but not sufficient for DNN dynamic-behavior
characterization. Five gaps to close, in order of architectural risk:

| # | operator | architectural sensitivity it exposes |
|---|---|---|
| 1 | activations (atan/sigmoid/tanh) | **SFU sparsity** -- dramatic AI swing from approximation choice; SFUs are a sparse silicon resource (1/SM on Jetson, 0 on KPU?) |
| 2 | matrix_add (2D element-wise) | **layout & access patterns** -- same AI as vector_add but rectangular access exposes stride/coalescing |
| 3 | softmax | **reduction + composite kernels** -- naive 5-pass vs FlashAttention-style 1-pass differs 2.5x in BW |
| 4 | LayerNorm | **two-pass reductions + scale/bias broadcast** -- common transformer hot path |
| 5 | attention head | **composition validation** -- does the analyzer correctly compose matmul + softmax + matmul? |

## Cross-cutting work (applies to every phase)

For each new op, six pieces must land together:

| piece | location |
|---|---|
| `op_footprint` formula | `validation/model_v4/sweeps/classify.py::OP_FOOTPRINT` |
| `ReuseModel` (for tier-aware roofline) | `src/graphs/estimation/reuse_models.py` |
| Sweep generator | `validation/model_v4/sweeps/_generate.py` |
| Workload builder (PyTorch CPU + CUDA + Jetson) | `validation/model_v4/workloads/<op>.py` |
| Subgraph builder (analyzer-side) | `validation/model_v4/harness/runner.py::_build_subgraph` |
| Visualizer marker + comparison | `visualize_baseline.py`, `compare_hardware.py` |

Plus: **regime classifier rules** for composite ops (which roofline regime
should softmax/LayerNorm/attention be expected to validate?).

---

## Phase 1 -- Activation functions (atan, sigmoid, tanh)

**Scope**: 3 element-wise operators with non-trivial FLOPS-per-element.

**Why first**: smallest extension on top of existing `vector_add` infrastructure,
but exposes the most interesting architectural difference (SFU bandwidth).
Establishes the pattern for all subsequent phases.

**Key technical decision -- how to model FLOPS-per-element**:
- Hardware-accelerated path (CPU SVML, CUDA SFU, KPU LUT): ~5-10 FLOPS/element
- Software polynomial / CORDIC: ~30-100 FLOPS/element
- Recommendation: characterize each hardware's actual implementation via
  micro-bench, report FLOPS-per-element as a per-(hw, op) calibration constant.
  Two operators per AI regime: `sigmoid` (cheap on most HW) and `atan` (often
  more expensive due to range reduction).

**Footprint** (per-element):

```
sigmoid: flops = N * sigmoid_flops_per_elem(hw)
atan:    flops = N * atan_flops_per_elem(hw)
bytes  = 2 * N * bpe   (in-place: read + write; non-in-place: 3*N*bpe)
```

**Reuse model**: same as `VectorAddReuseModel` (zero reuse, no useful tile
choice).

**Architectural addition -- SFU modeling**: needs a small
`HardwareResourceModel` extension:

```python
# Number of Special Function Units per compute unit, and their throughput
# relative to the FMA pipeline. e.g. Jetson SM has 1 SFU per 4 fp32 lanes.
sfu_units_per_compute_unit: int = 0       # 0 = no hardware SFU
sfu_ops_per_unit_per_clock: int = 0       # transcendental throughput
```

This infrastructure lands in Phase 1 so subsequent phases inherit it.

**Deliverables** (3 PRs):
- PR-A: footprint + reuse model + classifier + sweep generator + SFU field
  on `HardwareResourceModel`
- PR-B: PyTorch CPU/CUDA/Jetson measurer + V4 baseline capture on i7 +
  Orin Nano
- PR-C: per-(hw, op) FLOPS-per-element calibration + visualizer integration

**Estimated scope**: ~3-5 days.

---

## Phase 2 -- Element-wise matrix (matrix_add)

**Scope**: 2D version of `vector_add`. Tests rectangular access patterns and
layout assumptions.

**Why second**: Trivially extends Phase 1's element-wise infrastructure.
Interesting because same AI as vector_add but exposes coalescing differences
(e.g. transposed B operand on row-major hardware).

**Footprint**:
```
matrix_add (M, N): flops = M*N, bytes = 3*M*N*bpe → AI = 1/(3*bpe)
matrix_add_transposed (M, N): same FLOPS, but B accessed column-strided
```

**Reuse model**: same as vector_add (zero reuse).

**Decision needed**: do we model **transposed-B** as a separate op, or just
expect the analyzer's prediction to match measured (and watch the
latency-band fail when transposition matters)? Recommend: **separate op**
so we can validate the "transposed memory penalty" model explicitly.

**Deliverables** (1 PR):
- footprint + reuse model + sweep generator + measurer + i7/Orin baselines
  + visualizer

**Estimated scope**: ~1-2 days.

---

## Phase 3 -- Softmax

**Scope**: First composite operator. Two implementations to consider: naive
(5-pass) vs online (1-pass / FlashAttention-style).

**Why third**: Required building block for attention. Forces us to design how
composite ops express their working set, FLOPS, and reuse.

**Key technical decisions**:

1. **Implementation choice**: model **both** naive and online softmax as
   separate ops? Or just one?
   - Recommend: **separate ops** (`softmax_naive`, `softmax_online`). The
     whole point is to validate that the analyzer correctly captures the
     2.5x BW difference.
2. **Reuse model**: needs to walk the multi-pass kernel chain. Either:
   - (a) Sum over individual passes:
     `bytes_loaded = sum(pass_bytes_loaded)` per pass.
   - (b) Treat softmax as an "atomic op" with a closed-form `bytes_loaded`
     formula.
   - Recommend: **(a)** -- composes naturally with the existing per-op
     `ReuseModel` interface; each "pass" is a sub-op that the analyzer
     chains.
3. **Numerical stability** in the measurer: real softmax subtracts the max
   for stability. The FLOPS count should reflect this.

**Footprint** (B rows x N softmax dim, fp16):

| variant | FLOPS | bytes (binding tier) | AI |
|---|---|---|---|
| naive (5 passes) | ~5*B*N + (B*log_2(N) reductions) | ~5*B*N*bpe | 1/bpe |
| online (1 pass) | ~5*B*N | ~2*B*N*bpe | 2.5/bpe |

**Deliverables** (2 PRs):
- PR-A: naive softmax (footprint + reuse model + sweep + measurer + baselines)
- PR-B: online softmax + side-by-side comparison plot

**Estimated scope**: ~5-7 days.

---

## Phase 4 -- LayerNorm

**Scope**: Composite op with two-pass reduction (mean, var) + scale/bias
broadcast.

**Why fourth**: After softmax, this is a smaller delta. Reuses Phase 3's
"composite op" pattern but with 2 reductions and a stable inv_sqrt path.

**Footprint** (B rows x F features, fp16):

```
flops = B * (~10 * F + 2 * log_2(F))    # ~10 FLOPS/elem + 2 reduction trees
bytes = 4 * B * F * bpe                  # input + output + gamma + beta
AI ≈ 2.5 / bpe
```

**Reuse model**: 2-pass (mean+var → normalize+scale+bias).

**Deliverables** (1-2 PRs):
- PR-A: LayerNorm + RMSNorm (only 1 reduction, simpler -- common in modern
  transformers like Llama)
- Optionally PR-B: comparison panel

**Estimated scope**: ~2-4 days.

---

## Phase 5 -- Attention head (the capstone)

**Scope**: Validate the full **non-fused attention head** as a computational
graph composed of operators we have already characterized in Phases 1-4 (plus
the BLAS L3 `linear` and `matmul` already in V4).

**Modeling stance**: first-order, **decomposition-first**, no fusion. Why this
is the right first cut while the compiler and runtime are still under
development:

* The fused-kernel reality (FlashAttention, FusedMHA, etc.) is the long-term
  optimization target -- it's a *property of the deployment stack*, not the
  hardware.
* Before any compiler optimizations land, the realistic baseline is
  "PyTorch eager attention" -- which is exactly the unfused decomposition.
* Validating the decomposed graph proves that the analyzer **composes**:
  per-op accuracy + correct dataflow plumbing = correct end-to-end
  prediction. If the composed model is off, we know which sub-op to
  re-calibrate.
* When fused kernels do land (FlashAttention as a separate op in a
  follow-up), we can quantify the gap: measured-fused vs predicted-unfused
  is the **fusion uplift**, an architecturally meaningful number.

### The decomposition

A standard single-head attention layer (Q, K, V, O projections + scaled-dot
attention), expressed as a graph of operators we already model:

```
input: x ∈ (B, S, D_model)
weights: W_Q, W_K, W_V, W_O ∈ (D_model, D_model)

  +--------------+      +--------------+      +--------------+
  |   linear     |      |   linear     |      |   linear     |
  |  Q = x·W_Q   |      |  K = x·W_K   |      |  V = x·W_V   |
  |  (Phase 0)   |      |  (Phase 0)   |      |  (Phase 0)   |
  +--------------+      +--------------+      +--------------+
         |                     |                     |
         v                     v                     |
         +-------- matmul ─────+                     |
         |   scores = Q·K^T                          |
         |    (Phase 0 - matmul)                     |
         +-----------+                               |
                     |                               |
                     v                               |
              +-----------------+                    |
              | scale (1/√D)    |                    |
              | element-wise    |                    |
              | (Phase 1/2)     |                    |
              +-----------------+                    |
                     |                               |
                     v                               |
              +-----------------+                    |
              |    softmax      |                    |
              |    (Phase 3)    |                    |
              +-----------------+                    |
                     |                               |
                     +-------- matmul ───────────────+
                              |   attn = scores·V
                              |    (Phase 0 - matmul)
                              +--------+
                                       |
                                       v
                              +-----------------+
                              |    linear       |
                              |  out = attn·W_O |
                              |    (Phase 0)    |
                              +-----------------+
                                       |
                                       v
                                output: (B, S, D_model)
```

Every node is **one of**:
* `linear` (Phase 0, already characterized in V4)
* `matmul` (Phase 0)
* `element-wise scale` (Phase 1 / Phase 2 -- same kernel class as vector_add)
* `softmax` (Phase 3)

So the attention head IS the computational graph -- there is no new "atomic
op" to characterize. Phase 5 validates that running the analyzer over this
graph produces a prediction that matches the measured PyTorch eager
implementation (which is exactly this graph, no fusion).

### What the decomposition baseline does NOT model

This is honest about the gap:

* **FusedMHA / FlashAttention**: avoid materializing the (B, H, S, S)
  attention scores by tiling Q into blocks and streaming K, V through L1.
  Measured fused latency can be 2-5x lower than the decomposed prediction
  for long sequences.
* **Kernel-launch amortization**: the decomposed graph pays N × launch
  overhead for N kernels (Q, K, V, scale, softmax, attn, output, +
  intermediate writes). A fused kernel pays one launch.
* **Inter-op reuse**: scores tile that softmax reads is the SAME tile
  that the attn matmul reads. The decomposed model double-counts the
  read; a fused kernel keeps it in shared memory.

These gaps are *quantifiable once the baseline lands*. They become the
prioritization signal for compiler / runtime optimization work: where the
unfused-vs-fused gap is biggest, that's where fusion has the highest
leverage.

### Variants to support

1. **Standard single-head** (B=1 or small) -- v1 deliverable.
2. **Multi-head batched** (split D_model into H heads, batch the matmuls
   over H) -- v2 deliverable.
3. **Grouped-query attention (GQA)** (Llama-style: K, V have fewer heads
   than Q) -- v3, lower priority.
4. **Causal masking** (~50% FLOPS on the QK^T): skip in v1, add in v2.

### Footprint (single-head, decomposed, no fusion)

For input shape `(B, S, D_model)` and `D_head = D_model` (single-head case):

| sub-op | FLOPS | bytes (per kernel, no inter-op reuse) |
|---|---|---|
| Q = x·W_Q (linear) | 2*B*S*D² | 2*B*S*D + D² + 2*B*S*D |
| K = x·W_K (linear) | 2*B*S*D² | 2*B*S*D + D² + 2*B*S*D |
| V = x·W_V (linear) | 2*B*S*D² | 2*B*S*D + D² + 2*B*S*D |
| scores = Q·K^T (matmul) | 2*B*S²*D | 2*B*S*D + 2*B*S*D + 2*B*S² |
| scale (element-wise) | B*S² | 2*B*S² (in-place if measurer fuses) |
| softmax | ~5*B*S² | 2*B*S² (online) or 5*B*S² (naive) |
| attn = softmax·V (matmul) | 2*B*S²*D | 2*B*S² + 2*B*S*D + 2*B*S*D |
| out = attn·W_O (linear) | 2*B*S*D² | 2*B*S*D + D² + 2*B*S*D |

Total FLOPS (matmul-dominated for D >= 16):
```
≈ 8*B*S*D² + 4*B*S²*D + 5*B*S² ≈ 8*B*S*D² + 4*B*S²*D
```

Total bytes loaded (worst case, no reuse):
```
≈ ~16*B*S*D + 4*D² + ~10*B*S²
```

For typical configs (B=1, S=1024, D=128):
* FLOPS ~= 8.6 GF (compute side)
* Bytes ~= 11 MB intermediates (scores S² dominates: 4 MB; rest 7 MB)
* On Orin Nano: scores S² (4 MB) doesn't fit L2 (2 MB) → DRAM round-trips

This is exactly the symptom FlashAttention solves -- the predicted latency
will reflect that pain, providing the calibrated "before" for fusion gains.

### Deliverables (4 PRs)

* **PR-A: Composite-op infrastructure**. Add a `CompositeReuseModel` /
  graph-walker so the analyzer can chain per-op `ReuseModel`s with
  inter-op dependencies (intermediate tensor lifetimes). This is the
  enabling piece; required before any composite predictions are
  meaningful.
* **PR-B: Single-head attention baseline**. Decomposition graph builder +
  V4 sweep generator over `(B, S, D)` + measurer + i7 + Orin Nano
  baselines.
* **PR-C: Multi-head batched attention** + causal masking variant.
* **PR-D: FlashAttention as a separate op** (optional, if PR-A's
  composition holds). Measured-fused vs predicted-unfused gives the
  quantified fusion uplift.

**Estimated scope**: ~7-14 days.

---

## Suggested ordering and total scope

```
Phase 1 (activations)      3-5 days   ─┐ smallest delta, sets pattern
Phase 2 (matrix_add)       1-2 days   ─┘ for all subsequent phases
Phase 3 (softmax)          5-7 days   ── building block for attention
Phase 4 (layernorm)        2-4 days   ── small delta from softmax
Phase 5 (attention head)   7-14 days  ── capstone composition
─────────────────────────────────────
Total                     18-32 days  (~3-6 weeks of focused work)
```

## Risks and unknowns

1. **Composite-op modeling** is the biggest unknown. The current `ReuseModel`
   interface is per-op; chaining for softmax/attention needs a design pass
   before Phase 3.
2. **SFU modeling** is hardware-dependent and may need a new `SFUResource`
   field on `HardwareResourceModel` (analog of `compute_units` but for
   transcendentals). Lands in Phase 1.
3. **FLOPS-per-element calibration** for transcendentals must be measured
   per hardware -- there's no universal answer.
4. **Energy model accuracy** for composite ops compounds errors: if
   vector_add energy is off 10%, layernorm energy could be off 30%
   (3 sub-passes).
5. **Decomposition vs fusion gap** for attention is the *whole point* of
   tracking attention as a separate phase -- but it does mean the
   decomposed prediction will look "wrong" against fused-runtime
   measurements. Document the gap clearly so the V4 floor tests don't
   misfire.

## Cross-link

* PR #115 / #116 / #117 / #118 / #119 / #120 -- V5 follow-ups that
  proved the per-op + tier-aware modeling story works for BLAS L1/L2/L3.
* Tracking issues (filed 2026-05-08): #123 (Phase 1), #124 (Phase 2),
  #125 (Phase 3), #126 (Phase 4), #127 (Phase 5).

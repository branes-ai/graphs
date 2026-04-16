# TPU Modeling

This document describes how the `graphs` repository models **Google Tensor
Processing Units (TPU v1, v3, v4, v5p)** and the Google Coral Edge TPU
— all in the `SYSTOLIC_ARRAY` architecture class. A hypothetical
`TPU Edge Pro @ 30 W` SKU is also modeled for apples-to-apples
comparison against the KPU T256 and Jetson Orin AGX at the same thermal
envelope. Read
[`micro-architecture-modeling-methodology.md`](./micro-architecture-modeling-methodology.md)
first for the shared latency/energy methodology; this document only
covers TPU-specific mechanisms.

The TPU implementation lives in:

- `src/graphs/hardware/mappers/accelerators/tpu.py` — `TPUMapper`,
  `TPUAllocation`, factory functions
- `src/graphs/hardware/models/datacenter/tpu_{v1,v3,v4,v5p}.py` —
  datacenter TPU SKUs
- `src/graphs/hardware/models/edge/{coral_edge_tpu,tpu_edge_pro}.py` —
  edge SKUs
- `src/graphs/hardware/architectural_energy.py` —
  `SystolicArrayEnergyModel` (first-principles control-overhead model)
  and `TPUTileEnergyModel` (tile-based data-movement energy)

---

## 1. Architecture Summary

The TPU is a fixed-schedule **systolic array** processor. It is the
most energy-efficient class in the architectural taxonomy
(0.10–0.20× of stored-program baseline) because the schedule is
preloaded at kernel launch and data flows through predetermined paths
with no per-op instruction fetch.

Key modeling facts:

1. **MXU (Matrix Multiplier Unit) as the unit of compute** — each MXU
   is a 128×128 (16 K MACs) or 256×256 (65 K MACs) weight-stationary
   systolic array. A chip has 1 or 2 MXUs.
2. **Vector units as fallback** — element-wise ops (ReLU, Add) can't
   use the MXU. They run on auxiliary vector units at ~10× lower
   throughput.
3. **Unified Buffer (UB) + Accumulators + Weight Memory** — simple
   three-stage memory hierarchy. Weights are loaded once and
   **stationary** in the array; activations stream in.
4. **Deep pipeline** — a 128-cycle fill/drain before steady-state
   throughput. Small matrices (M, N < 128) incur significant pipeline-
   fill overhead.
5. **Batch-loving** — weight tiles are amortized across the batch. A
   TPU at batch=1 often runs < 20% efficiency; at batch ≥ 64 it
   approaches peak.
6. **BF16 is native** — FP32 is emulated on most TPUs; INT8 gets a 2×
   speedup over BF16. TPU v1 is INT8-only; v5p adds FP8 and structured
   sparsity.
7. **No dynamic control** — no branches, no dynamic resource
   arbitration. This is what makes it efficient *and* what limits it
   to regular dense workloads.

---

## 2. Device Coverage

Factory functions in `mappers/accelerators/tpu.py`:

### Datacenter

| Device | MXUs | Array | Clock | Peak | Memory | TDP | Precision |
|--------|------|-------|-------|------|--------|-----|-----------|
| **TPU v1** (ISCA 2017) | 1 | 256×256 | 700 MHz | 92 TOPS INT8 | 8 GB DDR3 (off-chip) | ~75 W | INT8 only |
| **TPU v3** | 2 | 128×128 | 940 MHz | 123 TFLOPS BF16 | 16 GB HBM | ~200 W | BF16, INT8 |
| **TPU v4** | 2 | 128×128 | 1050 MHz | 275 TFLOPS BF16 | 32 GB HBM2e | ~250 W | BF16, INT8 (2×) |
| **TPU v5p** | 2+ | 128×128 | 1100 MHz | 459 TFLOPS BF16 | HBM3 | ~450 W | FP8, BF16, INT8 + sparsity |

### Edge

| Device | MXU | Peak | Power | Notes |
|--------|-----|------|-------|-------|
| **Coral Edge TPU** | 1 | 4 TOPS INT8 | 2 W | USB/M.2 INT8-only inference |
| **TPU Edge Pro @ 30 W** (hypothetical) | 1 | 128×128 array, FP32/BF16/INT8 | 15/30/45 W | Comparison-only SKU matched to KPU T256 / Jetson Orin AGX |

Factory functions: `create_tpu_v1_mapper`, `create_tpu_v3_mapper`,
`create_tpu_v4_mapper`, `create_tpu_v5p_mapper`,
`create_coral_edge_tpu_mapper`, `create_tpu_edge_pro_mapper`.

---

## 3. Latency Model

The TPU mapper, like the GPU mapper, runs in one of two modes.

### 3.1 Mode Selection

`TPUMapper.should_use_sequential_execution(fusion_report, batch_size)`:

- **Sequential mode** (per-kernel, 1 or 2 MXUs each) when:
  - `batch_size < 16`, **and**
  - average FLOPs per subgraph < 500 M.
- **Parallel mode** otherwise.

The sequential mode threshold is **stricter than the GPU's** (500 M
vs. 200 M) because a TPU needs substantially more work to fill its
pipeline and saturate its single 128×128 array.

### 3.2 Operation Routing (MXU vs. Vector)

`_analyze_operation_type(subgraph) -> TPUAllocation`:

- If `operation_type ∈ {conv2d, linear, matmul, mm, bmm}`:
  - Routed to the **systolic array**.
  - Estimate matrix dimensions from FLOPs assuming a square-ish
    matmul: `M ≈ N ≈ sqrt(FLOPs / (2 × K))`, with `K ≈ 64` as a
    Conv-like default.
  - **Systolic utilization**:
    ```
    util_M = min(1.0, M / array_size)
    util_N = min(1.0, N / array_size)
    systolic_array_utilization = util_M × util_N
    ```
    A 32×32 matmul on a 128×128 array is at 25% × 25% = 6.25%
    utilization.
  - **Pipeline-fill overhead**:
    ```
    overhead = pipeline_depth / (K + pipeline_depth)
    ```
    Small-K matmuls pay the worst overhead (the pipeline-fill cost is
    comparable to the steady-state compute time).
- Otherwise (ReLU, Add, LayerNorm, softmax):
  - Routed to **vector units**.
  - Effective ops × 10 (≈ 10% of systolic-array throughput).

### 3.3 MXU Allocation

```
parallelism     = subgraph.parallelism.total_threads
mxus_needed     = ceil(parallelism / threads_per_mxu)     # threads_per_mxu = 16,384 (128×128)
mxus_allocated  = min(mxus_needed, num_mxus)              # num_mxus = 1 or 2
mxus_allocated  = max(1, mxus_allocated)
```

### 3.4 Roofline With Systolic Adjustments

```
effective_ops = ops
if not uses_systolic_array:
    effective_ops = ops × 10.0        # vector fallback penalty

if uses_systolic_array:
    effective_ops = ops × (1.0 + pipeline_fill_overhead)
    # in sequential mode: effective_ops = ops / systolic_array_utilization

compute_time, memory_time, bottleneck =
    _calculate_latency(effective_ops, bytes, mxus_allocated,
                       occupancy, precision)
```

Occupancy for systolic-routed ops **is** `systolic_array_utilization`
(unlike GPU where occupancy is a warp-scheduler metric) — small
matrices under-fill the array and both occupancy and utilization
reflect it.

### 3.5 Array Setup Overhead

Every kernel pays `ARRAY_SETUP_OVERHEAD = 64 ns` per dispatch (the
~128-cycle pipeline fill at ~2 GHz equivalent). This is the TPU
analog of the GPU's kernel-launch overhead, but two orders of
magnitude smaller because the schedule is already compiled into the
array hardware — only the weight-pointers and accumulator reset need
to be issued.

### 3.6 Sequential vs. Parallel Array Allocation

`determine_array_allocation(subgraph)`:

- `< 100 M FLOPs` → 1 MXU (typical ResNet-18 layer).
- `≥ 100 M FLOPs` → 2 MXUs.

In sequential mode, each subgraph is its own stage and latencies sum.
In parallel mode, stages run concurrently and within-stage latency is
`max(latency_i)`.

### 3.7 Small-Matrix Penalty

For systolic routing in sequential mode, unused MACs are modeled
explicitly:

```
if systolic_array_utilization < 1.0:
    effective_ops = ops / max(0.1, systolic_array_utilization)
```

A 50%-filled array doubles the effective op count — the MACs that
cycle empty still consume time. This is why a ResNet-18 at batch=1 on
TPU v4 runs nowhere near its 275 TFLOPS peak.

---

## 4. Energy Model

### 4.1 Baseline Three-Component Energy

Same three-component model (compute + memory + static idle) as every
other mapper. `TPUMapper.compute_energy_with_idle_power` uses
`IDLE_POWER_FRACTION = 0.5`.

### 4.2 Detailed: `TPUTileEnergyModel`

When a resource model has `tile_energy_model` attached, the mapper
overrides `_calculate_energy` to use the detailed tile-based model
(`TPUTileEnergyModel.compute_tile_energy`). The model captures tile-
level data movement through the full TPU memory hierarchy:

```
Weight Memory → Weight FIFO → Matrix Unit → Accumulators → Unified Buffer
```

Per-tile energy components:

1. **Weight tile loading** (DRAM/HBM → Weight FIFO → Matrix Unit),
   amortized by batch.
2. **Input activation streaming** (Unified Buffer → Matrix Unit).
3. **Systolic array computation** (MACs in weight-stationary dataflow).
4. **Accumulator management** (partial-sum staging).
5. **Output write** (Accumulators → UB; then UB → DRAM on eviction).
6. **Pipeline fill/drain overhead** (first and last `pipeline_depth`
   cycles produce no useful output).

Architectural parameters are generation-specific:

| Parameter | TPU v1 | TPU v3+ |
|-----------|--------|---------|
| Array size | 256×256 | 128×128 |
| Number of arrays | 1 | 2 |
| Weight tile size | 64 KiB | 32 KiB |
| Weight FIFO depth | 4 tiles | 2 tiles |
| Pipeline fill cycles | 256 | 128 |
| Accumulator size | 4 MiB | 2 MiB per MXU |
| Clock | 700 MHz | 940 MHz (v3) / 1050 MHz (v4) / 1100 MHz (v5p) |

Weights are **reused across the batch** — this is the structural
reason TPUs dominate energy comparisons at large batch.

### 4.3 Legacy: `SystolicArrayEnergyModel`

For resource models without a tile energy model (older TPU profiles
and the generic "TPU" reference), `SystolicArrayEnergyModel` provides
a first-principles control-overhead model. Its role is to explain
**why** systolic arrays are cheap, by enumerating every control signal
involved in a matrix op:

| Control domain | Energy event |
|----------------|--------------|
| Instruction decode | One decode per matrix op, **not per MAC** — `instructions_per_MAC ≈ 1 / 16,384`. |
| DMA controller | `dma_descriptor_setup` (~10 ns) + `dma_address_gen_per_cacheline`. |
| Weight loading sequencer | `weight_shift_control_per_element` + `weight_column_select_per_cycle`. |
| Unified Buffer controller | `unified_buffer_address_gen` + `unified_buffer_arbitration`. |
| Accumulator controller | `accumulator_read_control` + `accumulator_write_control` + `accumulator_address_gen`. |
| Tile iteration control | `tile_loop_control_per_tile`. |
| Data injection / extraction | Per-element into/out of the array edges. |

Aggregate efficiency vs. CPU baseline:

- `compute_efficiency = 0.15` (85% reduction in per-op overhead).
- `memory_efficiency = 0.20` (80% reduction — the UB + weight-
  stationary pattern avoids the cache-hierarchy cascade).

### 4.4 Why the TPU Wins Large-Batch Energy Comparisons

The `SystolicArrayEnergyModel` breakdown makes the reasons explicit:

1. **One instruction decode per 16 K MACs** vs. ~0.5 for a CPU, ~0.1
   for a GPU. Instruction-decode energy vanishes.
2. **Weight-stationary dataflow** — weights loaded once, multiplied
   N × batch times. Weight bandwidth amortizes linearly with batch.
3. **No cache coherence, no speculative fetch** — the entire memory
   path is DMA-driven and scheduled at compile time.
4. **No branch prediction, no OoO reorder buffer** — the hardware
   doesn't contain these structures to leak power.

Typical modeled pJ/MAC:

- **TPU (fixed systolic): ~0.8 pJ/MAC** (best).
- KPU (Domain Flow): ~1.1 pJ/MAC (slight overhead for programmability).
- GPU (SIMT): ~1.5 pJ/MAC.
- CPU (stored program): ~5 pJ/MAC.

---

## 5. Resource Model Specification

A `tpu_<gen>_resource_model()` factory typically declares:

1. **Compute units / MXUs**:
   - `compute_units = num_mxus` (1 for v1/Coral, 2 for v3/v4/v5p).
   - `threads_per_unit = array_width × array_height` (16,384 for
     128×128, 65,536 for 256×256).
   - `warp_size = array_size` (128 or 256) — reused as the per-
     dimension SIMD width.
2. **Precision profiles** — `PrecisionProfile` per supported precision
   with sustained peak.
3. **Memory hierarchy**:
   - `l1_cache_per_unit` = Unified Buffer size.
   - `l2_cache_total` = accumulator+UB total.
   - `main_memory` = HBM/DDR capacity.
   - `peak_bandwidth` = HBM bandwidth.
4. **Thermal operating points** (optional for datacenter, used for
   `tpu_edge_pro` at 15 W / 30 W / 45 W).
5. **Wave quantization = 1** (MXUs aren't grouped like GPU waves).
6. **Tile energy model** — attached via
   `model.tile_energy_model = TPUTileEnergyModel(...)` with the
   generation-specific parameters (array size, tile size, clock,
   accumulator dimensions).

The factory then attaches the architectural energy model:

```python
model.architecture_energy_model = SystolicArrayEnergyModel(
    tech_profile=DATACENTER_7NM_DDR5
)
```

(For v4+, the tile energy model is authoritative; the legacy
`SystolicArrayEnergyModel` remains attached for backward compatibility
with older comparison tools.)

---

## 6. TPU Edge Pro — Rationale for a Hypothetical SKU

`create_tpu_edge_pro_mapper` models a **non-existent** Google Edge TPU
at 30 W with FP32/BF16/INT8 support. Its purpose is **methodological
fairness**: the real Coral Edge TPU is INT8-only and 2 W, which makes
direct comparisons against a 30 W KPU-T256 or a 30 W Jetson Orin AGX
misleading. TPU Edge Pro answers "what would a 30 W Google systolic
array look like?" — letting the Embodied-AI-Architect orchestrator
compare architectures on a level playing field.

The SKU is marked `HYPOTHETICAL` in the resource model; any reports
that use it must flag the caveat.

---

## 7. Known Limitations (improvement backlog)

1. **Matrix dimension estimation is heuristic** — `K = 64` is a Conv-
   like default; large GEMMs with K = 4096 get mis-estimated.
   Upstream the subgraph descriptor should carry the actual (M, N, K)
   and the mapper should consume them directly.
2. **Vector-unit penalty is fixed 10×** — in reality vector-unit
   throughput varies with op type (softmax is worse than ReLU, which
   is worse than Add). A per-op table would help.
3. **Tile energy model is GEMM-only** — non-GEMM ops fall back to
   the baseline three-component model, missing UB staging energy.
4. **Batch size is hardcoded to 1 in `_calculate_energy`** — the
   `TPUTileEnergyModel.compute_tile_energy` call uses `batch_size=1`
   even when `map_graph` is given a larger batch. This under-credits
   weight amortization. Fix: propagate the batch size through the
   subgraph-mapping context.
5. **Weight/activation/output split is a fixed 50/30/20** heuristic
   — actual ratio depends on op shape. Should read from the subgraph
   descriptor.
6. **Sparsity support is not modeled** — TPU v5p has structured
   sparsity that can ~2× throughput on sparse weights. No mapper
   knob for this yet.
7. **Pipeline-fill overhead uses `K = 64` default** — if the real K
   were passed in, pipeline cost would be far more accurate
   (especially for the very small K that shows up in depthwise
   convs).
8. **`TPU Edge Pro` uses estimated efficiency factors** — not backed
   by measurement; flagged in the code comments.
9. **No multi-chip / pod modeling** — real TPU v4/v5p pods use
   dedicated interconnect (ICI) for model-parallel training. Only
   single-chip behavior is modeled.

---

## 8. Validation Hooks

- `validation/hardware/test_all_hardware.py` — includes TPU v4 and
  Coral in the cross-platform comparison.
- `cli/analyze_comprehensive.py --hardware tpu_v4` — end-to-end
  sanity run.
- `cli/compare_architectures.py` — TPU vs. GPU vs. KPU vs. CPU
  comparison tool; the architectural-energy breakdown here is what
  `SystolicArrayEnergyModel` and `TPUTileEnergyModel` feed.

When modifying the mapper, check that the batch=1 ResNet-18 latency
on TPU v4 stays within the known real-hardware ballpark (~1–3 ms),
and that large-batch BF16 throughput does not exceed 275 TFLOPS peak.

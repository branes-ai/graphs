# V5: Operational-Analysis Memory Hierarchy Plan

**Status**: draft for review
**Scope**: replace `_get_bandwidth_efficiency_scale` (the single-scalar memory model) with a per-tier, queueing-theory-grounded model that explains how a workload uses each tier of the memory hierarchy.

## Why this rewrite

The V4 validation harness surfaced that the current memory model is calibrated by op family rather than by physics:

* The GPU branch was calibrated against MobileNet-V2 conv2d on Jetson Orin AGX. Matmul sweeps from V4 land in the same curve and get conv-tuned numbers.
* The CPU branch was calibrated against i7-12700K. A Ryzen, M3, or EPYC reads `hw_type == 'CPU'` and gets i7 numbers.
* Cache-resident shapes can exceed peak DRAM bandwidth. The current model caps at 0.85 because it has no concept of "this op isn't DRAM-bound, it's L2-bound."
* Special cases (depthwise conv = 0.02) are hardcoded enumerations, not consequences of a model.

What we lose: the analyzer can't *explain* a prediction. It can't say "this 1024^3 matmul on i7 fits its inner tile in L1, achieves K-loop reuse, so the binding tier is L2 BW at ~150 GB/s." It can only say "0.7 of 75 GB/s = 52.5 GB/s, take it or leave it."

Issue #61 already shipped the schema fields for on-chip BW peaks. V5 builds the model that uses them.

## Conceptual model

### Operational analysis primer

Treat each memory tier as a **server** in the queueing-theory sense:

* **Service rate** μ = peak bandwidth of the tier (bytes/sec)
* **Arrival rate** λ = bytes/sec the kernel demands from the tier
* **Utilization** ρ = λ / μ
* **Effective service time** for a request = bytes / μ_effective

A kernel runs through a hierarchy of these servers. The throughput is bounded by the slowest server in the chain — Little's Law / bottleneck analysis. The art is figuring out **which tier is the binding constraint** and **how much traffic the kernel actually generates at that tier** (which depends on the kernel's reuse pattern and the tile-size choice).

### Two distinct tier roles per kernel

Two terms have to stay clearly separated throughout the rest of this doc:

* **Residency tier** — the smallest tier whose total (chip-aggregate) capacity holds the working set the kernel keeps alive while computing on a tile. Identified by walking innermost-out and stopping at the first tier that fits the residency window. The kernel's *cache hits* come from this tier.
* **Binding tier** — the tier the kernel *streams from* to refresh the residency window. Always the next-larger tier after the residency tier (or DRAM if the residency tier is already DRAM). Its effective bandwidth gates throughput.

Where this doc says "binding tier" it means the streaming source whose BW we charge memory traffic against. The `tier_picker` returns the binding tier (not the residency tier).

### What replaces `bw_efficiency_scale`

Three pieces:

1. **`MemoryTier`**: capacity, peak BW, achievable-BW factor, access latency. Exists per-tier on the hardware.
2. **`ReuseModel(op, shape, residency_tier)`**: how many times each input byte is read from the *binding* tier while the kernel is computing on a residency window held in the inner tier. Pure function of op type + shape + tile size + residency-tier capacity.
3. **`tier_picker(op, shape, hw) -> MemoryTier`**: walks the hierarchy innermost-out, finds the residency tier, returns the **binding tier** (the next-larger one).

The replacement contract:

```python
def memory_time(op, shape, dtype, hw) -> float:
    binding_tier = tier_picker(op, shape, hw)        # streaming source
    bytes_at_tier = ReuseModel(op, shape, binding_tier).bytes_loaded()
    return bytes_at_tier / binding_tier.effective_bandwidth_bps
```

The narrative payoff: the analyzer can return alongside the latency a one-line explanation, e.g.

> `(1024, 1024, 1024) fp32 matmul on i7-12700K: tile (128, 128, 32) fits in 32 KB L1 (per-core, residency tier), K-loop reuse 1024x, binding tier = L3 (LLC, 25 MB), effective L3 BW = 180 GB/s × 0.78 = 140 GB/s, memory_time = 12 MB / 140 GB/s = 86 µs.`

That's exactly the story the user asked for.

## Memory tier abstraction

This is what V5-1 actually shipped (PR #96):

```python
@dataclass(frozen=True)
class MemoryTier:
    name: str                          # "L1", "L2", "L3", "DRAM", "scratchpad"
    capacity_bytes: int                # per-unit if is_per_unit else aggregate
    is_per_unit: bool
    num_units: int                     # compute_units when per-unit; else 1

    peak_bandwidth_bps: float          # ALWAYS aggregate -- per-unit BW gets
                                       # multiplied by num_units when constructed
    access_latency_ns: float           # first-request startup latency (V5-3 will
                                       # use this as the memory-side LAUNCH_BOUND)
    achievable_fraction: float = 1.0   # V5-5 calibration knob; default = ideal

    @property
    def total_capacity_bytes(self) -> int:  # is_per_unit ? cap*num_units : cap
        ...

    @property
    def effective_bandwidth_bps(self) -> float:  # peak * achievable_fraction
        ...
```

`achievable_fraction` is a single scalar in V5-1, not a per-access-pattern function. The tradeoff: simpler to calibrate (one knob per (hw, tier)), simpler to reason about, matches the data we'll actually capture in V5-2's vector-add microbenchmarks. Per-access-pattern variation (coalesced vs strided, read-only vs read-write) can be added in V5-5+ when the data justifies it.

A hardware mapper exposes the list of tiers via the `memory_hierarchy` derived property:

```python
hw.memory_hierarchy == [
    MemoryTier("L1",   32*1024,      is_per_unit=True,  num_units=12, peak_bw=5.18e12, ...),
    MemoryTier("L3",   25*1024**2,   is_per_unit=False, num_units=1,  peak_bw=200e9,   ...),
    MemoryTier("DRAM", 64*1024**3,   is_per_unit=False, num_units=1,  peak_bw=75e9,    ...),
]
```

Existing fields populate this:
* `l1_cache_per_unit`, `l1_bandwidth_per_unit_bps` (from #61) → L1 tier
* `l2_cache_total`, `l2_bandwidth_bps` (from #61) → L2 tier
* `l3_cache_total`, `l3_bandwidth_bps` (from #61) → L3 tier (CPU only)
* `main_memory`, `peak_bandwidth` → DRAM tier

Backward compat: when an on-chip BW field is `None`, the tier is omitted from the hierarchy and the analyzer falls back to the existing scalar path.

## Per-operator reuse models

For each op kind we derive: given a *tile size* that fits in tier T, how many bytes load from tier T+1 (the next-larger tier) per FLOP of work?

### Vector add `c[i] = a[i] + b[i]` on N elements

* Working set: 3N bytes (a, b, c)
* Reuse: **none**. Each byte read once, each result written once.
* Bytes loaded at any tier: 3N (modulo cache line size)
* Effective compute/byte = 1/(3*bpe) — extremely low
* Always memory-bound

This is the *ground truth zero-reuse op*. Vector add benchmark = pure tier-BW measurement.

### Matmul `C = A @ B`, shape (M, K, N)

Working set with naive untiled execution: `(MK + KN + MN) * bpe`. With outer-product or row-major tiling at tile size (Mt, Nt, Kt):

* A tile: Mt × K bytes (streaming K dimension)
* B tile: K × Nt bytes
* C tile: Mt × Nt bytes (output, accumulator)

For a tiled matmul where the (Mt, Nt) C-tile fits in the binding tier:

* C is read once and written once: 2 × Mt × Nt × bpe bytes
* A is reused **N/Nt times** along the inner-K loop: Mt × K × bpe × (N/Nt)
* B is reused **M/Mt times**: K × Nt × bpe × (M/Mt)

**Key insight**: the larger the C-tile (Mt, Nt) that fits in the binding tier, the more reuse, the lower the bytes-per-flop, the more compute-bound the kernel becomes. This is what tile sizing buys you, and what the analyzer needs to model.

### Linear `y = x @ W^T + b`

Same reuse structure as matmul with `M = batch_size`. Reduces to matmul algebraically.

### Conv2d (standard, dense)

Output (B, Co, Ho, Wo), input (B, Ci, Hi, Wi), filter (Co, Ci, Kh, Kw).

Per output element: `Ci * Kh * Kw` MACs.

* **Spatial reuse**: each input pixel is read by `Kh * Kw` output positions (sliding window) — but only across overlapping receptive fields, not full Kh*Kw multiplier in cache-resident tiles.
* **Channel reuse**: each filter weight reused across `B * Ho * Wo` output positions.
* **Im2col / GEMM lowering**: most modern kernels reduce conv2d to matmul; the matmul reuse model applies to the lowered shape.

For the V4 sweep this is out of scope (V4 stays matmul + linear). Document the model so it can extend later.

### Depthwise conv2d

Same as conv2d but **no channel reuse**: each filter is `(Ci, Kh, Kw)` with one output channel per input channel. Reuse drops by factor `Co`. That's why `bw_efficiency_scale` had to hardcode 0.02 — depthwise's per-output-element bytes-per-FLOP is naturally 30-50x worse than dense conv2d. With a proper reuse model, depthwise becomes a special *case of the model* (no Co reuse), not a hardcoded special case in the analyzer.

### Attention (deferred)

Attention has multiple matmul stages with intermediate softmax. FlashAttention-style fused kernels change the reuse pattern dramatically. Defer to a V5.5 or later.

## Tier-picking algorithm

```python
def pick_binding_tier(op, shape, dtype, hw) -> MemoryTier:
    # Use total_capacity_bytes (which handles is_per_unit -> capacity *
    # num_units aggregation) so per-unit and shared tiers compare
    # apples-to-apples.
    tiers = sorted(hw.memory_hierarchy, key=lambda t: t.total_capacity_bytes)
    # Walk from innermost (L1) outward; find the residency tier (the
    # smallest tier whose AGGREGATE capacity holds the residency window
    # the op needs to keep alive to maximize reuse). The binding tier
    # we return is the next-larger tier -- the streaming source.
    for tier in tiers:
        residency = op.residency_window(shape, dtype, tier_capacity=tier.total_capacity_bytes)
        if residency <= tier.total_capacity_bytes:
            inner_idx = tiers.index(tier)
            if inner_idx + 1 < len(tiers):
                return tiers[inner_idx + 1]   # streaming source = binding tier
            return tier   # already at outermost; binding tier is itself
    return tiers[-1]   # outermost (DRAM) -- working set doesn't fit anywhere
```

`op.residency_window` is the per-op question: "given a tile of capacity C bytes in a cache tier, what's the tile size we'd choose, and what working set does it imply?"

For matmul: `residency_window(C) = sqrt(C / (3 * bpe))` for the C-tile, optimal-square tile heuristic.

The picked tier's BW × the op's bytes-per-flop at that tier × flops = memory_time.

## Microbenchmark validation strategy

For each (hardware, tier) pair, capture an empirical **achievable BW curve**:

### Vector add at increasing N

```text
N = 256  → fits in L1 → measures L1 BW
N = 64K  → fits in L2 → measures L2 BW
N = 4M   → fits in LLC → measures LLC BW
N = 1G   → DRAM-bound → measures effective DRAM BW
```

The slope `latency / N` at each plateau is the inverse achievable BW for that tier. This becomes the calibration anchor for `tier.achievable_fraction`.

### Matmul reuse vs tile size

```text
(M, K, N) = (64, 64, 64)        WS=48 KB → L1-resident → expect ALU-bound
(M, K, N) = (256, 256, 256)     WS=768 KB → L2-resident
(M, K, N) = (1024, 1024, 1024)  WS=12 MB → LLC-resident (i7) / DRAM (Orin)
(M, K, N) = (4096, 4096, 4096)  WS=192 MB → DRAM-bound on every target
```

The achieved GFLOPS at each WS bucket exposes the binding tier's effective BW + the reuse the kernel achieves. Two unknowns per shape (tile efficiency, tier BW); two equations from the WS sweep; the reuse model derives the kernel-tile-size assumption.

### Microbenchmark deliverables

Add `validation/model_v4/workloads/vector_add.py` (new) that builds a 1-op vector-add workload. Then sweep entries spanning the cache hierarchy:

```yaml
{shape: (1024,),    dtype: fp32, expected_tier: L1}
{shape: (16384,),   dtype: fp32, expected_tier: L2}
{shape: (1048576,), dtype: fp32, expected_tier: LLC}
{shape: (1<<28,),   dtype: fp32, expected_tier: DRAM}
```

Each entry validates one tier's BW directly. Failure on the L2 entry → L2 BW is wrong; failure on the LLC entry → LLC BW is wrong; etc. Pinpoint diagnostics.

## Migration plan

V5 lands as a sequence of PRs, each independently mergeable, each leaving the harness green.

| Phase | Scope | Exit criterion |
|---|---|---|
| **V5-0** | This plan + reuse-model derivations + microbenchmark spec — user review | User signs off; ✓ → proceed |
| **V5-1** | Add `MemoryTier` dataclass + `hw.memory_hierarchy` builder. Populate from existing fields (no analyzer changes). Backward compat: empty hierarchy if on-chip BW peaks missing. | All existing tests pass; new tests assert tier fields populate correctly on i7, H100, Orin |
| **V5-2** | Add `vector_add` workload + sweep + capture-on-target runbook. Capture i7 + Jetson baseline microbenchmarks. | New CSVs committed; visualizer renders per-tier BW from vector-add data |
| **V5-3** | Implement `pick_binding_tier` + per-op `residency_window` for matmul, linear. New `_get_effective_memory_time` co-exists with `_get_bandwidth_efficiency_scale`; analyzer uses new path when `hw.memory_hierarchy` is populated, falls back to scalar otherwise. | Matmul + linear V4 floor pass-rates equal or improve on every hw target |
| **V5-4** | Add the explainability output: `LatencyDescriptor.memory_explanation` returns a structured breakdown (binding tier, residency window, bytes_loaded_at_tier, effective_bw). Surface it in V4 reports. | A new V4 report column shows the binding tier per shape; visualizer adds a per-shape annotation mode |
| **V5-5** | Calibrate the tier-specific `achievable_fraction` per hardware target from the V5-2 microbenchmark CSVs. | Per-mapper calibration PRs (one per HW target); each one bumps V4 floors |
| **V5-6** | Remove `_get_bandwidth_efficiency_scale` once every active mapper has a populated `memory_hierarchy` + tier-specific calibrations. Decommission the scalar path. | bw_efficiency_scale deleted; V4 floors hold |

V5-1 through V5-3 can land in parallel with V4 follow-up calibration work since they don't break the existing path.

## What this gives the V4 validation harness

* **L1_BOUND finally fires** on hardware where it's reachable. Right now it's structurally inaccessible on flagship accelerators (per V4-#61 doc-test) because the AI breakpoint formulation is one-bottleneck-fits-all. With per-tier modeling, an L1-resident matmul that's BW-bound at L1 lands in L1_BOUND for real.

* **Per-shape failure attribution works at the tier level.** Today V4 says "this shape's predicted latency is 30% off." Tomorrow it can say "this shape's predicted latency is 30% off because the L2 BW assumption is 180 GB/s and measured was 140 GB/s." Calibration becomes a one-knob fix per failure.

* **Microbenchmarks become first-class V4 fixtures.** Vector add and small matmul become the *primary* validation shapes, with full DNN models as integration tests on top.

* **The story of how the workload uses the hardware** comes out of the analyzer for free, as a side-effect of the tier-picking algorithm. Reports become explanatory, not just predictive.

## Decision records (resolved)

The original draft surfaced six open questions. Decisions made during user review on 2026-05-06 -- downstream implementation PRs should rely on these defaults rather than reinterpret.

| # | Question | Decision | Owner | Date | Reversible? | Symbol(s) |
|---|---|---|---|---|---|---|
| 1 | `MemoryTier` first-class field vs derived view | **Derived view** (`@property`) -- shipped in V5-1 | stillwater | 2026-05-06 | Yes (could move to first-class field with a refactor) | `HardwareResourceModel.memory_hierarchy` |
| 2 | Tile-size oracle for `residency_window` | **Optimal-square heuristic** (`sqrt(C/(3*bpe))` for matmul C-tile), with hardware-specific overrides exposed as Optional fields | stillwater | 2026-05-06 | Yes (override per-mapper, or swap to lookup table) | `op.residency_window`, override fields TBD |
| 3 | Spatial-dataflow scratchpads (KPU, TPU) | **Same tier model** -- scratchpad maps to L1 tier with `is_per_unit=True`. The conceptual difference (HW cache vs SW resource manager staging) is encoded in the *staging* model, not the tier abstraction. Block linear algebra reuse decomposition applies the same way. | stillwater | 2026-05-06 | Yes (could subclass for a different model later) | `MemoryTier(is_per_unit=True)` |
| 4 | Multi-kernel streaming (network of queues) | **Deferred** -- out of scope for V5; revisit when the C++ concurrency analyzer needs it | stillwater | 2026-05-06 | Yes (additive when the time comes) | n/a |
| 5 | CPU SMT / hyperthreading | **Deferred** -- captured separately as `instruction_efficiency` if needed | stillwater | 2026-05-06 | Yes | `instruction_efficiency` |
| 6 | Latency-bound floor | **First-class** -- per-tier `access_latency_ns` field shipped in V5-1; the V5-3 tier picker uses it as the memory-side LAUNCH_BOUND for vector / matvec / small-matmul ops where single-stream startup dominates BW-limited service time | stillwater | 2026-05-06 | Yes (can disable by setting fields to 0) | `MemoryTier.access_latency_ns`, V5-3 tier picker |

V5-1 (PR #96) implements decisions 1, 3, and 6. Decisions 2, 4, and 5 are honored in subsequent phases (V5-3 for #2, deferred for #4 and #5).

## Tentative PR sequence

1. `feat(estimation): MemoryTier dataclass + hw.memory_hierarchy view (V5-1)`
2. `feat(validation): vector_add workload + sweep (V5-2a)`
3. `feat(validation): vector_add baseline capture for i7 + Jetson (V5-2b)` — needs hardware
4. `feat(estimation): per-op reuse models for matmul, linear (V5-3a)`
5. `feat(estimation): tier_picker + new _get_effective_memory_time (V5-3b)`
6. `feat(reporting): memory_explanation breakdown in LatencyDescriptor (V5-4)`
7. `fix(hardware): per-mapper tier achievable_fraction calibrations (V5-5, one per hw)`
8. `refactor(estimation): retire bw_efficiency_scale (V5-6)`

Each PR ships with regression tests. The V4 floor tests at each step protect against regressions.

## Out of scope for V5

* New op categories (attention, normalization, softmax) — these need their own reuse models
* Streaming-pipeline / multi-kernel-in-flight modeling
* Latency-bound kernels (small-message regime)
* Energy modeling at the per-tier level (currently energy uses lumped energy_per_byte; the same tier-aware approach could apply but it's a separate axis)

## Definition of done (V5)

* `_get_bandwidth_efficiency_scale` no longer exists.
* Every hw target with a V4 baseline has a populated `memory_hierarchy` with tier-specific `achievable_fraction` calibrated against vector-add microbenchmarks.
* `LatencyDescriptor` includes a structured `memory_explanation` field.
* The V4 report's per-shape table shows the binding tier alongside latency and energy.
* Adding a new hw target requires: (a) populating tier capacities + peak BWs, (b) running the vector-add sweep, (c) the calibration is otherwise automatic.

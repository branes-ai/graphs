# v8 unification: vendor-neutral ``compute_block_common`` -- paper exercise

Status: draft
Date: 2026-05-17
Tracking issue: graphs#208
Predecessors:
  - GPU sprint #171 (closed), CPU sprint #182, NPU sprint #187,
    CGRA sprint #196, DPU sprint #200, TPU sprint #204 (all closed)

## Purpose

Issue #208 ("v8 unification: vendor-neutral compute_block_common
module") proposes a **refactoring sprint** -- not architecture-
extension -- to collapse the cross-block-kind reuse patterns that
7 prior sprints have accumulated. This document is **PR 1 of that
sprint** -- the paper exercise: enumerate every shared primitive
and shared type shape across the 7 existing block kinds (KPU + GPU +
CPU + NPU + CGRA + DPU + TPU), classify them by how safely they can
be unified, and propose the migration strategy + backward-compat
guarantees for the new ``compute_block_common`` module.

The structure of this document differs from the prior 6 paper
exercises because the work is fundamentally different:

  - Prior paper exercises: audit ONE new SKU's features against the
    existing schema, propose NEW types
  - This paper exercise: audit EXISTING types across 7 block kinds,
    propose to UNIFY shared shapes

The output is a recommended ``compute_block_common.py`` module
structure + a per-block-kind migration plan + risk assessment.

## Method

For each shared primitive (already-reused enum / type) and each
recurring type shape (per-block-kind duplicated structure), this doc
records:

- Which modules currently define / reuse it
- How many block kinds use the pattern
- Whether the shapes are *literally identical*, *near-identical with
  minor variations*, or *similar with significant variations*
- Safety classification: SAFE_TO_UNIFY / NEAR_UNIFIABLE / KEEP_SEPARATE
- For SAFE_TO_UNIFY: proposed unified type + migration strategy

Out of scope for this exercise (deferred):

- **KPU schema unification** -- KPU is the oldest (v1), pre-dates
  the cross-block-kind reuse pattern, and has its own conventions
  (KPUTileSpec / KPUNoCSpec / KPUSiliconBin etc.). Unifying KPU
  requires its own refactor that touches all 12 KPU YAML SKUs.
  Defer to v10+.
- **DSP schema extension** -- separate v9 sprint (10 SKUs)
- **``has_external_dram`` vs ``has_host_dram`` naming reconciliation**
  (NPU/DPU/TPU vs CGRA) -- semantically real distinction (chip-
  attached vs host-bus). Defer to v9 after the unified base types land.
- **Per-architecture fabric kind enums** (NPUDataflowKind,
  CGRAFabricKind, DPUFabricKind, TPUFabricKind, GPUFabricKind) --
  intentionally architecture-specific; collapsing into a single
  enum loses precision. KEEP_SEPARATE.
- **Multi-block-per-die** (TPU SparseCore, Versal ARM control complex)
  -- separate v9+ feature

## Reference: current v7 schema state

After 7 sprints, ``embodied-schemas/src/embodied_schemas/`` contains:

```
process_node.py       -- defines DataConfidence, CircuitClass, ProcessNodeEntry
gpu.py                -- defines MemoryType (used by 5 other modules)
kpu.py                -- KPUTileSpec, KPUNoCSpec, KPUMemorySubsystem,
                         KPUSiliconBin, KPUClocks, KPUTheoreticalPerformance,
                         KPUThermalProfile (pre-dates the pattern)
gpu_block.py          -- GPUBlock + 5 supporting types + 4 enums + ClockDomain
cpu_block.py          -- CPUBlock + 5 supporting types + 4 enums (reuses ClockDomain)
npu_block.py          -- NPUBlock + 5 supporting types + 3 enums + KVCacheSpec
cgra_block.py         -- CGRABlock + 5 supporting types + 2 enums
dpu_block.py          -- DPUBlock + 5 supporting types + 2 enums
tpu_block.py          -- TPUBlock + 6 supporting types + 2 enums
compute_product.py    -- BlockKind enum + AnyBlock discriminated union + spine
```

That's ~30 per-block-kind support types across 6 modern block kinds
(everything except KPU). The pattern: each block kind defines its own
``*ComputeFabric``, ``*MemorySubsystem``, ``*OnDieFabric``,
``*ThermalProfile``, ``*TheoreticalPerformance``.

## Audit: shared primitives (already reused)

These are primitives currently defined in one module and imported
by others. v8 should centralize them in ``compute_block_common``
via re-export (no behavior change).

| Primitive | Source module | Reused by | Verdict |
|---|---|---|---|
| ``ClockDomain`` | ``gpu_block`` | cpu_block | SAFE_TO_UNIFY (re-export) |
| ``DataConfidence`` | ``process_node`` | npu, cgra, dpu, tpu | SAFE_TO_UNIFY (re-export) |
| ``MemoryType`` | ``gpu`` | npu, cgra (as host_dram_type), dpu, tpu | SAFE_TO_UNIFY (re-export) |
| ``CircuitClass`` | ``process_node`` | gpu, cpu, npu, cgra, dpu, tpu | SAFE_TO_UNIFY (re-export) |

**Recommendation**: ``compute_block_common`` re-exports all 4 from
their current source modules. Existing imports continue to work
(no breakage); new code uses ``from compute_block_common import ...``
for a single canonical import point.

## Audit: shared type shapes (per-block-kind duplicates)

These are types whose **structure** is duplicated across 6 modern
block kinds with varying degrees of similarity.

### 1. ``*TheoreticalPerformance`` -- LITERALLY IDENTICAL

```python
# In GPU/CPU/NPU/CGRA/DPU/TPU block modules:
class XXXTheoreticalPerformance(BaseModel):
    peak_ops_per_sec_by_precision: dict[str, float] = Field(...)

    @model_validator(mode="after")
    def _validate_positive(self) -> "XXXTheoreticalPerformance":
        for prec, value in self.peak_ops_per_sec_by_precision.items():
            if value < 0:
                raise ValueError(...)
        return self
```

**6 copies, byte-identical except for the class name.** SAFE_TO_UNIFY.

Recommendation: define ``compute_block_common.TheoreticalPerformance``
exactly once. Each block module aliases:

```python
# In each *_block.py:
from embodied_schemas.compute_block_common import TheoreticalPerformance
XXXTheoreticalPerformance = TheoreticalPerformance  # backward-compat alias
```

Existing imports (``from embodied_schemas.npu_block import NPUTheoreticalPerformance``)
continue to work via the alias. PR 3 of the sprint migrates NPU as
proof of concept; subsequent issues migrate the other 5 block kinds.

### 2. ``*ThermalProfile`` -- NEAR-IDENTICAL (one variation)

```python
class XXXThermalProfile(BaseModel):
    name: str
    tdp_watts: float
    cooling_solution_id: str
    clock_mhz: float | ClockDomain   # <-- variation: scalar vs ClockDomain
    dvfs_enabled: bool = False        # <-- present on NPU/CGRA/DPU/TPU; absent on GPU/CPU (use ClockDomain instead)
    efficiency_factor_by_precision: dict[str, float]
    instruction_efficiency_by_precision: dict[str, float]
    memory_bottleneck_factor_by_precision: dict[str, float]
    vdd_v: Optional[float]
```

**Difference**: GPU/CPU use ``ClockDomain`` (base/boost/sustained
+ dvfs_enabled inside ClockDomain) for multi-profile DVFS. NPU/CGRA/
DPU/TPU use scalar ``clock_mhz`` + top-level ``dvfs_enabled`` because
they're typically single-profile.

**Verdict**: NEAR_UNIFIABLE. Propose:

```python
class ThermalProfile(BaseModel):
    name: str
    tdp_watts: float
    cooling_solution_id: str
    # Use the union: scalar clock OR ClockDomain
    clock_mhz: float | None = None
    clock_domain: ClockDomain | None = None
    dvfs_enabled: bool = False
    efficiency_factor_by_precision: dict[str, float] = {}
    instruction_efficiency_by_precision: dict[str, float] = {}
    memory_bottleneck_factor_by_precision: dict[str, float] = {}
    vdd_v: float | None = None

    @model_validator(mode="after")
    def _validate_clock(self):
        # Exactly one of clock_mhz / clock_domain must be set
        ...
```

This is mergeable but riskier than ``TheoreticalPerformance`` because
the existing GPU/CPU YAMLs use one shape and NPU/CGRA/DPU/TPU YAMLs
use another. **Recommendation**: defer to v9 PR; not in scope for v8.

### 3. ``*OnDieFabric`` -- NEAR-IDENTICAL (topology enum varies)

```python
class XXXOnDieFabric(BaseModel):
    topology: XXXNoCTopology   # <-- per-arch enum
    bisection_bandwidth_gbps: float
    unit_count: int
    flit_size_bytes: int
    mesh_rows: int | None
    mesh_cols: int | None
    hop_latency_ns: float
    pj_per_flit_per_hop: float
    routing_distance_factor: float
    confidence: DataConfidence
```

**Difference**: each block kind has its own ``XXXNoCTopology`` enum
(MESH_2D vs AIE_MESH vs CROSSBAR vs etc.). The actual topology
semantics differ enough that collapsing the enum loses precision.

**Verdict**: NEAR_UNIFIABLE on STRUCTURE, KEEP_SEPARATE on enum.
Propose a generic ``OnDieFabricBase`` mixin (the 9 non-enum fields)
that each block kind composes with its own topology enum:

```python
# In compute_block_common:
class OnDieFabricFields(BaseModel):
    """Mixin: shared on-die fabric fields. Each block kind adds its
    own `topology` field with the architecture-specific enum."""
    bisection_bandwidth_gbps: float = Field(..., gt=0)
    unit_count: int = Field(..., gt=0)
    # ... 7 more shared fields
```

Each block module:
```python
class NPUOnDieFabric(OnDieFabricFields):
    topology: NPUNoCTopology = Field(...)
```

**Recommendation**: defer to v9 PR; mixin pattern requires careful
testing across all 6 block kinds. Not in scope for v8.

### 4. ``*MemorySubsystem`` -- NEAR-IDENTICAL with naming variation

```python
class XXXMemorySubsystem(BaseModel):
    on_chip_bandwidth_gbps: float
    # NPU/DPU/TPU: has_external_dram + external_dram_*
    # CGRA:        has_host_dram     + host_dram_*       (semantic difference)
    has_external_dram: bool       # OR has_host_dram (CGRA)
    external_dram_type: MemoryType | None    # OR host_dram_type
    external_dram_size_gb: float | None
    external_dram_bandwidth_gbps: float | None
    external_dram_access_energy_pj_per_byte: float
    coherence_protocol: str
    # Per-arch on-chip memory hierarchy fields (sram_kib_per_unit,
    # pmu_kib_per_pcu, scratchpad_kib_per_tile, unified_buffer_size_kib)
```

**Difference**: CGRA's ``has_host_dram`` is intentionally distinct
from NPU/DPU/TPU's ``has_external_dram`` -- CGRA's DRAM is reached
via host bus (PCIe to system DDR4), while NPU/DPU/TPU have chip-
attached DRAM. Unifying the naming would lose this distinction.

Per-architecture on-chip memory fields differ structurally:
- NPU: ``sram_kib_per_unit`` (per dataflow unit)
- CGRA: ``pmu_kib_per_pcu`` (per Pattern Compute Unit)
- DPU: ``scratchpad_kib_per_tile`` (per AIE tile)
- TPU: ``unified_buffer_size_kib`` (chip-shared)

**Verdict**: KEEP_SEPARATE for v8. The structural variations and
naming distinctions are architecturally meaningful. v9 may explore
abstract mixins for the shared subset (bandwidth + coherence +
external DRAM gating pattern).

### 5. ``*ComputeFabric`` -- SKELETON SHARED, BASELINE PRECISION VARIES

```python
class XXXComputeFabric(BaseModel):
    fabric_kind: XXXFabricKind   # per-arch enum
    circuit_class: CircuitClass
    ops_per_unit_per_clock: dict[str, int]
    energy_per_op_<baseline>_pj: float  # baseline varies: INT8 / BF16
    energy_scaling: dict[str, float]
    # Plus per-arch additions:
    #   - DPU: fpga_fabric_overhead_factor
    #   - TPU: (none extra; tile energy lives on TPUBlock)
    #   - others: (none extra)
```

**Difference**: baseline precision varies (NPU/DPU/CGRA use INT8;
TPU uses BF16; GPU/CPU use FP32). Field name reflects this:
``energy_per_op_int8_pj`` vs ``energy_per_op_bf16_pj``.

**Verdict**: KEEP_SEPARATE for v8. Unifying would require renaming
the energy field (breaking change) and introducing a baseline-precision
indicator. v9 may add a ``BaselinePrecision`` enum + generic
``energy_per_op_baseline_pj`` field, but the migration is non-trivial.

## Summary table

| Pattern | Block kinds | Variation | Verdict | v8 PR 2 scope? |
|---|---|---|---|---|
| Shared primitives (4 enums/types) | All | None -- already shared | SAFE_TO_UNIFY | YES (re-export) |
| ``TheoreticalPerformance`` | 6 | None -- byte-identical | SAFE_TO_UNIFY | YES (collapse) |
| ``ThermalProfile`` | 6 | scalar clock vs ClockDomain | NEAR_UNIFIABLE | NO (defer v9) |
| ``OnDieFabric`` | 6 | per-arch topology enum | NEAR_UNIFIABLE | NO (defer v9) |
| ``MemorySubsystem`` | 4 modern | has_external_dram naming | KEEP_SEPARATE for v8 | NO |
| ``ComputeFabric`` | 6 | baseline precision varies | KEEP_SEPARATE for v8 | NO |

## Recommended ``compute_block_common.py`` (PR 2 scope)

```python
"""Vendor-neutral primitives shared across compute block kinds.

Lands in v8 (graphs#208 sprint). Centralizes the shared primitives
that 7 prior sprints (KPU/GPU/CPU/NPU/CGRA/DPU/TPU) accumulated as
ad-hoc cross-block-kind reuse.

This module is **additive only**. Existing block modules continue to
work unchanged. New code can use this module as a single import
point; existing per-block-kind types remain available via
backward-compat aliases.
"""

# Re-exports (the 4 already-shared primitives)
from embodied_schemas.process_node import (
    CircuitClass,
    DataConfidence,
)
from embodied_schemas.gpu import MemoryType
from embodied_schemas.gpu_block import ClockDomain

# Unified type (collapses the 6 byte-identical *TheoreticalPerformance)
from pydantic import BaseModel, Field, model_validator

class TheoreticalPerformance(BaseModel):
    """Per-precision peak ops/sec roll-up. Identical shape across
    all 6 modern block kinds (CPU/GPU/NPU/CGRA/DPU/TPU); collapsed
    into one definition here.

    Per-block-kind aliases (e.g. ``NPUTheoreticalPerformance =
    TheoreticalPerformance``) preserve backward compat for existing
    callers."""

    peak_ops_per_sec_by_precision: dict[str, float] = Field(...)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_positive(self) -> "TheoreticalPerformance":
        for prec, value in self.peak_ops_per_sec_by_precision.items():
            if value < 0:
                raise ValueError(
                    f"peak_ops_per_sec_by_precision[{prec!r}] = {value} "
                    f"must be >= 0"
                )
        return self


__all__ = [
    # Shared primitives (re-exported)
    "CircuitClass",
    "DataConfidence",
    "MemoryType",
    "ClockDomain",
    # Unified types
    "TheoreticalPerformance",
]
```

## Migration strategy (PR 3 proof of concept)

PR 3 migrates **NPU** as the proof of concept because:
- NPU has the most cross-block-kind reuse already (3 primitives:
  DataConfidence, MemoryType, CircuitClass)
- NPU has 3 SKUs in the catalog (Hailo-8, Hailo-10H, Coral) that
  validate the migration end-to-end
- NPU has a clean ``NPUTheoreticalPerformance`` to collapse
- NPU's loader is the pattern source for DPU/TPU loaders

Migration steps for PR 3:
1. ``npu_block.py``: replace ``class NPUTheoreticalPerformance(BaseModel): ...``
   body with ``NPUTheoreticalPerformance = TheoreticalPerformance``
2. Update imports in ``npu_block.py`` to pull from ``compute_block_common``
   (cosmetic; existing source-module imports still work)
3. Test: all 3 NPU SKU YAMLs validate unchanged
4. Test: ``from embodied_schemas import NPUTheoreticalPerformance``
   still works (backward-compat alias)
5. Test: ``isinstance(npu_perf, TheoreticalPerformance)`` AND
   ``isinstance(npu_perf, NPUTheoreticalPerformance)`` -- the alias
   preserves identity

## Backward-compat guarantees

The sprint commits to **zero breaking changes**. Specifically:
1. Every existing YAML in ``data/compute_products/`` validates
   unchanged
2. Every existing per-block-kind type name remains importable
3. Every existing isinstance check continues to work (because
   aliases share the underlying class)
4. Every existing serialized JSON round-trips unchanged
5. The graphs-side YAML loaders (npu/cgra/dpu/tpu) continue to work
   unchanged (they consume the schema; the schema's external API
   doesn't change)

These guarantees are validated by the existing 450 embodied-schemas
tests + the 21 graphs YAML parity tests. **No new test files needed
for PR 2**; existing tests + a small smoke test for the unified
``TheoreticalPerformance`` are sufficient.

## Risks called out by this exercise

1. **Alias-vs-subclass semantics.** Using ``XYZ = TheoreticalPerformance``
   means ``XYZ`` IS ``TheoreticalPerformance`` (not a subclass).
   Pydantic discriminated-union dispatch keys on field types; aliases
   should work transparently but worth verifying in PR 2.

2. **JSON serialization.** Pydantic serializes by class structure,
   not class name. Aliased types serialize identically to the unified
   type. Existing YAML loads + JSON round-trips should be unaffected.

3. **Import order.** ``compute_block_common`` imports from
   ``process_node``, ``gpu``, and ``gpu_block``. None of these import
   from ``compute_block_common``, so no circular dependency.
   ``compute_product.py`` could optionally re-export from
   ``compute_block_common`` in v9.

4. **Defer NEAR_UNIFIABLE patterns to v9.** ``ThermalProfile`` /
   ``OnDieFabric`` / ``MemorySubsystem`` / ``ComputeFabric`` all have
   real architectural variations. Premature unification loses
   precision; deferring lets v9 design the right abstractions
   incrementally.

5. **KPU is deliberately excluded.** KPU's types (KPUTileSpec,
   KPUNoCSpec, etc.) pre-date the cross-block-kind reuse pattern and
   would require their own refactor across 12 SKU YAMLs. Defer to
   v10+. The v8 ``compute_block_common`` module documents that KPU
   migration is a separate concern.

6. **Per-block-kind enum collapse is OUT OF SCOPE.** Each block
   kind's fabric/NoC topology enum is architecturally meaningful
   (NPUDataflowKind has different semantics from CGRAFabricKind).
   Collapsing them would erase information; v8 explicitly does NOT
   do this.

7. **TPU's TPUTileEnergyCoefficients stays TPU-specific.** It's the
   only architecture with a fine-grained energy decomposition
   sub-type; no other block kind has a parallel. KEEP_SEPARATE.

## Next step

Sprint sequencing:

- **PR 1 (this document)** -- docs/design, graphs-side. Pure docs PR;
  no code changes.
- **PR 2 -- compute_block_common, embodied-schemas-side.** Implements
  the "Recommended ``compute_block_common.py``" above. Additive only;
  existing types untouched. Existing tests + a small smoke test for
  ``TheoreticalPerformance`` validate.
- **PR 3 -- NPU migration, embodied-schemas-side.** Migrates
  ``NPUTheoreticalPerformance`` to alias the unified type. All 3
  NPU SKU YAMLs validate unchanged; backward-compat alias preserves
  imports.

After this sprint closes, file follow-up issues per remaining block
kind:
- CPU migration (1 SKU)
- GPU migration (2 SKUs)
- CGRA migration (1 SKU)
- DPU migration (1 SKU)
- TPU migration (1 SKU)

Each is a small PR (~50 LOC change + 0 new tests). The 5 follow-ups
can be batched together if desired.

After the unification lands, the v9 sprint can tackle:
- DSP schema extension (the last schema gap, 10 SKUs) -- DSPBlock
  uses unified primitives from day 1, avoiding the 8th repetition
  of the per-architecture-types pattern
- ``ThermalProfile`` / ``OnDieFabric`` unification (NEAR_UNIFIABLE
  patterns) -- requires careful design to handle the variations
- ``has_external_dram`` vs ``has_host_dram`` naming reconciliation
  (touches SKU YAMLs)
- KPU unification (oldest module; 12 SKUs)

# Assessment: Unified ComputeProduct Schema for embodied-schemas

**Date:** 2026-05-08
**Context:** While extending `cli/list_hardware_mappers.py` to report die size and
transistor count alongside peak throughput, we surfaced a structural problem in
`embodied-schemas`: the `DieSpec` data we want (transistors, die size, foundry,
process node) is only defined and populated for GPUs. CPU/NPU/SoC schemas don't
even have those fields. This assessment captures the architectural conversation
about whether to unify the per-category schemas into a single `ComputeProduct`
schema, and what that migration would look like.

---

## Survey results: how badly is the data fragmented today?

### `embodied-schemas` data coverage (74 total YAML files)

| Type | Files | `transistors_billion` | `die_size_mm2` | Process node |
|------|-------|-----------------------|----------------|--------------|
| GPUs | 22 | 22/22 | 22/22 | `process_nm` int + `process_name` |
| CPUs | 36 | 0/36 | 0/36 | `process_node` enum only |
| NPUs | 4 | 0/4 | 0/4 | (none) |
| SoCs/Chips | 12 | 0/12 | 0/12 | `process_node_nm` only |

GPU YAMLs are gold-standard. H100 has die size 814 mm², 80B transistors, TSMC N4,
foundry, launch date, MSRP, full FP/INT peak tables. **The schema (`DieSpec`)
only exists for GPUs**: CPU/NPU/Chip Pydantic models in embodied-schemas don't
define `transistors_billion` or `die_size_mm2` fields at all.

### `hardware_registry/` (graphs repo) coverage

51 spec.json files. **Zero** have structured die size, transistor count, or
process node. One file (B100) mentions transistors in free-text notes. Schema is
purely operational: `ops_per_clock`, `theoretical_peaks`, `peak_bandwidth_gbps`,
`compute_units`, `tdp_watts`, clocks.

### Cross-repo wiring

`grep -rn "base_id\|embodied_schemas" src/graphs/hardware/` returns zero matches.
CLAUDE.md describes `base_id` linkage to embodied-schemas as the intended
architecture, but it isn't implemented in code. No mapper has any reference to
its embodied-schemas counterpart today.

---

## Why the schemas are split today

The categorical split looks like a historical artifact of how data was
bootstrapped, not a deliberate architectural choice:

- Each vendor publishes datasheets organized by product category (Intel ARK,
  NVIDIA GPU pages, Hailo NPU pages). Mirroring those pages into category-shaped
  Pydantic models was the path of least resistance.
- Each category emphasizes different facets in vendor marketing: CPUs talk about
  P-cores/E-cores/cache levels, GPUs about SMs/tensor cores, NPUs about TOPS.
- Schemas grew up at different times. `gpu.py` got a real `DieSpec`; `cpu.py`
  only got a `process_node` enum. Nobody backfilled.

The cost is now visible:
- DieSpec only exists on GPU; the same field set isn't reachable for
  CPUs/NPUs/SoCs even though the underlying physics is identical.
- Field names diverge: `process_nm` (GPU) vs `process_node` enum (CPU) vs
  `process_node_nm` (Chip).
- Bug fixes don't propagate. A `PowerSpec` definition in `gpu.py` is independent
  of the `PowerSpec` in `cpu.py`.

---

## The industry has already moved past clean categories

Modern compute products defy the categorical split:

| Product | What is it? |
|---------|-------------|
| Apple M3 Max | CPU + GPU + NPU + media engines, one die |
| AMD MI300A | CPU chiplets + GPU chiplets, MCM |
| NVIDIA Grace-Hopper | Grace CPU + Hopper GPU, MCM via NVLink-C2C |
| NVIDIA B100/B200 | dual-die GPU bridged by NV-HBI |
| Jetson Orin | CPU + GPU + 2x DLA + PVA + NVENC, all on one SoC |
| DGX H100 | 8x H100 + dual Sapphire Rapids + NVSwitch + ConnectX, board-level |
| Tesla Dojo | training tile that's neither chip-shaped nor server-shaped |

Forcing these into "GPU" or "CPU" buckets is increasingly false labelling. The
downstream consumers (graphs/ mappers, the agentic orchestrator) ultimately care
about *capabilities and resources* -- peak FP16, BW, TDP, die area, transistor
budget -- not whether vendor marketing called it a GPU.

---

## Proposed shape: `ComputeProduct` with discriminated blocks

A unified schema works only if we **don't** flatten everything into one bag.
Three concepts:

### 1. `ComputeProduct` (the spine) -- uniform across all products

```yaml
id: nvidia_h100_sxm5_80gb
name: NVIDIA H100 SXM5 80GB
vendor: nvidia
packaging:
  form_factor: sxm5            # chip | sxm | pcie_card | m2 | board | rack
  num_dies: 1                  # 2 for B100, 8+CPU+switch for DGX
  is_chiplet: false
  package_type: monolithic     # | mcm | board | system
physical:
  die_size_mm2: 814            # sum across dies
  transistors_billion: 80
  process_node_nm: 4
  process_node_name: "TSMC N4"
  foundry: tsmc
power:
  tdp_watts: 700
  modes: [...]                 # MAXN/30W/15W for Jetson; PL1/PL2 for Intel
memory:
  on_package_gb: 80
  type: hbm3
  bandwidth_gbps: 3350
peak_throughput:               # uniform precision dict, applies to whole product
  fp64: 33.45
  fp32: 66.91
  fp16: 989.4                  # tensor-core peak
  fp8:  1978.9
  int8: 1978.9
market:
  launch_date: 2022-09-20
  launch_msrp_usd: 30000
contains: []                   # for board/rack products: refs to child products
blocks:                        # category-specific detail, see below
  - kind: gpu
    ...
```

### 2. `blocks: list[Block]` -- discriminated union for category-specific detail

```yaml
blocks:
  - kind: gpu
    streaming_multiprocessors: 132
    cuda_cores: 16896
    tensor_cores: 528
    tensor_core_gen: 4

  - kind: cpu
    p_cores: 8
    e_cores: 4
    isa: x86_64
    vector_ext: [avx2, avx512_vnni]

  - kind: npu
    dataflow: systolic
    sparsity_support: true
```

A pure GPU has one `gpu` block. Jetson Orin has `[cpu, gpu, npu, npu]`
(CPU + Ampere GPU + 2x DLA). DGX H100 has zero blocks at the top level
and a populated `contains: [h100, h100, ..., sapphire_rapids, ...]`.

### 3. Hierarchy via `contains`

Board-level products reference chip-level products. The schema is recursive:
a rack contains boards contains chips. Top-level fields like `peak_throughput`
can be computed by summing children, or stated explicitly with a note.

This handles every product type uniformly: pure chip, MCM, multi-die,
multi-chip board, full rack.

---

## What it would take

Roughly a **3-4 week project** in embodied-schemas, plus consumer migration in
graphs/ and Embodied-AI-Architect.

### Phase 1 -- Design & RFC (3-5 days)
- Pydantic models for `ComputeProduct`, `Packaging`, `Physical`, `Power`,
  `Memory`, `Block` (CPUBlock/GPUBlock/NPUBlock/DSPBlock/CGRABlock/...).
- Decide on tricky cases up-front:
  - How does a CPU-with-iGPU split? (Two blocks under one product.)
  - How does B100 dual-die look? (One product, `num_dies: 2`, one logical
    GPUBlock; or two contained chip-products.)
  - How are thermal modes represented uniformly? (GPUs: single TDP. Jetson: 4.)
  - What lives in `peak_throughput` for a CPU-with-iGPU? Sum, or per-block only?
- Migration matrix: every old field -> new home, with explicit "drop" /
  "extras" decisions for orphans.
- Tests covering ~5 reference products spanning categories before bulk
  migration.

### Phase 2 -- Migration tooling (2-3 days)
- Converter script: reads old `gpu/*.yaml`, `cpu/*.yaml`, etc., emits new
  `products/<vendor>/<id>.yaml`.
- Round-trip validation: load old, convert, load new, assert structural
  equivalence.
- Flag low-confidence conversions for human review.

### Phase 3 -- Bulk migration (~1 week)
- Run converter on all 74 existing files.
- Hand-edit converter-flagged cases.
- **Backfill the gap data while we're in the file anyway** -- die sizes and
  transistor counts for CPUs/NPUs/SoCs from datasheets / Wikichip / Anandtech.
  This is the only chance to touch all those files at once.
- New folder layout: `data/products/<vendor>/<id>.yaml` (flat).

### Phase 4 -- Update consumers (1-2 weeks)
- embodied-schemas: deprecate `GPUEntry`/`CPUEntry`/`NPUEntry` exports with
  shim adapters. Add `ComputeProduct` exports.
- graphs/: implement `mapper.physical_spec` (or `mapper.compute_product`) that
  pulls from `ComputeProduct`. Build `cli/list_hardware_resources.py`.
- Embodied-AI-Architect: update prompts or hardware-selection logic that uses
  category-keyed fields.

### Phase 5 -- Sunset (1-2 weeks lag, parallel with normal work)
- Remove deprecated Pydantic models.
- Remove old `data/{gpus,cpus,npus,chips}/` folders after consumers cut over.

---

## Risks worth flagging up front

1. **Discriminated unions in YAML are not free.** Pydantic v2 handles them
   cleanly with a `kind:` discriminator; YAML editors and humans won't validate
   the union as easily as a flat shape. Plan for editor tooling (JSON Schema
   export) so authors get autocomplete.
2. **The 80/20 trap.** ~20% of fields are genuinely block-specific. Resist
   temptation to flatten them onto the spine -- that re-creates the original
   bag-of-optional-fields problem. Block discrimination is what makes this
   schema honest.
3. **Information loss.** Some old fields don't have an obvious home (e.g., GPU
   `nvenc: false`, CPU `socket: lga1700`). Decide policy: drop, move to a typed
   `block.platform_metadata`, or keep as `extras: dict[str, Any]`.
4. **External consumers.** If anything outside graphs/ and Embodied-AI-Architect
   consumes embodied-schemas, the deprecation window needs to cover them too.

---

## Recommended sequencing relative to the immediate deliverable

The unification project is the right answer but it's a 3-4 week investment.
Don't block `cli/list_hardware_resources.py` on it. Sequence:

1. **Now**: ship `PhysicalSpec` in graphs/ and the new CLI. Hand-populate from
   datasheets for non-GPU chips. This is the consumer-side abstraction we want
   regardless of how embodied-schemas evolves.
2. **Soon**: open an embodied-schemas RFC for `ComputeProduct`. Start with phase
   1 design while consumers (graphs/) are using `PhysicalSpec` as a stable
   interface.
3. **Later**: when `ComputeProduct` lands, `PhysicalSpec` becomes a thin loader
   that reads from a `ComputeProduct` YAML. Consumer code (mappers, the agentic
   system, the CLI) doesn't change -- only the data source behind
   `PhysicalSpec` changes.

This way the consumer interface is stable, the unification project doesn't
block today's deliverable, and the work invested now isn't thrown away later.

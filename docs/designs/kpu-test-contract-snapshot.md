# KPU Test Contract Snapshot — Phase 4b PR 1

Status: snapshot complete (2026-05-10)
Owner: Theo Omtzigt + Claude Code
Driver for: Phase 4b PRs 2–5 in `kpu-sku-and-process-node-plan.md`

## Purpose

Phase 4a landed `load_kpu_resource_model_from_yaml(base_id)` as a parallel
path next to the four hand-coded `kpu_t{64,128,256,768}_resource_model()`
factories. Phase 4b replaces the factories with thin wrappers that call
the loader. Before any of that can happen, we need a **frozen contract**
of every load-bearing numeric assertion the existing 51 KPU tests make
against `HardwareResourceModel`. That spreadsheet is this document.

For each contract we record:

- **Field**: which `HardwareResourceModel` (or `TileSpecialization` /
  `KPUTileEnergyModel`) attribute the test reads.
- **Test**: the assertion's location.
- **SKU(s)**: which factory's output is asserted on.
- **Factory value**: what the existing hand-coded factory produces.
- **Loader value**: what `load_kpu_resource_model_from_yaml(base_id)`
  produces against the `embodied-schemas` YAMLs.
- **Delta class**: matches / loader-only / factory-only / numeric-mismatch.
- **Action for Phase 4b**: which of PRs 2–5 must address it.

The numbers in the "Loader value" columns were computed by running the
loader against each SKU and comparing to the live factory. Reproducer:
`docs/designs/kpu-test-contract-snapshot.repro.py` (not committed —
inline in the PR description).

## TL;DR — go-no-go for each Phase 4b PR

| PR | Scope | Blocked by this snapshot? | Notes |
|---|---|---|---|
| 2 | Loader: extend energy + fabric + memory mappings | No -- additive, no test churn | Drop in `_build_tile_energy_model` and bring `tile_energy_model.mac_energy_*` from `process_node.energy_per_op_pj`. |
| 3 | YAML: add `efficiency_factor_by_precision` per profile | No -- additive YAML field, backward compatible | Extend `KPUThermalProfile`; backfill the four SKU YAMLs from current factory `efficiency_factor` values (catalogued below in §3). |
| 4 | Reconcile INT8 ops/clock + l2_cache_total convention | **Yes** -- needs explicit decision | See §5 "Resolution decisions required" -- the factory's INT8 numbers and the loader's l2 convention are both load-bearing. |
| 5 | Replace 4 factories with thin wrappers | Yes -- depends on PRs 2 + 3 + 4 | Pure mechanical once the deltas above are resolved. |

## 1. Per-precision peak performance (TOPS / TFLOPS)

Asserted in `tests/hardware/test_kpu_t64_precision_profiles.py`. The
T64 numbers below are **exact equality** assertions
(`pytest.approx(rel=1e-6)`); other SKUs are checked indirectly through
analyzer-driven tests that loosely bound the result.

### T64 -- exact values asserted

| Precision | Factory (TOPS) | Loader (TOPS) | Delta | Action |
|---|---:|---:|---|---|
| INT4 | 132.8 | 162.2 | +22% | PR 4: reconcile (M0.5 wins) |
| INT8 |  66.4 | 118.0 | +78% | PR 4: reconcile (M0.5 wins) |
| BF16 |  33.2 |  59.0 | +78% | PR 4: reconcile (M0.5 wins) |
| FP16 |  33.2 | n/a   | loader-missing | PR 3: extend YAML to declare FP16 (= BF16) |
| FP32 |   1.5 |   6.0 | factory uses pre-M0.5; loader uses M0.5 | PR 4: reconcile |

### Cross-SKU snapshot (peak TOPS)

| SKU  | Source  | INT4   | INT8   | BF16   | FP16  | FP32  |
|------|---------|-------:|-------:|-------:|------:|------:|
| T64  | factory | 132.8  |  66.4  |  33.2  |  33.2 |   1.5 |
| T64  | loader  | 162.2  | 118.0  |  59.0  |  n/a  |   6.0 |
| T128 | factory | 524.3  | 262.1  | 131.1  |  n/a  |  n/a  |
| T128 | loader  | 382.8  | 275.3  | 137.6  |  n/a  |  14.0 |
| T256 | factory | 401.0  | 286.7  |  43.1  |  n/a  |  14.3 |
| T256 | loader  | 401.0  | 286.7  | 143.4  |  n/a  |  14.3 |
| T768 | factory | 260.2  | 130.1  |  48.9  |  n/a  |  n/a  |
| T768 | loader  | 659.9  |1181.5  | 590.7  |  n/a  |  23.7 |

### Notable patterns

- **T256 BF16**: factory says 43.1 TFLOPS, loader says 143.4 TFLOPS. The
  loader correctly counts the BF16-fallback contribution from INT8-primary
  tiles (which the YAML declares as `bf16: 400` in `ops_per_tile_per_clock`);
  the factory's `ComputeFabric` path drops it. **This is a real bug in the
  factory's numbers, not a stale-pre-M0.5 artifact.** PR 4 should adopt the
  loader value as truth.
- **T128 INT4**: factory 524.3 TOPS, loader 382.8 TOPS. Factory's larger
  number includes BF16-tile and Matrix-tile INT4 contributions that the
  YAML doesn't declare (Matrix tiles in YAML have only `int8` and `bf16`,
  no `int4`). PR 4 decision: extend YAML or accept lower number.
- **T768**: factory 9× lower across the board. Factory's `ComputeFabric`
  uses tiny per-tile op counts that don't account for the Matrix tile's
  systolic op density (8192 INT8/clock per Matrix tile, per the YAML). PR 4
  must adopt loader values; the factory numbers are misleadingly small.
- **FP16**: factory exposes FP16 on T64; loader doesn't. KPU FP16 is
  mechanically the same path as BF16. PR 3 should extend the YAML to
  declare FP16 alongside BF16 wherever a tile supports BF16.

## 2. Architectural shape (compute_units, caches, memory)

| SKU  | Source  | compute_units | l1/unit | l2_total | main_mem | tile_dims |
|------|---------|--------------:|--------:|---------:|---------:|-----------|
| T64  | factory |  64           | 256 KiB |  4 MiB   |  8 GiB   | (32, 32)  |
| T64  | loader  |  64           | 256 KiB | 16 MiB   |  8 GiB   | (32, 32)  |
| T128 | factory | 128           | 256 KiB |  8 MiB   | 16 GiB   | (32, 32)  |
| T128 | loader  | 128           | 256 KiB | 32 MiB   | 16 GiB   | (32, 32)  |
| T256 | factory | 256           | 256 KiB | 16 MiB   | 32 GiB   | (20, 20)  |
| T256 | loader  | 256           | 256 KiB | 64 MiB   | 32 GiB   | (20, 20)  |
| T768 | factory | 768           | 256 KiB | 32 MiB   | 64 GiB   | ( 16, 8)  |
| T768 | loader  | 768           | 256 KiB |192 MiB   | 64 GiB   | ( 16, 8)  |

### Contracts touching these fields

- `tests/hardware/test_kpu_weight_stationarity.py:114` -- T64 expects
  `64 * 256 KB + 4 MB = 20 MB` aggregate on-chip. **Loader breaks this**
  (loader gives `64 * 256 KB + 16 MB = 32 MB`). PR 4 reconciliation
  needed: the loader's `l2_cache_total` convention is the chip-wide L3
  total; the factory's convention was a smaller "shared L2" number.
  Decision: which is l2_cache_total?
- `tests/hardware/test_kpu_domainflow_tile.py:35,44,55` -- T64/T128 = 32×32,
  T256 = 20×20 PE arrays. **Loader matches**. No action.
- `tests/hardware/test_kpu_domainflow_tile.py:214` -- `T128.compute_units == 128`.
  Both match.

## 3. Per-(profile, precision) efficiency factors

The factory hand-codes per-(thermal_profile, precision) `efficiency_factor`
values (range 0.65–0.85). The loader uses a flat 0.70 placeholder. No test
asserts a specific efficiency value directly, but downstream analyzer-driven
tests (`test_kpu_invariants.py`, `test_vit_b16_int8_on_kpu_t64_meets_stationarity_target`)
rely on these being right.

PR 3 needs to extend `KPUThermalProfile` with an optional
`efficiency_factor_by_precision: dict[str, float]` field and backfill the
four YAMLs with the values currently in the factories. Specific numbers to
backfill (from inspection of `kpu_t{64,128,256,768}.py`):

| SKU | Profile | Precision | efficiency_factor | tile_utilization |
|---|---|---|---:|---:|
| T64 | 3W | INT8 | 0.65 | 0.90 |
| T64 | 3W | BF16 | 0.60 | 0.85 |
| T64 | 6W (default) | INT8 | 0.70 | 0.92 |
| T64 | 6W | BF16 | 0.65 | 0.88 |
| ... | (see kpu_t64.py:204-265 for the full table per SKU) | | | |

## 4. Tile-level energy contracts

`tests/hardware/test_kpu_tile_energy.py` asserts:

| Test | Assertion | Source field |
|---|---|---|
| `test_kpu_t64_small_gemm:187` | `0.3e-12 < energy_per_mac_j < 1.5e-12` | `tile_energy_model.compute_gemm_energy(...)` |
| `test_kpu_t256_medium_gemm:216` | same range | same |
| `test_kpu_t768_large_gemm:245` | `0.4e-12 < energy_per_mac_j < 1.5e-12` | same |
| `test_fusion_benefits:295` | `savings_j > 0` | same |
| `test_batch_scaling:342` | `energy_per_inf[3] < energy_per_inf[0]` | same |
| `test_product_comparison:375` | `r768 < r64` (energy/MAC) | same |

These all read from `model.tile_energy_model` which is a `KPUTileEnergyModel`
instance. The loader **does not** populate `tile_energy_model` today --
PR 2 must add `_build_tile_energy_model(sku, node)` that derives:

- `mac_energy_int8/bf16/fp32` from `process_node.energy_per_op_pj["{class}:{precision}"]`
- `dram_read_energy_per_byte` from `_MEM_PHY_PJ_PER_BYTE_BY_TYPE[memory_type]`
- `l1/l2/l3_read_energy_per_byte` from a per-process-node SRAM table
- `num_tiles`, `pes_per_tile`, `tile_mesh_dimensions` from `kpu_architecture`

After PR 2 lands, the loader-produced model will satisfy these range
assertions because the energy values come from PDK figures (which already
sit in the documented range).

## 5. Cross-mapper invariants

`tests/validation_model_v4/test_kpu_invariants.py` runs the same
invariant set against every KPU mapper:

| Test | Asserts | Loader-affected? |
|---|---|---|
| `test_roofline_self_consistency` | RooflineAnalyzer self-agreement | No -- analyzer logic, not factory data |
| `test_latency_non_decreasing_in_size` | latency monotonicity | No |
| `test_memory_time_scales_with_bytes` | bandwidth math | Depends on `peak_bandwidth` -- both produce the same value |
| `test_achieved_compute_below_peak` | achieved ≤ peak | Loader changes peak (§1); could shift achieved ≤ peak comparison |
| `test_achieved_bw_below_peak` | achieved ≤ peak BW | No |
| `test_avg_power_below_tdp_known_violated` | xfail per #81 | No |
| `test_family_latency_non_increasing` | T64 ≥ T128 ≥ T256 ≥ T768 latency | Yes -- if loader changes peak ops, family ordering may shift |

The family-monotonicity test is the one to watch: if the loader's T768 peak
(1181 INT8 TOPS) replaces the factory's 130 TOPS, T768 will become much
faster than the factory said it was, which is the **right** outcome but
might break tests that compared T768 to a baseline calibrated to the older,
smaller number.

## 6. Resolution decisions required (the PR 4 spec)

Phase 4b PR 4 makes three explicit decisions. Recommendations in **bold**.

### Decision 1: T64 INT8 truth (and the rest of T64's per-precision values)

- **Adopt M0.5 (loader/YAML) values across the board.** Update
  `test_kpu_t64_precision_profiles.py` to assert the loader's numbers (118
  INT8, 162 INT4, 59 BF16, 6 FP32). Add FP16 = 59 once the YAML carries
  it (PR 3).
- Reasoning: the factory's `ComputeFabric` block carries pre-M0.5 16×16
  array math; its own `TileSpecialization` block in the same file already
  uses 32×32. The YAML, the validator framework, and the tile-mix
  consistency check all agree on M0.5. The factory's `ComputeFabric` is
  the inconsistent outlier.

### Decision 2: `l2_cache_total` convention

- **Keep the loader's "chip-wide L3 total" convention.** Update
  `test_kpu_weight_stationarity.py:114` to expect `64 * 256 KB + 16 MB =
  32 MB` (= `total_tiles × l3_kib_per_tile + 0` since L2 is per-tile, not
  shared).
- Reasoning: the factory's "4 MB shared L2" doesn't correspond to any
  silicon block in the M0.5 KPU architecture (L2 is per-tile, 32 KiB ×
  256 = 8 MiB; the chip-wide pool is L3 at 64 MiB). The factory number was
  a placeholder; the loader's number reflects the YAML's documented memory
  hierarchy.
- Alternative considered: introduce separate `l2_cache_per_tile` +
  `l3_cache_total` fields on `HardwareResourceModel`. Larger refactor;
  defer to a future cleanup.

### Decision 3: Cross-SKU peak ops normalization

- **Adopt loader values across all SKUs.** The largest deltas (T768 9×,
  T256 BF16 3.3×) are real bugs in factory accounting that the loader
  fixes. Tests that depend on the factory's old numbers (the family-latency
  invariant in particular) need re-baselining against the loader values.

## 7. Estimated test churn for PR 4

If decisions 1-3 are accepted as recommended:

- `test_kpu_t64_precision_profiles.py`: ~6 assertion updates
- `test_kpu_weight_stationarity.py`: 1 assertion update (line 114) +
  re-verify the ViT-B/16 latency band (currently 1.0–2.5 ms)
- `test_kpu_invariants.py::test_family_latency_non_increasing`: re-baseline
  against loader-produced peaks; possibly widen tolerance bands
- Other files: spot-checks rather than systematic updates expected

Total: ~10-15 test assertions to update. Mechanical given this snapshot.

## Out of scope for PR 1

This document records contracts and recommends decisions; it changes no
code or tests. PR 2-5 implement the changes. The `tile_energy_model`-level
contracts in §4 are summarized but not enumerated assertion-by-assertion
because PR 2's `_build_tile_energy_model` will reproduce them by
construction (energies come from the PDK/process-node figures, which are
already in the documented ranges).

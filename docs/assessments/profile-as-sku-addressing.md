# Assessment: Profile-as-SKU Addressing for Power-Modulated Edge SoCs

**Date:** 2026-05-08
**Tracking issue:** [#136](https://github.com/branes-ai/graphs/issues/136)
**Phase 1 PR:** [#137](https://github.com/branes-ai/graphs/pull/137)
**Status:** Phase 1 implemented; Phases 2-5 outlined.

---

## Background

While building out `PhysicalSpec` (PR #133) and the `cli/list_hardware_resources.py` spec-sheet view (PR #134), and then populating the four Jetson SKUs (PR #135), a structural question surfaced: how do we represent the fact that the same silicon ships in multiple persistent power profiles?

For embodied AI, this isn't a corner case -- it's the dominant deployment pattern. The Jetson Orin Nano alone has 4 power modes (7W, 15W, 25W, MAXN). Each mode:

- Is set via `nvpmodel` -- writes to the device's flash storage
- Requires a reboot to commit the new mode
- Locks CPU / GPU / DLA frequencies into a specific bin
- Yields materially different peak FP16, peak INT8, sustained throughput
- Implies different cooling solution (passive vs active fan vs thermal headroom)
- Implies different battery life on mobile platforms
- Has different software stack guarantees (some CUDA features are gated behind certain power profiles)

From the **deployment perspective**, "Jetson Orin Nano in 7W mode" and "Jetson Orin Nano in 15W mode" are different products that the agentic system should reason about distinctly. The roboticist deciding whether their drone-payload AI fits in 7W envelope or needs 15W shouldn't be asking "Orin Nano" -- they should be asking "Orin Nano at which mode?".

But from the **data perspective**, chip-level fab attributes (die size, transistor count, process node, foundry, architecture, launch date, MSRP) are identical across all modes. They're properties of the silicon, not properties of the runtime configuration.

This document captures how we resolved that tension.

---

## What the repo had before this work

Substantial pieces of the per-profile data layer already existed:

- **`ThermalOperatingPoint`** (`src/graphs/hardware/resource_model.py:733`) represents one mode -- `name`, `tdp_watts`, `cooling_solution`, and `performance_specs: Dict[Precision, PerformanceCharacteristics]`.
- **The Orin Nano resource model** (`src/graphs/hardware/models/edge/jetson_orin_nano_8gb.py`) enumerates the four canonical Orin Nano Super modes -- `7W`, `15W`, `25W`, `MAXN` -- with full per-profile clock and peak data populated from NVIDIA's published specs. (The original Phase 1 implementation registered three; the `25W` mode was added when reconciling this doc with the user-directed mode count.)
- **`HardwareMapper.__init__`** accepts `thermal_profile` to select which mode is active for the returned mapper.
- **`HardwareResourceModel.default_thermal_profile`** designates a sensible default for "give me one number" queries.

So the **per-profile data** was reachable; it just wasn't **addressable**. Three concrete gaps:

1. **No alias-style lookup**: `get_mapper_by_name("Jetson-Orin-Nano-8GB-7W")` returned `None`. The profile was a separate kwarg.
2. **No SKU enumeration that included profiles**: `list_all_mappers()` returned 46 silicon-bin names. The agentic system couldn't enumerate "all addressable deployment targets" without manually walking each mapper's `thermal_operating_points`.
3. **No clean home for module-level fields that aren't profile-dependent**: `memory_type` (LPDDR5 / HBM3 / etc.) and `memory_bus_width_bits` are silicon-bin properties (don't change across nvpmodel modes) but had no structured location in `PhysicalSpec`.

---

## The user-side ergonomics question

The original suggestion was: **make each (silicon × profile) a separate registry entry**. So instead of `Jetson-Orin-Nano-8GB`, the registry would have `Jetson-Orin-Nano-8GB-7W`, `Jetson-Orin-Nano-8GB-15W`, `Jetson-Orin-Nano-8GB-MAXN` as 3 distinct entries. Each gets its own full `PhysicalSpec`.

This was the right deployment-perspective intuition: power profiles ARE SKU-like for embodied AI. The question was the data-layer realization.

### Why "duplicate the registry entry" didn't survive review

Three concrete costs:

1. **Data duplication.** Across the registry, ~5 of the 46 mappers have multiple thermal profiles. Each has 3-4 modes. So ~5 × ~3.5 = ~17.5 SKU records would replace ~5 silicon-bin records. Each profile-SKU would carry the full `PhysicalSpec` -- die size, transistors, foundry, process node, architecture, launch date, MSRP, packaging metadata. That's ~10 fields × ~17.5 records = ~175 redundant cells, where ~140 are exact-copy duplications.

2. **Drift risk.** When NVIDIA discloses a new transistor count or someone corrects a die-size measurement, you'd have to update all profile variants in lockstep. Single source of truth says: chip-level data should live in exactly one place.

3. **Adding a new profile becomes a copy-paste exercise.** If NVIDIA ships a new mode (e.g., a hypothetical 50W envelope on a future Super refresh), you'd construct a brand-new full-spec entry. With the alias model, you just add a new `ThermalOperatingPoint` to the existing resource model and the alias machinery picks it up automatically. (Concrete example: the `25W` mode was added to Orin Nano under issue #136 simply by registering one extra `ThermalOperatingPoint`; no registry, factory, or PhysicalSpec changes were needed.)

### Why "leave it as-is and have callers know about thermal_profile" didn't survive either

The user's deployment intuition was correct. From the orchestrator's perspective, "give me the deployment SKU we should provision against" needs an answer that includes the power mode, because that's what the customer is deploying onto. Forcing every consumer to know to pass `thermal_profile=` is poor ergonomics, and it makes "list all SKUs we support" awkward (you'd have to walk per-mapper and union with thermal_operating_points keys).

---

## The chosen design: aliases at the registry layer

The registry continues to hold ONE entry per silicon-bin -- one PhysicalSpec, one factory, one set of category/vendor metadata. **Profile addressability is provided through alias resolution at lookup time**, not through duplicated registry entries.

Two alias forms are accepted:

| Form | Example | When to use |
|------|---------|-------------|
| Explicit separator | `Jetson-Orin-Nano-8GB@7W` | Unambiguous; recommended for code, scripts, agentic-system enumeration |
| Dashed | `Jetson-Orin-Nano-8GB-7W` | Convenience for human typing; resolved by longest-prefix-match against registry keys |

`get_mapper_by_name` resolves either form by:

1. Looking the name up directly in the silicon-bin registry. If hit -> use legacy code path with optional `thermal_profile` kwarg.
2. Otherwise, parse as `{silicon}@{profile}` first (explicit separator wins). Validate the profile against the silicon's registered `thermal_operating_points` before resolving.
3. Otherwise, parse as `{silicon}-{profile}` by trying every registered silicon-bin as a longest-prefix candidate.
4. If neither resolves, return `None` (preserves existing not-found semantics).

`list_all_skus(include_profile_aliases=True)` enumerates the silicon-bins plus one alias per registered profile, using the explicit-separator form. The agentic system iterating "all available deployment targets" calls this; legacy callers continue to use `list_all_mappers()` and see only silicon-bins.

### Why the kwarg always wins over an alias-encoded profile

If a caller passes BOTH a profile-suffixed name AND `thermal_profile=`, that's ambiguous. We resolve in favor of the kwarg because it's the more explicit user signal -- they specifically typed the parameter name. This is documented in `get_mapper_by_name`'s docstring and locked in by `TestKwargPrecedence` in `tests/hardware/test_profile_aliases.py`.

### Why `@` is the right separator

The alternatives are:
- `_` (underscore): would clash with profile names like `MAXN_Super` if vendors used underscores.
- `-` (dash): already used inside both silicon names (`Jetson-Orin-Nano-8GB`) and profile names (`MAXN-Super`). Parsing requires longest-prefix-match, which works but is not unambiguous.
- `:` (colon): unfriendly to filesystem-aware tools and copy-paste from URLs.
- `@`: not used anywhere else in our naming, can't appear in either silicon names or profile names. Matches conventions like Docker image references (`name@digest`) and `pip install package@version`.

`@` wins by elimination. The dashed form is supported as a convenience but the explicit form is what `list_all_skus` emits and what we recommend in documentation.

---

## Where each field belongs

| Field | Layer | Why |
|-------|-------|-----|
| `die_size_mm2`, `transistors_billion`, `process_node_nm`, `process_node_name`, `foundry`, `architecture` | `PhysicalSpec` (silicon-bin) | Properties of silicon. Unchanged across modes. |
| `launch_date`, `launch_msrp_usd` | `PhysicalSpec` (silicon-bin) | Module-level commercial property. Unchanged across modes. |
| `num_dies`, `is_chiplet`, `package_type` | `PhysicalSpec` (silicon-bin) | Packaging is silicon-level. |
| `memory_type` (LPDDR5 / HBM3 / etc.) | `PhysicalSpec` -- **to be added in Phase 3** | Module SKU constant. Doesn't change per nvpmodel mode. |
| `memory_bus_width_bits` | `PhysicalSpec` -- **to be added in Phase 3** | Module SKU constant. |
| `tdp_watts`, `cooling_solution` | `ThermalOperatingPoint` (already there) | Defines the mode. |
| Core base / boost clocks | `PerformanceCharacteristics` per `ThermalOperatingPoint` (already there as `core_frequency_hz` / `sustained_clock_hz`) | Varies by mode. |
| `memory_clock_mhz` | `ThermalOperatingPoint` -- **to be added in Phase 4** | Varies by mode on Jetson (lower power modes throttle DRAM clock). |
| Peak throughput per precision | `ThermalOperatingPoint.performance_specs` (already there) | Mode-dependent. |
| `name`, `category`, `vendor`, `description` | Registry metadata (already there) | Identity. |

The principle: **silicon properties on `PhysicalSpec`, mode properties on `ThermalOperatingPoint`**. Memory clock straddles the line because Jetson throttles DRAM in low-power modes, so it goes per-mode. Memory type is fixed by the package, so it stays at the silicon-bin level.

---

## Implementation phases

### Phase 1 (this commit, PR #137) -- Profile addressability via registry aliases

- `get_mapper_by_name` resolves both `{silicon}@{profile}` and `{silicon}-{profile}` forms.
- `list_all_skus(include_profile_aliases=True)` enumerates all addressable forms.
- 20 new tests in `tests/hardware/test_profile_aliases.py`.
- Backward-compatible: legacy callers see no behavior change.

### Phase 2 -- Spec-sheet expansion

`cli/list_hardware_resources.py` gains a `--profiles {default, all}` flag. `default` (current behavior) shows one row per silicon SKU using the default profile; `all` expands to one row per (silicon × profile) combination, with new columns for Mode, Base MHz, Boost MHz, Memory Clock, Memory Type. The `name` column shows the alias form when expanded so output is self-describing.

### Phase 3 -- Promote module-level fields to PhysicalSpec

Add `memory_type: Optional[str]` and `memory_bus_width_bits: Optional[int]` to `PhysicalSpec`. These are silicon-bin constants that today live nowhere structured (only embedded in resource_model internals). Backfill on H100 + the 4 Jetson SKUs we've populated so far.

### Phase 4 -- Add `memory_clock_mhz` to `ThermalOperatingPoint`

The base/boost core clocks already live in `PerformanceCharacteristics` per profile. Add memory clock as a typed field on `ThermalOperatingPoint` so renderers can read it without diving into PerformanceCharacteristics indirection. Per-profile because Jetson 7W mode throttles DRAM (~2133 MHz vs 3200 MHz at 15W).

### Phase 5 -- Update Embodied-AI-Architect orchestrator

The orchestrator should recommend full silicon-profile SKUs (`Jetson-Orin-Nano-8GB@7W`), not bare silicon names. This is a downstream-consumer change in a separate repo; the `list_all_skus()` API in this repo is what enables it.

---

## Cross-vendor applicability

The alias mechanism is a clean fit for vendors that ship enumerable persistent profiles (NVIDIA Jetson). It's inert for the rest:

| Vendor / Type | Profile semantics | Alias behavior |
|---------------|-------------------|----------------|
| NVIDIA Jetson (Orin, Thor) | nvpmodel; persistent boot-time configuration | Multiple aliases per silicon-bin |
| NVIDIA datacenter GPUs (H100 PCIe vs SXM5) | Form-factor + TDP are both fixed per SKU | One alias per silicon-bin (same as silicon-bin name + default profile) |
| Apple Silicon (M-series) | Dynamic thermal throttling; no enumerable profiles | One alias (default profile) |
| Server CPUs (Intel Xeon / AMD EPYC) | cTDP-up / cTDP-down via BIOS; coarse and platform-dependent | Could expose 2-3 aliases if the resource model registered them, but typically just one |
| Mobile SoCs (Qualcomm, MediaTek) | Dynamic; device-tree configurable in some Linux deployments | One alias per registered profile |
| Research accelerators (KPU, Plasticine v2) | Single TDP point in spec | One alias |

The mechanism doesn't impose itself where it's not useful: any chip with one thermal_operating_point gets exactly one alias (`{silicon}@{default_profile}`), and the bare silicon-bin name continues to be the canonical address. So Phase 1 doesn't churn the addressing surface for the ~41 chips that aren't power-modulated -- it just unlocks proper addressing for the ~5 that are.

---

## Decisions captured for later

These are open questions deferred to Phases 2-5 or to a future RFC:

- **Should we deprecate the dashed alias form?** The `@` form is unambiguous; the dashed form requires longest-prefix-match parsing. We support both for now because the dashed form is what the user originally proposed and it's friendlier to type. If we see the parsing become a maintenance burden, we can deprecate dashed in a future major version.
- **Should aliases include short forms or auto-generated TDP-based names?** Currently aliases use whatever's in `thermal_operating_points` as the key. We've already encountered the collision case in practice: Orin Nano's `25W` and `MAXN` modes both have `tdp_watts=25.0` (they differ in DVFS behavior, not envelope). If we ever auto-generate TDP-based aliases like `@25W` from raw watts, that approach would need a tiebreaker or compound key (e.g., `@25W-dvfs`, `@25W-locked`). Punted to a later phase.
- **Should `default_thermal_profile` be discoverable from the alias?** Today, bare `Jetson-Orin-Nano-8GB` returns the default profile silently. We could emit `Jetson-Orin-Nano-8GB@default` as an alias, but that adds noise. Status quo stands until someone asks for it.
- **How does this interact with the `embodied-schemas` ComputeProduct unification (RFC 0001)?** The `ComputeProduct` shape will need to express modes. We could put modes in a new `power_modes: list[PowerMode]` field at the spine level (mirroring `ThermalOperatingPoint` here), or as a property of each `Block` if modes are sometimes block-scoped. Decision deferred to Phase 1 of the unification RFC.

---

## Test artifacts

- `tests/hardware/test_profile_aliases.py` (20 tests, this commit)
- `tests/cli/test_list_hardware_resources.py` -- two stale assertions fixed (PR #135 invalidated H100-is-the-only-populated-GPU assumption when populating the 4 Jetsons; rewrote to test the actual contract that populated rows precede missing rows in any sort direction)

---

## Related work

- #133 (merged) -- PhysicalSpec foundation
- #134 (merged) -- `cli/list_hardware_resources.py` spec-sheet CLI
- #135 (merged) -- Backfill PhysicalSpec for 4 Jetson SKUs
- #136 (open umbrella) -- This work, full plan
- #137 (this PR) -- Phase 1 implementation
- #132 (open) -- Switch PhysicalSpec source to embodied-schemas ComputeProduct loader; ComputeProduct will need to express modes too
- embodied-schemas RFC 0001 (merged as #7) -- Unified ComputeProduct schema; the alias mechanism here is what the orchestrator will use to recommend full silicon-profile SKUs once the schema lands

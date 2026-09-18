# SoC study Phase 2: IP library, composition, die area -- execution plan

Tracking: branes-ai/graphs#269, Phase 2. Parent plan:
`docs/plans/soc-microarchitecture-study-plan.md`, section 7. Started 2026-09-18.

## Acceptance (from the parent plan)

1. An `orin_class_reference` design at `samsung_8lpp` lands within **15%** of
   Jetson AGX Orin's **455 mm^2 / 17 B transistors**. **Open by decision
   (P2-D7):** not achievable from public data. Of the target, 17 B is
   NVIDIA's own figure; 455 mm^2 is one third-party teardown.
2. Its GPU and DLA peak INT8 TOPS match the datasheet.
3. Re-run at `tsmc_n7` and `tsmc_n5`, area and energy fall monotonically, and
   IO / analog visibly scale worse than logic.

## PR sequence

| PR | Covers | Content |
|---|---|---|
| A | 2.1 + 2.2 | `hardware/soc/{ip_block,design,compose}.py`, `soc_designs/{ip,designs}/`, the sourced IP YAMLs, `compose_soc()` -> `SoCInstance`, and `silicon_math` generalized to dies without a KPU block and to multiple dies. The Orin reconstruction test lives here, because it is the first thing composition can be checked against. |
| B | 2.3 | `clocking.py` (alpha-power-law fmax per node and Vdd) plus the embodied-schemas release adding optional node speed fields. |
| C | 2.4 | `cli/show_soc.py`, the PHY-shoreline validator, and `validate_sku.py --soc`. |

2.1 and 2.2 ship together: an IP library cannot be checked until it composes,
and the Orin reconstruction is that check.

## Decisions made while executing (deviations from the parent plan)

### P2-D1. `to_compute_product()` moves to Phase 3

The parent plan has `SoCInstance.to_compute_product()` in 2.2, so the existing
validators run unchanged on the result. A `ComputeProduct` requires typed
blocks and a `Power` section: a `GPUBlock` needs `compute_fabrics`, `noc`,
`memory`, warp and wave figures, a `DSPBlock` needs thermal profiles, and
`Power` needs thermal profiles and TDPs. Phase 2 has none of that -- power is
Phase 3 -- and filling it with placeholders would put invented figures into a
catalog-shaped object, the "third source of truth" the parent plan warns
against in section 2.

So Phase 2's artifact is `SoCInstance` itself, and its validators run on it
directly. `to_compute_product()` lands in Phase 3 with the power model that
supplies the missing half.

### P2-D2. Per-block W/mm^2 moves to Phase 3

It needs per-block power. The PHY-shoreline check stays in 2.4: it is purely
geometric.

### P2-D3. IP budgets are sourced independently and never tuned to Orin

Third-party IP transistor budgets are not public. Each IP silicon line cites
an **independent** anchor -- a die shot, a vendor disclosure, a different chip
-- and its confidence. The Orin reconstruction then reports per-block and
total error. A miss outside 15% is reported, not fitted: tuning the budgets
until Orin lands would make the acceptance test circular.

### P2-D4. The anchor is the teardown's die, not the catalog's silicon bin

`nvidia_jetson_agx_orin_64gb` in the catalog declares 455 mm^2 / 17.0 B, but
its own silicon bin prices to **579 mm^2 at samsung_8lpp** (+27%). Most of the
excess is a 1,200 Mtx LPDDR5 PHY priced as 200 mm^2 of analog and 200 Mtx of
IO pads priced as 111 mm^2, while 10.3 B of the 17 B sits in one
`other_tegra_soc_blocks` lump. The acceptance targets 455 / 17 directly, and
the discrepancy is reported to the catalog rather than inherited.

**Correction, 2026-09-18 (revised the same day):** the two figures have
different provenance.

- **17 B transistors is NVIDIA's own figure.** NVIDIA's DRIVE AGX Orin press
  release (17 December 2019) states the Orin SoC "consists of 17 billion
  transistors"; the DRIVE and Jetson parts use the same SoC. The first
  research pass missed this and attributed both figures to a teardown;
  CodeRabbit caught it on #303, and the press release was checked directly.
- **455 mm^2 is not.** NVIDIA's documents -- the Jetson AGX Orin Technical
  Brief (TB_10749-001 v1.2), the datasheet and the press release -- do not
  state the die area. It traces to one third-party teardown by TechAnaLye,
  reported via @SkyJuice60 in June 2022. No second independent measurement was
  found; TechInsights DFR-2206-809 (paid) is the likely stronger source.

### P2-D5. Peaks are dense

Vendor INT8 figures for Orin are quoted with 2:4 structured sparsity, which
doubles them by construction. `IPCompute.ops_per_clock` holds dense peaks, and
the datasheet comparison halves the sparse figure rather than doubling ours.

### P2-D6. "Energy falls monotonically" means per-op energy, until Phase 3

Phase 2 has no power roll-up. Energy per op per engine at a node comes straight
from `ProcessNodeEntry.energy_per_op_pj` -- ALU only, so it understates CPU and
GPU energy until the Phase 3 architectural overhead lands -- and that is what
the monotonic check covers. It is labelled as ALU-only wherever it is shown.

### P2-D7. The area criterion stays open; the composition is a lower bound

Research for independent anchors (2026-09-18) found credible public areas for
three blocks only -- NVDLA **v1** synthesis results, a 65 nm still-camera ISP
and a 28 nm HEVC encoder -- and none for the blocks that carry Orin's die: the
Ampere SM, the Cortex-A78AE core, NVDLA **v2**, the PVA, the LPDDR5, MIPI and
PCIe PHYs, or 8LPP SRAM density. Those three anchors are in the library as
their own IPs, not passed off as Orin's.

So `IPSilicon` can be **unanchored**: it names its library and states what is
missing and where it could come from, but carries no figure. A composition with
any unanchored line reports `complete = False` and lists every gap, and its
areas are lower bounds. The Orin design prices only its SRAM -- derived from
NVIDIA's published capacities as 6T cells -- at **14.6 mm^2 / 1.58 B, about 3%
of the target**, with 14 named gaps.

Decided with the user: ship Phase 2 with the machinery, the passing TOPS
check and the lower-bound composition, and leave criterion 1 open on #269 until
anchors exist. **Nothing is estimated to make it pass**: an estimate chosen
after seeing 455 would make the check circular. The obvious route to closing it
is the TechInsights report, which tabulates Orin's functional-block sizes.

Criteria 2 and 3 pass: GPU 85.2 and DLA 52.4 dense INT8 TOPS against 85.0 and
52.5 derived from NVIDIA's own 275 / 105 sparse figures, and the priced area and
per-op energy fall monotonically from 8LPP to N7 to N5.

### P2-D8. Clocks retarget along foundry-stated relations only; no Vdd law

The parent plan proposed optional node speed fields in embodied-schemas and an
alpha-power law, fmax(Vdd) ~ (Vdd - Vth)^alpha / Vdd. Research (2026-09-18)
found:

- **Sourced:** TSMC's own iso-power speed gains for N16 -> N7 (~30-35%; two
  TSMC documents disagree), N7 -> N5 (15%) and N5 -> N4P (11%).
- **Not found:** threshold voltages for any of these FinFET nodes, a sourced
  alpha exponent for sub-20 nm devices, Vdd ranges for N5, 8LPP or GF 12nm,
  a Samsung 10LPP -> 8LPP speed figure, and any Samsung-vs-TSMC relation (no
  foundry benchmarks against a competitor).

So `clocking.retarget_fmax` walks only the stated relations, carries each as
a low-high range so the TSMC disagreement stays visible, and quotes the
conservative end. Where no chain of relations connects two nodes it returns
None, and composition marks the block `unretargetable` and lists it in
`off_reference_clocks` instead of inventing a clock. The Vdd law is not
modeled: with Vth and alpha unsourced it would have two free parameters and
no anchor.

With three relations, all TSMC, the table stays graphs-local in
`soc_designs/node_speed.yaml` rather than becoming an embodied-schemas schema
change (parent-plan decision D2: graphs-local first, upstream in Phase 5).

Consequence for the acceptance design: Orin's GPU, DLA and CPU clocks are
8LPP figures, and nothing connects 8LPP to TSMC, so its N7 and N5 peaks are
provisional and flagged. Its 8LPP peaks -- the ones criterion 2 checks -- are
unaffected.

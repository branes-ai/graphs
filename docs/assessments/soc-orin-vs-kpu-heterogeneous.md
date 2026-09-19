# Orin-class vs KPU-heterogeneous at N16 / N7 / N5: what the data decides

graphs#269 PR 4.3, the first Phase 4 study. 2026-09-18.
Study: `soc_designs/studies/orin_vs_kpu_heterogeneous.yaml`.
Reproduce: `python cli/sweep_soc.py --study orin_vs_kpu_heterogeneous --pareto area,power --union`.

## The question and the answer

The question is whether an Orin-class SoC is beaten on the autonomy workload
by the same platform with a heterogeneous KPU in place of its GPU, DLAs and
PVA, at TSMC N16, N7 and N5.

**On the data this repository holds today, the answer is: not decidable,
except for one thing.** Neither design can meet far flight on its memory
system alone. Every other comparison, on area, compute and power, is
between lower bounds. So the study reports it as undecided and names no
winner. This follows the strict rule (Phase 2 P2-D3, Phase 3 P3-D1 and
P3-D8): a gap is reported, never filled.

What the study does establish is exactly which data would decide it. That
is listed at the end.

## The designs

| | `orin_class_reference` | `kpu_heterogeneous_h64` |
|---|---|---|
| Accelerators | 16 Ampere SMs, GPU L2, 2 NVDLA v2, PVA | One H64 KPU core, generated from ComputeProduct `kpu_h64_auto1_lp5x4_7nm_tsmc_hpc` |
| Everything else | 3 x 4 Cortex-A78AE, ISP, codec, system cache, safety island, 256-bit LPDDR5 (204.8 GB/s), IO, fabric | Identical IP, so differences come from the accelerators |
| Layout | whitespace 0.15, IO ring 0.3 mm | Same |

The KPU core is built by `tools/generate_kpu_ip.py` from the SKU's own
`silicon_bin` and tile table:

- Its peak is exact: 81,920 INT8 ops per clock (24 PE tiles x 2,048 plus 4
  systolic tiles x 8,192) and 24,576 FP16 ops per clock.
- It has **no FP32**, so every stage with a Class C share fails closed onto
  the CPU. That is 14 of the 19 stages (`soc_designs/mappings/`).

The SKU also leaves five tile classes unpriced. The LNS PEs, the systolic
array and the ISP, SGM and VIO fixed-function tiles have no `silicon_bin`
line. The generator emits them as unanchored lines, so the KPU is a lower
bound too.

## Results

### Die area: lower bounds on both sides, no comparison

| Node | Orin-class die (LB) | Orin accelerators priced | KPU-het die (LB) | KPU core priced |
|---|---|---|---|---|
| N16 | 37.3 mm^2 | 10.9 mm^2 (SRAM only) | 48.4 mm^2 | 19.6 mm^2 (partial) |
| N7 | 9.9 mm^2 | 2.3 mm^2 | 13.7 mm^2 | 5.0 mm^2 |
| N5 | 7.8 mm^2 | 1.7 mm^2 | 9.8 mm^2 | 3.1 mm^2 |

Orin's accelerators are priced only in their NVIDIA-stated SRAM; the SM,
DLA and PVA logic are unanchored (Phase 2). The KPU core is priced except
for its five tile classes. **A smaller lower bound is not a smaller die.**
The KPU-het figure is larger here only because more of its silicon has a
figure. The bound-aware Pareto classifies every point as `undecided`
(P4-D2).

### Compute: the pooled model cannot tell them apart, and per engine nothing is priced

| Model | What it gives |
|---|---|
| `annex_v1` (pooled) | The annex's own machine: far flight 16.18 and air superiority 16.62 s/s for **both** designs at every node. It never looks at the design. |
| `default_v1` (per engine) | Every stage is a gap on both designs. No measured efficiency exists for INT8 on the Orin GPU or for any KPU kernel. |

The peaks differ, and the difference is not small:

| Node | Orin dense INT8 | KPU-het dense INT8 | Orin FP16 | KPU-het FP16 |
|---|---|---|---|---|
| N7 | 139.3 TOPS (clocks provisional) | 63.1 TOPS | 0.8 TFLOPS (CPU only) | 19.3 TFLOPS |

- **The H64 is an automotive-class part with less than half Orin's INT8
  peak.** This is not an iso-throughput comparison. A T128-based core is
  the natural next variant (`kpu_t128_core` is generated and complete).
- **Orin's SM template stated no FP16 rate when this study ran** (Phase 3
  P3-D1). Its Class B work therefore ran in FP32, and Orin's FP16 column
  reflects that gap, not the silicon. *Update 2026-09-19:* the template now
  carries NVIDIA's dense FP16 tensor rate, 2,048 ops per SM per clock
  (42.6 TFLOPS on the 16-SM design), sourced from the AGX Orin data sheet and
  the Orin Nano Super blog.

### DRAM: the one decided result

Far flight demands 306 GB/s of DRAM traffic, from the annex's own per-stage
bytes. Both designs keep the reference's 204.8 GB/s LPDDR5. At the 65%
sustained fraction, utilization is **2.30** on both, so **both are
infeasible in far flight on memory alone, whatever the compute
efficiencies turn out to be.** Air superiority (102 GB/s, utilization
0.76) is not memory-bound.

### Power: lower bounds, TOPS/W withheld

Under `annex_v1` no op lands on a datapath, so dynamic power is a gap.
DRAM I/O is priced at N16 and N7 but not at N5, where the catalog gives no
figure. Leakage covers anchored area only. Every power figure is a lower
bound, and useful TOPS/W is withheld (P3-D8).

## What would decide it

In order of leverage:

1. **Per-engine efficiencies for the stages that dominate.**
   - For Orin: INT8 on the GPU (`det`, `sdfenc`, `policy`, `vlm`, `vla`).
     These can be measured on the Orin AGX, which is rentable, or on the Orin
     Nano, which is available.
   - For the KPU: its dense and attention kernels, from the Phase 5
     domain-flow cost model or from silicon.
   - Either set turns the `default_v1` gaps into service times.
2. **FP32 on the KPU, or an explicit decision to keep it off.** Without FP32
   the KPU cannot take any Class C stage, and those are 14 of 19. Its
   comparison with Orin then comes down to what the shared CPUs can carry.
3. **Silicon for the unpriced blocks.**
   - Orin's SM, DLA and PVA logic, and the CPU cores. For each, the Phase 2
     plan lists where a figure could come from.
   - The H64's systolic, LNS and fixed-function tiles. These need a budget in
     the KPU SKU itself (embodied-schemas).
4. **A memory system sized for far flight.** It needs at least 471 GB/s
   of peak DRAM bandwidth (305.9 / 0.65 = 470.6), or a workload whose DRAM
   traffic is cut, for example by keeping the TSDF/ESDF working set on chip.
   No design in either family meets it on today's 256-bit LPDDR5.

## Findings from building the study

- **The H64 SKU is incomplete.** Five of its tile classes carry no
  transistors. `validate_sku` does not flag this today, because the
  `silicon_bin` lines it does have are consistent. The generator surfaces
  it.
- **`--gate-idle` was treating an unknown engine as an idle one.** An engine
  whose stages were all gaps had utilization 0, so it was power-gated. It is
  now gated only when no stage is mapped to it (fixed in #306).

# The KPU x CPU ladder at N7: what each configuration would have to achieve

graphs#269 Phase 6, PR 6.3. 2026-09-19.
Study: `soc_designs/studies/kpu_cpu_ladder.yaml`.
Reproduce:

```bash
python cli/analyze_required_efficiency.py \
    --design kpu_heterogeneous_h64 kpu_uniform_t64 kpu_uniform_t128 kpu_uniform_t256 \
             orin_class_reference \
    --override block:cpu.count=1,2,3 --regime "air superiority" --regime "far flight" \
    --node tsmc_n7 --mapping capability --efficiency orin_nano_measured_v1
python cli/sweep_soc.py --study kpu_cpu_ladder
```

## The question and the answer

Which CPU + KPU configuration carries the autonomy workload, and how does it
compare with the Orin-class reference?

**No configuration can be shown to carry it, and two can be shown not to.**
No efficiency exists for any KPU kernel class, so the usual comparison --
service time, power, TOPS/W -- is unavailable on every KPU rung (P6-D1).
What the ladder gives instead is **what each configuration would have to
achieve**, which is exact arithmetic on the workload's ops and each design's
dense peaks, and that turns out to separate the rungs cleanly.

Two verdicts are decided, both against measured data:

- **Orin misses air superiority on its own measured efficiencies.** Its GPU
  needs 18.2% of dense peak; the Nano's measured figures put utilization at
  1.55 on the three stages they price. A lower bound above 1 is a proven
  violation, and the seven unpriced stages only add to it.
- **The H64 rung with one CPU cluster misses it too.** Its CPU needs 98.7%
  of dense peak, and at measured efficiencies utilization is 1.37 on six of
  thirteen stages.

## The rungs

| Design | Accelerator | Dense peak (ops per clock, whole accelerator) | FP32 | Silicon |
|---|---|---|---|---|
| `orin_class_reference` | 16 Ampere SMs, 2 NVDLA, PVA | 65,536 INT8 / 32,768 FP16 / 4,096 FP32 | yes | SM, DLA, PVA logic unanchored |
| `kpu_heterogeneous_h64` | H64 KPU core | 81,920 INT8 / 24,576 FP16 | **no** | 5 tile classes unanchored |
| `kpu_uniform_t64` | T64 KPU core | 131,072 / 65,536 / 6,656 | yes | fully anchored |
| `kpu_uniform_t128` | T128 KPU core | 262,144 / 131,072 / 13,312 | yes | fully anchored |
| `kpu_uniform_t256` | T256 KPU core | 524,288 / 262,144 / 26,112 | yes | fully anchored |

The DLA is INT8-only and the GPU is faster per server, so the capability
rule puts every stage on the GPU; the DLA takes none. Every rung keeps the
reference's CPU complement, ISP, codec, system cache,
safety island, 256-bit LPDDR5 and IO, so differences are the accelerator's
and the CPU count's.

## What each configuration must achieve

Air superiority and far flight, at N7, capability mapping (each stage on the
engine that would take the fewest seconds at dense peak):

| Rung | CPU clusters | Stages on the accelerator | Accelerator needs | CPU needs |
|---|---|---|---|---|
| Orin | any | 16 of 16 | 18.2% | - |
| H64 | 1 | 3 of 16 | 26.7% | **98.7%** |
| H64 | 2 | 3 of 16 | 26.7% | 49.3% |
| H64 | 3 | 3 of 16 | 26.7% | 32.9% |
| T64 | any | 16 of 16 | 16.0% | - |
| T128 | any | 16 of 16 | 8.0% | - |
| T256 | any | 16 of 16 | 4.0% | - |

Far flight asks slightly less of the datapath (14.4% on the T64, 7.2% on the
T128, 3.6% on the T256) because its stage rates are lower -- and it is out
of reach anyway, on memory. See below.

Three things follow.

**FP32 decides the shape of the design.** The H64 has no FP32 rate, so 13 of
16 air-superiority stages cannot run on it and fall to the CPU. With one
cluster that CPU would have to sustain 98.7% of its dense peak, which is not
an efficiency question. Three clusters bring it to 32.9%. Every T-series
rung states FP32, takes all 16 stages, and leaves the CPU complement free
for whatever else the platform runs.

**The ladder halves the ask per doubling:** 16.0%, 8.0%, 4.0%. Not to the
last digit -- the 70/20/10 tile mix rounds to 13 / 26 / 51 BF16-primary
tiles, so the T256's FP32 rate is 1.96x the T128's rather than 2x.

**A T64 asks less than Orin** (16.0% against 18.2%) on the same 16 stages.
That is a like-for-like comparison, because both are requirements against
dense peaks, and it is the one thing the ladder decides in the KPU's favour.
It says nothing about area: the die figures below are lower bounds with
different coverage, not an area ratio.

## What measured data already decides

`orin_nano_measured_v1` prices GPU and CPU kernels only (27 CALIBRATED
entries from the Orin Nano). Utilization at those figures, air superiority:

| Rung | Engine | Needs | At measured efficiencies | Priced |
|---|---|---|---|---|
| Orin | GPU | 18.2% | **1.55** | 3 of 16 stages |
| H64, 1 cluster | CPU | 98.7% | **1.37** | 6 of 13 |
| H64, 2 clusters | CPU | 49.3% | 0.68 | 6 of 13 |
| H64, 3 clusters | CPU | 32.9% | 0.46 | 6 of 13 |
| T64 / T128 / T256 | KPU | 16.0 / 8.0 / 4.0% | nothing prices a KPU kernel | 0 |

Every figure omits the stages nothing prices, so each is a lower bound: the
two above 1 are proven violations, and the others are open.

**Two stages are over on one core in every rung.** `lio` and `ba` exceed
their period on a single A78AE core at the measured efficiencies, whatever
the accelerator is, so every configuration reports `feasible: NO` on the
measured table for that reason alone. They need more cores, a different
mapping or a faster implementation, not a better accelerator.

## Far flight is a memory problem on every rung

- DRAM demand is 305.9 GB/s against 133.1 GB/s sustained (204.8 GB/s peak at
  the 0.65 fraction): utilization **2.30** on every rung, because they all
  keep the reference's 256-bit LPDDR5.
- `vlm` moves more bytes per call than the sustained bandwidth delivers
  within its period, so it is out of reach at **any** efficiency, on every
  rung.

No accelerator choice changes either. Far flight needs at least 471 GB/s of
peak DRAM, or a workload whose traffic is cut.

## Area and power: lower bounds, and one trap

Air superiority at N7, `annex_v1`, whole SoC:

| Rung | Die area (LB) | Power (LB), annex | Power (LB), measured |
|---|---|---|---|
| Orin | 8.0 mm2 | 0.53 W | 3.24 W |
| H64 | 11.8 mm2 | 0.59 W | 0.61 W |
| T64 | 19.9 mm2 | 0.73 W | 0.75 W |
| T128 | 33.7 mm2 | 0.92 W | 0.95 W |
| T256 | 60.6 mm2 | 1.33 W | 1.35 W |

(One CPU cluster; each further cluster adds about 0.9 mm2 and under 0.01 W
of leakage.)

**Do not read the KPU rungs as lower-power.** Their power is lower on the
measured table only because their stages are gaps: no op lands on a
datapath, so no dynamic energy is charged. Orin's 3.24 W is higher precisely
because three of its stages *are* priced. The comparison is between a
partial figure and a more partial figure.

**Nor as smaller dies.** The T-series cores are fully anchored while Orin's
SM, DLA and PVA logic is not, so a larger figure here is more knowledge, not
more silicon. The bound-aware Pareto classifies every pair as `undecided`
(P4-D2), which is correct.

## What the ceilings decide (PR 6.4)

`cli/analyze_kpu_ceiling.py` puts the domain-flow ceilings
(`soc_designs/ceilings/`, from `graphs.estimation.soc.domainflow`) against
these requirements. A ceiling is the least of three upper bounds, each
sound on its own: the output-stationary wavefront's passes, fill and drain;
the compulsory DRAM traffic; and, where a tile class states its
interconnect, operand delivery. **Utilization at the ceiling above 1 is a
decided no** -- not even a flawless schedule on that silicon carries the
profile.

Ceilings on the T128's fabric, at N7:

| Kernel class | Kernel | Ceiling (FP16) | What holds it there |
|---|---|---|---|
| dense_conv_gemm | 2048^3 GEMM | 0.94 | compulsory DRAM traffic |
| dense_conv_gemm | 3x3 conv, 256 to 256 at 80x80 | 0.95 | wavefront |
| attention_prefill | 16 heads x 1024 x 64 | 0.71 | compulsory DRAM traffic |
| weight_stream_decode | GEMV 1x4096 . 4096x4096 | 0.0014 | compulsory DRAM traffic |
| elementwise_norm | layernorm 4096x4096 | 0.0017 | compulsory DRAM traffic |

Every bound is a time against one reference -- the same dense peak the
requirement is measured against -- and the ceiling is the ideal time over
the longest of them. The schedule pays the SKU's own issue interval: a
32x32 tile that states 512 FP16 MACs per clock issues every other clock,
so a pass of K steps takes 2K cycles.

Read against the ladder:

| Rung | Air superiority | Far flight |
|---|---|---|
| H64 | 0.27 -- open | **51.2 -- decided no** |
| T64 | 0.15 (LB) -- open | **58.1 (LB) -- decided no** |
| T128 | 0.075 (LB) -- open | **29.0 (LB) -- decided no** |
| T256 | 0.038 (LB) -- open | **14.5 (LB) -- decided no** |

- **Far flight is decided against every KPU rung**, and one stage does it:
  `vlm` streams its weights, so its ceiling is 0.14% of dense peak and it
  needs 58 times its period on a T64 even at that ceiling. Doubling the
  tiles halves the figure and never reaches 1: the fabric grows, the DRAM
  does not. This is the Phase 4 memory finding again, now attributed to a
  kernel class rather than an aggregate.
- **Air superiority stays open on every rung**, and comfortably: a T128
  needs 8.0% of dense peak where its dense-GEMM ceiling is 94%. Nothing
  here says it will achieve that, only that the geometry does not forbid it.
- **Dense work is not what threatens these designs.** A 2048^3 GEMM and a
  3x3 convolution sit at 94-95% of dense peak; attention reaches 71%, and
  its own schedule would allow 79% -- a 64-deep inner dimension against 64
  cycles of fill and drain. Streaming work -- decode and normalization --
  is two orders of magnitude below, and it is bandwidth, not the array,
  that puts it there. A layernorm's wavefront bound is exactly 0.5,
  because an elementwise pass uses the adder and leaves the multiplier
  idle.
- **13 of 16 stages have no ceiling.** Their kernel classes have no
  domain-flow schedule stated (factor graphs, raycast, wavefront, graph
  search, scatter-add, FFT) or no representative kernel at all, so every
  "open" verdict above is a lower bound.

## What the ladder does not decide

1. **Whether any KPU rung actually meets its requirement.** Nothing measures
   a KPU kernel. 16.0% of dense peak on a T64 is a target, not a result.
2. **Seven kernel classes have no representative kernel at all** -- cost
   volume DP, feature tracking, kNN, raycast, wavefront, graph search, pixel
   fixed-function -- so they are unpriced on the CPU too.
3. **The Nano's efficiencies are applied to a 16-SM design.** They were
   measured on 8 SMs. The per-server normalization assumes efficiency does
   not depend on SM count.
4. **INT8 on the CPU is priced at PyTorch's `_int_mm`**, 0.12 of peak, which
   is a library figure, not a ceiling.

## What would decide it

In order of leverage:

1. **KPU service times.** PR 6.4's ceilings say the geometry allows a T128
   to carry air superiority's compute (8.0% needed against a 94% dense
   ceiling) but they cannot say it will: a ceiling is not an efficiency.
   That still needs silicon, or a scheduler that produces service times
   rather than bounds.
2. **A memory system for far flight**, or less traffic. 471 GB/s of peak
   DRAM, or keeping the TSDF/ESDF working set on chip.
3. **`lio` and `ba` off a single core.** They are the reason every rung
   fails on measured data today.
4. **The remaining kernel classes measured**, on the hardware that is
   available (Orin Nano) or rentable (Orin AGX).
5. **Orin's unanchored silicon priced**, so that area and power can be
   compared rather than bounded.

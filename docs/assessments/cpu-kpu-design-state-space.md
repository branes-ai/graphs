# CPU + KPU: the design state space against 18 robot mission capabilities

For discussion with CPU silicon IP partners. Stillwater Supercomputing,
2026-09-20. Generated from `branes-ai/graphs` at the commit this document
ships in; every figure is reproducible with the commands in
`docs/guides/cpu-kpu-state-space-howto.md`.

## Start with the picture

`mission-frontier.html`, beside this file, is the same analysis as charts:
one per mission, energy per operation across, real-time factor up, with the
envelope over the catalogue and the band a configuration is already proven
short of. Open it first -- it carries the model and the shape of the trade
in a way a table of 3,672 rows cannot.

```bash
python cli/report_mission_frontier.py -o docs/assessments/mission-frontier.html
```

Two things that page shows and this one cannot:

- **For most missions there is no trade to make.** One configuration is
  cheaper *and* no slower than every other, so the envelope is a single
  point. The process node moves energy per operation; the busiest engine --
  almost always the CPU -- decides the rate, and the node does not move
  that.
- **Where a trade does exist it is vertical.** In edge-AI supervisory
  control, the same 0.2 pJ per operation buys anything from 0.23x to 1.59x
  real time depending on the fabric and the cores. Energy is not what is
  being spent to get performance there; silicon area is.

## Then look one level down

A configuration's position on that plane is decided one stage at a time.
`pipeline-demand.html`, also beside this file, is that level: one row per
pipeline stage, in pipeline order and banded by tier, with what the stage
demands per second of mission -- operations, bytes, and the precision
classes it needs -- and then, for **every** engine and not only the one the
schedule picked, the share of that engine one second of mission would
consume there.

```bash
python cli/report_pipeline.py -o docs/assessments/pipeline-demand.html
```

It is where the headline of this document becomes legible. For the humanoid
cobot on a T128 core at N7 with twelve Cortex-A78AE cores:

| stage | class | on the CPU | on the KPU |
|---|---|---|---|
| `tsdf` | raycast | **1.03x one core** | no schedule |
| `esdf` | wavefront | 93% of one core | no schedule |
| `mpc` | small QP | 92% of one core | no schedule |
| `det` | dense conv/GEMM | no INT8 figure | 5.2% (ceiling) |
| `vla` | attention prefill | no INT8 figure | 8.2% (ceiling) |

Three CPU stages at or near a whole core each, while the accelerator sits
near idle: that is the same conclusion the matrix reaches as "the CPU is
the sole reason in 2,108 of 3,104 decided points", stated in a form a
silicon architect can act on. The work to be done is not a larger fabric.
It is a domain-flow schedule for `raycast`, `wavefront` and `small_qp`, or
CPU cores that run them faster.

A cell with no bar says which figure is missing and why, in the two forms
that call for different work: *no INT8 figure (FP32 measured)* is a
benchmark run nobody has done, and *no FP32 figure, no schedule* is a
kernel class the domain-flow model cannot place on the fabric at all.

Still missing, and the reason this document is not yet the whole story: a
picture of each pipeline showing what every stage demands and what each
engine gives it, and an interactive filter over the state space. Both are
planned on top of the same data.

## The question, and the short answer

> Which KPU fabric, with how many CPU cores, on which memory system, in
> which process node, carries which robot mission?

We swept 3,672 configurations against 18 mission capabilities. **Five
missions have a configuration that survives every test we can apply. The
other thirteen are ruled out across the entire space, and the CPU is a
reason in 2,924 of the 3,104 decided points -- the only reason in 2,108 of
them.** Not the accelerator, not the process node, and not the memory
system, which is never a point's sole reason.

That is the conversation we would like to have. The accelerator side is
ours to build. The missions are gated by what a general-purpose core can
sustain on a handful of irregular kernels, and today's measured figures
are one to two orders of magnitude short of what the missions ask.

**How to read any number here.** Nothing in this document says a
configuration *works*. It says either "this is ruled out, here is the
proof" or "this is still open". A complete proof of sufficiency needs a
schedule and a power figure we do not yet have gap-free inputs for, and we
would rather hand you a bounded claim than a confident one.

## The workload

18 mission capabilities across 6 form factors, each a sensor
configuration, a rate per pipeline stage, a power budget and a
sense-to-act deadline. Each mission runs up to 19 stages; each stage is
classified into one of 15 kernel classes and into precision classes A
(INT8-eligible), B (FP16 floor) and C (FP32 floor).

The division of labour we assume: **the KPU takes the kernel classes our
domain-flow fabric has a schedule for** -- dense GEMM and convolution,
attention, weight-stream decode, elementwise normalization -- **and the
CPU carries everything else**: factor graphs and bundle adjustment,
SGM cost volumes, raycasting a TSDF, ESDF wavefront propagation, graph
search, feature tracking, scatter-add, FFT, QP solves and the pixel
front-end.

That split is the crux. It is not a modelling convenience: those classes
are irregular, pointer-light but memory-heavy, or triangular in their
dependences, and a regular wavefront fabric has no schedule for them.

## What each mission asks of the CPU

Per core, over 12 Cortex-A78AE cores, with a T128-class KPU carrying its
share. "Sustained GOP/s per core" is the IP-neutral form: it does not
depend on our peak, only on the workload's operation counts and the
mission's rates.

| Mission | Power budget | CPU: sustained GOP/s per core | KPU: sustained TOPS | DRAM |
|---|---|---|---|---|
| Edge AI, event detection | 2 W | 0.1 | 0.1 | 0 GB/s |
| Edge AI, multi-stream tracking | 5 W | 0.4 | 0.7 | 4 |
| Edge AI, supervisory control | 8 W | 0.2 | 1.0 | 27 |
| AMR, warehousing | 30 W | 2.0 | 1.5 | 16 |
| AV, SAE L2 | 30 W | 2.7 | 1.0 | 14 |
| Drone, inspection | 25 W | 3.4 | 1.9 | 24 |
| Quadruped, inspection | 30 W | 3.9 | 4.1 | 32 |
| Quadruped, surveillance | 40 W | 4.3 | 7.4 | 38 |
| AMR, logistics | 40 W | 4.4 | 5.7 | 35 |
| Drone, ISR endurance | 25 W | 5.6 | 13.3 | 306 |
| AMR, loading and manipulation | 75 W | 7.1 | 13.7 | 93 |
| Quadruped, ISR dismounted | 60 W | 8.2 | 12.5 | 198 |
| Humanoid, industrial cell | 60 W | 8.5 | 13.0 | 71 |
| Drone, interceptor | 25 W | 11.6 | 14.0 | 102 |
| Humanoid, cobot | 90 W | 11.7 | 22.6 | 131 |
| AV, SAE L3 | 100 W | 15.0 | 7.0 | 99 |
| Humanoid, house work | 150 W | 18.8 | 31.8 | 347 |
| AV, SAE L4/L5 | 800 W | 42.5 | 28.3 | 386 |

## What a core delivers today

Measured on an Orin Nano, clocks pinned and verified, one thread on one
Cortex-A78AE core. The core's dense FP32 peak is 35.2 GOP/s.

| Kernel class | Measured fraction of peak | Sustained GOP/s per core |
|---|---|---|
| Dense GEMM (FP32) | 0.89 | 31.3 |
| Attention prefill (FP32) | 0.50 | 17.7 |
| FFT | 0.20 | 7.1 |
| QP solve (KKT) | 0.13 | 4.6 |
| Factor graph / Cholesky | 0.085 | 3.0 |
| Weight-stream decode (FP32) | 0.058 | 2.0 |
| Pixel front-end (ISP) | 0.045 | 1.6 |
| ESDF wavefront | 0.030 | 1.1 |
| Feature tracking (KLT) | 0.022 | 0.8 |
| SGM cost volume | 0.0098 | 0.34 |
| Graph search | 0.0095 | 0.33 |
| Scatter-add | 0.0066 | 0.23 |
| **TSDF raycast** | **0.0034** | **0.12** |

The library-grade kernels reach 50-90% of peak. **The kernels the missions
actually need from the CPU reach 0.3-3%.** A TSDF raycast sustains 0.12
GOP/s on a core whose peak is 35.2.

That is the gap: missions ask 2 to 42 GOP/s per core; the classes they ask
it on deliver 0.12 to 3 GOP/s today.

## What that does to the state space

| | Points | Outcome |
|---|---|---|
| Configurations swept | 3,672 | 17 KPU designs x 3 CPU counts x 4 memory systems x 18 missions |
| Decided against | 3,104 | a point can fail for more than one reason |
| ... the CPU is one of the reasons | 2,924 | of these, **2,108 fail on the CPU alone** |
| ... our fabric's own ceiling is one | 996 | the KPU could not carry its share even at its geometric limit |
| ... the memory system is one | 459 | never on its own: every one of them also fails on the CPU or the ceiling |
| Still open | 568 | five missions: the three edge-AI capabilities, AMR warehousing and AV L2 |

The 17 designs are five KPU fabrics (H64, T64, T128, T256, T512) at the
nodes their SKUs state a clock for, plus the four hand-written designs of
the earlier ladder study.

Missions with **no** surviving configuration anywhere in the space: every
drone mission, every quadruped mission, two of three AMR missions, all
three humanoid missions, and AV L3 and L4/L5 -- thirteen of eighteen.

**The accelerator is not the binding constraint, and neither is the
process.** Moving a T128 from 16 nm to 7 nm changes what we ask of our own
fabric (14.4% of dense peak to 11.5%, because the 7 nm SKU clocks at 750
MHz against 600) and cuts the datapath energy floor from 0.43 W to 0.17 W.
It does not move the CPU requirement at all, and the CPU requirement is
what decides the mission.

## What the memory system decides

DRAM demand comes from the workload's own per-stage byte counts, against
peak bandwidth at a 0.65 sustained fraction for scattered access.

| Memory system | Peak | Missions whose DRAM traffic it carries |
|---|---|---|
| LPDDR5, 256-bit | 204.8 GB/s | 14 of 18 |
| LPDDR5X, 256-bit | 273 GB/s | 14 of 18 |
| LPDDR5X, 512-bit | 546 GB/s | 17 of 18 |
| HBM3, one stack | 819 GB/s | 18 of 18 |

Three missions need more than a 256-bit LPDDR5X part can deliver --
quadruped ISR (198 GB/s), drone ISR endurance (306) and humanoid house
work (347) -- and AV L4/L5 at 386 GB/s needs HBM-class bandwidth. For the
other 14, the platform's existing 256-bit LPDDR5 is not the limit.

## What we are asking of a CPU IP partner

In order of leverage:

1. **Sustained throughput on irregular kernels, not on GEMM.** The
   benchmark suite that produced the table above is in the repository
   (`graphs.benchmarks.soc_kernels`): SGM, KLT, TSDF raycast, ESDF
   wavefront, graph frontier relaxation and an ISP pipeline, each sized
   like the stage it stands for. We would like to run it on your core, and
   to know which of the six your microarchitecture already addresses.
2. **What limits them on your core.** Our measurements are of one PyTorch
   implementation on one core; we are not claiming they are the best
   achievable. Where a gather, a predicated select, a scatter-add or a
   wider load would change the number, we would rather model your core
   than our implementation.
3. **Cores against area and power.** Twelve cores is the Orin-class
   complement we assumed. The requirement scales exactly with core count,
   so 24 cores halves it -- but then the area and power budget has to
   absorb them, alongside the accelerator and the memory system.
4. **Which process node you would target.** The node changes our fabric's
   clock and the energy per operation by about 2.5x between 16 nm and
   7 nm. It does not change the CPU requirement, so the node decision is
   driven by your core's frequency and energy, not by the arithmetic.

## What this analysis does not decide, and why

We hold to one rule: a missing figure stays a gap. It is never filled with
an estimate to make a number come out.

- **No KPU kernel has been measured.** No silicon has run them. Every KPU
  figure here is either a requirement (what it would have to sustain) or a
  ceiling from geometry and bandwidth (what it could not exceed). The 996
  points where the ceiling contributes are ruled out by that bound, which
  is sound; nothing here claims the fabric achieves anything.
- **Three stages are still unpriced on the CPU.** SGM, radar FFT and the
  visual front-end each run an INT8 class we have no INT8 kernel for, so
  the measured column omits them, and every CPU utilization above is a
  lower bound. The true figures are worse, not better.
- **Power is a floor, and a partial one.** The datapath energy floor
  charges every operation once at the node's energy per operation. It
  excludes fetch, control and memory-hierarchy energy, and for a T-series
  KPU core it excludes the fabric's own operations, because that core's
  tiles span two logic libraries and its template states no single
  datapath class.
- **No verdict is `yes`.** Utilization at or under 1 is necessary, not
  sufficient: it does not prove a schedule meets its deadlines.

## Reproducing this

```bash
# The whole space, as a spreadsheet
python cli/analyze_mission_matrix.py --state-space --all --output state-space.csv

# One configuration against one mission, with the reasons
python cli/analyze_mission_matrix.py --design kpu_t128_n7 \
    --mission humanoid_cobot_human_adjacent_contact_rich \
    --cpu-clusters 3 --memory lpddr5x_phy_512b --verbose

# What a design must achieve, stage by stage
python cli/analyze_required_efficiency.py --design kpu_t256_n16 \
    --regime "air superiority" --efficiency orin_nano_measured_v1 --verbose

# What our fabric could do at best
python cli/analyze_kpu_ceiling.py --design kpu_t256_n7 --all --verbose
```

The method, the axes and the meaning of each column are in
`docs/guides/cpu-kpu-state-space-howto.md`. The measured efficiencies and
how they were taken are in
`docs/guides/soc_kernel_benchmarks_on_jetson.md`.

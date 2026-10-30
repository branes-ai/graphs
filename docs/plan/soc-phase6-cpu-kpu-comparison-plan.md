# SoC study Phase 6: contrasting CPU + KPU configurations -- execution plan

Tracking: branes-ai/graphs#269, Phase 6. Parent plan:
`docs/plans/soc-microarchitecture-study-plan.md`. The strict rule of Phases
2-5 carries over. Started 2026-09-19.

## The problem

Phase 4's study (`docs/assessments/soc-orin-vs-kpu-heterogeneous.md`) could
not decide the Orin-vs-KPU question, and a ladder of CPU + KPU
configurations runs into the same wall, harder:

- On a CPU+KPU design with the measured table, **6 of 18 far-flight stages
  are priced, all of them on the CPU** (the A78AE figures measured on the
  Orin Nano). Every KPU stage is a gap.
- No efficiency exists for any KPU kernel class. No silicon has run them,
  and the SKU's `efficiency_factor_by_precision` (0.55-0.7, hand-authored)
  is not a source. Under the strict rule it cannot fill the gap.
- Seven kernel classes have no representative kernel even on the CPU
  (cost-volume DP, feature tracking, kNN, raycast, wavefront, graph search,
  pixel fixed-function).

So a straight comparison of service times, power and TOPS/W between CPU+KPU
configurations is not available, and will not be until the KPU has either
silicon or a cost model.

## P6-D1, decided 2026-09-19: break-even first, model later

Two ways to get a comparison. The decision is to do both, in this order.

**1. Required efficiency (this phase's first PR).** Invert the question.
Instead of pricing a stage at an efficiency nobody has, solve for the
efficiency that would make the profile fit:

    required efficiency = sum over the engine's stages of
                          rate x sum over the stage's precision classes of
                          (share x ops per call / dense peak of that class's
                           format), all over the engine's servers

That is exact arithmetic on the workload's ops and the design's peaks. It
invents nothing, it is defined for an engine nothing has measured, and it
answers the comparison question directly: **which configuration asks least
of the KPU?** Above 1 it says no efficiency suffices -- the silicon is too
small, not too slow. `memory_occupancy` says the same for DRAM, which no
efficiency changes.

Implemented in `estimation/soc/breakeven.py`, with `cli/analyze_required_efficiency.py`.

**2. The domain-flow model (PR 6.4, done).** A SURE/SARE wavefront model
could give KPU service times directly, but a service time is a claim about
what the silicon *will* do, and nothing supports one. So 6.4 produces
**ceilings** instead: the least of the wavefront schedule, the compulsory
DRAM traffic and operand delivery, each an upper bound sound on its own.
A ceiling below the requirement is a decided no; above it, the verdict
stays open. The ceilings are deliberately not an `EfficiencyTable` and do
not live in `soc_designs/efficiency/`, so no analysis can price a stage
with them (P5-D1 stands).

## P6-D2, decided 2026-09-19: the first configuration space

A KPU x CPU ladder at N7, against the Orin reference as the anchor:

| Axis | Values |
|---|---|
| KPU core | H64 (no FP32), T64, T128, T256 |
| CPU | 1, 2, 3 clusters of 4 Cortex-A78AE |
| Node | tsmc_n7 |
| Profiles | the two regimes with per-stage data |

The FP32 axis is the interesting one: the H64 has none, so every Class C
stage falls to the CPU (14 of 19 stages), while the T-series cores do have
FP32 and can take them. The ladder shows what that costs in CPU complement.

Iso-area and iso-power comparisons against Orin need the unanchored Orin
silicon priced first (Phase 2), so they are not in this phase.

## PR sequence

| PR | Content |
|---|---|
| 6.1 | `estimation/soc/breakeven.py`, `cli/analyze_required_efficiency.py`, tests. |
| 6.2 | The ladder: T64 / T128 / T256 cores from `generate_kpu_ip.py`, one design per KPU core, the CPU-cluster override, the `kpu_cpu_ladder` study. **Done.** |
| 6.3 | The assessment: what the ladder decides, and the KPU efficiency each configuration would need. **Done:** `docs/assessments/soc-kpu-cpu-ladder.md`. |
| 6.4 | The domain-flow model, as **ceilings** read against 6.3's requirements (not an efficiency table: a ceiling is not an efficiency). **Done:** `estimation/soc/domainflow.py`, `soc_designs/ceilings/`, `cli/analyze_kpu_ceiling.py`. |

## What the requirement already says

At N7, with the shipped mappings (`orin_nano_measured_v1` for comparison):

| Design | Profile | Engine | Needs | At measured efficiencies |
|---|---|---|---|---|
| `orin_class_reference` | far flight | GPU | 16.3% | utilization 0.42 (LB, 3 of 10 stages priced) |
| | | CPU | 2.3% | utilization 0.23 (LB, 6 of 8 priced) |
| `orin_class_reference` | air superiority | GPU | 17.8% | utilization 1.09 (LB, 3 of 8 priced) |
| | | CPU | 4.1% | utilization 0.46 (LB, 6 of 8 priced) |
| `kpu_heterogeneous_h64` | far flight | KPU | 28.0% | nothing prices a KPU kernel |
| | | CPU | 16.0% | utilization 0.23 (LB, 6 of 14 priced) |
| `kpu_heterogeneous_h64` | air superiority | KPU | 26.7% | nothing prices a KPU kernel |
| | | CPU | 32.9% | utilization 0.46 (LB, 6 of 13 priced) |

The measured column is a lower bound twice over: it omits the stages the
table does not price, and a stage counts only when the table prices *every*
format it runs in.

Orin's GPU is over 1 in air superiority on three priced stages alone, so
the measured Nano efficiencies already miss what that profile needs.

Read: the H64 would have to sustain about 27% of its dense peak on the
stages it takes. Whether it can is exactly what the cost model, or silicon,
has to answer -- and 27% is the number to answer against.

## What the ladder says (PR 6.2)

`soc_designs/studies/kpu_cpu_ladder.yaml` for area, power and DRAM;
`cli/analyze_required_efficiency.py --design ... --override block:cpu.count=1,2,3`
for what each rung would have to achieve. At N7, capability mapping:

| Rung | Stages on the KPU | KPU needs (air / far) | CPU needs at 1 / 2 / 3 clusters (air) |
|---|---|---|---|
| H64 | 3 of 16, 4 of 18 | 26.7% / 28.0% | 98.7% / 49.3% / 32.9% |
| T64 | all | 16.0% / 14.4% | the CPU takes nothing |
| T128 | all | 8.0% / 7.2% | the CPU takes nothing |
| T256 | all | 4.0% / 3.6% | the CPU takes nothing |

- **FP32 is the whole story.** The H64 has none, so 13 of 16 air-superiority
  stages fall to the CPU, and with one cluster that CPU would have to run at
  98.7% of its dense peak. Three clusters bring it to 32.9%. The T-series
  cores state FP32, take every stage, and leave the CPU complement free.
- **The ladder halves the ask.** T64 16.0%, T128 8.0%, T256 4.0% in air
  superiority. Not quite exactly: the 70/20/10 tile mix rounds to 13 / 26 /
  51 BF16-primary tiles, so the T256's FP32 rate is 1.96x the T128's.
- **Far flight stays out of reach on memory.** `vlm` moves more bytes per
  call than the sustained bandwidth delivers in its period, on every rung,
  and no efficiency changes that. It is the same finding as Phase 4's: the
  memory system, not the datapath, decides far flight.
- **Area** at N7 (annex_v1, lower bounds throughout): Orin 8.0 mm2, H64
  11.8, T64 19.9, T128 33.7, T256 60.6. The T-series cores are fully
  anchored while Orin's SM, DLA and PVA logic is not, so a larger figure
  here is more knowledge, not more silicon. The Phase 4 caveat stands.

The open question is unchanged and now quantified per rung: a T128 has to
sustain 8% of its dense peak, a T64 16%, an H64 27% plus a 3-cluster CPU at
33%. PR 6.4's cost model answers against these.

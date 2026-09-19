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

**2. The domain-flow cost model (later).** A SURE/SARE wavefront model gives
KPU service times directly. It is research-scale, and it is model output:
THEORETICAL, in its own efficiency table, never merged into `default_v1`
(P5-D1 stands). The break-even figures are the yardstick its outputs are
read against -- a model that lands above the requirement says the
configuration does not work, and one that lands below says by how much.

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
| 6.2 | The ladder: T64 / T128 / T256 cores from `generate_kpu_ip.py`, one design per KPU core, the CPU-cluster override, a study. |
| 6.3 | The assessment: what the ladder decides, and the KPU efficiency each configuration would need. |
| 6.4 | The domain-flow cost model as a THEORETICAL table, read against 6.3's requirements. |

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

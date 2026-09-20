# Energy Accuracy Assessment

Session complete — 15 PRs merged across 2 repos, all CI-green

Starting from "what is next," this session resolved the entire KPU/CPU energy-and-accuracy thread plus its follow-ups, end to end:

|        Theme         |               PRs               |           Issues            |
|----------------------|---------------------------------|-----------------------------|
| KPU energy model     | #251 → #252(+ES #84/#85) → #253 | #177, #121, #154, #81       |
| CPU mapper accuracy  | #257 → #259 → #260              | #175, #178, #78             |
| Estimate confidence  | #263                            | #79                         |
| Tooling / references | #254, #255                      | #122, #176                  |
| Docs & hygiene       | #256, #258, #261, #262, #264    | session docs + test cleanup |

The arc: the KPU energy model went from physically-impossible (1175 W on a 6 W part) → thermal-clamped → Vdd-aware leakage → dynamic energy single-sourced
from the ProcessNode. The CPU mapper went from claiming 192× fanout on a batch=1 matvec → geometrically capped + L1-spill-honest + package-level static
power. And every energy estimate now carries an honest EstimationConfidence.

Process highlights: consulted you on each consequential modeling/scoping decision (clamp vs recalibrate, ProcessNode vs domain-flow source, round-TDP
preservation via Vdd re-tune, plumb-vs-registry for confidence); resolved CodeRabbit reviews on #260 and #263 (all findings fixed or justifiably deferred,
with CodeRabbit concurring); and kept the changelog, session log, contribution log, and a decision record current throughout.

Documented open follow-ups (noted in the session log, none blocking):
- Latency/Memory descriptor confidence audit (same gap #79 fixed for energy).
- CPU batch=1 levers: the ~6.5 µs/op launch overhead and per-operator LLC-bandwidth scaling.
- CPU_L1_SPILL_EXPONENT and the leakage_vdd_exponent values are THEORETICAL pending hardware calibration (Orin Nano / AGX).


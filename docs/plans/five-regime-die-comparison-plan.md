# Die comparison on the autonomy workload: T64 against kpu_h64_auto1

Execution plan for the last open item on branes-ai/graphs#268: *comparing the
two dies on the five operating regimes*, deferred there because it needs the
SoC study analyzer's workload (#269) and the Autonomy Workload Data Annex.

Both arrived on 2026-09-17, and with them the model that derives them. The
comparison therefore runs on **the 18 mission profiles the model derives**,
with the two that are named regimes labelled as such (decided 2026-09-17;
see "The regime axis").

## Sources, and when they arrived

| Document | In repo | Dated | What it supplies |
|---|---|---|---|
| `docs/BranesAI-Autonomy-Compute-Requirements.pdf` | 2026-09-13 | -- | **The argument.** 17 stages at the sizing regime (ops/s, op per byte, numeric floor, latency budget); the five operating regimes with sustained TOP/s, Class A and Class C shares, governing constraint; per-regime DRAM traffic and sensor ingress; achieved-to-peak bands (2% realistic, 5% optimistic); per-regime stages-over counts and binding tiers; compute seconds per second of flight (2.7 / 9.2 / 16.2 / 12.9 / 17.3). |
| `docs/BranesAI-Workload-Data-Annex.pdf` | **2026-09-17** | **2026-09-17** | **The inputs.** 19 costed stages over 7 tiers: costing unit, precision-class split (A/B/C %), ops per call, op per byte. Counting conventions (one MAC is two ops; bytes at the DRAM interface, cache-line granular when scattered). Effective throughput: Class A 2,000 GOP/s, Class B 300, Class C 15. The service-time / occupancy / oversubscription model. 18 mission profiles across 6 form factors with sustained TOP/s, GB/s, oversubscription, reactive-chain ms, deadline ms and stages-over. Its own validation against the argument document, and an exposure register of seven stages that disagree with it by more than 2x. |
| `docs/BranesAI-Workload-Data-Annex.xlsx` | **2026-09-17** | -- | The generated workbook: the itemized unit-cost sheet, precision classes, the eighteen profiles, validation and sensitivity. Emitted by `book.py`, not maintained by hand. |
| `docs/workload-model/` (10 Python files, ~1,900 lines) | **2026-09-17** | -- | **The model itself.** `derive.py` (forward unit-cost derivations, nothing back-solved), `pipeline.py` (cost, class, service-time and occupancy model), `profiles.py` (the 18 missions as explicit sensor suites and update rates), `book.py` (emits the workbook), plus `tiles.py` / `feasibility.py` (a coarse five-architecture assessment) and the graph renderers. |

**The workbook and its source arrived with the annex on 2026-09-17**, which
removes the gap this plan was originally written around. Per-stage update
rates per mission are now available (`profiles.py`), so regime demand is
**derived** rather than declared, and the whole chain from sensor
configuration to ops/s is re-runnable. Verified on arrival: running
`profiles.py` reproduces the annex's published drone rows exactly -- ISR
10.47 TOP/s / 306 GB/s / 16.2 s per second, Interceptor 12.46 / 102 / 16.6.

**What the model does not contain: three of the five regimes.** `profiles.py`
carries 18 mission profiles, of which exactly two are the argument
document's regimes, and say so in their own notes: *Drone / ISR (endurance,
wide-area search)* is far flight, *Drone / Interceptor (terminal engagement)*
is air superiority. Minimum usable, contested and fast flight exist only as
published aggregates in the 09-13 argument document. See "The regime axis"
below.

## What the two documents disagree about, and how this plan treats it

These are not editorial differences; they decide what the model can claim.

- **19 stages, not 17.** The annex costs 19 (it adds a mono camera front-end
  and a vision-language-action policy, and splits some tiers differently).
  The YAML carries the annex's 19 as the stage list, and records the
  argument document's 17-row sizing-regime table as a separate declared
  view. Neither is silently reshaped into the other.
- **18 mission profiles, not 5 regimes.** They are different axes: the
  profiles are form-factor missions (drone ISR, humanoid house work, SAE
  L4), the regimes are flight regimes of one platform. Both are modeled;
  only the five regimes answer #268's question.
- **The annex reproduces the regimes to within 3-4%** (far flight 10.47 vs
  10.24 TOP/s, +2.3%; air superiority 12.46 vs 12.81, -2.7%; oversubscription
  -0.1% and -3.9%). That agreement is a **test**, not a construction: the two
  derivations share no coefficient.
- **Seven stages disagree by more than 2x** (annex section 7). Five are cases
  where the annex is cheaper and under half a percent of any mission total;
  two are reparameterizations. The register is encoded as data so the
  disagreement is visible in the model rather than in a footnote.

## The regime axis (decided 2026-09-17)

The model's axis is 18 mission profiles; #268 asks for five regimes, of which
the model contains two -- *Drone / ISR* is far flight and *Drone /
Interceptor* is air superiority, each labelled so in its own note.

**Decision: report the axis the model supports.** The comparison covers all
18 profiles and labels those two. Everything stays derived from the annex and
nothing is authored: reconstructing minimum usable, contested and fast flight
would mean fitting sensor suites to published totals, which is the
back-solving the annex explicitly refuses to do. Those three regimes keep
only their published aggregates, and this plan does not give them per-stage
verdicts.

The 18 profiles are also the better question for silicon selection: they span
six form factors and a 500x range of demand (0.05 to 25.06 TOP/s), where the
five regimes are one platform's flight envelope.

## Phase A -- the workload as data (#269 items 1.1 and 1.2, scoped)

The model is the source of truth; the graphs port must reproduce it, not
paraphrase it.

- `src/graphs/core/pipeline_workload.py`: `Stage` (id, pipeline_tier, name,
  costing unit, `class_split`, `ops_per_call`, `bytes_per_call`, provenance),
  `MissionProfile` (sensor configuration and per-stage rates), the
  service-time / occupancy / oversubscription model, and the aggregation
  roll-ups. No hardware in this module.
- `workloads/pipelines/autonomy/branes_7tier_v1.yaml`: **generated from
  `docs/workload-model/`**, not transcribed, with provenance naming the
  model file and the 2026-09-17 revision. The generator is checked in so a
  changed rate in `profiles.py` regenerates the YAML.
- `cli/show_pipeline_workload.py` with `--profile` / `--stages` / `--regime`.

**Acceptance -- parity with the reference implementation.** For all 18
profiles, the port reproduces `profiles.py`'s sustained TOP/s, GB/s,
occupancy, class shares and reactive chain to within floating-point
tolerance. Then the annex's own published checks follow: the drone rows
(10.47 / 306 / 16.2 and 12.46 / 102 / 16.6), the argument document's
sizing-regime stage table summing to 12.81 TOP/s, and Exhibit B's class
shares (83.4% A, 16.0% B, 0.6% C).

## Phase B -- the comparison (#268's deferred item)

`cli/compare_kpu_regimes.py`, over the two 64-site dies at both nodes
(`kpu_t64_32x32_lp5x4_*`, `kpu_h64_auto1_lp5x4_*`), reusing the E3 engine
descriptors rather than re-deriving capability:

1. **Precision-floor feasibility per class.** Class C needs an FP32/FP64
   floor. The T64 has 3.16 FP32 TFLOPS on its BF16-primary class; the
   heterogeneous die has no FP32 at all, so every Class C stage is
   infeasible on it -- 0.6% of the arithmetic and, by Exhibit B, 30% of the
   sizing-regime schedule. This is the trade the refactor made, quantified.
2. **Class A / B / C demand against capability**, per stage now rather than
   per regime aggregate, at peak and at the annex's achieved-to-peak bands
   (2% and 5%). Peak is reported as the upper bound it is. Per-stage demand
   means a verdict can name the stage that breaks, which is what the
   argument document's "stages over" column reports.
3. **Fixed-function absorption, measured instead of assumed.** This is where
   the two repositories meet. `docs/workload-model/tiles.py` assigns each
   stage a *hand-set* throughput and energy gain for a hypothetical KPU tile
   -- 8x / 80x for stereo SGM, 30x / 250x for the FP64 solvers, 25x / 330x
   for state estimation anchored on Navion. The graphs side has the real
   thing: `kpu_h64_auto1` carries ISP, SGM and VIO function cores with cited
   work-unit throughput, pJ per unit and silicon area, and the E2 tile
   ladder already prices them. The comparison replaces the assumed
   multipliers with the modeled dies and **reports the difference**, which
   is a check on the annex's tile model as much as on the dies.
4. **Bandwidth.** Both dies sit on the same 64 GB/s LPDDR5 bus against
   regime demands of 16 / 156 / 289 / 182 / 100 GB/s. Only *minimum usable*
   fits; the rest are bus-bound before either fabric matters, which is a
   result about the SKUs' memory system rather than about tile kinds.
5. **Verdict per regime**: the binding constraint, the deficit factor, and
   the confidence of each number.

**Acceptance.** Each regime gets a verdict from both dies with its binding
constraint named; the FP32 finding and the bandwidth ceiling are asserted by
tests; `--output` emits JSON/CSV/MD through the shared formatter.

## Phase C -- write-up

`docs/assessments/five-regime-die-comparison.md`: the table, the three
findings above, and an explicit list of what the model does not include
(annex section 8: kernel launch and scheduling overhead, IPC, cold caches
and contention, drivers and thermal throttling, redundancy) plus what this
plan adds to that list (no per-stage rates without the workbook, so regime
demand is declared).

## Where the reference implementation lives

`docs/workload-model/` stays as the reference implementation and is not
modified by this work. The graphs port is checked against it, so if the two
disagree the model wins and the port is wrong. Its `tiles.py` and
`feasibility.py` are a different kind of artifact -- an assessment with
assumed gains -- and are inputs to Phase B's comparison, not to Phase A's
data.

## Out of scope

The rest of #269: the IP-block library, `compose_soc`, multi-block
`silicon_math`, the SoC mapper and scheduler, and the Orin-class
reconstruction. This plan uses only the two existing KPU dies and the
engine descriptors that #268 E3 already landed.

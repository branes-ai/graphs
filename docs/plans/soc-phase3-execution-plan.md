# SoC study Phase 3: the L0 analyzer, end to end -- execution plan

Tracking: branes-ai/graphs#269, Phase 3. Parent plan:
`docs/plans/soc-microarchitecture-study-plan.md`, sections 5.5-5.8 and 7.
Phase 2's decisions (`soc-phase2-execution-plan.md`) carry over. Started
2026-09-18.

## What Phase 3 has to work with

| Input | State |
|---|---|
| Workload | `graphs.core.pipeline_workload`: 19 costed stages, 18 mission profiles, reproducing the Data Annex model to 3e-16. Two profiles are named regimes. |
| SoC | `compose_soc` -> `SoCInstance`: engines with dense peaks per format, clocks with their basis, and areas that are lower bounds wherever silicon is unanchored. |
| Annex efficiency | Stated, machine-level: Class A 2,000, B 300, C 15 GOP/s effective on an Orin-class part (Data Annex section 2). |
| Measured efficiency | Orin GPU and DLA calibration runs in `hardware_registry/` (January-February 2026, PyTorch and TensorRT). Some files carry zeroed aggregate fields, so each operation profile is read on its own merits. |
| Everything else | No measured efficiency for the CPU clusters, the PVA, or any KPU kernel. |

## Decisions

### P3-D1. Two efficiency models, kept apart

- **`annex_v1` is the annex's own model**, and treats the SoC as the annex
  does: one pooled machine with per-class effective throughput. It is the
  regression target. With it, the analyzer must reproduce the annex's
  oversubscription and stages-over for all 18 profiles, which the workload
  port already matches to 3e-16. That makes this a check that the analyzer
  adds nothing and loses nothing, not a new claim.
- **`default_v1` is per engine**, keyed `(kernel_class, engine_kind,
  precision)`. It only carries a number where one is measured: dense
  conv/GEMM on the Orin GPU and DLA, INTERPOLATED from the calibrations.
  Every other entry is **unknown**, not guessed. A stage mapped to an engine
  whose efficiency is unknown gets no service time and is reported as a gap
  -- the same discipline as Phase 2's unanchored silicon.

Supported / unsupported is different from efficiency and is known
structurally. An engine without a format at or above a stage's precision
floor cannot run it: **fail closed, never promote** (parent-plan D8).

What the calibrations support, read against the composed peaks:

| Pair | Efficiency | Basis |
|---|---|---|
| dense_conv_gemm / GPU / FP32 | 0.338 (0.280-0.397) | Two MAXN PyTorch FP32 GEMMs at n=2048, over 16 SM x 256 x 1.3 GHz |
| dense_conv_gemm / DLA / INT8 | 0.0226 (0.013-0.141) | Median of 18 batch-1 TensorRT conv layers that ran wholly on a DLA, over 16,384 x 1.6 GHz per DLA; launch overhead included |

The rest of the calibration corpus does not source an entry:

- There is no GPU INT8 run.
- The GPU FP16 runs cannot be placed against a peak, because the SM template
  states no FP16 rate and the registry `spec.json` ops-per-clock disagree
  with NVIDIA's 275 TOPS.
- The ResNet-18 DLA run fell back to the GPU for 76% of its layers.

**Consequence, recorded rather than worked around:** on Orin, every stage
under `default_v1` is a gap. The two measured pairs are not formats these
stages run in, because Class A runs in INT8 on the GPU. The per-engine path
is exercised end to end by tests with a synthetic table, and it produces
numbers for real designs only as measurements arrive. `annex_v1` is the
working path until then.

### P3-D2. Kernel classes are assigned from each stage's derivation

The parent plan keys efficiency on `kernel_class`; the annex's stages do not
carry one. Each of the 19 gets one from its derivation basis (SGM is
`cost_volume_dp`, detection `dense_conv_gemm`, MPC `small_qp`, and so on), in a
table with a one-line reason per stage. It is a classification, not a number,
so it needs a rationale rather than a source.

### P3-D6. Class C runs in FP32; engines are pools of servers

- The annex's Class C is "FP32/FP64" and does not say which stage needs
  which. FP32 is taken as the floor. That is optimistic for a stage that
  truly needs FP64, and is stated where the floor is defined.
- A class runs in the lowest format the engine has at or above its floor. An
  engine with no such format cannot run the class, so the stage is
  infeasible there.
- A GPU is one server, because a stage launches across all its SMs. A CPU
  core or a DLA instance is one server of a pool. A stage runs on one server
  and owns it, which is the annex's lower bound. Multithreaded CPU stages and
  contention come in Phase 5.
- Stage service time is `max(compute, DRAM)`: the stage owns the sustained
  bandwidth while it runs. The pooled table keeps the annex's compute-only
  service time, so the annex reproduces exactly. DRAM is checked
  separately, as utilization.

### P3-D7. Sources added to the IP library for this phase

- **Cortex-A78AE per-core peaks.** Phase 2 deferred these. They come from
  the Arm Cortex-A78AE Software Optimization Guide (PJDOC-466751330-14665,
  Issue 4.0), section 3.1: FMLA and SDOT each issue 2 per cycle on 128-bit
  vectors. That gives FP32 16, FP64 8, FP16 32 and INT8 64 ops per clock per
  core.
- **LPDDR5 bandwidth: 204.8 GB/s** (Orin Technical Brief, Table 1), on the
  PHY template as a `memory_interface`.

A first result from these: far flight's own DRAM demand (306 GB/s) exceeds
Orin's peak before any compute is considered. Its utilization against the
65% sustained fraction is 2.3.

### P3-D3. The acceptance, restated against the data that exists

The parent plan's criteria assumed five regimes and complete silicon.

1. **Annex reproduction:** with `annex_v1`, the analyzer reproduces the
   annex's oversubscription and stages-over on all 18 profiles, and on the
   two regimes it matches the argument document's 16.2 and 17.3 s/s within
   the annex's own 4% cross-check. Minimum usable, contested and fast flight
   have no per-stage data (Phase 1's axis decision).
2. **Stages over:** match the annex per profile.
3. **Orin at 25 W versus the measured 101.8 GOPS/W:** the parent plan asks
   for the same order of magnitude. Leakage comes from area, and Orin's
   area is a lower bound, so its power is a lower bound and its GOPS/W an
   upper bound. The comparison is reported as a bound and passes only if the
   order of magnitude holds anyway.

### P3-D4. Power uses the models the repo already has

Dynamic energy is the node's per-op energy times the architectural overhead
of the engine's class (`architectural_energy`). Without that overhead a CPU
looks 10-50x too efficient (parent-plan risk 1). Leakage is area times the
node's leakage density, which on Orin is a lower bound (P2-D7). Every power
figure says whether it is complete.

### P3-D5. What Phase 2 deferred to here

- `to_compute_product()` (P2-D1) needs the power section this phase
  produces. It lands in 3.3 if the emitted blocks can be filled from composed
  data alone; otherwise it stays deferred with the reason stated.
- Per-block W/mm^2 (P2-D2) lands in 3.3, for anchored blocks only.

## PR sequence

| PR | Content |
|---|---|
| 3.1 | Kernel classes on the 19 stages; the efficiency schema; `annex_v1`; `default_v1` from the Orin calibrations plus explicit unknowns. |
| 3.2 | `mapping.py` (explicit and greedy, one engine per stage) and `schedule.py` (utilization, shared DRAM against sustained bandwidth, constraint ratios, the reactive chain). |
| 3.3 | Power and energy roll-up with `--gate-idle` and the thermal check; W/mm^2; the `to_compute_product` decision. |
| 3.4 | `SoCAnalyzer`, `SoCAnalysisResult` (JSON-first, section 5.8), `cli/analyze_soc.py`, confidence propagation. |

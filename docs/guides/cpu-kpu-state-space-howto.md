# Generating the CPU + KPU design state space

How the figures in `docs/assessments/cpu-kpu-design-state-space.md` are
produced, what each one means, and the commands that regenerate them.
Everything here runs from a checkout with `embodied-schemas` installed; no
hardware is needed, because the measurement step has already happened and
its results are in the repository.

## The question the method answers

> Which KPU fabric, with how many CPU cores, on which memory system, in
> which process node, carries which mission capability?

It is answered in three parts, because no single number can carry it:

| Part | Question | Tool |
|---|---|---|
| **Requirement** | What would each engine have to sustain? | `cli/analyze_required_efficiency.py` |
| **Ceiling** | What could the KPU fabric sustain at best? | `cli/analyze_kpu_ceiling.py` |
| **Matrix** | Both, against every mission and every configuration | `cli/analyze_mission_matrix.py` |

The matrix is the join. The other two exist to show their working for one
design at a time.

## What is measured, modelled, and neither

The method never fills a gap with a plausible number. Each figure is one of
three things, and the tools label which:

- **Measured.** CPU and GPU kernel efficiencies, from an Orin Nano with
  clocks pinned and verified (`soc_designs/efficiency/orin_nano_measured_v1.yaml`,
  44 entries, 40 CALIBRATED). See
  `docs/guides/soc_kernel_benchmarks_on_jetson.md` for how they were taken.
- **Derived from stated data.** Requirements (workload ops over stated
  peaks), the domain-flow ceilings (tile geometry, clock and bandwidth),
  the datapath energy floor (the process node's energy per op).
- **A gap.** No KPU kernel has been measured, so no KPU service time
  exists. Gaps are reported, never filled, and any total computed over one
  is labelled a lower bound.

**No verdict is ever `yes`.** The matrix proves violations. A point that
survives every test is `open`, because a complete proof needs a schedule
and a power figure that no gap-free input supports yet.

## The axes

| Axis | Values | Where they come from |
|---|---|---|
| KPU fabric | H64, T64, T128, T256, T512 | `embodied-schemas` ComputeProducts |
| Process node | `gf_12fdx`, `tsmc_n16`, `tsmc_n7` | one design per fabric *at the node its SKU states a clock for* |
| CPU | 1, 2, 3 clusters of 4 Cortex-A78AE | `block:cpu.count` |
| Memory | LPDDR5 256b (204.8 GB/s), LPDDR5X 256b (273), LPDDR5X 512b (546), HBM3 one stack (819) | `block:memory.ip`, each sourced to JEDEC |
| Mission | 18 capabilities across 6 form factors | `branes_7tier_v1` |

13 designs x 3 CPU counts x 4 memory systems x 18 missions = 2,808 points,
plus the Phase 6 ladder's hand-written designs, in about 9 seconds.

**Fabric and node are paired, not crossed.** A SKU states its clock at one
node; composing that core elsewhere makes the clock provisional and every
service time with it. `tools/generate_kpu_soc_designs.py` writes one design
per (fabric, node) the catalog actually states, so no point rests on a
provisional clock.

## The division of labour between CPU and KPU

The default mapping is **schedule-aware**: the accelerator takes only the
kernel classes the domain-flow model has a schedule for -- dense GEMM and
convolution, attention, weight-stream decode, elementwise norm -- and the
CPU carries the rest.

This matters more than it sounds. The alternative rule, `--mapping
capability`, gives the accelerator every class whose precision it can run.
On a T-series KPU that is all of them, so the CPU requirement reads zero
and the state space says nothing about the CPU at all. The schedule-aware
rule is what a CPU IP partner has to size for, and it uses no new
assumption: the classes it withholds are exactly the ones
`graphs.estimation.soc.domainflow.NO_SCHEDULE` already names, with reasons.

## How each column is computed

**Required efficiency.** For a stage on one server of an engine,

```
dense_seconds = sum over precision classes of
                share x ops_per_call / peak(that class's format)
```

and the engine's requirement is `sum over its stages of
dense_seconds x rate / servers`: the fraction of dense peak it must hold
for the profile to fit. Exact arithmetic on the workload's ops and the
design's peaks. Above 1, no efficiency suffices -- the silicon is too
small, not too slow.

**Sustained throughput.** The same requirement in GOP/s per core and TOPS
on the fabric, so a partner can hold their own core against it without
adopting this model's peaks. This is the IP-neutral form of the ask.

**Utilization at measured efficiencies.** Each stage priced at the measured
table's own figures. A stage counts only when the table prices *every*
format it runs in, so the figure is a lower bound whenever a stage is
unpriced, and a lower bound above 1 is a proven violation.

**Ceiling.** The domain-flow model's upper bound per kernel class: the
least of the wavefront schedule, the compulsory DRAM traffic and operand
delivery. A stage's time at the ceiling is its occupancy over that ceiling;
above 1 for the engine, not even a flawless schedule carries the mission.

**DRAM utilization.** The profile's bytes per second against the memory
system's peak times the sustained fraction (0.65 for scattered access).

**Datapath energy floor.** Every op charged once at the process node's
energy per op for its engine's library and format. No efficiency enters,
so nothing can come in under it. A `+` in the text output marks a floor
with gaps -- a format the node states no figure for. A block whose tiles
span two logic libraries, as a T-series core's do, is priced by the share
each library issues (`datapath_mix`), so its ops are charged rather than
skipped.

## The commands

### The charts

```bash
python cli/report_mission_frontier.py -o docs/assessments/mission-frontier.html
python cli/report_mission_frontier.py --mission "air superiority" -o one.html
python cli/report_mission_frontier.py --format json -o points.json
```

```bash
python cli/report_pipeline.py -o docs/assessments/pipeline-demand.html
python cli/report_pipeline.py --design kpu_t256_n7 --cpu-clusters 3
python cli/report_pipeline.py --mission humanoid_cobot_human_adjacent_contact_rich -o one.html
```

One row per pipeline stage for the chosen mission, banded by tier, showing
what the stage demands per second of mission and what each engine gives it.
Every stage is costed on **every** engine, not only the one the schedule
picked, so the division of labour can be argued with rather than assumed:
the page states what the CPU would have cost for a stage the KPU took, and
what the KPU would have given for one the CPU kept.

Bars are shaded by provenance -- a measurement, or the domain-flow ceiling
-- and a bar past the full mark, in red, is a stage that alone needs more
than one whole engine, before anything else is scheduled beside it. A cell
with no bar names the missing figure: *no INT8 figure (FP32 measured)*
means the benchmark run exists for one format and not the other, while *no
FP32 figure, no schedule* means the domain-flow model has no schedule for
that kernel class on the fabric. Every figure is repeated in a table under
each chart.

```bash
python cli/report_state_space.py -o docs/assessments/state-space.html
python cli/report_state_space.py --headroom 2 -o with-margin.html
python cli/report_state_space.py --format json -o space.json
```

Every mission down, every configuration across, one cell per pair, shaded
by real-time factor. Because that factor uses the best efficiency anything
states -- a measurement where one exists, the domain-flow ceiling otherwise
-- a cell short of the bar is **proven short** and a cell that clears it is
only **not ruled out**. The legend says "still standing" for that reason
and never "feasible".

Columns are grouped by CPU core count and ordered by design within each
group, so a pattern that follows one dimension reads as a shape: survivors
filling the right-hand third mean the core count decided it, survivors in
narrow repeated stripes mean the design did. Dragging across the columns
selects a slab and reports what it shares and how many missions survive in
it; `--headroom` (and the control on the page) raises the bar from bare
real time to a multiple of it.

One chart per mission on the energy-performance plane. **Energy per
operation** is the mission's own arithmetic at the node's energy per op,
weighted over each stage's formats and each engine's libraries -- a floor,
because no efficiency and no overhead enter it. **Real-time factor** is the
share of the mission's required rate the configuration could carry, using a
measurement where one exists and the domain-flow ceiling otherwise -- a
ceiling, so a point below 1.0 is proven short. The **envelope** is the
staircase over the points: nothing in the catalogue sits above or left of
it.

A block whose datapath spans several libraries states the share each issues
(`datapath_mix`, generated from the SKU's tile classes), so a T-series
fabric's INT8 op is priced as 89.8% balanced-logic and 10.2% hp-logic
rather than being charged wholly to either, or left a gap.

### The whole state space, as a spreadsheet

```bash
python cli/analyze_mission_matrix.py --state-space --all --output state-space.csv
```

One row per (design, CPU count, memory, mission), with every column above.
This is the file the assessment's tables are cut from.

### One configuration against one mission, readably

```bash
python cli/analyze_mission_matrix.py \
    --design kpu_t128_n7 --mission humanoid_cobot_human_adjacent_contact_rich \
    --cpu-clusters 3 --memory lpddr5x_phy_512b --verbose
```

`--verbose` prints why each decided point failed.

### What one design would have to achieve, stage by stage

```bash
python cli/analyze_required_efficiency.py \
    --design kpu_t256_n16 --regime "air superiority" \
    --efficiency orin_nano_measured_v1 --verbose
```

### What the KPU fabric could do at best

```bash
python cli/analyze_kpu_ceiling.py --sku kpu_t256_32x32_lp5x16_7nm_tsmc_hpc --ceilings-only
python cli/analyze_kpu_ceiling.py --design kpu_t256_n7 --all --verbose
```

### Area, power and DRAM across the same space

```bash
python cli/sweep_soc.py --study kpu_cpu_ladder --pareto area,power --union
```

### Regenerating the inputs

```bash
python tools/generate_kpu_ip.py           # KPU cores from the SKUs
python tools/generate_kpu_soc_designs.py  # one design per core, at its node
python tools/generate_kpu_ceilings.py     # the domain-flow ceilings
```

Each takes `--check`, which fails if the committed file no longer matches
what the model produces. The test suite runs all three.

## Changing the space

- **A new KPU fabric or node:** add the SKU to `embodied-schemas`, then its
  core to `KPU_CORES` and `CORE_NODES` in
  `src/graphs/hardware/soc/kpu_cores.py`, then run the three generators.
- **A different CPU:** add an IP template under `soc_designs/ip/` with its
  per-core `ops_per_clock` and clock, and point a design's `cpu` block at
  it. The requirement in GOP/s per core does not change; what changes is
  how much of it one core covers.
- **A different memory system:** add an IP template with
  `memory_interface.peak_gb_per_s` and a source, then pass it to
  `--memory`.
- **A new mission:** add a profile to the workload with its sensors, rates,
  power budget and deadline.

## Reading a result honestly

1. **Check the confidence and the lower-bound flags first.** A utilization
   of 0.4 that omits ten stages says nothing about the design.
2. **A `no` is a proof; an `open` is an absence of one.** Nothing here
   licenses "this configuration works".
3. **Requirements are per engine and per server.** 32% of dense peak across
   12 cores is not the same ask as 32% on one.
4. **The KPU columns rest on a ceiling, not a measurement.** Until a KPU
   kernel is measured, every KPU verdict is either "the geometry forbids
   it" or "open".

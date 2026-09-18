# SoC study Phase 4: sweeps, reports, the first study -- execution plan

Tracking: branes-ai/graphs#269, Phase 4. Parent plan:
`docs/plans/soc-microarchitecture-study-plan.md`, section 7. The decisions
of Phases 2 and 3 carry over, above all the strict rule: a gap stays a gap
(P2-D3, P3-D1, P3-D8). Started 2026-09-18.

## What a study can decide today

| Quantity | Orin-class | A KPU-heterogeneous design |
|---|---|---|
| Die area | Lower bound. 14 of Orin's silicon lines are unanchored. | The KPU block is complete: each KPU ComputeProduct's lines carry generator transistor counts. Any Orin IP the design reuses (CPU, PHY, IO) brings its gaps along. |
| Compute and oversubscription | Pooled `annex_v1` only, and it ignores the design. `default_v1` prices nothing on Orin. | `default_v1` has no KPU entry, so all gaps. `annex_v1` gives the same number as Orin. |
| DRAM feasibility | Decidable: demand against the stated LPDDR5 bandwidth. | Decidable once the design states its memory interface. |
| Power | Lower bound. There is no dynamic term under `annex_v1`. | Same. |

**Consequence for 4.3.** Under the strict rule, the Orin-versus-KPU comparison
can be decided on area (when both sides are complete, or one side's lower
bound already exceeds the other's complete figure) and on DRAM. It cannot
yet be decided on compute or power. Deciding it needs per-engine
efficiencies for the KPU's kernels and the Orin engines. Those come from
measurement, or from the Phase 5 domain-flow cost model. 4.3 is written up
on what is decidable, and names what is not.

## Decisions

### P4-D1. Study axes are explicit overrides, not list-valued design fields

The parent plan has list-valued fields inside a design become sweep axes.
That would make a design file mean two things: a design, or a family of
designs. A study instead names a base design and lists **overrides**, each
of one block field (`count`, `clock_ghz`) or one layout field, with the
values to sweep. The cross product of the overrides, nodes, profiles,
efficiency tables and `gate_idle` settings is the set of points. Designs
stay single designs, and a study says what varies.

### P4-D2. Pareto dominance is proven, not assumed (4.2)

A lower bound cannot dominate anything: its true value may be larger. Point
A dominates B on a minimized metric only when A's value is known exactly and
B's is either exact and no better, or a lower bound already at least A's.
Points that no proven relation separates are reported as **undecided**,
never placed on or off the front by their bounds.

## PR sequence

| PR | Content |
|---|---|
| 4.1 | `estimation/soc/study.py` (study schema, axis expansion, sweep runner) and `cli/sweep_soc.py`; study YAMLs in `soc_designs/studies/`. |
| 4.2 | Bound-aware Pareto and "union of regimes" minimum-design reports; plots as an optional extra. |
| 4.3 | `tools/generate_kpu_ip.py` (KPU ComputeProduct -> IP template, unpriced tile classes as gaps); `kpu_heterogeneous_h64`; the study, written up in `docs/assessments/soc-orin-vs-kpu-heterogeneous.md`. |
| 4.4 | MCP tool registration (run `cross-repo-checker` first). |

### P4-D3. A KPU on an SoC is a generated core, one server

`tools/generate_kpu_ip.py` turns a KPU ComputeProduct into an IP template:

- **Silicon:** its `silicon_bin` transistors. A *core* drops the chip's own
  memory PHYs and pads, which belong to the SoC.
- **Peak:** the tile table's exact ops per clock, for modeled formats only.
- **Clock:** its boost clock at its own node.

Tile classes that no `silicon_bin` line prices become unanchored lines.
Without that, the H64 would have looked complete while missing five tile
classes. A KPU runs a stage across the whole fabric, so it is one server,
like a GPU. `IPCompute.datapath_class` names the library its ALUs are in,
since a KPU also has an HP-logic NoC.

### Result of 4.3

Undecided on area, compute and power, because both sides are lower bounds.
Decided on DRAM: both designs fail far flight on memory alone. The write-up
lists what would decide the rest.


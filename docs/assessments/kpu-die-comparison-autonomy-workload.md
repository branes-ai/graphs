# Two KPU dies against the autonomy workload

The last open item on [graphs#268](https://github.com/branes-ai/graphs/issues/268):
compare the uniform **T64** against the heterogeneous **kpu_h64_auto1**, the
reference design the tile refactor produced. Both are 64-site dies at
TSMC N16 on the same 64 GB/s LPDDR5 bus; the T64 is a 6 W part with FP32, the
H64 a 2.8 W part with an ISP, an SGM core and a Navion-class VIO core and no
FP32 at all.

Reproduce with:

```
python cli/analyze_dies_on_workload.py                     # the table below
python cli/analyze_dies_on_workload.py --profile "air superiority"
python cli/analyze_dies_on_workload.py --achieved-to-peak 0.05
```

## Method and inputs

| Input | Dated | Role |
|---|---|---|
| `docs/workload-model/` + `BranesAI-Workload-Data-Annex.pdf` / `.xlsx` | 2026-09-17 | 19 stages costed forward from algorithm structure; 18 missions as sensor suites and update rates; the service-time model |
| `BranesAI-Autonomy-Compute-Requirements.pdf` | 2026-09-13 | The five operating regimes, and the 2% / 5% achieved-to-peak bands |
| embodied-schemas catalog | v0.14.0 | Both dies, their tile classes and their function cores with cited throughput, energy and **contract limits** |

Demand comes from `graphs.core.pipeline_workload`, which reproduces the
reference model to 3e-16 on all 18 profiles. Capability comes from the E3
engine descriptors. A stage lands in one of three ways -- absorbed by a
function core within its contract, run on programmable tiles at its precision
class, or **not runnable at all** -- and the verdict names the first
constraint that binds: precision, contract, bandwidth, compute.

The axis is the 18 mission profiles the model derives, with the two that are
named regimes labelled. Minimum usable, contested and fast flight are not
here: reconstructing them would mean fitting sensor suites to published
totals, which is the back-solving the annex refuses to do.

## The comparison, at 2% of peak

| form_factor | mission | regime | demand_tops | baseline_s_per_s | T64 s/s | T64 binding | T64 unrunnable | H64 s/s | H64 binding | H64 unrunnable |
|---|---|---|---|---|---|---|---|---|---|---|
| Edge AI device | Event detection & classification |  | 0.05 | 0.1 | 0.1 | none | 0 | 0.2 | none | 0 |
| Edge AI device | Multi-stream tracking & anomaly |  | 0.63 | 0.9 | 0.7 | none | 0 | 1.3 | compute | 0 |
| Edge AI device | Supervisory control & command grounding |  | 0.8 | 1.3 | 0.9 | none | 0 | 1.9 | precision | 1 |
| Drone | Inspection (structure / asset) |  | 1.76 | 3.2 | 2.0 | compute | 0 | 3.1 | precision | 10 |
| Drone | ISR (endurance / wide-area search) | far flight | 10.47 | 16.2 | 11.4 | compute | 0 | 22.4 | precision | 11 |
| Drone | Interceptor (terminal engagement) | air superiority | 12.46 | 16.6 | 12.7 | compute | 0 | 21.2 | precision | 10 |
| Quadruped | Inspection (plant walkdown) |  | 3.71 | 5.2 | 3.8 | compute | 0 | 6.5 | precision | 10 |
| Quadruped | Surveillance (persistent patrol) |  | 6.53 | 8.1 | 6.5 | compute | 0 | 11.5 | precision | 10 |
| Quadruped | ISR (dismounted, comms-denied) |  | 10.48 | 15.3 | 11.1 | compute | 0 | 20.4 | precision | 11 |
| AMR | Warehousing (structured aisles) |  | 1.34 | 2.2 | 1.4 | compute | 0 | 2.4 | precision | 9 |
| AMR | Logistics (mixed / dynamic yard) |  | 5.09 | 6.9 | 5.2 | compute | 0 | 9.0 | precision | 11 |
| AMR | Loading / unloading (manipulation) |  | 11.32 | 16.2 | 11.9 | compute | 0 | 22.6 | precision | 9 |
| Humanoid | Industrial automation (structured cell) |  | 10.67 | 18.0 | 11.8 | compute | 0 | 21.4 | precision | 9 |
| Humanoid | Cobot (human-adjacent, contact-rich) |  | 18.24 | 28.9 | 19.9 | compute | 0 | 37.4 | precision | 9 |
| Humanoid | House work (open-world, long-horizon) |  | 25.06 | 42.4 | 28.1 | compute | 0 | 54.0 | precision | 10 |
| Autonomous vehicle | SAE L2 / L2+ (partial automation) |  | 0.95 | 2.7 | 1.3 | compute | 0 | 2.5 | precision | 6 |
| Autonomous vehicle | SAE L3 (conditional automation) |  | 6.44 | 13.4 | 7.6 | compute | 0 | 12.1 | precision | 10 |
| Autonomous vehicle | SAE L4 / L5 (high / full automation) |  | 25.06 | 48.0 | 28.9 | compute | 0 | 50.4 | precision | 11 |

## Findings

### 1. Precision decides this, not throughput

The H64 cannot run **9 to 11 stages of most missions**, because 14 of the 19
stages carry an FP32/FP64 floor and the die has no FP32 format on any engine.
Bundle adjustment, LiDAR-inertial odometry, TSDF integration, ESDF
propagation, the MPC solve and the barrier-function filter are all absent
rather than slow. This is the trade the refactor made, and it is visible in
the verdict column as `precision` on **16 of 18 profiles**; the two
exceptions are the smallest edge devices, which run no odometry, no mapping
and no control loop at all.

It does not move with speed. At 2% of peak the H64 is feasible on 1 of 18
profiles; at 5%, 2; at a physically impossible 100% of peak, still **2**. The
T64 goes 3 -> 6 -> 9 over the same range, because its constraint is rate and
rate responds to assumptions.

### 2. A function core is rated for a configuration, and these missions exceed it

The absorption advantage is real but partly unusable as specified:

| Core | Rated for | Air superiority asks | Verdict |
|---|---|---|---|
| SGM (Li et al. ISSCC 2017) | 1920x1080, **30 fps**, 128 disparities | 1440x1080 at **40 fps** | over contract |
| VIO (Navion, JSSC 2019) | **752x480**, 200 features | **1440x1080** | over contract |
| ISP (Darkroom) | no stated limit | 62.2 of 475 Mpx/s | absorbed, 13% used |

Throughput alone would have said yes to the VIO core: 20 frames per second
demanded against 540 available. The 4.3x pixel mismatch is only visible
because the core states `config_limits` and the mission states its sensor
suite. Far flight runs stereo at 20 fps and is inside the SGM core's
contract; air superiority at 40 fps is not.

### 3. Absorbed traffic stays off the bus, and it is not enough

On far flight the stream links keep **32.8 GB/s** off the DRAM bus (306 ->
273 GB/s). Both dies then sit on the same 64 GB/s LPDDR5 interface against
100-306 GB/s of demand, so bandwidth is 1.6x to 4.8x short before either
fabric matters. The heterogeneous die's advantage here is real and an order
of magnitude too small to change the verdict.

### 4. The assumption that dominates every compute verdict

Both dies are charged the same achieved-to-peak ratio, measured on
general-purpose embedded silicon (0.008% to 12%, median 0.31%). The KPU's
whole argument is that it achieves a higher fraction of peak on this work,
and **this comparison does not test that claim**.

What it would take: the T64 would need **22.8% of peak** on far flight and
**25.4%** on air superiority to close them -- roughly twice the best figure
in the measurement corpus and seventy times its median.

The annex's own tile model (`docs/workload-model/tiles.py`) assumes the
advantage rather than deriving it: hand-set gains of 8x throughput and 80x
energy for stereo SGM, 25x/330x for state estimation anchored on Navion,
30x/250x for the FP64 solvers. Even with those gains it reports far flight at
**5.9 seconds of compute per second of flight**, so on its own numbers a
single 64-site die is about 6x short of the mission it was drawn for.

### 5. Both dies are a generation too small for the hard missions

Demand spans 0.05 to 25.06 TOP/s. The dies peak at 62.3 (T64) and 38.9 (H64)
INT8 TOPS, which sounds sufficient and is not: at any defensible
achieved-to-peak the hard missions need several such dies, and the missions
they do close (edge detection, multi-stream tracking, warehouse AMR) are the
ones below about 1.5 TOP/s.

## What this does not model

From the annex (section 8), all of which add demand: kernel launch and
scheduling overhead, inter-process communication, cold caches and contention,
drivers and thermal throttling, redundancy beyond the stated SAE L4/L5 stack.
Added here: no DVFS or power budgeting (the dies' TDPs are not checked against
the missions' power budgets), no floorplan or placement effects, no
multi-die allocations, and no per-stage mapping onto individual tile classes
-- capability is aggregated per precision class.

The three regimes without per-stage data (minimum usable, contested, fast
flight) keep only their published aggregates and get no verdict here.

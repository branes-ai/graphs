# Embodied AI Benchmarks

Yes, but the landscape is fragmented and the "capability/efficiency/accuracy" split is cleaner in some domains than others. Perception has mature accuracy benchmarks; manipulation and navigation have capability benchmarks with well-understood metrics; efficiency (compute, energy, sample, data) is the weakest and least standardized axis. Here is a critical inventory.

**Navigation — the most mature embodied benchmarks**

The Habitat Challenge (annual, 2019–present) is the closest thing to a stable leaderboard. Metrics are well-defined:

- Success Rate (SR): fraction of episodes reaching the goal within a step budget and distance threshold.
- SPL (Success weighted by Path Length): SR weighted by geodesic-optimal path length. Penalizes detours.
- SoftSPL: continuous variant that credits partial progress.
- Distance-to-Goal (DTG): mean residual distance at episode end.
- Collision rate and human-collision rate (in Social-Nav).

Specific tracks and their current (as of early 2026) approximate headline numbers on held-out test sets:

- PointNav (Gibson, MP3D, HM3D): effectively saturated on noiseless sensing and actuation (>95% SR). Remains non-trivial under noisy GPS/compass or visual-only settings.
- ObjectNav (HM3D-Semantics): still unsaturated; best published methods are in the 60–75% SR / 30–45% SPL range depending on the split. Known evaluation noise of a few SR points run-to-run — a common failure mode in reported numbers.
- ImageNav and Instance-ImageNav: generally harder; SR in the 50–70% range.
- Social Navigation (Habitat 3.0): SR is not the whole story — success rates are reported alongside "human collision rate" and "personal-space violation" metrics. No single standard.
- Vision-Language Navigation (R2R, RxR, REVERIE): Success Rate, Navigation Error, SPL, and nDTW (normalized Dynamic Time Warping) for path fidelity. RxR adds multilingual fidelity metrics.

Main critique: all of these are simulator-bound. Sim-to-real transfer numbers are reported ad hoc per paper, not against a shared benchmark. Habitat-Sim vs. iGibson vs. AI2-THOR produce incomparable scores for nominally the same task.

**Manipulation**

No single leaderboard dominates. The ones that matter:

- Meta-World (ML1/ML10/ML45/MT50): task-success rate with train/test-task generalization splits. Well-defined, widely reported, but critiqued for being low-dimensional and not representative of real manipulation.
- RLBench (100+ tasks, CoppeliaSim): success rate per task, number of demos needed. Common baseline for imitation learning.
- CALVIN: long-horizon, language-conditioned; reports task-completion rate over sequences of 5 subtasks (the "sequential success rate" is the headline metric; chain length at failure is the secondary one).
- LIBERO: lifelong learning benchmark — reports forward transfer, backward transfer, and success rate across four task suites (Spatial, Object, Goal, Long).
- VIMA-Bench: multi-modal prompt manipulation; four generalization levels (placement, combinatorial, novel-object, novel-task) with per-level SR.
- ManiSkill / ManiSkill2 / ManiSkill3: contact-rich tasks with standardized success conditions and reward functions; widely used for RL.
- RoboCasa: large-scale kitchen tasks in MuJoCo/RoboSuite; success rate with demo-efficiency reporting.
- COLOSSEUM: an RLBench-extension that systematically perturbs 14 environmental factors (lighting, texture, distractor objects) and reports robustness drop.

The serious critique: in all of these, reported numbers are sensitive to demonstration quality, action-space choice, and random seed. Meta-World specifically has known seed variance of 10+ points on some ML10 splits. Cross-paper comparison is unreliable without rerunning baselines.

**Generalist / foundation-model robot benchmarks**

This is the fastest-moving area and the least stable:

- Open X-Embodiment (OXE): a dataset (2M+ trajectories, 22 embodiments) rather than a benchmark per se. Evaluations on RT-1, RT-2, RT-X, Octo, OpenVLA are typically done on held-out task sets with per-task SR, but protocols vary by paper.
- SimplerEnv: attempts to reduce the sim-to-real evaluation gap by tuning simulator parameters against real-world success rates on Google Robot and WidowX. Reports per-task SR with an explicit real-vs-sim correlation metric — one of the few benchmarks that takes sim-to-real accuracy seriously.
- RoboArena: crowd-evaluated real-robot head-to-head comparisons of generalist policies; reports Elo-style rankings rather than absolute SR. Useful for ordering methods, less useful for absolute capability claims.
- DROID: large-scale teleoperation dataset (76k trajectories, 564 scenes); used mainly for pretraining with downstream evaluation on sub-benchmarks.
- ALOHA / Mobile ALOHA task suites: bimanual, often reported per-task with wall-clock training time.

None of these has a stable leaderboard with frozen held-out sets. Numbers published six months apart are frequently incomparable.

**Long-horizon, language-conditioned, instruction-following**

- ALFRED: household tasks from natural-language instructions in AI2-THOR. Metrics: Task Success Rate, Goal-Condition Success, Path-Length-Weighted variants. Test-unseen SR has been climbing steadily — current strong methods are in the 60–75% range.
- TEACh, DialFRED: dialog-augmented variants.
- BEHAVIOR-1K (Stanford, OmniGibson): 1000 household activities with 50 scenes. Reports per-activity success and a "primitive-action success" decomposition. More representative but computationally expensive.
- ALFWorld: text-based abstraction of ALFRED, useful for LLM-agent evaluation.
- Habitat Rearrangement / Home-Robot: pick-and-place across rooms; reports SR and sub-task success.

**Driving**

Distinct category, but often lumped in:

- CARLA Leaderboard 2.0: Driving Score = Route Completion × Infractions Penalty; plus per-infraction breakdowns (collisions, red-light, off-road). Frozen routes and held-out towns.
- nuScenes detection and prediction: mAP, NDS (nuScenes Detection Score), ADE/FDE for prediction.
- Waymo Open Motion: minADE, minFDE, mAP, miss rate.
- Argoverse 2 forecasting and sensor benchmarks.
- nuPlan: closed-loop planning with scenario-based metrics (comfort, progress, rule compliance).

These are the closest to "engineering-grade" benchmarks in embodied AI — accuracy metrics are unambiguous and held-out sets are well-curated. Weakness: they don't stress contested or adversarial conditions.

**Perception accuracy (component-level, not end-to-end)**

Standard and well-understood, worth mentioning because embodied systems inherit them:

- Detection: COCO mAP, LVIS, Objects365.
- Segmentation: mIoU on Cityscapes, ADE20K, COCO-Stuff; PQ (panoptic quality).
- 3D: KITTI / nuScenes / Waymo 3D detection mAP, AMOTA for tracking.
- Depth: AbsRel, δ<1.25 on KITTI/NYUv2.
- Pose estimation: AUC-ADD on YCB-Video, LM-O.
- Open-vocabulary: LVIS-minival, ODinW.

**Efficiency benchmarks — the weak link**

This is where honest reporting breaks down:

- Sample efficiency: sometimes reported as "SR at N demos" or "SR at M env-steps." Meta-World and RLBench papers do this inconsistently.
- Data efficiency: OpenVLA-style papers report performance vs. pretraining dataset size, but there is no standard dataset-size axis.
- Compute / latency / energy: no equivalent of MLPerf for embodied AI. MLPerf Inference (Edge) and MLPerf Tiny cover relevant hardware but not robotic workloads. A few academic efforts — EmbodiedGPT, RoboFlamingo, and some Isaac-Lab benchmarking posts — report tokens/sec or control-loop Hz, but nothing standardized.
- Training wall-clock and FLOPs: inconsistently reported; often absent in robot-learning papers despite being central to reproducibility.
- Memory footprint and on-device inference: essentially unreported in published leaderboards.

This is a genuine gap and a defensible claim for an invention disclosure. If your invention has a concrete latency, power, or sample-efficiency advantage, there is no accepted yardstick to compare against — which is both an opportunity (you can define the metric) and a liability (reviewers will push back on self-defined metrics).

**Honest assessment of the state of play**

Three critiques worth stating plainly:

1. **Reproducibility is poor.** Most embodied benchmarks have simulator-version, GPU-determinism, and seed-variance issues that make published numbers differ by 5–15 points across runs. The Habitat team has published explicitly on this; most others have not.

2. **Sim-to-real correlation is weakly established.** SimplerEnv is the exception, not the rule. Reported "99% success in sim" means little about field performance. For defense use cases this matters more than leaderboard position.

3. **No accepted adversarial or robustness benchmark exists for embodied policies** in the way RobustBench or ImageNet-C does for classification. COLOSSEUM is the closest attempt; MITRE ATLAS provides a taxonomy but not an evaluation harness. Claims of "robust to contested environments" therefore have no shared scoring methodology today.

If you are sizing an invention against these: perception accuracy metrics are solid, navigation capability benchmarks are usable with caveats, manipulation benchmarks are noisy, generalist-model benchmarks are in flux, and efficiency benchmarks are effectively absent.
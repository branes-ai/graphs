# Industry Benchmark Assessment

There is no single, authoritative "embodied AI for military" benchmark suite today — which is itself the most important thing to state in this section of a disclosure. What exists is a patchwork of academic simulation benchmarks, DARPA challenges, MIL-STDs that predate modern AI, and emerging DoD/NIST AI policy. Below is a critical map of the landscape.

**Academic / open embodied-AI benchmarks commonly cited as baselines**

These are what competitors and research groups actually report numbers against, even for defense work, because nothing DoD-native has replaced them:

- Navigation and exploration: Habitat 3.0 / HM3D, Matterport3D, Gibson, and the PointNav / ObjectNav / ImageNav / SocialNav tracks. Metrics: Success Rate (SR), Success weighted by Path Length (SPL), SoftSPL.
- Manipulation: RLBench, Meta-World (ML1/ML10/ML45), CALVIN, LIBERO, VIMA-Bench, ARNOLD, BEHAVIOR-1K. Metrics: task success, generalization to unseen tasks/objects.
- Driving / off-road: CARLA Leaderboard 2.0, nuScenes, Waymo Open Motion, and for unstructured terrain the RELLIS-3D and RUGD datasets. DARPA RACER reported against internal scenarios but did not publish a reusable benchmark.
- Aerial: AirSim, Flightmare, and the AlphaPilot / Lockheed DRL Racing benchmarks.
- Swarm / multi-agent: DARPA OFFSET scenarios, MAgent, Google Research Football, Neural MMO — none are military-validated.
- Foundation models for robots: Open X-Embodiment (RT-X), RoboArena, Eval-Bench. These are weeks-old by industry standards; expect churn.

Any claim of "state of the art" in a patent should be pinned to a specific version of one of these — these benchmarks change faster than patent prosecution timelines.

**DARPA programs that function as de facto benchmarks**

DRC (2015), SubT (2018–2021), OFFSET, RACER, ANSR, Squad X, AIR (AI Reinforcements), and Learning Introspection Control. These define capability bars (e.g., SubT's 60-minute autonomous subterranean run with degraded comms) that vendors cite, but the scoring rubrics and course layouts are not reproducible outside the program.

**Military and dual-use standards that bound the problem**

These are mostly platform/safety standards, not AI-performance standards:

- MIL-STD-882E — system safety, including software hazard severity/probability matrices. Applied to autonomy but awkwardly — it assumes deterministic failure modes.
- MIL-STD-810H — environmental (shock, vibration, thermal, humidity, salt fog). Relevant for embodiment, not cognition.
- MIL-STD-461G / MIL-STD-464D — EMI/EMC and E3. Matters for contested-EM claims.
- MIL-STD-1472H — human-systems integration, relevant for human-on-the-loop claims.
- MIL-STD-3022 / DoDI 5000.61 — V&V and accreditation of models and simulations (VV&A). This is the closest thing DoD has to a rulebook for how you'd qualify a simulation used to train or evaluate an embodied policy.
- DoDD 3000.09 (updated Jan 2023) — autonomy in weapon systems; mandates "rigorous hardware and software V&V and realistic system developmental and operational T&E." Intentionally vague on what "rigorous" means for learned systems.
- DoDI 5000.89 — T&E policy.
- CJCSI 3170.01 / JCIDS — capability requirements.
- JTA / DISR — technical architecture constraints.
- JAUS (SAE AS-4) and STANAG 4586 — interoperability for unmanned systems messaging. These predate modern ML.

**AI-specific standards — still maturing**

- NIST AI RMF 1.0 (Jan 2023) and the Generative AI Profile (NIST AI 600-1, 2024). Defines Govern/Map/Measure/Manage functions but is voluntary and non-quantitative.
- NIST AI 100-2 E2023/E2025 — Adversarial Machine Learning taxonomy (evasion, poisoning, privacy, abuse). Useful for structuring a threat model in the disclosure.
- MITRE ATLAS — adversarial ML TTPs catalog, increasingly cited in DoD red-team plans.
- ISO/IEC 22989 (concepts), 23053 (ML framework), 23894 (risk management), 25059 (quality model for AI systems), 5259-series (data quality for ML), TR 24027 (bias), TR 24028 (trustworthiness), 42001 (AI management system).
- IEEE 7000-2021, 7009 (fail-safe design of autonomous systems), 2857 (privacy), P2817 (verification of autonomous systems) — still largely aspirational.
- UL 4600 — Standard for Safety for the Evaluation of Autonomous Products. Automotive-origin but the only published safety-case standard for learned autonomy.

**Cyber / data-handling layer (non-optional for DoD acquisition)**

- NIST SP 800-37 RMF, 800-53 Rev. 5 controls, 800-207 Zero Trust, 800-218 SSDF, 800-218A (AI extension, draft).
- CMMC 2.0 (contractor) and DISA STIGs.
- DoDI 8500.01, 8510.01, 8520.03.
- For model supply chain: EO 14110 artifacts and the DoD CDAO Responsible AI Toolkit (T&E portion).

**How competitors / legacy systems actually perform against this**

Honestly: most deployed military autonomy (Switchblade, ALIAS, MQ-9 autonomy kits, Bolt-On Autonomy Kit for ground vehicles, RCV-M/L autonomy stacks from Kodiak/Forterra/Overland AI) is qualified through bespoke operational test scripts and MIL-STD environmental compliance — not against published ML benchmarks. Vendors typically self-report sim metrics (CARLA, Isaac Sim, AFSIM-coupled) and a handful of controlled field trials. Published numbers are sparse because scenarios are classified or proprietary. Academic benchmark scores are almost never reported in DoD source selections.

**Concrete gaps an invention in this space can legitimately claim to address**

- No standardized benchmark for DDIL comms (Denied, Degraded, Intermittent, Limited) — policies are usually tested with ideal connectivity.
- No widely accepted metric for sim-to-real transfer under contested RF / GNSS-denied conditions.
- No public adversarial-robustness benchmark for embodied perception stacks comparable to RobustBench in image classification.
- No standard for multi-modal, multi-platform real-time data fusion latency and accuracy under jamming.
- VV&A frameworks (MIL-STD-3022) do not address learned components — specifically, there is no accepted method for accrediting a simulation used to train an RL policy that will be fielded.
- DoDD 3000.09's "rigorous V&V" has no quantitative test floor; there is room for a proposed metric suite.
- No standard secure-enclave / TEE benchmark for on-platform inference combining latency, power, and attestation.
- MITRE ATLAS tactics are cataloged but there are no standard evaluation harnesses that exercise them against embodied policies end-to-end.
- UL 4600 style safety cases exist for AVs but no military analog for lethal or semi-lethal autonomy.

One critical caveat for the disclosure: if you cite a specific benchmark version or standard revision, pin the date. These documents — especially NIST AI RMF profiles, DoDD 3000.09 implementation, and the embodied-AI academic benchmarks — are revised on timelines shorter than patent pendency, and claims that depend on "current SOTA on Benchmark X" age badly. Better to frame the gap in capability terms (e.g., "no published benchmark evaluates policy performance under simultaneous GNSS denial and intermittent C2") than in named-benchmark terms.
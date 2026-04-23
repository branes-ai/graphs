# SAE Autonomy Taxonomy

I need to push back on the premise before answering. **SAE has not published compute or energy requirements tied to the L0–L5 taxonomy.** Those numbers are widely circulated but they come from chip vendors and analyst reports, not from SAE standards.

**What SAE has actually published**

- **SAE J3016** — *"Taxonomy and Definitions for Terms Related to Driving Automation Systems for On-Road Motor Vehicles."* Current revision is J3016_202104 (April 2021), superseding 2014/2016/2018 versions. This is the L0–L5 document. It is strictly definitional: it describes how the dynamic driving task (DDT), DDT fallback, and operational design domain (ODD) are allocated between driver and system at each level. It contains no numerical compute, memory, power, or energy targets. It is not even a performance specification — SAE explicitly states it is not a safety standard.
- **SAE J3131** — *"Definitions for Terms Related to Automated Driving System Reference Architecture."* Architectural vocabulary, no quantitative requirements.
- **SAE J3018** — *"Safety-Relevant Guidance for On-Road Testing of Prototype Automated Driving System (ADS)-Operated Vehicles."* Process guidance, not compute budgets.
- **SAE J3197, J3237, J3163** — related taxonomies for shared mobility, ADS-DV cybersecurity terminology, and vehicle-to-infrastructure concepts.
- **ISO/SAE 21434** — cybersecurity engineering for road vehicles (replaced the withdrawn SAE J3061).
- **ISO 26262** (functional safety) and **ISO 21448 SOTIF** (safety of the intended functionality) are the adjacent standards that actually drive engineering requirements — and these are ISO, not SAE.

SAE *does* publish peer-reviewed technical papers through **SAE MobilityRxiv** and the **SAE International Journal of Connected and Automated Vehicles** in which individual researchers have estimated compute and energy requirements by level. But these are author-authored technical papers, not SAE-ratified standards, and they disagree with each other.

**Where the "compute per SAE level" numbers actually originate**

The commonly repeated figures (e.g., L2 ≈ 10 TOPS, L3 ≈ 30–60 TOPS, L4 ≈ 300–500 TOPS, L5 ≈ 1000+ TOPS; power envelopes of 100 W at L2 rising to 500–2000 W at L5) trace to:

- NVIDIA Drive platform marketing and whitepapers (Xavier ≈ 30 TOPS, Orin ≈ 254 TOPS, Thor targeting ~2000 TFLOPS FP8).
- Mobileye EyeQ generational roadmaps (EyeQ5 ≈ 24 TOPS, EyeQ6H ≈ 34 TOPS, EyeQ Ultra ≈ 176 TOPS).
- Lin, Wang, Xu, Kang, Mars, Tang — *"The Architectural Implications of Autonomous Driving: Constraints and Acceleration"*, ASPLOS 2018 — one of the most-cited academic sources for end-to-end pipeline budgets; reports L4 pipelines at several hundred watts.
- Liu, Tang, Zhang, Yuan, Gaudiot — *"Computing Systems for Autonomous Driving: State of the Art and Challenges"*, IEEE IoT Journal 2020.
- Industry analyst reports: McKinsey, Yole Développement, S&P Global Mobility (ex-IHS Markit), Strategy Analytics, Gartner.
- Tesla AI Day 2021/2022 slides for HW3/HW4 envelopes.
- Qualcomm Snapdragon Ride roadmap disclosures.

If you want a defensible citation in patent prosecution, the Lin et al. ASPLOS 2018 paper and the Liu et al. IEEE IoT 2020 survey are the strongest academic sources. The NVIDIA and Mobileye numbers are useful industry anchors but are vendor-published.

One tangentially relevant SAE effort worth naming: **SAE J3237** (*Capabilities and Requirements for Autonomous Vehicle Testing*) and the draft **J3400** documents — but neither prescribes compute/energy by level.

A separate standard that *does* attempt level-based capability mapping for unmanned systems is **NIST Special Publication 1011** — the **Autonomy Levels for Unmanned Systems (ALFUS) Framework.** It predates J3016, covers unmanned ground/air/maritime systems, and scores autonomy along three axes (mission complexity, environmental complexity, human independence). It never achieved J3016-level traction, and it too does not prescribe compute/energy budgets.

If you want, I can run a web search for any SAE document published between my knowledge cutoff (May 2025) and today (April 2026) — it is possible SAE has started a new work item I am not aware of, but as of mid-2025 no such standard existed.

**Why automotive AV is farther along than embodied AI and drone autonomy**

The honest answer is that the gap is real in some dimensions, overstated in others, and the reasons are mostly structural rather than technical.

*Structural reasons the automotive stack is ahead:*

1. **Capital concentration.** Waymo, Cruise (pre-2023), Tesla, Mobileye, NVIDIA, Aurora, Zoox, Pony, Baidu, Huawei, plus every OEM's internal program — cumulative investment exceeds $150B over roughly 15 years. Embodied AI and drone autonomy together have received a small fraction of that.
2. **Constrained ODD.** Roads are geometrically simple (effectively 2D manifolds), governed by codified rules, and instrumented with signage, lane markings, HD maps (TomTom, Here, DeepMap, Mobileye REM), and GNSS coverage. A drone operating BVLOS in contested airspace has none of those priors.
3. **Data scale.** Waymo has ~30M autonomous miles and billions of simulated miles; Tesla's FSD fleet has accumulated >5B miles of behavioral data; Mobileye REM has crowd-mapped hundreds of millions of road-km. No drone operator has anything remotely comparable, and no military embodied-AI program does either.
4. **Sensor standardization.** A few camera form factors, a handful of radar bands (77 GHz dominant), a converging lidar ecosystem. SWaP envelopes are generous — a car can carry 500 W of compute and 10 kg of sensors without affecting range meaningfully. A 2 kg quadrotor cannot.
5. **Regulatory scaffolding.** UNECE WP.29 (R157 on ALKS, R155 on cybersecurity, R156 on software updates), NHTSA ADS guidance, California DMV reporting, EU AI Act's high-risk categorization — automotive has regulatory pathways, however imperfect. Small UAS BVLOS autonomy under FAA Part 108 is still being finalized; military drone autonomy is governed by DoDD 3000.09 which is deliberately non-prescriptive on V&V.
6. **Safety-engineering frameworks.** ISO 26262, ISO 21448 SOTIF, UL 4600, and the ISO 8800 (AI safety in road vehicles) drafts give automotive an actual assurance vocabulary. There is no equivalent mature framework for drone autonomy or general embodied AI.
7. **Tier-1 supplier ecosystem.** Bosch, Continental, ZF, Denso, Aptiv, Valeo — mature suppliers of perception stacks, E/E architectures, and safety-rated compute. Drones have DJI and a fragmented long tail; military drone autonomy has mostly bespoke primes.
8. **Compute vendors built the products.** NVIDIA Drive, Qualcomm Ride, Mobileye EyeQ, Tesla HW4, Ambarella CV — automotive-grade inference silicon with ISO 26262 ASIL-D support is a real product category. Nothing comparable exists at the sub-100 g, sub-20 W drone tier; Jetson Orin Nano is the closest and it is not automotive-grade certified.

*Reasons the gap is narrower than the framing suggests:*

- Drone autonomy in narrow ODDs is arguably ahead of automotive L4. Skydio's obstacle-avoidance stack, ETH Zürich / UZH Robotics and Perception Group's FPV work (beating professional pilots in racing, *Nature* 2023), and industrial inspection drones operate with high autonomy in visually rich, unstructured environments on-device at <15 W. Automotive L4 still relies on HD maps, redundant lidar, and 300+ W of compute.
- Military loitering munitions with terminal autonomy (Switchblade 600, Lancet-3, some HERO variants, and published capabilities of newer systems) perform last-mile visual servoing under degraded GNSS — a capability automotive stacks do not require.
- Manipulation and dexterous control have advanced outside the automotive lineage entirely; modern VLA models (RT-2, OpenVLA, π0) handle task generality that no driving stack attempts.

*Real, specific reasons drones lag automotive on the axes you care about:*

- **SWaP**. A 2 kg quadrotor has a power budget in the 5–30 W range for all non-propulsion compute. That is 10–100× tighter than automotive, and it dominates every architectural choice. Running a 7B-parameter VLA on-board is not feasible at that envelope without aggressive quantization and custom silicon.
- **Operating envelope**. 3D motion with nonlinear aerodynamics (ground effect, prop wash, wind gusts), short reaction times (control loops at 500–1000 Hz vs. automotive 10–100 Hz), and catastrophic failure modes (loss of thrust = crash) narrow the design margin dramatically.
- **No equivalent of HD maps or V2X**. Automotive leans heavily on HD maps and increasingly on V2X. Drones operating BVLOS in contested or GNSS-denied environments cannot assume either.
- **Data poverty**. The public drone-autonomy dataset landscape (UZH-FPV, Blackbird, EuRoC MAV, Mid-Air, AirSim synthetic sets) is orders of magnitude smaller than automotive. There is no "Waymo Open Dataset" for drones.
- **Market fragmentation**. Automotive converges on a small number of stack architectures; drones span hobbyist, commercial inspection, delivery, defense ISR, and loitering munitions, each with incompatible requirements. No economies of scale on safety-rated compute.
- **Regulatory uncertainty**. FAA Part 108 BVLOS rule has been "coming soon" for years. Military autonomy rules under DoDD 3000.09 require human judgment over force but leave V&V unspecified. Without a regulatory target, engineering requirements cannot be fixed.
- **Contested environment requirements that automotive simply does not have**. Jamming, spoofing, active adversaries, denied communications, and operation in degraded visual environments are mandatory for military drone autonomy and absent from the automotive problem statement.

**One critical caveat on the SAE levels themselves**

The J3016 taxonomy is widely used but has been criticized — notably by Philip Koopman and others — as operationally ambiguous and unhelpful as a safety framework, because the level alone does not determine the ODD or the failure behavior. If your disclosure leans on SAE levels as a capability axis for embodied AI, be aware that even the automotive community treats them as a communication device rather than a rigorous engineering requirement, and applying them by analogy to drones or general embodied AI inherits all their ambiguity plus new problems (what does L3 mean for a manipulator? for a swarm?).

If you want, I can search for any SAE or ISO work items published since mid-2025 that might have introduced quantitative compute/energy requirements — but based on the state of the standards as of my cutoff, the claim that "SAE has published compute and energy requirements for L0–L5" is not accurate and should not be cited as such in a disclosure.

## SAE/ISO work items published since mid-2025

Search result: **no SAE or ISO standard has been published in the mid-2025 through April 2026 window that prescribes quantitative compute or energy requirements tied to autonomy level.** Several adjacent standards were published or are in active work, but none of them set numerical TOPS, watts, or joules-per-km targets. Details below.

**What was actually published in the relevant window**

*ISO/PAS 8800:2024 — "Road vehicles — Safety and Artificial Intelligence"* (published December 2024). This is the headline new document for AI in road vehicles and is the standard most likely to be cited against an invention in this space. It introduces an AI/ML product-development lifecycle, data-management requirements (provenance, quality, generalization validation), and treatment of systematic and random faults of AI elements. It complements ISO 26262 and ISO 21448 rather than replacing them. It does not quantify compute, memory, power, or energy requirements — it is a process and argument standard, not a hardware specification.

*ISO 34505:2025 — "Road vehicles — Test scenarios for automated driving systems — Scenario evaluation and test case generation"* (published 2025). Completes the ISO 34502/34503/34504 scenario framework. Scope is scenario-based V&V methodology; no compute or energy numbers.

*ISO 34504:2024* — scenario categorization with "tag"-based meta-attributes. Same caveat.

*SAE J3237 — "Operational Safety Metrics for Verification & Validation of ADS"* appears to have moved from WIP to a published Recommended Practice recently (cited as "recently published" in UNECE GRVA March 2025 workshop materials). It defines Dynamic Driving Task Assessment (DA) metrics for safety performance quantification. These are behavioral/safety metrics — time-to-collision, minimum distance, rule compliance — not compute or energy metrics.

*AVSC Best Practice for ADS-DV Immediate Post-Crash Behaviors* (February 2025). Behavioral guidance for what an ADS does after a crash. Not compute-related.

*SAE J3131_202203 — "Automated Driving Systems Reference Architecture"* remains the 2022 revision; no 2025/2026 re-issue located. Marked as WIP for further revision.

*SAE J3016* remains at the April 2021 revision. No update has been published since.

*UL 4600* remains at Edition 3 (March 2023). No fourth edition located.

*IEEE 2846* remains at the 2022 edition. No 2025/2026 revision located.

**What was published that actually discusses compute and energy — but as technical papers, not standards**

SAE Technical Paper **2025-01-8598**, *"Assessing the Energy Consumption Impact of Intelligent Driving Technologies on Electric Vehicles: A Comprehensive Review"* (Liu, Qi, Ou, April 2025). Literature review over 59 studies. Confirms that compute can dominate non-propulsion energy on CAEVs and surveys V2V/V2I offsets. It is the most direct recent SAE publication on the topic but is an author-authored technical paper, not a standards document — it has no normative force and sets no requirements.

SAE Technical Paper **2026-26-0691**, *"Software-Defined Vehicles: Architecting the Future of Intelligent and Connected Mobility"* (2026). Comparative analysis of NVIDIA Drive, Tesla FSD, and Qualcomm Snapdragon Ride HPC platforms. Discusses ISO 26262 and ISO/SAE 21434 compliance. Again, a technical paper surveying the landscape — not a normative standard.

Industry-side, the **S-CORE Project** launched in June 2025 (Bosch, QNX, Mercedes-Benz, others) aiming at an open-source core stack for software-defined vehicles. This is a software initiative, not a compute/energy standard.

**Observed industry compute envelope (vendor-reported, as of early 2026)**

For context, not as a standard: NVIDIA Drive Thor targets ~2000 TFLOPS FP8; Horizon Robotics Journey 6P ~560 TOPS; AutoX Gen5 claims ~2200 TOPS for driverless L4. These are vendor-published marketing figures and should not be cited as compliance targets.

**Assessment relative to your earlier claim**

The earlier answer stands: no SAE standard maps L0–L5 to quantitative compute or energy requirements, and the mid-2025 to April 2026 window does not change that. The active standards work in this window has gone into (a) AI safety lifecycle (ISO/PAS 8800), (b) scenario-based V&V (ISO 34502/34504/34505), and (c) operational safety metrics (SAE J3237). Compute and energy remain the domain of vendor spec sheets and peer-reviewed technical papers, not ratified standards.

For a disclosure, this pattern is itself defensible content: the standards ecosystem has prioritized safety-argument and V&V scaffolding over hardware budget specification, which leaves an open gap for quantitative compute/energy-per-capability requirements — particularly for embodied systems outside the automotive ODD.

Sources:
- [SAE J3016 update blog](https://www.sae.org/blog/sae-j3016-update)
- [ISO/PAS 8800:2024 — Road vehicles — Safety and artificial intelligence](https://www.iso.org/standard/83303.html)
- [Introducing ISO/PAS 8800 (SGS)](https://www.sgs.com/en/news/2025/04/safeguards-04625-introducing-iso-pas-8800-functional-safety-for-ai-in-road-vehicles)
- [Safety-Related Systems in Road Vehicles with AI — ISO/PAS 8800:2024 (UL Solutions)](https://www.ul.com/sis/blog/safety-related-systems-road-vehicles-artificial-intelligence-are-addressed-isopas-88002024)
- [ISO 34502:2022 — Scenario-based safety evaluation framework](https://www.iso.org/standard/78951.html)
- [ISO 34504:2024 — Scenario categorization](https://www.iso.org/standard/78953.html)
- [ISO 34505:2025 — Scenario evaluation and test case generation](https://www.iso.org/standard/78954.html)
- [SAE J3131_202203 — ADS Reference Architecture definitions](https://www.sae.org/standards/content/j3131_202203/)
- [SAE J3237 (WIP) — Dynamic Driving Task Assessment metrics](https://www.sae.org/standards/content/j3237/)
- [SAE 2025-01-8598 — Energy Consumption Impact of Intelligent Driving Technologies (technical paper)](https://saemobilus.sae.org/papers/assessing-energy-consumption-impact-intelligent-driving-technologies-electric-vehicles-a-comprehensive-review-2025-01-8598)
- [SAE 2026-26-0691 — Software-Defined Vehicles architecture paper](https://saemobilus.sae.org/papers/software-defined-vehicles-architecting-future-intelligent-connected-mobility-2026-26-0691)
- [AVSC releases best practice for ADS-DV post-crash behaviors (Feb 2025)](https://www.sae.org/news/press-room/2025/02/avsc-ads-dv)
- [AVSC-D-02-2024 — Best Practice for Core Automated Driving](https://www.sae.org/standards/content/avsc-d-02-2024/)
- [UL 4600 Edition 3 announcement (UL Solutions)](https://www.ul.com/news/ul-4600-edition-3-updates-incorporate-autonomous-trucking)
- [UL 4600 — Second Edition (UL Standards & Engagement)](https://ulse.org/standards-and-engagement/presenting-standard-safety-evaluation-autonomous-vehicles/second-edition)
- [IEEE 2846-2022 — Assumptions in Safety-Related Models for ADS](https://standards.ieee.org/ieee/2846/10831/)
- [SAE ORAD Committee ADS Standardization Activity (UNECE GRVA)](https://unece.org/sites/default/files/2024-05/GRVA-19-19e_0.pdf)

## Compute requirements for non-AV form factors

The premise is mostly right but needs one correction before the rest of the analysis holds up. The non-automotive form factors don't just have *higher* compute demand than AVs — they have *higher demand at lower power budgets*. It's a scissors, not a ramp. That scissors is the reason the incumbent story is more nuanced than a clean "Mobileye/Qualcomm/NVIDIA lose this market."

**Why the AV compute curve does not extrapolate cleanly**

An automotive AV stack is shaped by a small number of constraints that don't apply elsewhere. It operates on a 2D manifold with codified rules. It has HD-map priors and lane geometry. It has GNSS-plus-IMU-plus-wheel-odometry redundancy. Its control loops run at 10–100 Hz. It carries 300–800 W of compute in a 1,800 kg vehicle with forced-air cooling. It is certified under ISO 26262 / SOTIF, which assumes largely deterministic perception pipelines.

Take those constraints away and three things change simultaneously. The algorithmic complexity rises because the environment is 3D, unstructured, and without priors. The platform power budget collapses because you are now running on a 2 kg to 50 kg machine with passive or semi-passive cooling and a battery. And the safety regime changes from ISO 26262 to ISO 13482 / ISO 10218 / ASTM F3442 / RTCA DO-365, which are either less mature or fundamentally different in shape. The compute you actually need is not just "more" — it is structurally different: multi-timescale, mixed-precision, memory-bandwidth bound for transformer reasoning, deterministic at the low tier, flexible at the upper tier.

**Delivery drones**

Current platforms (Zipline Gen 2, Wing Hummingbird-class, Amazon MK30, Flytrex, Matternet M2) sit in the 20–80 TOPS / 10–25 W envelope, almost uniformly on Jetson Orin Nano or NX class silicon with a safety-certified flight controller underneath (Pixhawk-class STM32 or a custom ASIL-equivalent MCU).

Realistic extrapolation for certified BVLOS operation at scale under FAA Part 108 and ASTM F3442/F3442M-23 is in the 100–300 TOPS / 15–30 W range by 2028. Drivers: vision-based Detect-and-Avoid that has to meet RTCA DO-365B equivalency, multi-modal weather/wind prediction, terminal-area VIO under GNSS-degraded conditions, and all-weather precision landing.

This is the form factor most *compatible* with the incumbent automotive stack. NVIDIA Jetson already serves it. Qualcomm Flight RB5 targets it explicitly. Ambarella CV5/CV7 are credible. Mobileye is not a natural player because their stack is locked to automotive APIs. The binding constraint here is not TOPS — it is aviation-grade certification of the compute + software together, which neither NVIDIA nor Qualcomm has delivered yet. The winner in this segment is likely whoever first ships a DO-178C / DO-254 equivalent pipeline on a 30W inference SoC.

**Quadrupeds for surveillance**

Current platforms (Boston Dynamics Spot, ANYbotics ANYmal D/X, Ghost Robotics Vision 60, Unitree B2, DeepRobotics X30) are almost all on Jetson AGX Orin (275 TOPS / 60 W) paired with a separate real-time controller for locomotion. Locomotion itself — MPC-based whole-body control at 500–1000 Hz — is solved with a few CPU cores and is not compute-hungry. The compute demand comes from perception and mission autonomy.

Realistic extrapolation for 2027–2029 is 500–1500 effective TOPS / 60–100 W, driven by three things: semantic scene understanding (open-vocabulary detection, 3D scene graphs), natural-language mission tasking via VLM-class reasoning, and multi-robot coordination. Drivers are not locomotion — they are the upper tier.

Here the NVIDIA story still works in the short term because the Jetson AGX Orin is already the de facto platform and Thor will extend the envelope. But a 7B-parameter VLA at FP16 is 14 GB of weights alone; a 70 GB/s LPDDR5 bus on a Jetson is limiting even for inference-only workloads. This is where Hailo-15, SiMa.ai MLSoC, Ambarella N1, and Rockchip RK3588/3588S are eating into the Jetson position from below on TOPS-per-watt — they beat Orin on perception inference efficiency but lack the general-purpose Ampere GPU for VLA work.

**Humanoids for the home**

This is where your "not viable" claim is most defensible, and where the numbers get genuinely hard.

A humanoid for home deployment has to support simultaneously: whole-body control at 500–1000 Hz across 30–50 DoF (deterministic, safety-critical); contact-rich bimanual manipulation with visuo-tactile feedback at 30–100 Hz; a vision-language-action policy for task generality (currently 1–7 B parameters, trending toward 20–70 B); an LLM-class planner for language understanding and long-horizon task decomposition; and human-robot interaction (speech, gesture, intent) at interactive latency. All of this on a machine with a 1–3 kWh battery and roughly a 2–5 hour runtime target, which implies a sustained compute power ceiling of 100–200 W.

Current humanoid platforms have chosen divergent paths:

- Figure 02 and Apptronik Apollo: NVIDIA-class embedded GPU (Orin or Thor) for perception/VLA with cloud offload for heavy reasoning.
- 1X Neo and, reportedly, Figure's next generation: custom inference silicon co-designed with the VLA model architecture.
- Tesla Optimus: FSD chip (HW4-derived) with aggressive quantization.
- Boston Dynamics Atlas (electric): hybrid NVIDIA + custom control SoC.
- Unitree H1 / G1: Jetson Orin NX.

Realistic extrapolation for 2028–2030 production humanoids targeting home deployment: 2–10 effective TFLOPS (FP8/FP16) at 100–200 W, with 64–256 GB of high-bandwidth memory for VLA inference, plus a separate safety-rated real-time controller (FPGA or automotive-grade MCU) for locomotion and collision-force limiting. That is a different class of silicon from anything in the Drive, EyeQ, or Snapdragon Ride lineup. It is closer to a mobile workstation than to an ADAS SoC, and it has safety certification requirements that automotive silicon does not address (ISO 13482, ISO/TS 15066 for collaborative operation, and the emerging ISO/DIS 25785 for service robots).

**Assessment of the incumbents**

*Mobileye.* Effectively no credible extrapolation path into drones, quadrupeds, or humanoids. EyeQ is an automotive-perception fixed-function accelerator. The stack is closed, the APIs are automotive-specific, and Intel's internal robotics effort is organizationally elsewhere. Expect Mobileye to remain AV-only.

*Qualcomm.* Credible for drones (Flight RB5) and low-end mobile robots (QCS6490, QCS8550, Robotics RB series). Their mobile-SoC heritage (low-power, 5G-integrated, ISP-rich) is a genuine advantage at the sub-20 W tier. Not credible as the prime compute for humanoids — the memory bandwidth and matrix-math capacity are too far below the VLA inference requirement. Likely role: the "eyes and ears" SoC in a multi-chip humanoid, not the brain.

*NVIDIA.* The only incumbent with a serious full-stack robotics play (Jetson + Isaac Sim + Isaac ROS + GR00T foundation model + Thor SoC explicitly pitched as dual-use for AV and humanoid). Jetson AGX Orin is already the de facto quadruped brain; Thor is architecturally better suited to transformer inference than Orin. But three weaknesses are worth naming plainly. First, Thor is still optimized around automotive workloads — deterministic camera-radar-lidar fusion at ASIL-D — not around 10 B parameter VLA inference at low latency. Second, NVIDIA has no ISO 13482 / IEC 61508 certification track record comparable to their ISO 26262 maturity, which matters for consumer humanoids. Third, memory capacity and bandwidth at the 100 W envelope are not increasing as fast as model sizes; at some point the humanoid primes will build custom silicon because Thor's memory subsystem won't keep up with their model roadmap. This is exactly the Tesla-vs-NVIDIA pattern from the automotive era, and it will likely repeat.

*The real challengers.* The companies to watch are not the automotive incumbents but the edge-AI specialists and the vertically integrated humanoid primes. Hailo-15, SiMa.ai MLSoC, Ambarella N1, Kinara Ara-2, Rebellions Atom, and FuriosaAI RNGD are pushing TOPS-per-watt past anything NVIDIA ships below 50 W. BrainChip Akida and SynSense are betting on event-based / spiking approaches that could leapfrog conventional inference for specific sensor pipelines. On the integration side, Figure, 1X, Apptronik, and Tesla are all plausibly building custom humanoid silicon on 3–5 year horizons, following the Tesla Dojo / FSD precedent.

**Bottom line**

The extrapolation is: drones stay inside the incumbent envelope with aviation-certification as the differentiator; quadrupeds stay on Jetson-class silicon for 2–3 more years and then bifurcate between "cheap Rockchip/Hailo-based fleets" and "premium Thor-based inspection units"; humanoids break the automotive silicon model entirely and either (a) consume workstation-class silicon that nobody currently ships in a 150 W thermal envelope, or (b) drive a custom-silicon wave from the primes themselves. Your instinct that the current providers are "not viable solutions for these other robotic form factors" is correct for humanoids, partially correct for quadrupeds, and wrong for drones. NVIDIA is the only incumbent with a path to remain relevant in the top-end robotics tier, and that path depends on them executing on memory bandwidth and a non-automotive safety-certification stack — both of which are open questions. Mobileye has no path here. Qualcomm has a subordinate role as a low-power sensor-processing SoC, not as the reasoning brain.

The deeper point is that the humanoid compute tier is a new category of silicon that doesn't exist yet in shipping form. Whoever builds a 100–200 W, 256 GB-memory, ISO-13482-certifiable inference SoC with hard-real-time safety partitioning is building a product that has no current competitor. That is the defensible gap.

## Drone Autonomy

You're correct to push back, and the Jetson delivered-performance gap deserves to be stated plainly before any first-principles exercise is credible. Let me set the floor on the measurement problem first, then build the compute budget from scratch.

**Why the marketed TOPS is almost useless for system sizing**

The 1–2% figure is an extreme case but the direction is right. The compounding losses on any Jetson-class SoC under realistic robotics operating conditions break down roughly as follows.

Marketed peak is INT8 peak theoretical MAC throughput summed across GPU Tensor Cores + DLA0 + DLA1 + (on some SKUs) the PVA, assuming all units fully utilized simultaneously on perfectly-shaped workloads. No real model saturates all three engines at once. Realistic TensorRT coverage on DLA is partial — many ops fall back to GPU, collapsing the DLA contribution. That alone drops you to 40–60% of peak.

Memory bandwidth then dominates. Orin AGX has 204 GB/s LPDDR5 shared across the CPU cluster, GPU, DLA, ISP, VIC, and encoders. Any realistic perception pipeline plus VIO plus planner is memory-bound well before it is compute-bound; sustained is typically 20–40% of the kernel-bound number.

Thermal envelope in a drone or cobot enclosure with no forced air and ambient temperatures above 25 °C forces MAXQ or a custom power mode; Orin AGX dropping from 60 W to 15–30 W sustained takes another 40–60% off.

Safety partitioning — required for delivery drones under any DO-178C / DO-254 equivalency pathway, and for cobots under ISO 13482 or ISO/TS 15066 — requires running lockstep redundant pipelines, hardware monitoring, or freedom-from-interference partitioning. Depending on the scheme, the usable compute drops by another 30–50%.

Model precision and workload mix matters. FP16/BF16 peak on Orin AGX is about 85 TFLOPS; FP32 is about 5 TFLOPS. Transformer-heavy VLA workloads running FP16 cannot use the INT8 headline number at all.

Multiply these together and you get the 1–5% sustained-delivered-to-application figure in the worst case. A more optimistic realistic case for perception-heavy INT8 workloads is 10–20% of marketed peak. Either way, **system sizing against marketed TOPS is negligent engineering.** The rest of this analysis uses *sustained effective GFLOPS / TFLOPS at the target precision* as the unit.

**First-principles compute decomposition**

Any autonomy stack is the sum of seven workload classes. Each has a principled complexity scaling.

*Perception.* Camera at resolution R × framerate F using backbone with G GFLOPs/frame. A YOLOv8-s-class detector at 1080p/30: ≈ 30 × 8 = 240 GFLOPS. A modern open-vocabulary detector (Grounding-DINO, YOLO-World) at the same rate: 1–3 TFLOPS. SAM-class segmentation at 5 Hz: 0.5–1 TFLOPS. Multi-camera (6 cameras as on a humanoid or drone): multiply by 6.

*State estimation / VIO / SLAM.* Feature-based VIO (VINS-Fusion, ORB-SLAM3) at 30 Hz: 1–3 TFLOPS sustained. Deep VIO (DROID-SLAM, NeRF-SLAM): 10–50 TFLOPS. Dense volumetric mapping in a multi-camera drone at 10 Hz: 5–15 TFLOPS.

*Prediction.* Agent forecasting for N agents at 10 Hz with transformer predictors (Wayformer-class): 0.1–1 TFLOPS for small N, scales linearly. For a delivery drone in urban airspace with ~5–20 tracked agents: 0.5–2 TFLOPS.

*Planning.* MPC at 100–500 Hz for a 6–50 DoF system: 0.1–2 TFLOPS sustained. Sampling-based planners (MPPI): similar. Learning-based planners (Diffusion Policy, ACT): 1–10 TFLOPS at 30 Hz depending on model size.

*Control.* Low-level control (stabilization, joint PID, motor commutation) runs on an MCU and consumes single-digit MFLOPS. Irrelevant to the budget. Whole-body MPC at 1 kHz for a 50-DoF humanoid: 0.1–0.5 TFLOPS sustained, typically on a dedicated RT core.

*Reasoning and VLA.* This dominates everything else at the upper autonomy levels. Decoder-only transformer inference is memory-bandwidth-bound, not compute-bound, so the correct metric is (model parameters × 2 bytes × tokens/sec ÷ memory bandwidth). A 7B-parameter VLA at FP16 running at 10 Hz requires ≈ 140 GB/s of weight reads before you count KV cache or activations. Compute-wise: 7B × 2 FLOPs × 10 Hz = 140 GFLOPS per trajectory step for a single forward pass, but the memory bottleneck pushes effective utilization to 5–15%, so the sustained *compute requirement* is roughly 1–3 TFLOPS to actually hit 10 Hz. A 70B reasoning model at 1 Hz needs ≈ 1.4 TB/s of memory bandwidth — outside any edge envelope. In practice that tier is cloud-offloaded or heavily distilled.

*Safety and redundancy.* ASIL-B / DAL-C equivalency for drones and cobots typically requires a 1.5× to 3× duplication factor and runtime assurance monitoring at 10–30% overhead. Assume a 2× multiplier on the above tiers as a baseline.

**First-principles budgets by form factor and autonomy level**

The levels below are my construction, not SAE or ASTM. I'm defining them because the SAE framework doesn't cleanly map to drones, quadrupeds, or humanoids. All numbers are *sustained effective TFLOPS at the target precision needed for the workload*, including the 2× safety multiplier.

*Delivery drones (BVLOS, safety-certified):*

- D-L1 — ADS-B DAA, pre-planned route, visual precision landing (Zipline Gen 2, Wing Hummingbird class today). Perception ≈ 0.3 TF, VIO ≈ 0.5 TF, control ≈ 0.1 TF, safety 2×: **~2 TFLOPS sustained**.
- D-L2 — Visual DAA with conflict prediction, GNSS-degraded VIO, weather adaptation. Perception ≈ 1–2 TF (multi-camera detection + classification), VIO ≈ 2 TF, prediction ≈ 0.5 TF, planning ≈ 0.3 TF, safety 2×: **~8–10 TFLOPS sustained**.
- D-L3 — Urban autonomous delivery, certifiable multi-agent DAA, complex landing zones. Perception ≈ 5 TF, dense mapping ≈ 3 TF, prediction ≈ 2 TF, planner ≈ 1 TF, small VLM for scene reasoning ≈ 2 TF, safety 2×: **~25–30 TFLOPS sustained**.
- D-L4 — Unconstrained airspace, peer coordination, adaptive mission. Adds ≈ 20 TF for reasoning and coordination: **~70–100 TFLOPS sustained**.

*Quadrupeds for surveillance:*

- Q-L1 — Teleoperation with obstacle avoidance. Perception ≈ 0.5 TF, locomotion MPC ≈ 0.2 TF, safety 2×: **~1.5 TFLOPS sustained**.
- Q-L2 — Waypoint autonomy, terrain-aware locomotion. + SLAM ≈ 2 TF, terrain understanding ≈ 1 TF: **~5–7 TFLOPS sustained**.
- Q-L3 — Mission autonomy with semantic scene understanding. + open-vocab detection ≈ 2 TF, anomaly detection ≈ 1 TF, small VLM ≈ 3 TF: **~15–20 TFLOPS sustained**.
- Q-L4 — Long-horizon autonomous patrol with multi-robot coordination. + VLA planning ≈ 5 TF, coordination ≈ 2 TF: **~40–60 TFLOPS sustained**.
- Q-L5 — Natural-language mission tasking with adaptive decomposition. + large VLM ≈ 30 TF: **~150–300 TFLOPS sustained**, with cloud offload for reasoning likely.

*Humanoids in the home:*

- H-L1 — Teleoperated bimanual manipulation with collision safety. Perception ≈ 2 TF, whole-body control ≈ 0.5 TF, safety monitoring ≈ 1 TF, safety 2×: **~7–10 TFLOPS sustained**.
- H-L2 — Supervised single-task automation (load dishwasher, fold laundry). + manipulation policy ≈ 5 TF, object-centric perception ≈ 3 TF: **~30–40 TFLOPS sustained**.
- H-L3 — Multi-step task chaining with natural-language instructions. + small VLA (2–7 B params) ≈ 10 TF, language grounding ≈ 5 TF: **~80–120 TFLOPS sustained**.
- H-L4 — General household assistant with on-the-job learning. + large VLA (10–20 B params) ≈ 40 TF, long-context reasoning ≈ 30 TF, safety runtime monitor ≈ 10 TF: **~300–500 TFLOPS sustained**.
- H-L5 — Full household autonomy with long-horizon planning and world modeling. + 70 B-class reasoning, multi-modal world model ≈ 200–500 TF (cloud-offloaded in practice): **~1,000–3,000 TFLOPS effective**, with at least half of that plausibly off-board.

**Mapping to what Jetson AGX Orin actually delivers**

Take Orin AGX in a realistic drone/cobot enclosure. Peak marketed 275 INT8 TOPS, 85 FP16 TFLOPS. Apply the compound derate (DLA underutilization, memory bandwidth, thermal, safety partitioning, precision mix): a defensible sustained effective number for robotics workloads is **3–10 TFLOPS at FP16 and 15–40 TOPS at INT8**, and that is *before* you reserve headroom for worst-case behavior.

Against the budgets above:

- D-L1 (2 TFLOPS): Orin NX or Orin AGX delivers. Current Zipline/Wing-class operations fit.
- D-L2 (8–10 TFLOPS): Orin AGX is marginal; one Orin per drone is at the edge of the thermal envelope. Two Orins or a Thor is more honest.
- D-L3 (25–30 TFLOPS): Orin AGX is not sufficient. Requires Thor-class (marketed 1000+ TOPS FP8; realistic sustained probably 50–150 TFLOPS) or a multi-SoC architecture.
- Q-L3 (15–20 TFLOPS): Orin AGX is marginal.
- Q-L4 (40–60 TFLOPS): Orin AGX inadequate. Thor-class or cloud offload required.
- H-L2 (30–40 TFLOPS): Orin AGX inadequate.
- H-L3 (80–120 TFLOPS): Thor-class minimum, and Thor at realistic sustained rates may not get there either.
- H-L4+ (300–3000 TFLOPS): no existing edge SoC, Thor included, can sustain this in a 100–200 W humanoid thermal envelope. Full stop.

**What this implies for silicon and suppliers**

The honest read on the current market is:

*NVIDIA Jetson (Orin line)* is adequate for D-L1 drones and Q-L1/L2 quadrupeds. It is marginal-to-inadequate for anything above that. The "we'll scale up with Thor" argument is credible in marketing-TOPS space and uncertain in sustained-delivered space. Thor's memory bandwidth (likely 273–512 GB/s depending on SKU) is still insufficient for 20B+ VLA inference at interactive rates.

*Qualcomm Robotics (Ride Flex, QCS8550)* is in the 20–50 effective TOPS INT8 sustained range, suitable for D-L1 and Q-L1. Not credible for humanoid L3+.

*Mobileye* is not a player in any of these.

*The gap that needs to be filled* is a 100–500 sustained-effective-TFLOPS, 100–200 W, 256 GB-HBM-class, safety-certifiable inference SoC with hard-real-time partitioning. That product does not exist today. Thor is the closest attempt and it is still a compromise architecture optimized for AV. Specialist inference silicon (Hailo-15, SiMa MLSoC, Ambarella N1, FuriosaAI RNGD, Tenstorrent Blackhole) gets closer on TOPS-per-watt efficiency but lacks the safety-certification path.

*The most likely trajectory* is that humanoid primes — Figure, 1X, Apptronik, Tesla, Sanctuary — build custom inference silicon co-designed with their VLA model architecture, because no merchant silicon will be economically viable at their volume targets in the 2028–2030 window. This is the direct analog of Tesla building FSD chip because NVIDIA didn't deliver what they needed. It will repeat.

**One honest caveat on the whole exercise**

First-principles compute estimates like this are sensitive to assumed model sizes and precisions. If the field discovers that a 1 B-parameter distilled VLA is sufficient for H-L3 household tasks — possible; this is an active research question — the humanoid budgets drop by 10×. If large world-model-based planning (DeepMind Genie 3-class, VJEPA-2) becomes the accepted architecture, the budgets go up by 10×. The order-of-magnitude ratios between levels are more defensible than the absolute numbers. The headline — that current incumbent edge silicon cannot deliver sustained compute for humanoid-class autonomy in a 200 W envelope — is robust to reasonable variation in those assumptions.
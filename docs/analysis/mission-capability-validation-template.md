# Mission-Capability Validation Template

**Purpose:** Sharpen the TOPS and non-compute-power assumptions in
`docs/analysis/mission-capability-per-watt.md` and
`src/graphs/reporting/mission_capability.py` by sourcing each number
from domain experts and workload measurements.

The goal is to move every catalogued mission from **first-principles
engineering estimate** to **workload-measured** with a citation that
survives due diligence.

---

## How to fill in

For each mission below:

1. Replace `TBD` in the *Workload breakdown* column with the actual
   model(s) needed and their per-inference ops counts at target
   resolution / frame rate.
2. Cite a reference (paper, published model card, vendor datasheet, or
   internal measurement).
3. Record the domain-expert name + organization who signed off on the
   numbers.
4. Update `src/graphs/reporting/mission_capability.py` if the final
   TOPS or non-compute-power differs materially (> 30 %) from the
   current assumption, then regenerate
   `reports/microarch_model/m05_preview/mission_capability.html`.
5. Bump the mission's *Validation status*: `estimate` -> `expert
   review` -> `measured`.

---

## Mission validation table

Values in parentheses are the current first-principles estimates from
`default_mission_catalog()` in `mission_capability.py`.

### 1. Urban counter-SUAS nano-swarm

| Parameter | Current estimate | Workload breakdown | Reference | Expert | Status |
|---|---|---|---|---|---|
| Required TOPS | 35 TOPS INT8 | VIO (~5), YOLOv8-n / similar (~15), multi-target track (~10), swarm coord (~5) | TBD | TBD | estimate |
| Duration | 10 min | flight time of 30-50 g quadcopter | TBD (airframe lead) | TBD | estimate |
| Battery Wh | 10 Wh | 2S Li-HV, ~1400 mAh @ 7.6 V | Commercial FPV drone BOM | TBD | estimate |
| Non-compute W | 15 W | 4 motors hover + IMU + optic flow cam | TBD | TBD | estimate |
| Payload mass budget | 15 g | after 25 g airframe + 10 g battery from 50 g AUW | TBD (airframe) | TBD | estimate |

**Validators to engage:** counter-UAS / defense primes (Anduril, Skydio
defense, Teal), academic swarm labs (GRASP, NIMBUS).

### 2. Body-worn tactical / medical exoskeleton

| Parameter | Current estimate | Workload breakdown | Reference | Expert | Status |
|---|---|---|---|---|---|
| Required TOPS | 20 TOPS | gait prediction model, terrain classifier, intent CNN, reactive LSTM | TBD | TBD | estimate |
| Duration | 12 h | one shift | TBD (medical / defense) | TBD | estimate |
| Thermal envelope | 3 W compute | IEC 60601-1 skin-contact limit for applied parts | IEC 60601-1 | TBD | estimate |
| Non-compute W | 3 W | joint encoders + IMU + low-duty actuators | TBD | TBD | estimate |

**Validators to engage:** Ekso Bionics, Sarcos, ReWalk, Wyss Institute,
ARPA-H biomedical programs.

### 3. 30-day abyssal AUV

| Parameter | Current estimate | Workload breakdown | Reference | Expert | Status |
|---|---|---|---|---|---|
| Required TOPS | 20 TOPS | side-scan sonar CNN (~8), vision pipeline inspection (~5), anomaly detection (~3), SLAM (~4) | TBD | TBD | estimate |
| Duration | 720 h | 30-day transit-and-inspect | Oil & gas AUV programs | TBD | estimate |
| Battery Wh | 15 kWh | 50 kg AUV class | Woods Hole AUV notes | TBD | estimate |
| Non-compute W | 18 W | slow-transit thruster + sonar + lights | TBD | TBD | estimate |

**Validators to engage:** Kongsberg HUGIN, Bluefin / General Dynamics,
Saab Sabertooth, MBARI, Ocean Infinity.

### 4. 6-month oceanographic glider fleet

| Parameter | Current estimate | Workload breakdown | Reference | Expert | Status |
|---|---|---|---|---|---|
| Required TOPS | 5 TOPS | species classifier (~2), plankton counter (~1), anomaly / event detection (~2) | TBD | TBD | estimate |
| Duration | 4320 h | 6 months | Slocum / Seaglider mission logs | TBD | estimate |
| Battery Wh | 10 kWh | typical alkaline or primary-Li pack | Teledyne Slocum datasheet | TBD | estimate |
| Non-compute W | 0.4 W | buoyancy pumps intermittent + hotel | Published Slocum power budget | TBD | estimate |

**Validators to engage:** Teledyne Webb Research, Kongsberg Seaglider,
NOAA, OceanScan, MBARI.

### 5. 72-hour autonomous UGV pack mule

| Parameter | Current estimate | Workload breakdown | Reference | Expert | Status |
|---|---|---|---|---|---|
| Required TOPS | 100 TOPS | multi-camera stereo (~20), LIDAR fusion (~15), object+person detection (~25), path plan (~10), cooperative behavior (~15), VLM (~15) | TBD | TBD | estimate |
| Duration | 72 h | three-day escort | Army TARDEC / DARPA programs | TBD | estimate |
| Battery Wh | 10 kWh | 100 kg ground robot typical | Ghost Robotics Vision 60 notes | TBD | estimate |
| Non-compute W | 100 W | leg actuators average + sensors + comms | TBD (vendor) | TBD | estimate |

**Validators to engage:** Ghost Robotics, Boston Dynamics (Spot
Enterprise), ARL, DARPA Tactical Technology Office.

### 6. 7-day sealed USAR micro-robot

| Parameter | Current estimate | Workload breakdown | Reference | Expert | Status |
|---|---|---|---|---|---|
| Required TOPS | 10 TOPS | SLAM (~3), victim CNN (~4), acoustic event detection (~2), comms efficient coding (~1) | TBD | TBD | estimate |
| Duration | 168 h | 7-day endurance | FEMA INSARAG | TBD | estimate |
| Battery Wh | 200 Wh | small Li-ion pack, no recharge | TBD | TBD | estimate |
| Non-compute W | 0.5 W | duty-cycled motors + sensors | TBD | TBD | estimate |
| Thermal envelope | 3 W compute | sealed IP68 case passive dissipation | TBD | TBD | estimate |

**Validators to engage:** CRASAR (Texas A&M), FEMA Urban SAR task
forces, Recon Robotics, Gostai, Boston Dynamics Spot-lite programs.

### 7. 24/7 HALE solar pseudosatellite

| Parameter | Current estimate | Workload breakdown | Reference | Expert | Status |
|---|---|---|---|---|---|
| Required TOPS | 30 TOPS | wide-area imagery (~15), target detection (~8), moving-target tracker (~5), comms relay (~2) | TBD | TBD | estimate |
| Duration (night phase) | 12 h | night energy budget is binding | Zephyr program public data | TBD | estimate |
| Battery Wh (reserve) | 500 Wh | night battery reserve | TBD (Zephyr, PHASA-35) | TBD | estimate |
| Thermal envelope (night) | 5 W compute | cold battery discharge + thermal | TBD | TBD | estimate |

**Validators to engage:** Airbus Zephyr, BAE Systems (PHASA-35),
Aurora Flight Sciences, AeroVironment.

### 8. On-board AI for small LEO imaging satellite

| Parameter | Current estimate | Workload breakdown | Reference | Expert | Status |
|---|---|---|---|---|---|
| Required TOPS | 100 TOPS | hi-res scene classifier (~30), ship/vehicle detection (~25), change detection (~20), on-orbit compression (~25) | TBD | TBD | estimate |
| Duration (per orbit) | 24 h | continuous operation | TBD | TBD | estimate |
| Thermal envelope | 40 W compute | payload power allocation | Planet Labs public data | TBD | estimate |
| Non-compute W | 10 W | optical bench + cryocooler + downlink amp floor | TBD | TBD | estimate |

**Validators to engage:** Planet Labs, Satellogic, Capella Space,
SpaceX Starlink, AWS Ground Station, Ubotica (CV on orbit).

### 9. All-day autonomous agricultural tractor

| Parameter | Current estimate | Workload breakdown | Reference | Expert | Status |
|---|---|---|---|---|---|
| Required TOPS | 200 TOPS | 12-cam stereo + segmentation (~80), 2 LIDAR fusion (~40), crop/weed classifier (~30), humans-in-field detector (~20), path planner (~20), row-level weed sprayer control (~10) | TBD | TBD | estimate |
| Duration | 10 h | workday | John Deere, Bear Flag Robotics | TBD | estimate |
| Thermal envelope | 15 W compute | dust-sealed fanless cab slot | TBD (Deere, AGCO) | TBD | estimate |

**Validators to engage:** John Deere Autonomous, AGCO, Bear Flag
Robotics (acquired by Deere), Blue River Technology, Carbon Robotics.

### 10. 8-hour-shift industrial humanoid robot

| Parameter | Current estimate | Workload breakdown | Reference | Expert | Status |
|---|---|---|---|---|---|
| Required TOPS | 500 TOPS | body-control (~50), multi-cam perception (~100), VLM grounding (~200), task planner (~100), speech/dialogue (~50) | TBD | TBD | estimate |
| Duration | 8 h | one shift | Figure 01, Tesla Optimus, Apptronik | TBD | estimate |
| Battery Wh | 3 kWh | Figure 01 class | Figure product briefs | TBD | estimate |
| Non-compute W | 250 W | actuator duty cycle average | TBD | TBD | estimate |

**Validators to engage:** Figure AI, Tesla Optimus team, Apptronik,
1X Technologies, Sanctuary AI, Boston Dynamics Atlas, Agility
Robotics Digit.

---

## Cross-cutting validation questions

These apply to every mission and should be answered up-front before
drilling into individual TOPS numbers.

1. **Quantization mix.** Is the TOPS budget INT8-dominant, FP16-mixed,
   FP32-tail? Current model assumes INT8 dense; mixed-precision
   workloads shift the TOPS/W ceiling proportionally.
2. **Duty cycle.** Non-compute-power numbers assume *average* load.
   Missions with bursty actuator duty cycles (UGV pack mule climbing
   terrain, humanoid picking a dropped item) have non-compute power
   that varies 10:1.  Either refine the mission to *peak* non-compute
   or factor the average more carefully.
3. **Model size growth.** Many of these missions will add VLM or
   world-model components over the next 3 years. Each 10x in model
   size maps to roughly 10x in TOPS at held-equal latency. The
   catalog is a snapshot of today's target workloads.
4. **Accuracy floor.** Mission-capability flips *at a specific model
   accuracy*. If quantizing to INT8 drops accuracy below the
   mission's acceptable bound, the effective TOPS requirement must be
   scaled up for FP16 or a bigger model at INT8.
5. **Cloud / edge hybrid.** Some missions (HALE, LEO sat) tolerate
   periodic downlink; others (denied environments, AUV, USAR) cannot.
   Missions explicitly assumed to be edge-only should be marked.

---

## How to use this template

- Fork this document per validation session; do not overwrite.
- Engagements log: when a domain expert is interviewed, link the
  engagement notes from the *Expert* cell.
- Status progression: `estimate` -> `expert review` -> `measured`.
  Measured means end-to-end on representative hardware + inputs.
- When any mission reaches *measured*, update the
  `default_mission_catalog()` TOPS / battery / non-compute numbers in
  `mission_capability.py` and bump the `citation` field on the
  Mission to reference the measurement source.
- Regenerate the HTML after any catalog change; the feasibility tally
  is the visible "how many missions are still KPU-exclusive"
  indicator.

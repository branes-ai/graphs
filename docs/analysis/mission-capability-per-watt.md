# Mission Capability per Watt

**How the 10× TOPS/W architectural advantage of the KPU translates into
black-and-white mission capability for embodied AI platforms.**

Companion to `docs/analysis/speed-of-light-analysis.md` (raw silicon
ceiling) and `reports/microarch_model/m05_preview/silicon_composition.html`
(per-architecture hierarchy). Where those quantify *the gap*, this doc
answers *why the gap matters to a customer*.

---

## 1. The framing

"Intelligence per Watt" (Aschenbrenner, McKinsey-NVDA analyses, and a
growing LLM-capability literature) works for cloud LLMs because the
compute and the energy grid are both scalable — the metric just tells
you how much model quality you buy per kWh. **For embodied AI it's the
wrong abstraction: the platform's watts are not scalable.** A 30 W
battery budget is a hard wall imposed by payload mass, airframe lift,
battery chemistry, or skin-contact thermal limits. Past that wall, the
mission doesn't get "slower" — it physically cannot happen.

The right translation is **Mission Capability per Watt**, with a
discrete step function: at some TOPS/W threshold, a whole class of
missions flips from impossible to possible. That's where a 10×
architectural efficiency advantage stops being a benchmark-marketing
number and starts being a **go / no-go** for a customer's product.

---

## 2. Three physics thresholds that create black-and-white outcomes

The 10× TOPS/W gap maps to different customer-visible thresholds
depending on which physical law is binding:

1. **Energy-budget thresholds** (battery × hours). Compute wattage ×
   mission duration competes with motors, comms, sensors, heating for
   the same battery joules. Cross a threshold and the mission shortens
   from days to hours.
2. **Thermal-envelope thresholds** (W dissipated / cm^2 / K). Sealed
   canisters, body-worn devices, solar-only platforms, and high-altitude
   craft have hard cooling ceilings below what a GPU dissipates at full
   TOPS.
3. **Payload-mass thresholds** (grams of chip + grams of battery +
   grams of heatsink). Below ~50 g of total avionics, no GPU-class
   solution exists on the market at all.

### The 10× in numbers

At the canonical dense-INT8 operating point:

| Architecture | TOPS / W (dense INT8) |
|---|---|
| NVIDIA Jetson Orin AGX (any power mode) | ~2.3 |
| KPU T128 target (canonical 32x32 tile @ 8 nm, 12 W) | ~22 |
| Ratio | **~10x** |

So for a given required throughput `N TOPS`:

- GPU wattage: `N / 2.3 W`
- KPU wattage: `N / 22 W`

And the 10x matters most where platform budgets are 5–60 W.

---

## 3. Ten missions where 10x flips impossible -> possible

The missions are grouped by which physical threshold is binding.
Numbers are defensible first-principles estimates; refine per-platform.

### Group A — payload mass is the binding constraint

#### 1. Urban counter-SUAS nano-swarm (building clearing, first responders)

- **Platform:** 30–50 g quadcopter, palm-sized, 5–10 min flight.
- **Compute need:** VIO SLAM + obstacle avoidance + target detection +
  swarm coordination ~35 TOPS concurrent.
- **GPU reality:** Orin Nano module alone is 35–50 g at 7 W sustained,
  delivering ~15 TOPS. The compute package is heavier than the airframe.
- **KPU enabler:** ~1.6 W, <5 g package, delivers 35 TOPS concurrent.
- **Black-and-white:** Indoor palm-sized autonomous swarm is
  **impossible** with GPU, **possible** with KPU.

#### 2. Body-worn tactical / medical exoskeleton (skin-safe thermal)

- **Platform:** Worn against skin 12 h/day, joint-integrated.
- **Compute need:** Adaptive gait + terrain prediction + intent
  recognition ~20 TOPS continuous.
- **GPU reality:** 9 W dissipation under a uniform fails IEC 60601 skin-
  temperature limits; requires an external heatsink that breaks form.
- **KPU enabler:** 0.9 W, integrates into joint actuator housing, no fan.
- **Black-and-white:** All-day body-worn cognitive assist is
  **impossible** with GPU, **possible** with KPU.

### Group B — energy x mission duration is binding

#### 3. 30-day abyssal AUV (pipeline inspection, deep-ocean ISR)

- **Platform:** 50 kg AUV, 30-day mission, battery-only (no recharge).
  20 kWh total.
- **Compute need:** 20 TOPS continuous for acoustic + vision fusion,
  SLAM, anomaly detection.
- **GPU reality:** 8.8 W x 720 h = 6.3 kWh — compute eats 32% of
  battery; thrusters can't complete the transit.
- **KPU enabler:** 0.9 W x 720 h = 660 Wh (~3%).
- **Black-and-white:** 30-day deep-ocean AI survey is **impossible**
  with GPU, **possible** with KPU.

#### 4. Six-month oceanographic glider fleet (climate monitoring)

- **Platform:** 60 kg buoyancy-driven glider, 6-month deployment, fleet
  of 100 for basin-scale monitoring.
- **Compute need:** 5 TOPS continuous (species classification,
  underwater vision).
- **GPU reality:** 2.2 W x 4320 h = 9.5 kWh per glider — entire battery,
  no budget for pumps, sensors, or satcom.
- **KPU enabler:** 0.23 W x 4320 h = 990 Wh (~10% of battery budget).
- **Black-and-white:** Six-month autonomous oceanographic fleet is
  **impossible** with GPU, **possible** with KPU.

#### 5. 72-hour autonomous UGV pack mule (small-unit military escort)

- **Platform:** 100 kg ground robot, 72 h autonomous escort with useful
  payload.
- **Compute need:** 100 TOPS for multi-sensor perception + cooperative
  behavior + path planning.
- **GPU reality:** 44 W x 72 h = 3.2 kWh compute + ~8 kWh motors ->
  ~65 kg battery, consuming payload capacity.
- **KPU enabler:** 4.6 W x 72 h = 330 Wh compute; ~15 kg recovered as
  payload.
- **Black-and-white:** Multi-day UGV with useful payload is
  **impossible** with GPU, **marginal-to-possible** with KPU.

#### 6. Seven-day sealed USAR micro-robot (earthquake rubble)

- **Platform:** Sealed enclosure, no ventilation, deployed in collapsed
  structure, no recharge.
- **Compute need:** 10 TOPS for SLAM + victim detection + acoustic
  localization.
- **GPU reality:** 4.4 W x 168 h = 740 Wh plus sealed-case thermal
  buildup breaks operating limits.
- **KPU enabler:** 0.46 W x 168 h = 77 Wh, fits thermally-sealed
  housing below 40 degC.
- **Black-and-white:** Seven-day rubble-deployed search robot is
  **impossible** with GPU, **possible** with KPU.

### Group C — thermal envelope is binding (solar / altitude / space)

#### 7. 24/7 HALE solar pseudosatellite (ISR, telecom relay)

- **Platform:** 75 kg fixed-wing solar UAV at 60,000 ft. Night power
  budget ~5 W reserve; day budget ~50 W net after avionics.
- **Compute need:** ~30 TOPS continuous for wide-area target detection.
- **GPU reality:** 13 W compute exceeds night budget; craft falls out of
  operational altitude during nocturnal station-keeping.
- **KPU enabler:** 1.4 W, well inside night budget.
- **Black-and-white:** 24/7 solar-HALE ISR is **impossible** with GPU,
  **possible** with KPU.

#### 8. On-board AI for small LEO imaging satellite

- **Platform:** 150 kg LEO sat, 50 W total power, ~40 W for payload.
- **Compute need:** 100 TOPS for real-time ship / vehicle / change
  detection (so only pre-classified imagery hits the limited downlink).
- **GPU reality:** 44 W compute consumes the entire payload envelope;
  no energy left for optical bench, cryocooler, or downlink amplifier.
- **KPU enabler:** 4.6 W, leaves 35 W for sensors and comms.
- **Black-and-white:** On-orbit real-time imagery classification on a
  small LEO sat is **impossible** with GPU, **possible** with KPU.

### Group D — concurrent-model density is binding (bigger platforms)

#### 9. All-day autonomous agricultural tractor (200 TOPS concurrent)

- **Platform:** 5000 kg tractor; 12-camera + 2-LIDAR + IMU fusion in
  RF-denied fields (no cloud offload possible).
- **Compute need:** 200 TOPS concurrent for sensor fusion, path
  planning, crop/weed recognition, humans-in-field detection.
- **GPU reality:** 88 W sustained requires 3–4 Orin modules + active
  cooling + sealing against field dust; thermal density fights ingress
  protection ratings.
- **KPU enabler:** 9 W single-chip, fanless, passively cooled, dust-
  sealed in the cab.
- **Black-and-white:** Single-chip multi-sensor fusion for full-day ag
  ops is **impossible** with GPU, **possible** with KPU.

#### 10. Eight-hour-shift industrial humanoid robot

- **Platform:** 30–80 kg humanoid, 8 h shift, battery-powered.
- **Compute need:** 500 TOPS for full-body motor control + multimodal
  perception + VLM grounding + task planning, concurrent.
- **GPU reality:** 220 W sustained — half the total platform power
  budget; halves shift duration or doubles battery mass, which changes
  the kinematics.
- **KPU enabler:** 23 W, <5% of platform power. Full 8 h shift with
  actuators getting their due.
- **Black-and-white:** Full-shift humanoid with on-board full cognition
  is **impossible** with GPU, **possible** with KPU.

---

## 4. How this reframes the investor / customer conversation

A 10× TOPS/W advantage, stated as a benchmark number, invites the
response *"interesting but we'll wait for next-gen silicon."* The same
advantage stated as mission capability gives the customer a binary
choice they cannot punt on: either their product can do 24/7 HALE ISR /
30-day AUV missions / body-worn exoskeletons / nano-swarms, or it
cannot. The sales cycle collapses because there is no competitive
parity to negotiate against — there is only **"our KPU makes this
product category possible, or you ship no product."**

Three recurring patterns customers respond to:

- **Endurance multiplication** (missions 3–6). "Our mission is now
  measured in days and weeks instead of hours." Battery BOM drops; ops
  tempo of the program doubles.
- **Platform miniaturization** (missions 1, 2, 7). "We have a product
  class at 50 g / 1 kg / 75 kg that no incumbent can field." The gap
  isn't measurable in AI performance — it is the *existence* of the
  product.
- **Concurrent-model deployment** (missions 9, 10). "Multiple large
  models run on a single platform with no fan, no data-link." The
  system-integration story simplifies by an order of magnitude.

All three patterns map to the same underlying mechanism: the energy per
inference, times the inferences per mission, fell inside the platform's
physical budget. That is the Mission Capability per Watt story, and it
is the form of the argument that cuts through incumbency.

---

## 5. Next steps

- Derive **mission-hours enabled / W** curves for each of the 10
  missions directly from the KPU T128 energy model in
  `building_block_energy.py`.
- Produce a visual comparison artifact under `reports/microarch_model/`
  plotting each mission's GPU-side and KPU-side wattage on the same
  platform-budget axis, with the binding physical threshold annotated.
- Validate the compute requirements per mission with domain experts
  (first responders, subsea robotics, precision ag, humanoid teams) so
  the TOPS numbers are workload-driven, not model-size-driven.

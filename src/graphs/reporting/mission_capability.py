"""
Mission-capability analysis for embodied AI.

Operationalizes ``docs/analysis/mission-capability-per-watt.md``: for a
catalogued set of embodied-AI missions, quantify whether a given
architecture (GPU-class or KPU-class) fits the platform's physical
budget, and derive "mission-hours enabled" curves for energy-bound
missions.

The core argument: Intelligence per Watt as an LLM-style scalar metric
understates the step-function nature of embodied-AI deployments.
Batteries, airframes, skin-contact thermal limits, and solar-only
power envelopes are hard walls. Below a TOPS/W threshold, a whole
class of missions physically cannot happen. A 10x architectural TOPS/W
advantage (KPU vs GPU at matched process) doesn't make the mission
"faster" - it flips feasibility.

This module evaluates that flip quantitatively:

  p_compute_w          = required_TOPS / arch.tops_per_watt
  p_total_w            = p_compute_w + non_compute_power_w
  e_required_wh        = p_total_w * mission_hours
  energy_feasible      = e_required_wh <= battery_wh
  thermal_feasible     = p_total_w <= thermal_envelope_w
  mass_feasible        = arch.min_module_mass_g <= payload_mass_budget_g
  mission_hours_enabled= battery_wh / p_total_w  (for energy-bound)

Each mission flags its own binding physical threshold. A mission is
"black-and-white" when GPU violates its binding threshold and KPU
satisfies it with meaningful headroom.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ======================================================================
# Enums
# ======================================================================


class PhysicsThreshold(Enum):
    """Which physical law binds this mission's feasibility."""
    PAYLOAD_MASS = "payload mass"
    ENERGY_BUDGET = "energy x duration (battery)"
    THERMAL_ENVELOPE = "thermal dissipation"
    CONCURRENT_DENSITY = "concurrent-model thermal density"


class MissionCategory(Enum):
    AERIAL_SMALL = "Small aerial / UAV"
    AERIAL_HALE = "HALE solar"
    GROUND_UGV = "Ground UGV"
    MARITIME = "Maritime / subsea"
    BODY_WORN = "Body-worn"
    SPACE = "Space"
    INDUSTRIAL = "Industrial"
    USAR = "Search and rescue"
    AGRICULTURE = "Agriculture"
    HUMANOID = "Humanoid"


# ======================================================================
# Data model
# ======================================================================


@dataclass
class Mission:
    """One catalogued embodied-AI mission with quantified physical budget."""
    name: str
    category: MissionCategory
    description: str
    compute_tops: float                 # required INT8 TOPS concurrent
    mission_hours: float                # duration target
    binding_threshold: PhysicsThreshold
    battery_wh: float                   # available energy (0 if not battery-bound)
    non_compute_power_w: float          # motors + sensors + comms baseline
    payload_mass_budget_g: float        # total avionics mass budget
    thermal_envelope_w: float           # max dissipation allowed
    citation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "compute_tops": self.compute_tops,
            "mission_hours": self.mission_hours,
            "binding_threshold": self.binding_threshold.value,
            "battery_wh": self.battery_wh,
            "non_compute_power_w": self.non_compute_power_w,
            "payload_mass_budget_g": self.payload_mass_budget_g,
            "thermal_envelope_w": self.thermal_envelope_w,
            "citation": self.citation,
        }


@dataclass
class ArchProfile:
    """Compute architecture characterized by its SoL operating point."""
    name: str
    tops_per_watt: float             # dense INT8, sustained
    min_module_mass_g: float         # lightest practical package (module + heatsink)
    min_module_power_w: float        # floor TDP (idle + always-on)
    module_thermal_density_w_per_cm2: float  # sustained dissipation per package surface

    def compute_power_w(self, tops: float) -> float:
        """Power consumed to deliver `tops` TOPS, clamped to module floor."""
        return max(tops / self.tops_per_watt, self.min_module_power_w)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tops_per_watt": self.tops_per_watt,
            "min_module_mass_g": self.min_module_mass_g,
            "min_module_power_w": self.min_module_power_w,
            "module_thermal_density_w_per_cm2":
                self.module_thermal_density_w_per_cm2,
        }


@dataclass
class MissionResult:
    mission: Mission
    arch: ArchProfile
    compute_power_w: float
    total_power_w: float
    energy_required_wh: float
    mission_hours_enabled: float      # battery_wh / total_power_w
    energy_feasible: bool
    thermal_feasible: bool
    mass_feasible: bool
    feasible: bool                    # all three AND hours >= mission_hours
    binding_violation: Optional[str]
    headroom: float                   # budget / required; <1.0 = violates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission": self.mission.name,
            "arch": self.arch.name,
            "compute_power_w": self.compute_power_w,
            "total_power_w": self.total_power_w,
            "energy_required_wh": self.energy_required_wh,
            "mission_hours_enabled": self.mission_hours_enabled,
            "energy_feasible": self.energy_feasible,
            "thermal_feasible": self.thermal_feasible,
            "mass_feasible": self.mass_feasible,
            "feasible": self.feasible,
            "binding_violation": self.binding_violation,
            "headroom": self.headroom,
        }


# ======================================================================
# Evaluation
# ======================================================================


def evaluate(mission: Mission, arch: ArchProfile) -> MissionResult:
    """Return pass/fail + quantified headroom for one mission + arch pair."""
    p_compute = arch.compute_power_w(mission.compute_tops)
    p_total = p_compute + mission.non_compute_power_w
    e_required = p_total * mission.mission_hours
    hours_enabled = (
        mission.battery_wh / p_total if p_total > 0 and mission.battery_wh > 0
        else float("inf")
    )

    energy_feasible = (
        mission.battery_wh == 0 or e_required <= mission.battery_wh
    )
    # Thermal envelope applies to the COMPUTE module specifically
    # (the chip's heat budget). Motor / actuator dissipation lands
    # elsewhere on the platform and is accounted for in the non-
    # compute-power baseline, not in this envelope.
    thermal_feasible = (
        mission.thermal_envelope_w == 0
        or p_compute <= mission.thermal_envelope_w
    )
    mass_feasible = (
        mission.payload_mass_budget_g == 0
        or arch.min_module_mass_g <= mission.payload_mass_budget_g
    )
    feasible = (
        energy_feasible and thermal_feasible and mass_feasible
        and (hours_enabled >= mission.mission_hours
             or mission.battery_wh == 0)
    )

    # Identify which threshold binds (the one with lowest headroom).
    candidates: List[Tuple[str, float]] = []
    if mission.battery_wh > 0 and e_required > 0:
        candidates.append((
            "energy_budget",
            mission.battery_wh / e_required,
        ))
    if mission.thermal_envelope_w > 0 and p_compute > 0:
        candidates.append((
            "thermal_envelope",
            mission.thermal_envelope_w / p_compute,
        ))
    if mission.payload_mass_budget_g > 0 and arch.min_module_mass_g > 0:
        candidates.append((
            "payload_mass",
            mission.payload_mass_budget_g / arch.min_module_mass_g,
        ))
    if candidates:
        binding_name, headroom = min(candidates, key=lambda c: c[1])
    else:
        binding_name, headroom = "unbound", float("inf")

    violation = None
    if not feasible:
        if not energy_feasible:
            violation = (
                f"energy: {e_required:.0f} Wh needed vs "
                f"{mission.battery_wh:.0f} Wh battery"
            )
        elif not thermal_feasible:
            violation = (
                f"thermal: {p_compute:.1f} W compute dissipation vs "
                f"{mission.thermal_envelope_w:.1f} W envelope"
            )
        elif not mass_feasible:
            violation = (
                f"mass: module {arch.min_module_mass_g:.0f} g vs "
                f"{mission.payload_mass_budget_g:.0f} g budget"
            )

    return MissionResult(
        mission=mission,
        arch=arch,
        compute_power_w=p_compute,
        total_power_w=p_total,
        energy_required_wh=e_required,
        mission_hours_enabled=hours_enabled,
        energy_feasible=energy_feasible,
        thermal_feasible=thermal_feasible,
        mass_feasible=mass_feasible,
        feasible=feasible,
        binding_violation=violation,
        headroom=headroom,
    )


def hours_enabled_curve(
    mission: Mission, arch: ArchProfile,
    tops_range: Tuple[float, float] = (1.0, 1000.0),
    n_points: int = 40,
) -> List[Tuple[float, float]]:
    """For an energy-bound mission, return (TOPS, mission-hours-enabled)
    points across the TOPS axis. Useful for overlaying GPU vs KPU."""
    if mission.battery_wh <= 0:
        return []
    lo, hi = tops_range
    import math
    out: List[Tuple[float, float]] = []
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    for i in range(n_points):
        frac = i / (n_points - 1)
        tops = 10 ** (log_lo + frac * (log_hi - log_lo))
        p_compute = arch.compute_power_w(tops)
        p_total = p_compute + mission.non_compute_power_w
        hours = mission.battery_wh / p_total if p_total > 0 else 0.0
        out.append((tops, hours))
    return out


# ======================================================================
# Default architecture profiles + mission catalog
# ======================================================================


def gpu_profile() -> ArchProfile:
    """NVIDIA Jetson Orin-class GPU (Ampere @ 8 nm).

    TOPS/W anchored to Orin AGX dense INT8 published number across
    all power modes (~2.3 TOPS/W). See
    docs/analysis/mission-capability-per-watt.md for derivation."""
    return ArchProfile(
        name="NVIDIA Jetson Orin-class GPU",
        tops_per_watt=2.27,
        min_module_mass_g=50.0,      # Orin Nano module with heatsink
        min_module_power_w=5.0,
        module_thermal_density_w_per_cm2=0.5,
    )


def kpu_profile() -> ArchProfile:
    """KPU T128 target (hypothetical 8 nm, 32x32 canonical tile)."""
    return ArchProfile(
        name="KPU T128 target",
        tops_per_watt=22.0,
        min_module_mass_g=5.0,       # small BGA + passive heatsink
        min_module_power_w=0.5,
        module_thermal_density_w_per_cm2=0.3,
    )


def default_mission_catalog() -> List[Mission]:
    return [
        # === Group A: payload mass binding ===
        Mission(
            name="Urban counter-SUAS nano-swarm",
            category=MissionCategory.AERIAL_SMALL,
            description=(
                "30-50 g quadcopter, indoor building clearing, 5-10 min "
                "flight. VIO SLAM + obstacle avoidance + target detection "
                "+ swarm coordination."
            ),
            compute_tops=35.0,
            mission_hours=0.17,
            binding_threshold=PhysicsThreshold.PAYLOAD_MASS,
            battery_wh=10.0,
            non_compute_power_w=15.0,
            payload_mass_budget_g=15.0,   # compute mass budget inside 50 g airframe
            thermal_envelope_w=5.0,
            citation=(
                "DARPA OFFSET / indoor-swarm programs. Target 50 g AUW "
                "leaves ~15 g for avionics + compute."
            ),
        ),
        Mission(
            name="Body-worn tactical / medical exoskeleton",
            category=MissionCategory.BODY_WORN,
            description=(
                "Skin-contact wearable, 12 h shift, gait + terrain + "
                "intent recognition."
            ),
            compute_tops=20.0,
            mission_hours=12.0,
            binding_threshold=PhysicsThreshold.THERMAL_ENVELOPE,
            battery_wh=50.0,
            non_compute_power_w=3.0,
            payload_mass_budget_g=200.0,
            thermal_envelope_w=3.0,       # IEC 60601 skin-contact limit
            citation=(
                "IEC 60601-1 skin-temperature limits for applied parts. "
                "~3 W sustained dissipation against body surface."
            ),
        ),

        # === Group B: energy x duration binding ===
        Mission(
            name="30-day abyssal AUV",
            category=MissionCategory.MARITIME,
            description=(
                "50 kg AUV, pipeline inspection, 30-day mission, battery-"
                "only (no recharge). Acoustic + vision fusion + SLAM + "
                "anomaly detection."
            ),
            compute_tops=20.0,
            mission_hours=720.0,
            binding_threshold=PhysicsThreshold.ENERGY_BUDGET,
            battery_wh=15000.0,            # 15 kWh Li-ion for 50 kg class
            non_compute_power_w=18.0,      # slow-transit thrusters + sensors
            payload_mass_budget_g=5000.0,
            thermal_envelope_w=40.0,
            citation=(
                "Oil and gas ROV / AUV inspection missions. 15 kWh Li-ion "
                "typical for 50 kg class. Slow transit ~1 m/s, continuous "
                "sonar + vision hotel load."
            ),
        ),
        Mission(
            name="6-month oceanographic glider fleet",
            category=MissionCategory.MARITIME,
            description=(
                "60 kg buoyancy-driven glider, 6-month deployment, fleet "
                "of 100 for basin-scale monitoring. Species classification "
                "+ underwater vision."
            ),
            compute_tops=5.0,
            mission_hours=4320.0,
            binding_threshold=PhysicsThreshold.ENERGY_BUDGET,
            battery_wh=10000.0,
            non_compute_power_w=0.4,       # buoyancy pumps intermittent
            payload_mass_budget_g=3000.0,
            thermal_envelope_w=10.0,
            citation=(
                "Teledyne Slocum / Kongsberg Seaglider class. ~10 kWh "
                "battery for 6-month endurance at ~0.4 W hotel load."
            ),
        ),
        Mission(
            name="72-hour autonomous UGV pack mule",
            category=MissionCategory.GROUND_UGV,
            description=(
                "100 kg ground robot, small-unit escort, 72 h autonomous "
                "mission with useful payload. Multi-sensor perception + "
                "cooperative behavior + path planning."
            ),
            compute_tops=100.0,
            mission_hours=72.0,
            binding_threshold=PhysicsThreshold.ENERGY_BUDGET,
            battery_wh=10000.0,
            non_compute_power_w=100.0,     # motors + sensors (average)
            payload_mass_budget_g=10000.0,
            thermal_envelope_w=150.0,
            citation=(
                "Ghost Robotics Vision 60 / Boston Dynamics Spot-class "
                "pack robots. Battery-to-payload tradeoff is the main "
                "design axis."
            ),
        ),
        Mission(
            name="7-day sealed USAR micro-robot",
            category=MissionCategory.USAR,
            description=(
                "Sealed enclosure, deployed in collapsed structure, no "
                "recharge. SLAM + victim detection + acoustic "
                "localization."
            ),
            compute_tops=10.0,
            mission_hours=168.0,
            binding_threshold=PhysicsThreshold.ENERGY_BUDGET,
            battery_wh=200.0,
            non_compute_power_w=0.5,       # sleep-between-events duty cycle
            payload_mass_budget_g=500.0,
            thermal_envelope_w=3.0,        # sealed case, limited dissipation
            citation=(
                "FEMA / ISAR-class sealed micro-robots deployed post-"
                "collapse. Typical Li-ion pack <250 Wh, no forced "
                "cooling."
            ),
        ),

        # === Group C: thermal envelope binding ===
        Mission(
            name="24/7 HALE solar pseudosatellite",
            category=MissionCategory.AERIAL_HALE,
            description=(
                "75 kg fixed-wing solar UAV at 60 kft. Continuous day + "
                "night station-keeping for wide-area ISR."
            ),
            compute_tops=30.0,
            mission_hours=12.0,            # night phase
            binding_threshold=PhysicsThreshold.THERMAL_ENVELOPE,
            battery_wh=500.0,
            non_compute_power_w=3.0,       # minimal avionics at night
            payload_mass_budget_g=2000.0,
            thermal_envelope_w=5.0,        # night power budget from battery reserve
            citation=(
                "Airbus Zephyr / BAE PHASA-35 class. Night phase battery "
                "reserve is the binding design constraint."
            ),
        ),
        Mission(
            name="On-board AI for small LEO imaging satellite",
            category=MissionCategory.SPACE,
            description=(
                "150 kg LEO sat, ship / vehicle / change detection for "
                "downlink reduction. 50 W total power budget."
            ),
            compute_tops=100.0,
            mission_hours=24.0,
            binding_threshold=PhysicsThreshold.THERMAL_ENVELOPE,
            battery_wh=0.0,                # solar + orbit-averaged, not Wh-bound
            non_compute_power_w=10.0,      # optical bench + cryocooler + comms floor
            payload_mass_budget_g=10000.0,
            thermal_envelope_w=40.0,       # payload power allocation
            citation=(
                "Planet Labs / Satellogic-class small sats. Payload power "
                "envelope typically 30-50 W."
            ),
        ),

        # === Group D: concurrent-model density binding ===
        Mission(
            name="All-day autonomous agricultural tractor",
            category=MissionCategory.AGRICULTURE,
            description=(
                "5000 kg tractor, 12 cameras + 2 LIDAR + IMU fusion, crop/"
                "weed recognition, humans-in-field, path planning - all "
                "concurrent in RF-denied fields."
            ),
            compute_tops=200.0,
            mission_hours=10.0,
            binding_threshold=PhysicsThreshold.CONCURRENT_DENSITY,
            battery_wh=0.0,                # diesel fuel, not Wh-bound
            non_compute_power_w=0.0,
            payload_mass_budget_g=5000.0,
            thermal_envelope_w=15.0,       # single-chip dust-sealed cab slot
            citation=(
                "John Deere Autonomous 8R and similar. Dust-sealed (IP65+) "
                "cab enclosures limit single-chip dissipation to ~15 W "
                "without active cooling."
            ),
        ),
        Mission(
            name="8-hour-shift industrial humanoid robot",
            category=MissionCategory.HUMANOID,
            description=(
                "30-80 kg humanoid, 8 h shift, battery-powered. Full-body "
                "motor control + multimodal perception + VLM grounding + "
                "task planning."
            ),
            compute_tops=500.0,
            mission_hours=8.0,
            binding_threshold=PhysicsThreshold.ENERGY_BUDGET,
            battery_wh=3000.0,
            non_compute_power_w=250.0,     # actuators average (conservative)
            payload_mass_budget_g=3000.0,
            thermal_envelope_w=80.0,
            citation=(
                "Figure 01 / Tesla Optimus / Boston Dynamics Atlas-class "
                "humanoids. Typical 2.5-3 kWh battery for 8 h shift."
            ),
        ),
    ]


# ======================================================================
# Report aggregation
# ======================================================================


@dataclass
class MissionCapabilityReport:
    missions: List[Mission]
    archs: List[ArchProfile]
    results: Dict[Tuple[str, str], MissionResult]  # (mission, arch) -> result
    generated_at: str = ""

    def result(self, mission_name: str, arch_name: str) -> MissionResult:
        return self.results[(mission_name, arch_name)]

    def feasibility_ratio(self, arch: ArchProfile) -> Tuple[int, int]:
        """(feasible count, total) for this arch across all missions."""
        n_total = len(self.missions)
        n_feasible = sum(
            1 for m in self.missions
            if self.result(m.name, arch.name).feasible
        )
        return n_feasible, n_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "missions": [m.to_dict() for m in self.missions],
            "archs": [a.to_dict() for a in self.archs],
            "results": [
                {
                    "mission": m_name, "arch": a_name,
                    **r.to_dict(),
                }
                for (m_name, a_name), r in self.results.items()
            ],
            "generated_at": self.generated_at,
        }


def build_default_report() -> MissionCapabilityReport:
    from datetime import datetime, timezone
    missions = default_mission_catalog()
    archs = [gpu_profile(), kpu_profile()]
    results: Dict[Tuple[str, str], MissionResult] = {}
    for m in missions:
        for a in archs:
            results[(m.name, a.name)] = evaluate(m, a)
    return MissionCapabilityReport(
        missions=missions,
        archs=archs,
        results=results,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ======================================================================
# HTML rendering
# ======================================================================


_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _feasibility_badge(feasible: bool) -> str:
    if feasible:
        return ('<span class="cat" style="background:#3fc98a;">'
                'CAN DO</span>')
    return ('<span class="cat" style="background:#d4860b;">'
            'CANNOT DO</span>')


def _render_summary_table(report: MissionCapabilityReport) -> str:
    rows = []
    for m in report.missions:
        gpu_r = report.result(m.name, report.archs[0].name)
        kpu_r = report.result(m.name, report.archs[1].name)
        rows.append(
            f'<tr>'
            f'<td class="name">{html.escape(m.name)}</td>'
            f'<td>{html.escape(m.category.value)}</td>'
            f'<td class="num">{m.compute_tops:.0f}</td>'
            f'<td class="num">{m.mission_hours:.1f}</td>'
            f'<td>{html.escape(m.binding_threshold.value)}</td>'
            f'<td class="num">{gpu_r.compute_power_w:.1f} W</td>'
            f'<td>{_feasibility_badge(gpu_r.feasible)}</td>'
            f'<td class="num">{kpu_r.compute_power_w:.2f} W</td>'
            f'<td>{_feasibility_badge(kpu_r.feasible)}</td>'
            f'</tr>'
        )
    gpu_f, total = report.feasibility_ratio(report.archs[0])
    kpu_f, _ = report.feasibility_ratio(report.archs[1])
    rows.append(
        f'<tr class="total-row">'
        f'<td colspan="5"><strong>Feasibility tally</strong></td>'
        f'<td></td>'
        f'<td><strong>{gpu_f} / {total} missions feasible</strong></td>'
        f'<td></td>'
        f'<td><strong>{kpu_f} / {total} missions feasible</strong></td>'
        f'</tr>'
    )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>Mission</th><th>Category</th>'
        '<th>Req. TOPS</th><th>Hours</th>'
        '<th>Binding threshold</th>'
        '<th>GPU compute W</th><th>GPU verdict</th>'
        '<th>KPU compute W</th><th>KPU verdict</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def _render_mission_card(m: Mission, gpu_r: MissionResult,
                         kpu_r: MissionResult) -> str:
    gpu_verdict = (
        f"CAN DO ({gpu_r.mission_hours_enabled:.1f} h enabled, "
        f"{gpu_r.headroom:.1f}x budget)"
        if gpu_r.feasible else
        f"CANNOT DO - {gpu_r.binding_violation}"
    )
    kpu_verdict = (
        f"CAN DO ({kpu_r.mission_hours_enabled:.1f} h enabled, "
        f"{kpu_r.headroom:.1f}x budget)"
        if kpu_r.feasible else
        f"CANNOT DO - {kpu_r.binding_violation}"
    )
    return f"""
<section class="mission-card">
  <h3>{html.escape(m.name)}</h3>
  <div class="meta">
    <strong>Category:</strong> {html.escape(m.category.value)}
    | <strong>Required TOPS:</strong> {m.compute_tops:.0f}
    | <strong>Duration:</strong> {m.mission_hours:.1f} h
    | <strong>Binding:</strong> {html.escape(m.binding_threshold.value)}
  </div>
  <p class="mission-desc">{html.escape(m.description)}</p>
  <div class="mission-grid">
    <div class="arch-col gpu">
      <h4>GPU ({m.compute_tops:.0f} TOPS @ {gpu_r.compute_power_w:.1f} W)</h4>
      <p>{html.escape(gpu_verdict)}</p>
    </div>
    <div class="arch-col kpu">
      <h4>KPU ({m.compute_tops:.0f} TOPS @ {kpu_r.compute_power_w:.2f} W)</h4>
      <p>{html.escape(kpu_verdict)}</p>
    </div>
  </div>
  <p class="mission-cite"><em>{html.escape(m.citation)}</em></p>
</section>
"""


def _render_hours_curve_js(report: MissionCapabilityReport) -> str:
    """For energy-bound missions, render a mission-hours vs TOPS curve
    comparing GPU and KPU. One chart per mission."""
    charts: Dict[str, Any] = {}
    for m in report.missions:
        if m.battery_wh <= 0:
            continue
        gpu = next(a for a in report.archs if "GPU" in a.name)
        kpu = next(a for a in report.archs if "KPU" in a.name)
        gpu_curve = hours_enabled_curve(m, gpu, tops_range=(1.0, 2000.0))
        kpu_curve = hours_enabled_curve(m, kpu, tops_range=(1.0, 2000.0))
        mission_point_x = m.compute_tops
        mission_point_y = m.mission_hours
        traces = [
            {
                "type": "scatter", "mode": "lines",
                "name": "GPU hours-enabled",
                "x": [p[0] for p in gpu_curve],
                "y": [p[1] for p in gpu_curve],
                "line": {"color": "#5b8ff9", "width": 2},
            },
            {
                "type": "scatter", "mode": "lines",
                "name": "KPU hours-enabled",
                "x": [p[0] for p in kpu_curve],
                "y": [p[1] for p in kpu_curve],
                "line": {"color": "#3fc98a", "width": 2},
            },
            {
                "type": "scatter", "mode": "markers+text",
                "name": "Mission requirement",
                "x": [mission_point_x], "y": [mission_point_y],
                "marker": {"size": 14, "color": "#d4860b",
                           "symbol": "star"},
                "text": ["required (TOPS, hours)"],
                "textposition": "top right",
            },
        ]
        chart_id = (
            f"chart_mission_{abs(hash(m.name)) & 0xFFFFFFFF:x}"
        )
        charts[chart_id] = {
            "data": traces,
            "layout": {
                "title": (
                    f"{m.name} - mission-hours enabled vs required TOPS"
                ),
                "xaxis": {
                    "title": "Required TOPS (log)", "type": "log",
                },
                "yaxis": {
                    "title": "Mission-hours enabled (log)",
                    "type": "log",
                },
                "margin": {"t": 50, "b": 60, "l": 70, "r": 20},
                "legend": {"orientation": "h", "y": -0.2},
                "shapes": [
                    {
                        "type": "line",
                        "x0": 0.9, "x1": 2100,
                        "y0": mission_point_y, "y1": mission_point_y,
                        "xref": "x", "yref": "y",
                        "line": {"color": "#d4860b", "dash": "dot",
                                 "width": 1},
                    },
                    {
                        "type": "line",
                        "x0": mission_point_x, "x1": mission_point_x,
                        "y0": 0.001, "y1": 1e6,
                        "xref": "x", "yref": "y",
                        "line": {"color": "#d4860b", "dash": "dot",
                                 "width": 1},
                    },
                ],
            },
        }
    return (
        f"const MISSION_CHARTS = {json.dumps(charts)};\n"
        "for (const [id, spec] of Object.entries(MISSION_CHARTS)) {\n"
        "  const el = document.getElementById(id);\n"
        "  if (el) Plotly.newPlot(id, spec.data, spec.layout, "
        "{displayModeBar: false, responsive: true});\n"
        "}\n"
    )


def _render_wattage_chart_js(report: MissionCapabilityReport) -> str:
    """Single wattage-comparison chart across all missions: GPU compute W
    and KPU compute W on the same axis with the platform's compute budget
    (derived from binding threshold) overlaid."""
    gpu = next(a for a in report.archs if "GPU" in a.name)
    kpu = next(a for a in report.archs if "KPU" in a.name)
    labels = [m.name for m in report.missions]
    gpu_w = [
        report.result(m.name, gpu.name).compute_power_w
        for m in report.missions
    ]
    kpu_w = [
        report.result(m.name, kpu.name).compute_power_w
        for m in report.missions
    ]
    # Derived per-mission compute budget (what the arch can spend on
    # compute without violating the binding threshold).
    budgets = []
    for m in report.missions:
        if m.binding_threshold == PhysicsThreshold.ENERGY_BUDGET:
            # budget = battery / hours - non_compute
            budget = (m.battery_wh / m.mission_hours) - m.non_compute_power_w
        elif m.binding_threshold == PhysicsThreshold.THERMAL_ENVELOPE:
            budget = m.thermal_envelope_w - m.non_compute_power_w
        else:
            budget = m.thermal_envelope_w - m.non_compute_power_w
        budgets.append(max(budget, 0.0))
    traces = [
        {
            "type": "bar", "name": "GPU compute wattage",
            "x": labels, "y": gpu_w,
            "marker": {"color": "#5b8ff9"},
            "text": [f"{v:.1f} W" for v in gpu_w],
            "textposition": "outside",
        },
        {
            "type": "bar", "name": "KPU compute wattage",
            "x": labels, "y": kpu_w,
            "marker": {"color": "#3fc98a"},
            "text": [f"{v:.2f} W" for v in kpu_w],
            "textposition": "outside",
        },
        {
            "type": "scatter", "mode": "markers",
            "name": "Platform compute budget",
            "x": labels, "y": budgets,
            "marker": {"size": 16, "color": "#d4860b",
                       "symbol": "line-ew-open",
                       "line": {"width": 4}},
        },
    ]
    chart = {
        "data": traces,
        "layout": {
            "title": (
                "Compute wattage vs platform budget per mission "
                "(log-scale; anything above the orange bar is infeasible)"
            ),
            "xaxis": {"title": "", "tickangle": -30},
            "yaxis": {"title": "Wattage (W)", "type": "log"},
            "barmode": "group",
            "margin": {"t": 50, "b": 160, "l": 70, "r": 20},
            "legend": {"orientation": "h", "y": -0.4},
        },
    }
    payload = {"chart_wattage_vs_budget": chart}
    return (
        f"const WATT_CHARTS = {json.dumps(payload)};\n"
        "for (const [id, spec] of Object.entries(WATT_CHARTS)) {\n"
        "  const el = document.getElementById(id);\n"
        "  if (el) Plotly.newPlot(id, spec.data, spec.layout, "
        "{displayModeBar: false, responsive: true});\n"
        "}\n"
    )


def render_mission_capability_page(
    report: MissionCapabilityReport, repo_root: Path,
) -> str:
    from graphs.reporting.microarch_html_template import (
        _CSS, _load_logo,
        _render_brand_footer, _render_brand_header,
    )
    assets = _load_logo(repo_root)
    header = _render_brand_header(
        assets,
        "Mission Capability per Watt (embodied AI)",
        f"GPU vs KPU feasibility across 10 missions "
        f"| generated {report.generated_at}",
    )
    footer = _render_brand_footer("microarch-model-delivery-plan.md")

    extra_css = """
table.blocks { width: 100%; border-collapse: collapse; background: #fff;
               margin-bottom: 18px; font-size: 13px; }
table.blocks th, table.blocks td { padding: 7px 10px;
                                     border-bottom: 1px solid #e3e6eb;
                                     vertical-align: top; }
table.blocks th { font-size: 11px; text-transform: uppercase;
                  color: #586374; background: #f3f5f8; text-align: left; }
table.blocks td.name { font-weight: 600; color: #0a2540; }
table.blocks td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.blocks tr.total-row td { background: #eef6ea;
                                border-top: 2px solid #3fc98a;
                                border-bottom: 2px solid #3fc98a;
                                padding: 12px 10px; font-size: 14px; }
span.cat { display: inline-block; padding: 2px 8px; border-radius: 3px;
           color: white; font-size: 10px; font-weight: 600; }
section.mission-card { background: #fff; padding: 16px 20px;
                       border-radius: 6px; margin-bottom: 14px;
                       box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                       border-left: 4px solid #0a2540; }
section.mission-card h3 { margin: 0 0 6px; color: #0a2540; }
section.mission-card .meta { color: #586374; font-size: 13px;
                              margin-bottom: 8px; }
section.mission-card .mission-desc { color: #3a4452; font-size: 13px;
                                       margin: 6px 0 12px; }
section.mission-card .mission-cite { color: #586374; font-size: 11px;
                                       margin: 10px 0 0; }
.mission-grid { display: grid;
                grid-template-columns: 1fr 1fr; gap: 12px; }
.arch-col { background: #f3f5f8; padding: 10px 14px; border-radius: 4px; }
.arch-col h4 { margin: 0 0 6px; font-size: 13px; color: #0a2540; }
.arch-col p { margin: 0; font-size: 13px; color: #3a4452; }
.arch-col.gpu { border-left: 3px solid #5b8ff9; }
.arch-col.kpu { border-left: 3px solid #3fc98a; }
.chart-section { background: #fff; padding: 18px 22px; border-radius: 6px;
                 margin-bottom: 18px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.chart-section h3 { margin: 0 0 4px; color: #0a2540; }
.chart-section .chart-desc { color: #586374; font-size: 13px;
                             margin: 0 0 12px; }
.plot { width: 100%; min-height: 360px; }
section.method-note { background: #eef2f7; padding: 14px 18px;
                      border-left: 3px solid #0a2540; border-radius: 4px;
                      margin: 18px 0; }
a.nav-back { display: inline-block; color: #0a2540; text-decoration: none;
             font-weight: 600; margin-bottom: 10px; }
a.nav-back:hover { text-decoration: underline; }
"""

    summary_table = _render_summary_table(report)

    # Per-mission cards with hours-enabled charts for energy-bound ones
    cards_html_parts: List[str] = []
    for m in report.missions:
        gpu_r = report.result(m.name, report.archs[0].name)
        kpu_r = report.result(m.name, report.archs[1].name)
        cards_html_parts.append(_render_mission_card(m, gpu_r, kpu_r))
        if m.battery_wh > 0:
            chart_id = (
                f"chart_mission_{abs(hash(m.name)) & 0xFFFFFFFF:x}"
            )
            cards_html_parts.append(
                f'<div id="{chart_id}" class="plot" '
                f'style="min-height:340px; margin: -8px 0 18px;"></div>'
            )
    cards_html = "\n".join(cards_html_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Mission Capability per Watt</title>
  <script src="{_PLOTLY_CDN}"></script>
  <style>{_CSS}
{extra_css}
  </style>
</head>
<body>
{header}
<main>
  <p><a class="nav-back" href="index.html">&lt; Back to index</a></p>
  <section class="page-header">
    <h2>Mission Capability per Watt for embodied AI</h2>
    <div class="meta">Applies the silicon speed-of-light analysis
      to 10 catalogued embodied-AI missions. For each mission we
      quantify whether a GPU-class or KPU-class architecture fits
      the platform's physical budget (battery, thermal envelope,
      payload mass). The GPU/KPU feasibility column is a binary
      outcome driven by hard physical thresholds; the
      mission-hours-enabled curve shows the margin for energy-bound
      missions. See
      <a href="../../../docs/analysis/mission-capability-per-watt.md">
      docs/analysis/mission-capability-per-watt.md</a> for the
      narrative companion.</div>
  </section>

  <section class="chart-section">
    <h3>Feasibility summary (GPU vs KPU across 10 missions)</h3>
    <p class="chart-desc">One row per mission. The final "Feasibility
      tally" row is the investor headline - how many of the catalog's
      mission categories each architecture can address.</p>
    {summary_table}
  </section>

  <section class="chart-section">
    <h3>Compute wattage vs platform budget</h3>
    <p class="chart-desc">Per-mission compute wattage for GPU (blue)
      and KPU (green). Orange tick marks the maximum wattage the
      platform can spend on compute without violating its binding
      physical threshold. Bars above the orange tick are
      infeasible.</p>
    <div id="chart_wattage_vs_budget" class="plot"
         style="min-height:440px;"></div>
  </section>

  <section class="method-note">
    <h3 style="margin-top:0;">How to read the per-mission cards</h3>
    <p>Each card below shows the mission parameters, the GPU and KPU
      compute wattage at the required TOPS, and the verdict
      (CAN DO / CANNOT DO) for each architecture. For energy-bound
      missions, the chart below the card plots mission-hours enabled
      as a function of required TOPS, for both architectures. The
      orange star marks the mission's (required TOPS, required hours)
      point; curves above the star support the mission, curves below
      do not.</p>
  </section>

  {cards_html}

  <section class="method-note">
    <h3 style="margin-top:0;">Caveats</h3>
    <ul>
      <li>The required-TOPS numbers are engineering estimates, not
        workload-measured. See
        <a href="../../../docs/analysis/mission-capability-validation-template.md">
        mission-capability-validation-template.md</a> for the expert-
        validation template.</li>
      <li>GPU TOPS/W is fixed at 2.27 (Orin AGX dense INT8) across all
        power modes. Real GPUs vary modestly (2.0 - 2.5).</li>
      <li>KPU TOPS/W is the T128 target (22 TOPS/W dense INT8);
        calibration to measured silicon will sharpen the numbers.</li>
      <li>Non-compute power (motors, sensors, comms) is a flat
        average; real missions have duty cycles that would refine
        the mission-hours-enabled curves.</li>
    </ul>
  </section>
</main>
{footer}
<script>
{_render_wattage_chart_js(report)}
{_render_hours_curve_js(report)}
</script>
</body>
</html>
"""

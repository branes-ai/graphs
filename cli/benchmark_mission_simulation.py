#!/usr/bin/env python3
"""
Benchmark Mission Simulation CLI Tool

Simulate mission execution and track power/energy consumption over time.
Models duty cycles, state transitions, and battery depletion.

Usage:
    # Simulate a mission
    ./cli/benchmark_mission_simulation.py --profile drone-inspection --platform jetson-orin-nano --battery medium-lipo-4s

    # With specific duration
    ./cli/benchmark_mission_simulation.py --profile warehouse-amr --platform raspberry-pi-5 --battery large-lipo-6s --duration 4

    # JSON output
    ./cli/benchmark_mission_simulation.py --profile drone-inspection --platform jetson-orin-nano --battery medium-lipo-4s --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.mission.capability_tiers import (
    get_tier_for_power,
)
from graphs.mission.mission_profiles import (
    MISSION_PROFILES,
    get_mission_profile,
)
from graphs.mission.hardware_mapper import (
    get_platform,
    list_platforms,
)
from graphs.mission.battery import (
    BATTERY_CONFIGURATIONS,
    get_battery,
)


class MissionState(Enum):
    """Mission execution states."""
    IDLE = "idle"
    PERCEPTION = "perception"
    CONTROL = "control"
    MOVEMENT = "movement"
    CHARGING = "charging"
    COMPLETE = "complete"
    ABORTED = "aborted"


@dataclass
class SimulationStep:
    """A single step in the simulation."""
    time_minutes: float
    state: str
    power_w: float
    energy_consumed_wh: float
    battery_remaining_pct: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_minutes": self.time_minutes,
            "state": self.state,
            "power_w": self.power_w,
            "energy_consumed_wh": self.energy_consumed_wh,
            "battery_remaining_pct": self.battery_remaining_pct,
            "notes": self.notes,
        }


@dataclass
class MissionSimulationResult:
    """Result of mission simulation."""
    profile_name: str
    platform_name: str
    battery_name: str
    target_duration_hours: float

    # Configuration
    platform_tdp_w: float
    battery_capacity_wh: float

    # Simulation timeline
    steps: List[SimulationStep]

    # Summary
    actual_duration_hours: float
    total_energy_consumed_wh: float
    average_power_w: float
    mission_completed: bool
    abort_reason: Optional[str] = None

    # State statistics
    time_in_perception_pct: float = 0.0
    time_in_control_pct: float = 0.0
    time_in_movement_pct: float = 0.0
    time_in_idle_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configuration": {
                "profile": self.profile_name,
                "platform": self.platform_name,
                "battery": self.battery_name,
                "target_duration_hours": self.target_duration_hours,
                "platform_tdp_w": self.platform_tdp_w,
                "battery_capacity_wh": self.battery_capacity_wh,
            },
            "timeline": [s.to_dict() for s in self.steps],
            "summary": {
                "actual_duration_hours": self.actual_duration_hours,
                "total_energy_consumed_wh": self.total_energy_consumed_wh,
                "average_power_w": self.average_power_w,
                "mission_completed": self.mission_completed,
                "abort_reason": self.abort_reason,
            },
            "state_distribution": {
                "perception_pct": self.time_in_perception_pct,
                "control_pct": self.time_in_control_pct,
                "movement_pct": self.time_in_movement_pct,
                "idle_pct": self.time_in_idle_pct,
            },
        }


def simulate_mission(
    profile_name: str,
    platform_name: str,
    battery_name: str,
    duration_hours: float = 1.0,
    time_step_minutes: float = 5.0,
) -> MissionSimulationResult:
    """Simulate a mission execution."""
    profile = get_mission_profile(profile_name)
    if profile is None:
        raise ValueError(f"Unknown profile: {profile_name}")

    platform = get_platform(platform_name)
    if platform is None:
        raise ValueError(f"Unknown platform: {platform_name}")

    battery = get_battery(battery_name)
    if battery is None:
        raise ValueError(f"Unknown battery: {battery_name}")

    # Initial state
    usable_capacity = battery.capacity_wh * 0.9  # 90% usable
    remaining_energy = usable_capacity
    total_consumed = 0.0
    steps = []

    # State time tracking
    time_perception = 0.0
    time_control = 0.0
    time_movement = 0.0
    time_idle = 0.0

    # Power profiles by state
    perception_power = platform.tdp_w * 0.7  # Inference heavy
    control_power = platform.tdp_w * 0.4     # Control processing
    movement_power = platform.tdp_w * 0.3    # Movement overhead (compute portion)
    idle_power = platform.tdp_w * 0.15       # Idle

    # Duty cycles from profile
    perception_duty = profile.duty_cycle.perception
    control_duty = profile.duty_cycle.control
    movement_duty = profile.duty_cycle.movement

    # Simulation loop
    current_time = 0.0
    target_time = duration_hours * 60  # Convert to minutes
    battery_low_threshold = 10.0  # 10% battery = abort

    mission_completed = True
    abort_reason = None

    while current_time < target_time:
        # Determine current state based on duty cycle (simplified)
        # In reality, states would follow a more complex pattern
        cycle_position = (current_time / 10) % 1.0  # 10-minute cycles

        if cycle_position < perception_duty:
            state = MissionState.PERCEPTION
            power = perception_power
            time_perception += time_step_minutes
        elif cycle_position < perception_duty + control_duty:
            state = MissionState.CONTROL
            power = control_power
            time_control += time_step_minutes
        elif cycle_position < perception_duty + control_duty + movement_duty:
            state = MissionState.MOVEMENT
            power = movement_power
            time_movement += time_step_minutes
        else:
            state = MissionState.IDLE
            power = idle_power
            time_idle += time_step_minutes

        # Calculate energy for this step
        step_energy = power * (time_step_minutes / 60)  # Wh
        remaining_energy -= step_energy
        total_consumed += step_energy

        # Battery percentage
        battery_pct = (remaining_energy / usable_capacity) * 100

        # Check for low battery
        notes = ""
        if battery_pct < battery_low_threshold:
            mission_completed = False
            abort_reason = f"Battery depleted at {current_time:.0f} min"
            notes = "LOW BATTERY - ABORTING"
            steps.append(SimulationStep(
                time_minutes=current_time,
                state=MissionState.ABORTED.value,
                power_w=power,
                energy_consumed_wh=total_consumed,
                battery_remaining_pct=battery_pct,
                notes=notes,
            ))
            break

        # Record step (every N steps to keep output manageable)
        if len(steps) < 100 or current_time % 10 < time_step_minutes:
            steps.append(SimulationStep(
                time_minutes=current_time,
                state=state.value,
                power_w=power,
                energy_consumed_wh=total_consumed,
                battery_remaining_pct=battery_pct,
                notes=notes,
            ))

        current_time += time_step_minutes

    # Final step
    if mission_completed:
        steps.append(SimulationStep(
            time_minutes=current_time,
            state=MissionState.COMPLETE.value,
            power_w=0,
            energy_consumed_wh=total_consumed,
            battery_remaining_pct=(remaining_energy / usable_capacity) * 100,
            notes="Mission complete",
        ))

    # Calculate statistics
    actual_duration = current_time / 60  # Hours
    average_power = total_consumed / actual_duration if actual_duration > 0 else 0
    total_active_time = time_perception + time_control + time_movement + time_idle

    return MissionSimulationResult(
        profile_name=profile_name,
        platform_name=platform_name,
        battery_name=battery_name,
        target_duration_hours=duration_hours,
        platform_tdp_w=platform.tdp_w,
        battery_capacity_wh=battery.capacity_wh,
        steps=steps,
        actual_duration_hours=actual_duration,
        total_energy_consumed_wh=total_consumed,
        average_power_w=average_power,
        mission_completed=mission_completed,
        abort_reason=abort_reason,
        time_in_perception_pct=(time_perception / total_active_time * 100) if total_active_time > 0 else 0,
        time_in_control_pct=(time_control / total_active_time * 100) if total_active_time > 0 else 0,
        time_in_movement_pct=(time_movement / total_active_time * 100) if total_active_time > 0 else 0,
        time_in_idle_pct=(time_idle / total_active_time * 100) if total_active_time > 0 else 0,
    )


def format_simulation_result(result: MissionSimulationResult) -> str:
    """Format simulation result as text."""
    lines = []

    lines.append("=" * 80)
    lines.append("  MISSION SIMULATION RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Configuration
    lines.append("  Configuration:")
    lines.append(f"    Profile:          {result.profile_name}")
    lines.append(f"    Platform:         {result.platform_name} ({result.platform_tdp_w:.1f}W TDP)")
    lines.append(f"    Battery:          {result.battery_name} ({result.battery_capacity_wh:.0f}Wh)")
    lines.append(f"    Target Duration:  {result.target_duration_hours:.1f} hours")
    lines.append("")

    # Summary
    status = "COMPLETED" if result.mission_completed else "ABORTED"
    lines.append(f"  Mission Status: {status}")
    if result.abort_reason:
        lines.append(f"    Reason: {result.abort_reason}")
    lines.append("")

    lines.append("  Energy Summary:")
    lines.append(f"    Actual Duration:     {result.actual_duration_hours:.2f} hours ({result.actual_duration_hours*60:.0f} min)")
    lines.append(f"    Energy Consumed:     {result.total_energy_consumed_wh:.1f} Wh")
    lines.append(f"    Average Power:       {result.average_power_w:.1f} W")
    lines.append("")

    # State distribution
    lines.append("  State Distribution:")
    lines.append(f"    Perception: {result.time_in_perception_pct:>5.1f}%  {'#' * int(result.time_in_perception_pct/2)}")
    lines.append(f"    Control:    {result.time_in_control_pct:>5.1f}%  {'#' * int(result.time_in_control_pct/2)}")
    lines.append(f"    Movement:   {result.time_in_movement_pct:>5.1f}%  {'#' * int(result.time_in_movement_pct/2)}")
    lines.append(f"    Idle:       {result.time_in_idle_pct:>5.1f}%  {'#' * int(result.time_in_idle_pct/2)}")
    lines.append("")

    # Timeline (sampled)
    lines.append("  Timeline (sampled):")
    lines.append("  " + "-" * 70)
    lines.append(f"    {'Time':>8} {'State':<12} {'Power':>8} {'Energy':>10} {'Battery':>10}")
    lines.append("  " + "-" * 70)

    # Show first, middle, and last steps plus key transitions
    display_steps = []
    if len(result.steps) > 0:
        display_steps.append(result.steps[0])
    if len(result.steps) > 10:
        display_steps.extend(result.steps[1:10:3])
    if len(result.steps) > 20:
        mid = len(result.steps) // 2
        display_steps.extend(result.steps[mid-2:mid+2])
    if len(result.steps) > 2:
        display_steps.extend(result.steps[-3:])

    # Remove duplicates while preserving order
    seen = set()
    unique_steps = []
    for s in display_steps:
        key = s.time_minutes
        if key not in seen:
            seen.add(key)
            unique_steps.append(s)

    for s in sorted(unique_steps, key=lambda x: x.time_minutes):
        lines.append(
            f"    {s.time_minutes:>7.0f}m {s.state:<12} {s.power_w:>7.1f}W "
            f"{s.energy_consumed_wh:>9.1f}Wh {s.battery_remaining_pct:>9.1f}%"
        )

    lines.append("  " + "-" * 70)
    lines.append("")

    # Battery depletion curve (ASCII art)
    lines.append("  Battery Level Over Time:")
    lines.append("")
    bar_width = 50
    for step in result.steps[::max(1, len(result.steps)//8)]:  # Sample ~8 points
        bar_len = int(step.battery_remaining_pct / 100 * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        lines.append(f"    {step.time_minutes:>5.0f}m [{bar}] {step.battery_remaining_pct:.0f}%")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate mission execution and power consumption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulate a mission
  ./cli/benchmark_mission_simulation.py --profile drone-inspection --platform jetson-orin-nano --battery medium-lipo-4s

  # With specific duration
  ./cli/benchmark_mission_simulation.py --profile warehouse-amr --platform raspberry-pi-5 --battery large-lipo-6s --duration 4

  # JSON output
  ./cli/benchmark_mission_simulation.py --profile drone-inspection --platform jetson-orin-nano --battery medium-lipo-4s --format json
"""
    )

    parser.add_argument(
        "--profile", "-p",
        type=str,
        required=True,
        choices=list(MISSION_PROFILES.keys()),
        help="Mission profile"
    )
    parser.add_argument(
        "--platform",
        type=str,
        required=True,
        help="Hardware platform"
    )
    parser.add_argument(
        "--battery", "-b",
        type=str,
        required=True,
        help="Battery configuration"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=1.0,
        help="Target mission duration in hours (default: 1)"
    )
    parser.add_argument(
        "--step",
        type=float,
        default=5.0,
        help="Simulation time step in minutes (default: 5)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles"
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="List available platforms"
    )
    parser.add_argument(
        "--list-batteries",
        action="store_true",
        help="List available batteries"
    )

    args = parser.parse_args()

    # Handle list options
    if args.list_profiles:
        print("Available mission profiles:")
        for name in sorted(MISSION_PROFILES.keys()):
            print(f"  {name}")
        return 0

    if args.list_platforms:
        print("Available platforms:")
        for name in sorted(list_platforms()):
            platform = get_platform(name)
            if platform:
                print(f"  {name:<30} TDP: {platform.tdp_w:>6.1f}W")
        return 0

    if args.list_batteries:
        print("Available batteries:")
        for name, battery in sorted(BATTERY_CONFIGURATIONS.items()):
            print(f"  {name:<25} {battery.capacity_wh:>6.0f}Wh")
        return 0

    # Run simulation
    try:
        result = simulate_mission(
            profile_name=args.profile,
            platform_name=args.platform,
            battery_name=args.battery,
            duration_hours=args.duration,
            time_step_minutes=args.step,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Output
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_simulation_result(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())

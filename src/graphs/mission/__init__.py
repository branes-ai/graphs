"""
Mission planning and capability tier analysis.

This package provides tools for analyzing power budgets, mission durations,
and hardware selection for embodied AI systems across different capability tiers.

Capability Tiers:
    - Wearable AI (0.1-1W): Smartwatches, AR glasses, health monitors
    - Micro-Autonomy (1-10W): Drones, handheld scanners, smart IoT
    - Industrial Edge (10-30W): Factory AMRs, cobots, automated sorting
    - Embodied AI (30-100W+): Quadrupeds, humanoids, world-model simulation
    - Automotive AI (100-500W): L3/L3+/L4 autonomous driving
"""

from graphs.mission.capability_tiers import (
    CapabilityTier,
    CAPABILITY_TIERS,
    get_tier_by_name,
    get_tier_for_power,
    list_tier_names,
)

from graphs.mission.power_allocation import (
    PowerAllocation,
    SubsystemType,
    TYPICAL_ALLOCATIONS,
    get_typical_allocation,
)

from graphs.mission.mission_profiles import (
    MissionProfile,
    DutyCycle,
    MISSION_PROFILES,
    get_mission_profile,
    list_mission_profiles,
)

from graphs.mission.battery import (
    BatteryConfiguration,
    BatteryChemistry,
    BATTERY_CONFIGURATIONS,
    get_battery_by_name,
    find_batteries_for_mission,
)

__all__ = [
    # Capability Tiers
    "CapabilityTier",
    "CAPABILITY_TIERS",
    "get_tier_by_name",
    "get_tier_for_power",
    "list_tier_names",
    # Power Allocation
    "PowerAllocation",
    "SubsystemType",
    "TYPICAL_ALLOCATIONS",
    "get_typical_allocation",
    # Mission Profiles
    "MissionProfile",
    "DutyCycle",
    "MISSION_PROFILES",
    "get_mission_profile",
    "list_mission_profiles",
    # Battery
    "BatteryConfiguration",
    "BatteryChemistry",
    "BATTERY_CONFIGURATIONS",
    "get_battery_by_name",
    "find_batteries_for_mission",
]

"""
Battery configuration models for embodied AI systems.

Provides battery specifications and utilities for estimating runtime,
sizing batteries for missions, and comparing battery options.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from graphs.mission.capability_tiers import TierName


class BatteryChemistry(Enum):
    """Common battery chemistries for embodied AI."""
    LIPO = "lipo"  # Lithium Polymer - high energy density, drone-grade
    LI_ION = "li-ion"  # Lithium Ion - general purpose
    LIFEPO4 = "lifepo4"  # Lithium Iron Phosphate - safety, longevity
    LI_ION_HV = "li-ion-hv"  # High-voltage Li-ion - automotive grade
    NIMH = "nimh"  # Nickel-Metal Hydride - legacy, low cost


@dataclass
class BatteryConfiguration:
    """
    Battery configuration for an embodied AI system.

    Attributes:
        name: Configuration identifier
        chemistry: Battery chemistry type
        capacity_wh: Total capacity in watt-hours
        voltage_nominal: Nominal voltage
        weight_kg: Total battery weight in kg
        volume_cm3: Total volume in cubic centimeters
        max_discharge_rate_c: Maximum continuous discharge rate (C-rate)
        peak_discharge_rate_c: Peak discharge rate (short bursts)
        cycle_life: Expected cycle life at 80% DoD
        operating_temp_min_c: Minimum operating temperature
        operating_temp_max_c: Maximum operating temperature
        typical_tier: Typical capability tier for this battery
        description: Description of use case
    """
    name: str
    chemistry: BatteryChemistry
    capacity_wh: float
    voltage_nominal: float
    weight_kg: float
    volume_cm3: float
    max_discharge_rate_c: float = 1.0
    peak_discharge_rate_c: float = 2.0
    cycle_life: int = 500
    operating_temp_min_c: float = 0.0
    operating_temp_max_c: float = 45.0
    typical_tier: Optional[TierName] = None
    description: str = ""

    @property
    def energy_density_wh_per_kg(self) -> float:
        """Gravimetric energy density in Wh/kg."""
        return self.capacity_wh / self.weight_kg if self.weight_kg > 0 else 0.0

    @property
    def energy_density_wh_per_l(self) -> float:
        """Volumetric energy density in Wh/L."""
        volume_l = self.volume_cm3 / 1000.0
        return self.capacity_wh / volume_l if volume_l > 0 else 0.0

    @property
    def max_continuous_power_w(self) -> float:
        """Maximum continuous discharge power in watts."""
        return self.capacity_wh * self.max_discharge_rate_c

    @property
    def peak_power_w(self) -> float:
        """Peak discharge power in watts."""
        return self.capacity_wh * self.peak_discharge_rate_c

    def estimate_runtime_hours(self, average_power_w: float, safety_margin: float = 0.9) -> float:
        """
        Estimate runtime at a given average power draw.

        Args:
            average_power_w: Average power consumption in watts
            safety_margin: Usable capacity fraction (default 0.9 = 90%)

        Returns:
            Estimated runtime in hours
        """
        if average_power_w <= 0:
            return float('inf')

        usable_capacity = self.capacity_wh * safety_margin
        return usable_capacity / average_power_w

    def can_support_power(self, power_w: float, continuous: bool = True) -> bool:
        """Check if battery can support a given power level."""
        if continuous:
            return power_w <= self.max_continuous_power_w
        return power_w <= self.peak_power_w

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "chemistry": self.chemistry.value,
            "capacity_wh": self.capacity_wh,
            "voltage_nominal": self.voltage_nominal,
            "weight_kg": self.weight_kg,
            "volume_cm3": self.volume_cm3,
            "energy_density_wh_per_kg": self.energy_density_wh_per_kg,
            "energy_density_wh_per_l": self.energy_density_wh_per_l,
            "max_continuous_power_w": self.max_continuous_power_w,
            "peak_power_w": self.peak_power_w,
            "max_discharge_rate_c": self.max_discharge_rate_c,
            "cycle_life": self.cycle_life,
            "operating_temp_range_c": [self.operating_temp_min_c, self.operating_temp_max_c],
        }


# =============================================================================
# Battery Configuration Database
# =============================================================================

# Wearable AI Batteries (tiny, high density)
BATTERY_WEARABLE_SMALL = BatteryConfiguration(
    name="wearable-small",
    chemistry=BatteryChemistry.LIPO,
    capacity_wh=1.5,
    voltage_nominal=3.7,
    weight_kg=0.010,
    volume_cm3=3.0,
    max_discharge_rate_c=1.0,
    peak_discharge_rate_c=2.0,
    cycle_life=500,
    typical_tier=TierName.WEARABLE_AI,
    description="Smartwatch class battery (400mAh)",
)

BATTERY_WEARABLE_MEDIUM = BatteryConfiguration(
    name="wearable-medium",
    chemistry=BatteryChemistry.LIPO,
    capacity_wh=4.0,
    voltage_nominal=3.7,
    weight_kg=0.025,
    volume_cm3=8.0,
    max_discharge_rate_c=1.0,
    peak_discharge_rate_c=2.0,
    cycle_life=500,
    typical_tier=TierName.WEARABLE_AI,
    description="AR glasses class battery (1000mAh)",
)

# Micro-Autonomy Batteries (drone-grade LiPo)
BATTERY_DRONE_SMALL = BatteryConfiguration(
    name="drone-small",
    chemistry=BatteryChemistry.LIPO,
    capacity_wh=50.0,
    voltage_nominal=14.8,  # 4S
    weight_kg=0.35,
    volume_cm3=150.0,
    max_discharge_rate_c=25.0,
    peak_discharge_rate_c=50.0,
    cycle_life=200,
    operating_temp_min_c=-10.0,
    operating_temp_max_c=50.0,
    typical_tier=TierName.MICRO_AUTONOMY,
    description="Small drone battery (3300mAh 4S 25C)",
)

BATTERY_DRONE_MEDIUM = BatteryConfiguration(
    name="drone-medium",
    chemistry=BatteryChemistry.LIPO,
    capacity_wh=100.0,
    voltage_nominal=22.2,  # 6S
    weight_kg=0.65,
    volume_cm3=280.0,
    max_discharge_rate_c=25.0,
    peak_discharge_rate_c=50.0,
    cycle_life=200,
    operating_temp_min_c=-10.0,
    operating_temp_max_c=50.0,
    typical_tier=TierName.MICRO_AUTONOMY,
    description="Medium drone battery (4500mAh 6S 25C)",
)

BATTERY_HANDHELD = BatteryConfiguration(
    name="handheld-industrial",
    chemistry=BatteryChemistry.LI_ION,
    capacity_wh=40.0,
    voltage_nominal=10.8,  # 3S
    weight_kg=0.30,
    volume_cm3=120.0,
    max_discharge_rate_c=2.0,
    peak_discharge_rate_c=5.0,
    cycle_life=500,
    operating_temp_min_c=-20.0,
    operating_temp_max_c=60.0,
    typical_tier=TierName.MICRO_AUTONOMY,
    description="Industrial handheld battery pack",
)

# Industrial Edge Batteries (AMR/Cobot grade)
BATTERY_AMR_SMALL = BatteryConfiguration(
    name="amr-small",
    chemistry=BatteryChemistry.LIFEPO4,
    capacity_wh=500.0,
    voltage_nominal=25.6,  # 8S LFP
    weight_kg=4.5,
    volume_cm3=2500.0,
    max_discharge_rate_c=1.0,
    peak_discharge_rate_c=3.0,
    cycle_life=2000,
    operating_temp_min_c=-20.0,
    operating_temp_max_c=55.0,
    typical_tier=TierName.INDUSTRIAL_EDGE,
    description="Small AMR battery (20Ah LiFePO4)",
)

BATTERY_AMR_LARGE = BatteryConfiguration(
    name="amr-large",
    chemistry=BatteryChemistry.LIFEPO4,
    capacity_wh=1200.0,
    voltage_nominal=51.2,  # 16S LFP
    weight_kg=12.0,
    volume_cm3=6000.0,
    max_discharge_rate_c=1.0,
    peak_discharge_rate_c=2.0,
    cycle_life=3000,
    operating_temp_min_c=-20.0,
    operating_temp_max_c=55.0,
    typical_tier=TierName.INDUSTRIAL_EDGE,
    description="Large AMR battery (24Ah LiFePO4)",
)

# Embodied AI Batteries (robot-grade)
BATTERY_QUADRUPED = BatteryConfiguration(
    name="quadruped-standard",
    chemistry=BatteryChemistry.LI_ION,
    capacity_wh=800.0,
    voltage_nominal=48.0,
    weight_kg=6.0,
    volume_cm3=4000.0,
    max_discharge_rate_c=2.0,
    peak_discharge_rate_c=5.0,
    cycle_life=500,
    operating_temp_min_c=-10.0,
    operating_temp_max_c=45.0,
    typical_tier=TierName.EMBODIED_AI,
    description="Quadruped robot battery pack (Spot-class)",
)

BATTERY_HUMANOID = BatteryConfiguration(
    name="humanoid-standard",
    chemistry=BatteryChemistry.LI_ION,
    capacity_wh=2000.0,
    voltage_nominal=48.0,
    weight_kg=14.0,
    volume_cm3=9000.0,
    max_discharge_rate_c=2.0,
    peak_discharge_rate_c=4.0,
    cycle_life=500,
    operating_temp_min_c=-10.0,
    operating_temp_max_c=45.0,
    typical_tier=TierName.EMBODIED_AI,
    description="Humanoid robot battery system",
)

# Automotive AI Batteries (vehicle-integrated)
BATTERY_AUTOMOTIVE_COMPUTE = BatteryConfiguration(
    name="automotive-compute-module",
    chemistry=BatteryChemistry.LI_ION_HV,
    capacity_wh=5000.0,  # Dedicated compute battery
    voltage_nominal=400.0,
    weight_kg=35.0,
    volume_cm3=25000.0,
    max_discharge_rate_c=1.0,
    peak_discharge_rate_c=2.0,
    cycle_life=1500,
    operating_temp_min_c=-40.0,
    operating_temp_max_c=85.0,
    typical_tier=TierName.AUTOMOTIVE_AI,
    description="Automotive compute power module (from main pack)",
)


# =============================================================================
# Battery Registry
# =============================================================================

BATTERY_CONFIGURATIONS: Dict[str, BatteryConfiguration] = {
    # Wearable
    "wearable-small": BATTERY_WEARABLE_SMALL,
    "wearable-medium": BATTERY_WEARABLE_MEDIUM,
    # Micro-Autonomy
    "drone-small": BATTERY_DRONE_SMALL,
    "drone-medium": BATTERY_DRONE_MEDIUM,
    "handheld-industrial": BATTERY_HANDHELD,
    # Industrial Edge
    "amr-small": BATTERY_AMR_SMALL,
    "amr-large": BATTERY_AMR_LARGE,
    # Embodied AI
    "quadruped-standard": BATTERY_QUADRUPED,
    "humanoid-standard": BATTERY_HUMANOID,
    # Automotive
    "automotive-compute-module": BATTERY_AUTOMOTIVE_COMPUTE,
}


def get_battery_by_name(name: str) -> Optional[BatteryConfiguration]:
    """Get a battery configuration by name."""
    return BATTERY_CONFIGURATIONS.get(name.lower().replace(" ", "-").replace("_", "-"))


def find_batteries_for_mission(
    mission_hours: float,
    average_power_w: float,
    tier: Optional[TierName] = None,
    max_weight_kg: Optional[float] = None,
    safety_margin: float = 0.9,
) -> List[BatteryConfiguration]:
    """
    Find batteries that can support a mission.

    Args:
        mission_hours: Required mission duration in hours
        average_power_w: Average power consumption in watts
        tier: Optional capability tier filter
        max_weight_kg: Optional maximum weight constraint
        safety_margin: Usable capacity fraction (default 0.9)

    Returns:
        List of suitable battery configurations, sorted by weight
    """
    required_wh = (mission_hours * average_power_w) / safety_margin

    candidates = []
    for battery in BATTERY_CONFIGURATIONS.values():
        # Check capacity
        if battery.capacity_wh < required_wh:
            continue

        # Check tier if specified
        if tier is not None and battery.typical_tier != tier:
            continue

        # Check weight if specified
        if max_weight_kg is not None and battery.weight_kg > max_weight_kg:
            continue

        # Check power capability
        if not battery.can_support_power(average_power_w, continuous=True):
            continue

        candidates.append(battery)

    # Sort by weight (lighter first)
    return sorted(candidates, key=lambda b: b.weight_kg)


def find_batteries_for_tier(tier: TierName) -> List[BatteryConfiguration]:
    """Get all batteries suitable for a capability tier."""
    return [
        battery for battery in BATTERY_CONFIGURATIONS.values()
        if battery.typical_tier == tier
    ]


def estimate_battery_weight(
    mission_hours: float,
    average_power_w: float,
    energy_density_wh_per_kg: float = 150.0,
    safety_margin: float = 0.9,
) -> float:
    """
    Estimate battery weight for a mission.

    Args:
        mission_hours: Mission duration in hours
        average_power_w: Average power consumption in watts
        energy_density_wh_per_kg: Assumed energy density (default 150 Wh/kg)
        safety_margin: Usable capacity fraction

    Returns:
        Estimated battery weight in kg
    """
    required_wh = (mission_hours * average_power_w) / safety_margin
    return required_wh / energy_density_wh_per_kg

"""
Capability Tier definitions for embodied AI systems.

Capability tiers categorize embodied AI platforms by their power envelope
and typical application domains. Each tier has characteristic constraints
around thermal management, battery capacity, and compute density.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class TierName(Enum):
    """Capability tier identifiers."""
    WEARABLE_AI = "wearable-ai"
    MICRO_AUTONOMY = "micro-autonomy"
    INDUSTRIAL_EDGE = "industrial-edge"
    EMBODIED_AI = "embodied-ai"
    AUTOMOTIVE_AI = "automotive-ai"


@dataclass
class ThermalConstraints:
    """Thermal constraints for a capability tier."""
    max_ambient_temp_c: float = 40.0
    max_junction_temp_c: float = 85.0
    typical_cooling: str = "passive"  # passive, active-fan, liquid
    sustained_power_derating: float = 0.8  # Fraction of peak power sustainable


@dataclass
class FormFactorConstraints:
    """Physical form factor constraints."""
    max_compute_weight_kg: Optional[float] = None
    max_compute_volume_cm3: Optional[float] = None
    max_total_system_weight_kg: Optional[float] = None
    typical_form_factors: List[str] = field(default_factory=list)


@dataclass
class CapabilityTier:
    """
    Definition of a capability tier for embodied AI systems.

    Capability tiers represent classes of systems with similar power envelopes,
    application domains, and operational constraints.

    Attributes:
        name: Tier identifier (e.g., "micro-autonomy")
        display_name: Human-readable name
        power_min_w: Minimum power envelope (watts)
        power_max_w: Maximum power envelope (watts)
        description: Detailed description of the tier
        typical_applications: List of typical application domains
        example_platforms: List of example hardware platforms
        thermal: Thermal constraints for this tier
        form_factor: Physical form factor constraints
        typical_mission_hours: Typical mission duration range
        typical_battery_wh_per_kg: Typical battery energy density available
    """
    name: TierName
    display_name: str
    power_min_w: float
    power_max_w: float
    description: str
    typical_applications: List[str]
    example_platforms: List[str] = field(default_factory=list)
    thermal: ThermalConstraints = field(default_factory=ThermalConstraints)
    form_factor: FormFactorConstraints = field(default_factory=FormFactorConstraints)
    typical_mission_hours: tuple = (1.0, 8.0)  # (min, max)
    typical_battery_wh_per_kg: float = 150.0  # Wh/kg for typical batteries

    @property
    def power_range_str(self) -> str:
        """Format power range as string."""
        if self.power_max_w < 1.0:
            return f"{self.power_min_w*1000:.0f}-{self.power_max_w*1000:.0f}mW"
        elif self.power_min_w < 1.0:
            return f"{self.power_min_w:.1f}-{self.power_max_w:.0f}W"
        return f"{self.power_min_w:.0f}-{self.power_max_w:.0f}W"

    @property
    def typical_power_w(self) -> float:
        """Typical operating power (geometric mean of range)."""
        return (self.power_min_w * self.power_max_w) ** 0.5

    def contains_power(self, power_w: float) -> bool:
        """Check if a power level falls within this tier."""
        return self.power_min_w <= power_w <= self.power_max_w

    def estimate_battery_weight_kg(self, mission_hours: float) -> float:
        """Estimate battery weight for a mission duration."""
        energy_wh = self.typical_power_w * mission_hours
        return energy_wh / self.typical_battery_wh_per_kg

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name.value,
            "display_name": self.display_name,
            "power_min_w": self.power_min_w,
            "power_max_w": self.power_max_w,
            "power_range": self.power_range_str,
            "description": self.description,
            "typical_applications": self.typical_applications,
            "example_platforms": self.example_platforms,
            "typical_mission_hours": list(self.typical_mission_hours),
            "thermal": {
                "max_ambient_temp_c": self.thermal.max_ambient_temp_c,
                "max_junction_temp_c": self.thermal.max_junction_temp_c,
                "typical_cooling": self.thermal.typical_cooling,
                "sustained_power_derating": self.thermal.sustained_power_derating,
            },
        }


# =============================================================================
# Capability Tier Definitions
# =============================================================================

WEARABLE_AI = CapabilityTier(
    name=TierName.WEARABLE_AI,
    display_name="Wearable AI",
    power_min_w=0.1,
    power_max_w=1.0,
    description=(
        "Ultra-low power wearable devices with severe thermal and battery constraints. "
        "Requires aggressive model optimization (quantization, pruning) and often "
        "relies on sensor fusion with minimal on-device inference."
    ),
    typical_applications=[
        "Smartwatches with health monitoring",
        "AR/VR glasses with eye tracking",
        "Hearing aids with noise cancellation",
        "Fitness trackers with activity recognition",
        "Smart rings with gesture detection",
    ],
    example_platforms=[
        "Coral Micro (Edge TPU)",
        "MAX78000 (Analog Devices)",
        "GAP9 (GreenWaves)",
        "Syntiant NDP120",
    ],
    thermal=ThermalConstraints(
        max_ambient_temp_c=35.0,  # Body-worn, lower ambient
        max_junction_temp_c=70.0,  # Lower for skin contact
        typical_cooling="passive",
        sustained_power_derating=0.9,  # Less derating needed at low power
    ),
    form_factor=FormFactorConstraints(
        max_compute_weight_kg=0.02,  # 20g max for compute module
        max_compute_volume_cm3=5.0,  # Very compact
        max_total_system_weight_kg=0.1,  # 100g total
        typical_form_factors=["watch", "glasses", "earbuds", "ring"],
    ),
    typical_mission_hours=(8.0, 24.0),  # All-day wear
    typical_battery_wh_per_kg=200.0,  # Small Li-Po cells
)

MICRO_AUTONOMY = CapabilityTier(
    name=TierName.MICRO_AUTONOMY,
    display_name="Micro-Autonomy",
    power_min_w=1.0,
    power_max_w=10.0,
    description=(
        "Small autonomous systems with significant weight and thermal constraints. "
        "Typical use cases include aerial drones, handheld industrial devices, and "
        "smart IoT sensors requiring real-time inference."
    ),
    typical_applications=[
        "Inspection drones",
        "Handheld barcode/defect scanners",
        "Smart security cameras",
        "Agricultural monitoring drones",
        "Delivery micro-drones",
    ],
    example_platforms=[
        "Jetson Orin Nano (8GB)",
        "Hailo-8",
        "Intel Myriad X",
        "Google Coral Edge TPU",
        "Qualcomm QCS6490",
    ],
    thermal=ThermalConstraints(
        max_ambient_temp_c=45.0,
        max_junction_temp_c=85.0,
        typical_cooling="passive",
        sustained_power_derating=0.75,
    ),
    form_factor=FormFactorConstraints(
        max_compute_weight_kg=0.1,  # 100g
        max_compute_volume_cm3=50.0,
        max_total_system_weight_kg=2.0,  # Small drone
        typical_form_factors=["drone", "handheld", "camera", "sensor-box"],
    ),
    typical_mission_hours=(0.5, 4.0),  # Short missions
    typical_battery_wh_per_kg=180.0,  # Drone-grade LiPo
)

INDUSTRIAL_EDGE = CapabilityTier(
    name=TierName.INDUSTRIAL_EDGE,
    display_name="Industrial Edge",
    power_min_w=10.0,
    power_max_w=30.0,
    description=(
        "Industrial-grade edge computing for continuous operation in factory and "
        "warehouse environments. Emphasis on reliability, deterministic latency, "
        "and integration with industrial control systems."
    ),
    typical_applications=[
        "Warehouse AMRs (Autonomous Mobile Robots)",
        "Collaborative robots (cobots)",
        "Automated sorting systems",
        "Quality inspection stations",
        "Inventory management robots",
    ],
    example_platforms=[
        "Jetson Orin NX (8GB/16GB)",
        "Jetson Orin AGX (32GB)",
        "Hailo-8L PCIe",
        "Intel NUC with Arc GPU",
        "Qualcomm RB5",
    ],
    thermal=ThermalConstraints(
        max_ambient_temp_c=50.0,  # Industrial environment
        max_junction_temp_c=95.0,
        typical_cooling="active-fan",
        sustained_power_derating=0.85,
    ),
    form_factor=FormFactorConstraints(
        max_compute_weight_kg=0.5,
        max_compute_volume_cm3=500.0,
        max_total_system_weight_kg=50.0,  # AMR
        typical_form_factors=["amr", "cobot", "rack-mount", "embedded-box"],
    ),
    typical_mission_hours=(4.0, 12.0),  # Shift-based
    typical_battery_wh_per_kg=150.0,  # Industrial Li-ion
)

EMBODIED_AI = CapabilityTier(
    name=TierName.EMBODIED_AI,
    display_name="Embodied AI",
    power_min_w=30.0,
    power_max_w=100.0,
    description=(
        "High-capability embodied AI systems including legged robots and humanoids. "
        "Requires sophisticated perception, world modeling, and real-time control "
        "with significant compute for dynamic movement and environmental interaction."
    ),
    typical_applications=[
        "Quadruped robots (Boston Dynamics Spot-class)",
        "Humanoid robots",
        "Advanced manipulation platforms",
        "Search and rescue robots",
        "Complex world-model simulation",
    ],
    example_platforms=[
        "Jetson Orin AGX (64GB)",
        "Jetson Thor",
        "Multiple Jetson Orin NX in cluster",
        "AMD Ryzen Embedded + Radeon",
        "Intel Core Ultra + Arc",
    ],
    thermal=ThermalConstraints(
        max_ambient_temp_c=40.0,
        max_junction_temp_c=100.0,
        typical_cooling="active-fan",
        sustained_power_derating=0.80,
    ),
    form_factor=FormFactorConstraints(
        max_compute_weight_kg=2.0,
        max_compute_volume_cm3=2000.0,
        max_total_system_weight_kg=100.0,  # Large robot
        typical_form_factors=["quadruped", "humanoid", "manipulation-platform"],
    ),
    typical_mission_hours=(1.0, 8.0),
    typical_battery_wh_per_kg=160.0,
)

AUTOMOTIVE_AI = CapabilityTier(
    name=TierName.AUTOMOTIVE_AI,
    display_name="Automotive AI",
    power_min_w=100.0,
    power_max_w=500.0,
    description=(
        "Automotive-grade AI compute for advanced driver assistance and autonomous "
        "driving (L3/L3+/L4). Requires redundancy, functional safety certification, "
        "and operation across wide temperature ranges."
    ),
    typical_applications=[
        "Level 3 Highway Autopilot",
        "Level 3+ Urban Autonomy",
        "Level 4 Robotaxi",
        "Advanced ADAS with surround perception",
        "Autonomous trucking",
    ],
    example_platforms=[
        "NVIDIA DRIVE Orin",
        "NVIDIA DRIVE Thor",
        "Qualcomm Snapdragon Ride",
        "Mobileye EyeQ6",
        "Tesla FSD Computer",
        "AMD Versal AI Edge",
    ],
    thermal=ThermalConstraints(
        max_ambient_temp_c=85.0,  # Automotive grade
        max_junction_temp_c=125.0,  # Automotive junction temp
        typical_cooling="liquid",
        sustained_power_derating=0.90,  # Better cooling
    ),
    form_factor=FormFactorConstraints(
        max_compute_weight_kg=5.0,
        max_compute_volume_cm3=5000.0,
        max_total_system_weight_kg=2000.0,  # Vehicle
        typical_form_factors=["vehicle-ecu", "trunk-mount", "distributed"],
    ),
    typical_mission_hours=(2.0, 16.0),  # Trip-based, can be long
    typical_battery_wh_per_kg=250.0,  # EV battery (not dedicated compute battery)
)


# =============================================================================
# Tier Registry
# =============================================================================

CAPABILITY_TIERS: Dict[TierName, CapabilityTier] = {
    TierName.WEARABLE_AI: WEARABLE_AI,
    TierName.MICRO_AUTONOMY: MICRO_AUTONOMY,
    TierName.INDUSTRIAL_EDGE: INDUSTRIAL_EDGE,
    TierName.EMBODIED_AI: EMBODIED_AI,
    TierName.AUTOMOTIVE_AI: AUTOMOTIVE_AI,
}

# Ordered list for iteration (by power envelope)
CAPABILITY_TIERS_ORDERED: List[CapabilityTier] = [
    WEARABLE_AI,
    MICRO_AUTONOMY,
    INDUSTRIAL_EDGE,
    EMBODIED_AI,
    AUTOMOTIVE_AI,
]


def get_tier_by_name(name: str) -> Optional[CapabilityTier]:
    """
    Get a capability tier by name.

    Args:
        name: Tier name (e.g., "micro-autonomy" or "MICRO_AUTONOMY")

    Returns:
        CapabilityTier or None if not found
    """
    # Try direct enum lookup
    try:
        tier_name = TierName(name.lower().replace("_", "-"))
        return CAPABILITY_TIERS.get(tier_name)
    except ValueError:
        pass

    # Try enum name lookup
    try:
        tier_name = TierName[name.upper().replace("-", "_")]
        return CAPABILITY_TIERS.get(tier_name)
    except KeyError:
        pass

    return None


def get_tier_for_power(power_w: float) -> Optional[CapabilityTier]:
    """
    Find the capability tier that contains a given power level.

    Args:
        power_w: Power in watts

    Returns:
        CapabilityTier that contains this power level, or None
    """
    for tier in CAPABILITY_TIERS_ORDERED:
        if tier.contains_power(power_w):
            return tier
    return None


def list_tier_names() -> List[str]:
    """Return list of all tier names."""
    return [tier.name.value for tier in CAPABILITY_TIERS_ORDERED]

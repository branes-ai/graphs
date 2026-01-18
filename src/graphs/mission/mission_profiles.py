"""
Mission profile definitions for embodied AI systems.

Mission profiles capture the operational characteristics of different
use cases, including duty cycles, power curves, and duration requirements.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from graphs.mission.capability_tiers import TierName


class MissionPhase(Enum):
    """Phases within a mission."""
    STARTUP = "startup"
    TRANSIT = "transit"
    ACTIVE = "active"
    IDLE = "idle"
    SHUTDOWN = "shutdown"


@dataclass
class DutyCycle:
    """
    Duty cycle specification for a subsystem or mission phase.

    Attributes:
        active_ratio: Fraction of time in active state (0.0-1.0)
        peak_ratio: Fraction of active time at peak power (0.0-1.0)
        description: Description of the duty cycle pattern
    """
    active_ratio: float  # Fraction of time subsystem is active
    peak_ratio: float = 0.2  # Fraction of active time at peak power
    description: str = ""

    def __post_init__(self):
        if not 0.0 <= self.active_ratio <= 1.0:
            raise ValueError(f"active_ratio must be 0.0-1.0, got {self.active_ratio}")
        if not 0.0 <= self.peak_ratio <= 1.0:
            raise ValueError(f"peak_ratio must be 0.0-1.0, got {self.peak_ratio}")

    @property
    def effective_ratio(self) -> float:
        """Effective power ratio considering peak vs typical."""
        # Simplified model: peak power is 1.5x typical
        peak_contribution = self.active_ratio * self.peak_ratio * 1.5
        typical_contribution = self.active_ratio * (1.0 - self.peak_ratio) * 1.0
        return peak_contribution + typical_contribution


@dataclass
class MissionProfile:
    """
    Mission profile for an embodied AI application.

    Captures the operational characteristics including duration, duty cycles,
    and power requirements for different mission phases.

    Attributes:
        name: Profile identifier
        display_name: Human-readable name
        tier: Associated capability tier
        description: Detailed description
        typical_duration_hours: Expected mission duration
        perception_duty: Duty cycle for perception subsystem
        control_duty: Duty cycle for control subsystem
        movement_duty: Duty cycle for movement subsystem
        peak_perception_power_w: Peak perception power requirement
        cruise_perception_power_w: Cruising perception power
        peak_movement_power_w: Peak movement power (sprinting, climbing)
        cruise_movement_power_w: Cruising movement power
        environment: Operating environment description
        constraints: Additional operational constraints
    """
    name: str
    display_name: str
    tier: TierName
    description: str
    typical_duration_hours: float

    # Duty cycles
    perception_duty: DutyCycle
    control_duty: DutyCycle
    movement_duty: DutyCycle

    # Power requirements (optional, for reference)
    peak_perception_power_w: Optional[float] = None
    cruise_perception_power_w: Optional[float] = None
    peak_movement_power_w: Optional[float] = None
    cruise_movement_power_w: Optional[float] = None

    # Environment and constraints
    environment: str = "indoor"
    constraints: List[str] = field(default_factory=list)

    def estimate_average_power_multiplier(self) -> float:
        """
        Estimate average power as multiplier of rated power.

        Returns a value < 1.0 representing the fraction of rated power
        expected during typical operation based on duty cycles.
        """
        # Weighted average based on typical power allocation
        # Assumes perception=35%, control=20%, movement=35%, overhead=10%
        perception_contrib = 0.35 * self.perception_duty.effective_ratio
        control_contrib = 0.20 * self.control_duty.effective_ratio
        movement_contrib = 0.35 * self.movement_duty.effective_ratio
        overhead_contrib = 0.10 * 1.0  # Overhead always on

        return perception_contrib + control_contrib + movement_contrib + overhead_contrib

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "tier": self.tier.value,
            "description": self.description,
            "typical_duration_hours": self.typical_duration_hours,
            "duty_cycles": {
                "perception": {
                    "active_ratio": self.perception_duty.active_ratio,
                    "peak_ratio": self.perception_duty.peak_ratio,
                },
                "control": {
                    "active_ratio": self.control_duty.active_ratio,
                    "peak_ratio": self.control_duty.peak_ratio,
                },
                "movement": {
                    "active_ratio": self.movement_duty.active_ratio,
                    "peak_ratio": self.movement_duty.peak_ratio,
                },
            },
            "environment": self.environment,
            "constraints": self.constraints,
            "average_power_multiplier": self.estimate_average_power_multiplier(),
        }


# =============================================================================
# Mission Profile Definitions
# =============================================================================

# Wearable AI Missions
PROFILE_HEALTH_MONITORING = MissionProfile(
    name="health-monitoring",
    display_name="Health Monitoring",
    tier=TierName.WEARABLE_AI,
    description="Continuous health monitoring with periodic inference",
    typical_duration_hours=16.0,
    perception_duty=DutyCycle(active_ratio=0.1, peak_ratio=0.05, description="Periodic sensor sampling"),
    control_duty=DutyCycle(active_ratio=0.05, peak_ratio=0.1, description="Alert processing"),
    movement_duty=DutyCycle(active_ratio=0.02, peak_ratio=0.5, description="Haptic feedback"),
    environment="body-worn",
    constraints=["skin-safe temperature", "all-day battery", "water resistant"],
)

PROFILE_AR_GLASSES = MissionProfile(
    name="ar-glasses",
    display_name="AR Glasses Assistant",
    tier=TierName.WEARABLE_AI,
    description="Augmented reality with visual recognition and display",
    typical_duration_hours=4.0,
    perception_duty=DutyCycle(active_ratio=0.6, peak_ratio=0.3, description="Continuous camera + eye tracking"),
    control_duty=DutyCycle(active_ratio=0.4, peak_ratio=0.2, description="Scene understanding"),
    movement_duty=DutyCycle(active_ratio=0.8, peak_ratio=0.1, description="Display rendering"),
    environment="indoor/outdoor",
    constraints=["head-mounted weight", "thermal comfort", "display power"],
)

# Micro-Autonomy Missions
PROFILE_DRONE_INSPECTION = MissionProfile(
    name="drone-inspection",
    display_name="Drone Inspection",
    tier=TierName.MICRO_AUTONOMY,
    description="Aerial inspection with high-resolution imaging and defect detection",
    typical_duration_hours=0.5,
    perception_duty=DutyCycle(active_ratio=0.9, peak_ratio=0.4, description="Continuous high-res capture + inference"),
    control_duty=DutyCycle(active_ratio=0.8, peak_ratio=0.3, description="Flight control + path planning"),
    movement_duty=DutyCycle(active_ratio=0.7, peak_ratio=0.3, description="Flight with hover periods"),
    peak_perception_power_w=6.0,
    cruise_perception_power_w=4.0,
    peak_movement_power_w=15.0,
    cruise_movement_power_w=8.0,
    environment="outdoor",
    constraints=["wind resistance", "GPS availability", "return-to-home reserve"],
)

PROFILE_DRONE_DELIVERY = MissionProfile(
    name="drone-delivery",
    display_name="Drone Delivery",
    tier=TierName.MICRO_AUTONOMY,
    description="Point-to-point delivery with obstacle avoidance",
    typical_duration_hours=0.75,
    perception_duty=DutyCycle(active_ratio=0.7, peak_ratio=0.2, description="Navigation + obstacle detection"),
    control_duty=DutyCycle(active_ratio=0.6, peak_ratio=0.2, description="Path planning"),
    movement_duty=DutyCycle(active_ratio=0.85, peak_ratio=0.4, description="Transit flight + landing"),
    peak_movement_power_w=20.0,
    cruise_movement_power_w=12.0,
    environment="outdoor-urban",
    constraints=["payload capacity", "range requirement", "landing precision"],
)

PROFILE_HANDHELD_SCANNER = MissionProfile(
    name="handheld-scanner",
    display_name="Handheld Industrial Scanner",
    tier=TierName.MICRO_AUTONOMY,
    description="Handheld device for barcode/defect scanning with AR overlay",
    typical_duration_hours=8.0,
    perception_duty=DutyCycle(active_ratio=0.3, peak_ratio=0.5, description="On-demand scanning"),
    control_duty=DutyCycle(active_ratio=0.2, peak_ratio=0.3, description="Processing + display"),
    movement_duty=DutyCycle(active_ratio=0.0, peak_ratio=0.0, description="Handheld - no movement"),
    environment="indoor-industrial",
    constraints=["shift-length battery", "drop resistance", "glove operation"],
)

# Industrial Edge Missions
PROFILE_WAREHOUSE_AMR = MissionProfile(
    name="warehouse-amr",
    display_name="Warehouse AMR",
    tier=TierName.INDUSTRIAL_EDGE,
    description="Autonomous mobile robot for warehouse goods transport",
    typical_duration_hours=8.0,
    perception_duty=DutyCycle(active_ratio=0.8, peak_ratio=0.2, description="Continuous SLAM + obstacle detection"),
    control_duty=DutyCycle(active_ratio=0.7, peak_ratio=0.3, description="Path planning + fleet coordination"),
    movement_duty=DutyCycle(active_ratio=0.6, peak_ratio=0.2, description="Transit with loading stops"),
    peak_movement_power_w=40.0,
    cruise_movement_power_w=20.0,
    environment="indoor-warehouse",
    constraints=["24/7 operation", "fleet management", "charging infrastructure"],
)

PROFILE_COBOT_ASSEMBLY = MissionProfile(
    name="cobot-assembly",
    display_name="Collaborative Assembly Robot",
    tier=TierName.INDUSTRIAL_EDGE,
    description="Collaborative robot for assembly tasks alongside humans",
    typical_duration_hours=8.0,
    perception_duty=DutyCycle(active_ratio=0.9, peak_ratio=0.3, description="Human detection + part recognition"),
    control_duty=DutyCycle(active_ratio=0.8, peak_ratio=0.4, description="Motion planning + safety"),
    movement_duty=DutyCycle(active_ratio=0.5, peak_ratio=0.3, description="Arm movements with pauses"),
    environment="factory-floor",
    constraints=["human safety", "precision requirements", "cycle time"],
)

# Embodied AI Missions
PROFILE_QUADRUPED_PATROL = MissionProfile(
    name="quadruped-patrol",
    display_name="Quadruped Security Patrol",
    tier=TierName.EMBODIED_AI,
    description="Legged robot for facility patrol and inspection",
    typical_duration_hours=4.0,
    perception_duty=DutyCycle(active_ratio=0.85, peak_ratio=0.3, description="Surround perception + anomaly detection"),
    control_duty=DutyCycle(active_ratio=0.9, peak_ratio=0.4, description="Locomotion control + navigation"),
    movement_duty=DutyCycle(active_ratio=0.7, peak_ratio=0.3, description="Walking with inspection stops"),
    peak_movement_power_w=80.0,
    cruise_movement_power_w=40.0,
    environment="indoor/outdoor",
    constraints=["terrain traversal", "stair climbing", "weather resistance"],
)

PROFILE_HUMANOID_MANIPULATION = MissionProfile(
    name="humanoid-manipulation",
    display_name="Humanoid Manipulation Tasks",
    tier=TierName.EMBODIED_AI,
    description="Humanoid robot performing manipulation in human environments",
    typical_duration_hours=2.0,
    perception_duty=DutyCycle(active_ratio=0.9, peak_ratio=0.4, description="Multi-modal perception + object recognition"),
    control_duty=DutyCycle(active_ratio=0.95, peak_ratio=0.5, description="Whole-body control + manipulation planning"),
    movement_duty=DutyCycle(active_ratio=0.6, peak_ratio=0.4, description="Walking + manipulation"),
    peak_movement_power_w=150.0,
    cruise_movement_power_w=60.0,
    environment="indoor-human",
    constraints=["human interaction", "dexterous manipulation", "balance control"],
)

# Automotive AI Missions
PROFILE_HIGHWAY_AUTOPILOT = MissionProfile(
    name="highway-autopilot",
    display_name="Highway Autopilot (L3)",
    tier=TierName.AUTOMOTIVE_AI,
    description="Highway driving with driver supervision",
    typical_duration_hours=4.0,
    perception_duty=DutyCycle(active_ratio=1.0, peak_ratio=0.3, description="Continuous 360-degree perception"),
    control_duty=DutyCycle(active_ratio=1.0, peak_ratio=0.2, description="Path planning + vehicle control"),
    movement_duty=DutyCycle(active_ratio=0.1, peak_ratio=0.5, description="Steering/braking assist"),
    environment="highway",
    constraints=["functional safety", "weather conditions", "driver handoff"],
)

PROFILE_URBAN_ROBOTAXI = MissionProfile(
    name="urban-robotaxi",
    display_name="Urban Robotaxi (L4)",
    tier=TierName.AUTOMOTIVE_AI,
    description="Fully autonomous urban driving without driver",
    typical_duration_hours=10.0,
    perception_duty=DutyCycle(active_ratio=1.0, peak_ratio=0.5, description="Maximum perception for complex urban"),
    control_duty=DutyCycle(active_ratio=1.0, peak_ratio=0.4, description="Complex decision making + prediction"),
    movement_duty=DutyCycle(active_ratio=0.15, peak_ratio=0.6, description="Active steering/braking"),
    environment="urban",
    constraints=["no driver backup", "pedestrian safety", "edge cases"],
)


# =============================================================================
# Mission Profile Registry
# =============================================================================

MISSION_PROFILES: Dict[str, MissionProfile] = {
    # Wearable AI
    "health-monitoring": PROFILE_HEALTH_MONITORING,
    "ar-glasses": PROFILE_AR_GLASSES,
    # Micro-Autonomy
    "drone-inspection": PROFILE_DRONE_INSPECTION,
    "drone-delivery": PROFILE_DRONE_DELIVERY,
    "handheld-scanner": PROFILE_HANDHELD_SCANNER,
    # Industrial Edge
    "warehouse-amr": PROFILE_WAREHOUSE_AMR,
    "cobot-assembly": PROFILE_COBOT_ASSEMBLY,
    # Embodied AI
    "quadruped-patrol": PROFILE_QUADRUPED_PATROL,
    "humanoid-manipulation": PROFILE_HUMANOID_MANIPULATION,
    # Automotive AI
    "highway-autopilot": PROFILE_HIGHWAY_AUTOPILOT,
    "urban-robotaxi": PROFILE_URBAN_ROBOTAXI,
}


def get_mission_profile(name: str) -> Optional[MissionProfile]:
    """Get a mission profile by name."""
    return MISSION_PROFILES.get(name.lower().replace(" ", "-").replace("_", "-"))


def list_mission_profiles(tier: Optional[TierName] = None) -> List[str]:
    """
    List available mission profile names.

    Args:
        tier: Optional filter by capability tier

    Returns:
        List of mission profile names
    """
    if tier is None:
        return list(MISSION_PROFILES.keys())

    return [
        name for name, profile in MISSION_PROFILES.items()
        if profile.tier == tier
    ]


def get_profiles_for_tier(tier: TierName) -> List[MissionProfile]:
    """Get all mission profiles for a capability tier."""
    return [
        profile for profile in MISSION_PROFILES.values()
        if profile.tier == tier
    ]

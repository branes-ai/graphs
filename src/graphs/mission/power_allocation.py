"""
Power allocation models for embodied AI systems.

Power in an embodied AI system is allocated across multiple subsystems:
- Perception: Sensors, preprocessing, DNN inference
- Control: Planning, state estimation, decision making
- Movement: Actuators, motors, servos
- Overhead: Communications, logging, thermal management
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any

from graphs.mission.capability_tiers import TierName, CAPABILITY_TIERS


class SubsystemType(Enum):
    """Primary subsystem categories for power allocation."""
    PERCEPTION = "perception"
    CONTROL = "control"
    MOVEMENT = "movement"
    OVERHEAD = "overhead"


@dataclass
class SubsystemPower:
    """Power characteristics for a subsystem."""
    peak_power_w: float
    typical_power_w: float
    idle_power_w: float = 0.0
    duty_cycle: float = 1.0  # Fraction of time active

    @property
    def average_power_w(self) -> float:
        """Calculate average power considering duty cycle."""
        active_power = self.typical_power_w * self.duty_cycle
        idle_power = self.idle_power_w * (1.0 - self.duty_cycle)
        return active_power + idle_power


@dataclass
class PowerAllocation:
    """
    Power allocation across subsystems for an embodied AI system.

    The allocation represents how the total power budget is distributed
    across perception, control, movement, and overhead subsystems.

    Attributes:
        perception_ratio: Fraction of power for perception (0.0-1.0)
        control_ratio: Fraction of power for control (0.0-1.0)
        movement_ratio: Fraction of power for movement (0.0-1.0)
        overhead_ratio: Fraction of power for overhead (0.0-1.0)
        description: Description of this allocation strategy
    """
    perception_ratio: float
    control_ratio: float
    movement_ratio: float
    overhead_ratio: float
    description: str = ""

    def __post_init__(self):
        """Validate that ratios sum to 1.0."""
        total = self.perception_ratio + self.control_ratio + self.movement_ratio + self.overhead_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Power allocation ratios must sum to 1.0, got {total:.3f}")

    def validate(self) -> bool:
        """Check if allocation is valid (sums to 1.0)."""
        total = self.perception_ratio + self.control_ratio + self.movement_ratio + self.overhead_ratio
        return abs(total - 1.0) < 0.01

    def get_ratio(self, subsystem: SubsystemType) -> float:
        """Get ratio for a specific subsystem."""
        mapping = {
            SubsystemType.PERCEPTION: self.perception_ratio,
            SubsystemType.CONTROL: self.control_ratio,
            SubsystemType.MOVEMENT: self.movement_ratio,
            SubsystemType.OVERHEAD: self.overhead_ratio,
        }
        return mapping[subsystem]

    def allocate_power(self, total_power_w: float) -> Dict[SubsystemType, float]:
        """
        Allocate a total power budget across subsystems.

        Args:
            total_power_w: Total power budget in watts

        Returns:
            Dictionary mapping subsystem to allocated power
        """
        return {
            SubsystemType.PERCEPTION: total_power_w * self.perception_ratio,
            SubsystemType.CONTROL: total_power_w * self.control_ratio,
            SubsystemType.MOVEMENT: total_power_w * self.movement_ratio,
            SubsystemType.OVERHEAD: total_power_w * self.overhead_ratio,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "perception": self.perception_ratio,
            "control": self.control_ratio,
            "movement": self.movement_ratio,
            "overhead": self.overhead_ratio,
            "description": self.description,
        }

    def format_table(self, total_power_w: Optional[float] = None) -> str:
        """Format allocation as a table string."""
        lines = []
        lines.append("Subsystem      | Ratio  | " + ("Power (W)" if total_power_w else ""))
        lines.append("-" * (40 if total_power_w else 25))

        allocated = self.allocate_power(total_power_w) if total_power_w else {}

        for subsystem in SubsystemType:
            ratio = self.get_ratio(subsystem)
            name = subsystem.value.capitalize().ljust(14)
            ratio_str = f"{ratio*100:5.1f}%"
            if total_power_w:
                power_str = f" | {allocated[subsystem]:6.1f}W"
                lines.append(f"{name} | {ratio_str}{power_str}")
            else:
                lines.append(f"{name} | {ratio_str}")

        return "\n".join(lines)


# =============================================================================
# Predefined Allocation Strategies
# =============================================================================

ALLOCATION_PERCEPTION_HEAVY = PowerAllocation(
    perception_ratio=0.55,
    control_ratio=0.15,
    movement_ratio=0.20,
    overhead_ratio=0.10,
    description="Perception-heavy: For sensor-rich systems with complex vision pipelines",
)

ALLOCATION_BALANCED = PowerAllocation(
    perception_ratio=0.35,
    control_ratio=0.20,
    movement_ratio=0.35,
    overhead_ratio=0.10,
    description="Balanced: Equal emphasis on perception and movement",
)

ALLOCATION_CONTROL_HEAVY = PowerAllocation(
    perception_ratio=0.25,
    control_ratio=0.35,
    movement_ratio=0.30,
    overhead_ratio=0.10,
    description="Control-heavy: For systems requiring complex planning and decision making",
)

ALLOCATION_MOVEMENT_HEAVY = PowerAllocation(
    perception_ratio=0.25,
    control_ratio=0.15,
    movement_ratio=0.50,
    overhead_ratio=0.10,
    description="Movement-heavy: For highly dynamic platforms (legged robots, drones)",
)

ALLOCATION_STATIONARY = PowerAllocation(
    perception_ratio=0.60,
    control_ratio=0.25,
    movement_ratio=0.05,
    overhead_ratio=0.10,
    description="Stationary: For fixed installations with no movement subsystem",
)


# =============================================================================
# Typical Allocations by Tier
# =============================================================================

TYPICAL_ALLOCATIONS: Dict[TierName, PowerAllocation] = {
    TierName.WEARABLE_AI: PowerAllocation(
        perception_ratio=0.50,
        control_ratio=0.20,
        movement_ratio=0.15,  # Haptics, small actuators
        overhead_ratio=0.15,
        description="Wearable: Perception-focused with minimal actuation",
    ),
    TierName.MICRO_AUTONOMY: PowerAllocation(
        perception_ratio=0.45,
        control_ratio=0.15,
        movement_ratio=0.30,  # Flight motors or locomotion
        overhead_ratio=0.10,
        description="Micro-Autonomy: Balance between perception and propulsion",
    ),
    TierName.INDUSTRIAL_EDGE: PowerAllocation(
        perception_ratio=0.40,
        control_ratio=0.20,
        movement_ratio=0.30,
        overhead_ratio=0.10,
        description="Industrial Edge: Robust perception with reliable movement",
    ),
    TierName.EMBODIED_AI: PowerAllocation(
        perception_ratio=0.30,
        control_ratio=0.25,
        movement_ratio=0.35,  # Complex locomotion
        overhead_ratio=0.10,
        description="Embodied AI: Complex control for dynamic movement",
    ),
    TierName.AUTOMOTIVE_AI: PowerAllocation(
        perception_ratio=0.55,
        control_ratio=0.20,
        movement_ratio=0.10,  # Steering/braking assist only
        overhead_ratio=0.15,  # Redundancy, safety systems
        description="Automotive AI: Perception-dominant with safety overhead",
    ),
}


# =============================================================================
# Allocation by Application Type
# =============================================================================

APPLICATION_ALLOCATIONS: Dict[str, PowerAllocation] = {
    # Aerial
    "drone-inspection": PowerAllocation(
        perception_ratio=0.50,
        control_ratio=0.15,
        movement_ratio=0.25,
        overhead_ratio=0.10,
        description="Drone inspection: High perception for defect detection",
    ),
    "drone-delivery": PowerAllocation(
        perception_ratio=0.30,
        control_ratio=0.15,
        movement_ratio=0.45,
        overhead_ratio=0.10,
        description="Drone delivery: Movement-heavy for range/payload",
    ),

    # Ground - Industrial
    "warehouse-amr": PowerAllocation(
        perception_ratio=0.40,
        control_ratio=0.20,
        movement_ratio=0.30,
        overhead_ratio=0.10,
        description="Warehouse AMR: Balanced for navigation and obstacle avoidance",
    ),
    "cobot-manipulation": PowerAllocation(
        perception_ratio=0.35,
        control_ratio=0.30,
        movement_ratio=0.25,
        overhead_ratio=0.10,
        description="Cobot: Control-heavy for precise manipulation",
    ),

    # Ground - Legged
    "quadruped-locomotion": PowerAllocation(
        perception_ratio=0.25,
        control_ratio=0.25,
        movement_ratio=0.40,
        overhead_ratio=0.10,
        description="Quadruped: Movement-heavy for dynamic locomotion",
    ),
    "humanoid-general": PowerAllocation(
        perception_ratio=0.30,
        control_ratio=0.30,
        movement_ratio=0.30,
        overhead_ratio=0.10,
        description="Humanoid: Balanced for general-purpose operation",
    ),

    # Automotive
    "highway-autopilot": PowerAllocation(
        perception_ratio=0.50,
        control_ratio=0.20,
        movement_ratio=0.10,
        overhead_ratio=0.20,
        description="Highway autopilot: Perception + safety overhead",
    ),
    "urban-robotaxi": PowerAllocation(
        perception_ratio=0.55,
        control_ratio=0.25,
        movement_ratio=0.05,
        overhead_ratio=0.15,
        description="Urban robotaxi: Maximum perception for complex scenes",
    ),

    # Stationary
    "inspection-station": PowerAllocation(
        perception_ratio=0.65,
        control_ratio=0.20,
        movement_ratio=0.05,
        overhead_ratio=0.10,
        description="Inspection station: Perception-dominant, minimal movement",
    ),
}


def get_typical_allocation(tier: TierName) -> PowerAllocation:
    """Get typical power allocation for a capability tier."""
    return TYPICAL_ALLOCATIONS.get(tier, ALLOCATION_BALANCED)


def get_application_allocation(application: str) -> Optional[PowerAllocation]:
    """Get power allocation for a specific application type."""
    return APPLICATION_ALLOCATIONS.get(application.lower().replace(" ", "-"))


def list_allocation_strategies() -> List[str]:
    """List all predefined allocation strategy names."""
    return [
        "perception-heavy",
        "balanced",
        "control-heavy",
        "movement-heavy",
        "stationary",
    ]


def get_allocation_strategy(name: str) -> Optional[PowerAllocation]:
    """Get a predefined allocation strategy by name."""
    strategies = {
        "perception-heavy": ALLOCATION_PERCEPTION_HEAVY,
        "balanced": ALLOCATION_BALANCED,
        "control-heavy": ALLOCATION_CONTROL_HEAVY,
        "movement-heavy": ALLOCATION_MOVEMENT_HEAVY,
        "stationary": ALLOCATION_STATIONARY,
    }
    return strategies.get(name.lower().replace("_", "-"))

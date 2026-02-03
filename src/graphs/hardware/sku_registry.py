"""
SKU Registry for Hardware Product Lines

This module provides a clean separation between:
1. SoC capabilities (defined in model files with ALL thermal profiles)
2. Comparison scenarios (named configurations for fair comparisons)
3. SKU metadata (part numbers, product positioning, typical deployments)

Usage:
    from graphs.hardware.sku_registry import JetsonRegistry, ComparisonScenario

    # Get profiles for a comparison scenario
    profiles = JetsonRegistry.get_comparison_profiles(ComparisonScenario.MATCHED_15W)
    # Returns: {'Nano': '15W', 'NX': '15W', 'AGX': '15W'}

    # Get info about a specific SKU
    sku = JetsonRegistry.get_sku('Jetson-Orin-NX-16GB')
    print(sku.typical_deployment_profile)  # '25W'
    print(sku.marketplace_position)  # 'mid-range'
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable


class ComparisonScenario(Enum):
    """
    Named scenarios for comparing hardware across a product line.

    Each scenario specifies which thermal profile to use for each device,
    enabling fair and meaningful comparisons for different use cases.
    """

    # Fair silicon comparisons (matched power envelopes)
    MATCHED_15W = "matched_15w"      # All devices at 15W - fair silicon comparison
    MATCHED_25W = "matched_25w"      # All devices at 25W (where supported)

    # Marketplace positioning (how products are sold/marketed)
    MARKETPLACE = "marketplace"      # Entry/Mid/Flagship positioning

    # Deployment scenarios (realistic use cases)
    BATTERY_DRONE = "battery_drone"  # Lowest power for battery applications
    EDGE_PASSIVE = "edge_passive"    # Passive cooling (no fan)
    EDGE_ACTIVE = "edge_active"      # Active cooling (with fan)
    INDUSTRIAL = "industrial"        # Industrial deployment with robust cooling
    MAX_PERFORMANCE = "max_perf"     # Maximum performance (benchmarking only)


class MarketPosition(Enum):
    """Product positioning in the marketplace."""
    ENTRY = "entry"           # Entry-level, devkits, hobbyist
    MID_RANGE = "mid-range"   # Production-worthy, edge deployment
    FLAGSHIP = "flagship"     # Top of line, maximum capability
    AUTOMOTIVE = "automotive" # Specialized automotive/industrial


@dataclass
class SKUInfo:
    """
    Complete information about a hardware SKU.

    Separates:
    - Identity (name, part number)
    - Capabilities (supported profiles)
    - Positioning (market segment, typical use)
    - Recommendations (which profile for which scenario)
    """

    # Identity
    name: str                              # e.g., "Jetson-Orin-NX-16GB"
    display_name: str                      # e.g., "NVIDIA Jetson Orin NX 16GB"
    part_number: Optional[str] = None      # e.g., "900-13767-0000-000"

    # Hardware classification
    family: str = ""                       # e.g., "Jetson Orin"
    soc: str = ""                          # e.g., "Orin" (shared across Nano/NX/AGX)
    form_factor: str = ""                  # e.g., "Module", "DevKit", "Industrial"

    # Capability (from model file)
    supported_profiles: List[str] = field(default_factory=list)  # ['10W', '15W', '25W', '40W']

    # Market positioning
    market_position: MarketPosition = MarketPosition.MID_RANGE
    typical_deployment_profile: str = ""   # What most customers actually use

    # Profile recommendations for different scenarios
    scenario_profiles: Dict[ComparisonScenario, str] = field(default_factory=dict)

    # Metadata
    release_year: int = 2023
    notes: str = ""


# =============================================================================
# Jetson Orin Product Line Registry
# =============================================================================

class JetsonRegistry:
    """
    Registry for NVIDIA Jetson product line SKUs and comparison scenarios.

    This centralizes the knowledge about:
    - Which power profiles each device supports
    - How to compare devices fairly
    - Product positioning and typical deployments
    """

    # SKU definitions
    SKUS: Dict[str, SKUInfo] = {
        "Jetson-Orin-Nano-8GB": SKUInfo(
            name="Jetson-Orin-Nano-8GB",
            display_name="NVIDIA Jetson Orin Nano 8GB",
            part_number="945-13730-0000-000",
            family="Jetson Orin",
            soc="Orin",
            form_factor="DevKit",
            supported_profiles=["7W", "15W"],
            market_position=MarketPosition.ENTRY,
            typical_deployment_profile="7W",
            scenario_profiles={
                ComparisonScenario.MATCHED_15W: "15W",
                ComparisonScenario.MATCHED_25W: "15W",  # Max available
                ComparisonScenario.MARKETPLACE: "7W",
                ComparisonScenario.BATTERY_DRONE: "7W",
                ComparisonScenario.EDGE_PASSIVE: "7W",
                ComparisonScenario.EDGE_ACTIVE: "15W",
                ComparisonScenario.INDUSTRIAL: "15W",
                ComparisonScenario.MAX_PERFORMANCE: "15W",
            },
            release_year=2023,
            notes="Entry-level devkit. Same SoC as NX but limited cooling/power delivery. "
                  "Not recommended for production deployments.",
        ),

        "Jetson-Orin-NX-16GB": SKUInfo(
            name="Jetson-Orin-NX-16GB",
            display_name="NVIDIA Jetson Orin NX 16GB",
            part_number="900-13767-0000-000",
            family="Jetson Orin",
            soc="Orin",
            form_factor="Module",
            supported_profiles=["10W", "15W", "25W", "40W"],
            market_position=MarketPosition.MID_RANGE,
            typical_deployment_profile="15W",  # Conservative production default
            scenario_profiles={
                ComparisonScenario.MATCHED_15W: "15W",
                ComparisonScenario.MATCHED_25W: "25W",
                ComparisonScenario.MARKETPLACE: "25W",  # Mid-range positioning
                ComparisonScenario.BATTERY_DRONE: "10W",
                ComparisonScenario.EDGE_PASSIVE: "15W",
                ComparisonScenario.EDGE_ACTIVE: "25W",
                ComparisonScenario.INDUSTRIAL: "25W",
                ComparisonScenario.MAX_PERFORMANCE: "40W",
            },
            release_year=2023,
            notes="Production-worthy module. Same SoC as Nano but with better thermal "
                  "design and power delivery. Recommended for edge robotics.",
        ),

        "Jetson-Orin-AGX-64GB": SKUInfo(
            name="Jetson-Orin-AGX-64GB",
            display_name="NVIDIA Jetson AGX Orin 64GB",
            part_number="900-13701-0040-000",
            family="Jetson Orin",
            soc="Orin",
            form_factor="Module",
            supported_profiles=["15W", "30W", "50W", "MAXN"],
            market_position=MarketPosition.FLAGSHIP,
            typical_deployment_profile="30W",  # Balanced flagship default
            scenario_profiles={
                ComparisonScenario.MATCHED_15W: "15W",
                ComparisonScenario.MATCHED_25W: "30W",  # Closest available
                ComparisonScenario.MARKETPLACE: "50W",  # Flagship positioning
                ComparisonScenario.BATTERY_DRONE: "15W",
                ComparisonScenario.EDGE_PASSIVE: "15W",
                ComparisonScenario.EDGE_ACTIVE: "30W",
                ComparisonScenario.INDUSTRIAL: "50W",
                ComparisonScenario.MAX_PERFORMANCE: "MAXN",
            },
            release_year=2022,
            notes="Flagship edge AI module. 2x the SMs of NX, 2x the bandwidth. "
                  "Recommended for autonomous vehicles and high-performance robotics.",
        ),
    }

    # Aliases for common names
    ALIASES: Dict[str, str] = {
        "nano": "Jetson-Orin-Nano-8GB",
        "orin-nano": "Jetson-Orin-Nano-8GB",
        "jetson-nano": "Jetson-Orin-Nano-8GB",
        "jetson-orin-nano": "Jetson-Orin-Nano-8GB",

        "nx": "Jetson-Orin-NX-16GB",
        "orin-nx": "Jetson-Orin-NX-16GB",
        "jetson-nx": "Jetson-Orin-NX-16GB",
        "jetson-orin-nx": "Jetson-Orin-NX-16GB",

        "agx": "Jetson-Orin-AGX-64GB",
        "orin-agx": "Jetson-Orin-AGX-64GB",
        "jetson-agx": "Jetson-Orin-AGX-64GB",
        "jetson-orin-agx": "Jetson-Orin-AGX-64GB",
    }

    @classmethod
    def resolve_name(cls, name: str) -> str:
        """Resolve aliases to canonical SKU name."""
        normalized = name.lower().strip()
        if normalized in cls.ALIASES:
            return cls.ALIASES[normalized]
        # Try direct match (case-insensitive)
        for sku_name in cls.SKUS:
            if sku_name.lower() == normalized:
                return sku_name
        return name  # Return as-is if not found

    @classmethod
    def get_sku(cls, name: str) -> Optional[SKUInfo]:
        """Get SKU info by name or alias."""
        canonical = cls.resolve_name(name)
        return cls.SKUS.get(canonical)

    @classmethod
    def get_comparison_profiles(
        cls,
        scenario: ComparisonScenario,
        devices: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Get the thermal profile to use for each device in a comparison scenario.

        Args:
            scenario: The comparison scenario
            devices: List of device names (defaults to all Orin devices)

        Returns:
            Dict mapping device name to thermal profile string

        Example:
            >>> JetsonRegistry.get_comparison_profiles(ComparisonScenario.MATCHED_15W)
            {'Jetson-Orin-Nano-8GB': '15W', 'Jetson-Orin-NX-16GB': '15W', 'Jetson-Orin-AGX-64GB': '15W'}
        """
        if devices is None:
            devices = list(cls.SKUS.keys())

        result = {}
        for device in devices:
            sku = cls.get_sku(device)
            if sku and scenario in sku.scenario_profiles:
                result[sku.name] = sku.scenario_profiles[scenario]

        return result

    @classmethod
    def get_scenario_description(cls, scenario: ComparisonScenario) -> str:
        """Get human-readable description of a comparison scenario."""
        descriptions = {
            ComparisonScenario.MATCHED_15W:
                "Fair silicon comparison with all devices at 15W power envelope",
            ComparisonScenario.MATCHED_25W:
                "Fair comparison at 25W (or max available for constrained devices)",
            ComparisonScenario.MARKETPLACE:
                "Marketplace positioning: Entry (Nano) / Mid-range (NX) / Flagship (AGX)",
            ComparisonScenario.BATTERY_DRONE:
                "Battery-powered drone deployment (minimum power modes)",
            ComparisonScenario.EDGE_PASSIVE:
                "Passive cooling deployment (no fan, limited power)",
            ComparisonScenario.EDGE_ACTIVE:
                "Active cooling deployment (fan-cooled, typical production)",
            ComparisonScenario.INDUSTRIAL:
                "Industrial deployment (robust cooling, extended temperature)",
            ComparisonScenario.MAX_PERFORMANCE:
                "Maximum performance (benchmarking only, not typical deployment)",
        }
        return descriptions.get(scenario, "Unknown scenario")

    @classmethod
    def list_skus(cls, family: Optional[str] = None) -> List[SKUInfo]:
        """List all SKUs, optionally filtered by family."""
        skus = list(cls.SKUS.values())
        if family:
            skus = [s for s in skus if s.family == family]
        return skus

    @classmethod
    def print_comparison_matrix(cls):
        """Print a matrix of scenarios vs devices."""
        print("=" * 80)
        print("JETSON ORIN COMPARISON MATRIX")
        print("=" * 80)
        print()

        # Header
        devices = ["Nano", "NX", "AGX"]
        print(f"{'Scenario':<20} ", end="")
        for d in devices:
            print(f"{d:<12}", end="")
        print()
        print("-" * 60)

        # Rows
        for scenario in ComparisonScenario:
            profiles = cls.get_comparison_profiles(scenario)
            print(f"{scenario.value:<20} ", end="")
            for d in devices:
                full_name = cls.resolve_name(d)
                profile = profiles.get(full_name, "N/A")
                print(f"{profile:<12}", end="")
            print()


# =============================================================================
# Convenience functions
# =============================================================================

def compare_at_matched_power(power_watts: int = 15) -> Dict[str, str]:
    """
    Get profiles for a matched-power comparison.

    This enables fair silicon-to-silicon comparisons by running all devices
    at the same power envelope.
    """
    if power_watts == 15:
        return JetsonRegistry.get_comparison_profiles(ComparisonScenario.MATCHED_15W)
    elif power_watts == 25:
        return JetsonRegistry.get_comparison_profiles(ComparisonScenario.MATCHED_25W)
    else:
        # Custom power level - find closest available profile for each device
        result = {}
        for name, sku in JetsonRegistry.SKUS.items():
            # Find closest profile
            available = [int(p.replace('W', '')) for p in sku.supported_profiles]
            closest = min(available, key=lambda x: abs(x - power_watts))
            result[name] = f"{closest}W"
        return result


def get_typical_deployment_profiles() -> Dict[str, str]:
    """
    Get the typical deployment profile for each device.

    This represents what most customers actually use in production.
    """
    return {
        name: sku.typical_deployment_profile
        for name, sku in JetsonRegistry.SKUS.items()
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    JetsonRegistry.print_comparison_matrix()

    print()
    print("=" * 80)
    print("TYPICAL DEPLOYMENT PROFILES")
    print("=" * 80)
    for name, profile in get_typical_deployment_profiles().items():
        sku = JetsonRegistry.get_sku(name)
        print(f"{name}: {profile} ({sku.market_position.value})")

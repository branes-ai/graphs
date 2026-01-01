"""
Hardware Registry Sync Module

Connects calibration results to hardware registry specifications,
enabling roofline analysis with empirical efficiency factors.

Key Features:
1. Maps hardware fingerprints to registry IDs
2. Provides unified view of spec + calibration data
3. Enables roofline analysis with calibrated peaks
4. Falls back to theoretical peaks when no calibration exists

Usage:
    from graphs.hardware.calibration.registry_sync import HardwareRegistry

    registry = HardwareRegistry()

    # Get hardware by ID
    hw = registry.get_hardware("nvidia_h100_sxm5_80gb")

    # Get hardware by fingerprint (from calibration)
    hw = registry.get_hardware_by_fingerprint("c3f840a080356806")

    # Get calibrated roofline parameters
    peak_gflops = hw.get_calibrated_peak("fp32")  # Uses measured if available
    bandwidth = hw.get_calibrated_bandwidth()
    efficiency = hw.get_efficiency("fp32")
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .calibration_db import CalibrationDB, CalibrationRun


# =============================================================================
# HARDWARE ENTRY
# =============================================================================

@dataclass
class HardwareEntry:
    """
    Unified hardware entry combining spec and calibration data.

    Provides calibrated peaks when available, falling back to theoretical.
    """

    # Identity
    id: str                         # Registry ID (e.g., "nvidia_h100_sxm5_80gb")
    vendor: str
    model: str
    device_type: str                # cpu, gpu, tpu, kpu, dpu, dsp, cgra
    product_category: str           # datacenter, edge, mobile, embedded

    # Theoretical specifications
    ops_per_clock: Dict[str, int]   # {precision: ops_per_clock}
    theoretical_peaks: Dict[str, float]  # {precision: GFLOPS}
    peak_bandwidth_gbps: float

    # Hardware details
    architecture: str
    compute_units: int
    memory_gb: float
    tdp_watts: int
    base_clock_mhz: float
    boost_clock_mhz: float
    platform: str                   # x86_64, aarch64, etc.

    # Power profiles
    power_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Calibration data (populated from DB or legacy files)
    calibrations: List['CalibrationData'] = field(default_factory=list)

    # Fingerprint mapping
    fingerprints: List[str] = field(default_factory=list)

    # Notes
    notes: str = ""

    def get_calibrated_peak(
        self,
        precision: str = "fp32",
        power_mode: Optional[str] = None
    ) -> float:
        """
        Get peak performance, preferring calibrated over theoretical.

        Args:
            precision: Precision to get peak for.
            power_mode: Optional power mode filter.

        Returns:
            Peak GFLOPS (calibrated if available, otherwise theoretical).
        """
        # Try to find calibration for this precision/power mode
        for cal in self.calibrations:
            if cal.precision == precision:
                if power_mode is None or cal.power_mode == power_mode:
                    if cal.peak_measured_gops > 0:
                        return cal.peak_measured_gops

        # Fall back to theoretical
        return self.theoretical_peaks.get(precision, 0.0)

    def get_calibrated_bandwidth(self, power_mode: Optional[str] = None) -> float:
        """
        Get memory bandwidth, preferring calibrated over theoretical.

        Returns:
            Bandwidth in GB/s (calibrated if available, otherwise theoretical).
        """
        for cal in self.calibrations:
            if power_mode is None or cal.power_mode == power_mode:
                if cal.stream_best_gbps > 0:
                    return cal.stream_best_gbps

        return self.peak_bandwidth_gbps

    def get_efficiency(
        self,
        precision: str = "fp32",
        power_mode: Optional[str] = None
    ) -> float:
        """
        Get measured efficiency (0-1) if calibrated.

        Returns:
            Efficiency ratio, or 1.0 if not calibrated.
        """
        for cal in self.calibrations:
            if cal.precision == precision:
                if power_mode is None or cal.power_mode == power_mode:
                    if cal.efficiency > 0:
                        return min(cal.efficiency, 1.0)  # Cap at 100%

        return 1.0  # No calibration = assume theoretical

    def get_latest_calibration(
        self,
        precision: str = "fp32"
    ) -> Optional['CalibrationData']:
        """Get the most recent calibration for a precision."""
        matching = [c for c in self.calibrations if c.precision == precision]
        if matching:
            return max(matching, key=lambda c: c.timestamp)
        return None

    def get_roofline_params(
        self,
        precision: str = "fp32",
        use_calibrated: bool = True
    ) -> Tuple[float, float, float]:
        """
        Get roofline model parameters.

        Args:
            precision: Precision for compute peak.
            use_calibrated: Whether to prefer calibrated values.

        Returns:
            Tuple of (peak_gflops, bandwidth_gbps, ridge_point).
        """
        if use_calibrated:
            peak = self.get_calibrated_peak(precision)
            bandwidth = self.get_calibrated_bandwidth()
        else:
            peak = self.theoretical_peaks.get(precision, 0.0)
            bandwidth = self.peak_bandwidth_gbps

        ridge_point = peak / bandwidth if bandwidth > 0 else 0.0

        return peak, bandwidth, ridge_point

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "vendor": self.vendor,
            "model": self.model,
            "device_type": self.device_type,
            "product_category": self.product_category,
            "ops_per_clock": self.ops_per_clock,
            "theoretical_peaks": self.theoretical_peaks,
            "peak_bandwidth_gbps": self.peak_bandwidth_gbps,
            "architecture": self.architecture,
            "compute_units": self.compute_units,
            "memory_gb": self.memory_gb,
            "tdp_watts": self.tdp_watts,
            "base_clock_mhz": self.base_clock_mhz,
            "boost_clock_mhz": self.boost_clock_mhz,
            "platform": self.platform,
            "power_profiles": self.power_profiles,
            "fingerprints": self.fingerprints,
            "calibration_count": len(self.calibrations),
            "notes": self.notes,
        }


@dataclass
class CalibrationData:
    """Simplified calibration data for HardwareEntry."""
    timestamp: datetime
    precision: str
    power_mode: str
    peak_measured_gops: float
    stream_best_gbps: float
    efficiency: float
    software_fingerprint: str
    framework: str
    notes: str = ""

    @classmethod
    def from_calibration_run(cls, run: CalibrationRun) -> 'CalibrationData':
        """Create from CalibrationRun."""
        return cls(
            timestamp=datetime.fromisoformat(run.timestamp),
            precision=run.precision,
            power_mode=run.power_mode,
            peak_measured_gops=run.peak_measured_gops,
            stream_best_gbps=run.stream_best_gbps,
            efficiency=run.efficiency,
            software_fingerprint=run.software_fingerprint,
            framework=run.blas_library,
            notes=run.notes,
        )

    @classmethod
    def from_legacy_json(cls, data: Dict[str, Any]) -> 'CalibrationData':
        """Create from legacy calibration JSON."""
        metadata = data.get("metadata", {})
        return cls(
            timestamp=datetime.fromisoformat(
                metadata.get("calibration_date", "2020-01-01T00:00:00")
            ),
            precision="fp32",  # Legacy files are typically FP32
            power_mode=metadata.get("cpu_clock", {}).get("governor", "unknown"),
            peak_measured_gops=data.get("best_measured_gflops", 0.0),
            stream_best_gbps=data.get("measured_bandwidth_gbps", 0.0),
            efficiency=data.get("best_efficiency", 0.0),
            software_fingerprint="legacy",
            framework=metadata.get("framework", "unknown"),
            notes=f"Legacy calibration from {metadata.get('calibration_date', 'unknown')}",
        )


# =============================================================================
# HARDWARE REGISTRY
# =============================================================================

class HardwareRegistry:
    """
    Unified hardware registry with calibration data.

    Combines:
    - Hardware specs from hardware_registry/
    - Legacy calibrations from hardware_registry/*/calibrations/
    - New versioned calibrations from calibration DB
    """

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        db_path: Optional[Path] = None
    ):
        """
        Initialize hardware registry.

        Args:
            registry_path: Path to hardware_registry directory.
            db_path: Path to calibration database.
        """
        # Find registry path
        if registry_path is None:
            # Try common locations
            candidates = [
                Path(__file__).parent.parent.parent.parent.parent / "hardware_registry",
                Path.cwd() / "hardware_registry",
            ]
            for p in candidates:
                if p.exists():
                    registry_path = p
                    break

        self.registry_path = registry_path
        self.db_path = db_path

        # Caches
        self._hardware_cache: Dict[str, HardwareEntry] = {}
        self._fingerprint_map: Dict[str, str] = {}  # fingerprint -> hardware_id

        # Load registry
        if self.registry_path and self.registry_path.exists():
            self._load_registry()

    def _load_registry(self):
        """Load all hardware specs from registry."""
        for category in ['cpu', 'gpu', 'accelerator', 'dsp']:
            category_path = self.registry_path / category
            if not category_path.exists():
                continue

            for hw_dir in category_path.iterdir():
                if not hw_dir.is_dir():
                    continue

                spec_path = hw_dir / "spec.json"
                if not spec_path.exists():
                    continue

                try:
                    entry = self._load_hardware_entry(spec_path, hw_dir)
                    if entry:
                        self._hardware_cache[entry.id] = entry
                except Exception as e:
                    print(f"Warning: Failed to load {spec_path}: {e}")

    def _load_hardware_entry(
        self,
        spec_path: Path,
        hw_dir: Path
    ) -> Optional[HardwareEntry]:
        """Load a single hardware entry."""
        with open(spec_path) as f:
            spec = json.load(f)

        # Create entry
        entry = HardwareEntry(
            id=spec.get("id", hw_dir.name),
            vendor=spec.get("vendor", "Unknown"),
            model=spec.get("model", hw_dir.name),
            device_type=spec.get("device_type", "unknown"),
            product_category=spec.get("product_category", "unknown"),
            ops_per_clock=spec.get("ops_per_clock", {}),
            theoretical_peaks=spec.get("theoretical_peaks", {}),
            peak_bandwidth_gbps=spec.get("peak_bandwidth_gbps", 0.0),
            architecture=spec.get("architecture", ""),
            compute_units=spec.get("compute_units", 0),
            memory_gb=spec.get("memory_gb", 0),
            tdp_watts=spec.get("tdp_watts", 0),
            base_clock_mhz=spec.get("base_clock_mhz", 0.0),
            boost_clock_mhz=spec.get("boost_clock_mhz", 0.0),
            platform=spec.get("platform", ""),
            power_profiles=spec.get("power_profiles", {}),
            notes=spec.get("notes", ""),
        )

        # Load legacy calibrations
        calibrations_dir = hw_dir / "calibrations"
        if calibrations_dir.exists():
            for cal_file in calibrations_dir.glob("*.json"):
                try:
                    with open(cal_file) as f:
                        cal_data = json.load(f)
                    cal = CalibrationData.from_legacy_json(cal_data)
                    entry.calibrations.append(cal)
                except Exception:
                    pass

        return entry

    def get_hardware(self, hardware_id: str) -> Optional[HardwareEntry]:
        """
        Get hardware entry by ID.

        Args:
            hardware_id: Registry ID (e.g., "nvidia_h100_sxm5_80gb").

        Returns:
            HardwareEntry or None if not found.
        """
        entry = self._hardware_cache.get(hardware_id)

        # If we have a DB, enrich with versioned calibrations
        if entry and self.db_path and Path(self.db_path).exists():
            self._enrich_with_db_calibrations(entry)

        return entry

    def get_hardware_by_fingerprint(
        self,
        fingerprint: str
    ) -> Optional[HardwareEntry]:
        """
        Get hardware entry by calibration fingerprint.

        Args:
            fingerprint: Hardware fingerprint from calibration.

        Returns:
            HardwareEntry or None if not found.
        """
        # Check cache
        if fingerprint in self._fingerprint_map:
            return self.get_hardware(self._fingerprint_map[fingerprint])

        # Search calibration DB for matching fingerprint
        if self.db_path and Path(self.db_path).exists():
            db = CalibrationDB(str(self.db_path))
            try:
                trajectory = db.get_trajectory(fingerprint)
                if trajectory:
                    # Try to match to registry by CPU/GPU model
                    run = trajectory[-1]
                    matched = self._match_to_registry(run)
                    if matched:
                        self._fingerprint_map[fingerprint] = matched.id
                        return matched
            finally:
                db.close()

        return None

    def _match_to_registry(self, run: CalibrationRun) -> Optional[HardwareEntry]:
        """Match a calibration run to a registry entry."""
        # Try to match by model name
        model_lower = run.cpu_model.lower()

        for hw_id, entry in self._hardware_cache.items():
            entry_model_lower = entry.model.lower()

            # Check for substring match
            if model_lower in entry_model_lower or entry_model_lower in model_lower:
                # Enrich with this calibration
                cal = CalibrationData.from_calibration_run(run)
                entry.calibrations.append(cal)
                entry.fingerprints.append(run.hardware_fingerprint)
                return entry

        return None

    def _enrich_with_db_calibrations(self, entry: HardwareEntry):
        """Add calibrations from DB to hardware entry."""
        if not self.db_path:
            return

        db = CalibrationDB(str(self.db_path))
        try:
            for fingerprint in entry.fingerprints:
                trajectory = db.get_trajectory(fingerprint)
                for run in trajectory:
                    # Check if we already have this calibration
                    exists = any(
                        c.timestamp == datetime.fromisoformat(run.timestamp)
                        for c in entry.calibrations
                    )
                    if not exists:
                        entry.calibrations.append(
                            CalibrationData.from_calibration_run(run)
                        )
        finally:
            db.close()

    def list_hardware(
        self,
        device_type: Optional[str] = None,
        category: Optional[str] = None,
        vendor: Optional[str] = None
    ) -> List[HardwareEntry]:
        """
        List hardware entries with optional filters.

        Args:
            device_type: Filter by device type (cpu, gpu, etc.)
            category: Filter by category (datacenter, edge, etc.)
            vendor: Filter by vendor name.

        Returns:
            List of matching HardwareEntry objects.
        """
        results = []

        for entry in self._hardware_cache.values():
            if device_type and entry.device_type != device_type:
                continue
            if category and entry.product_category != category:
                continue
            if vendor and vendor.lower() not in entry.vendor.lower():
                continue
            results.append(entry)

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary statistics."""
        summary = {
            "total_entries": len(self._hardware_cache),
            "by_device_type": {},
            "by_category": {},
            "by_vendor": {},
            "calibrated_count": 0,
        }

        for entry in self._hardware_cache.values():
            # Count by device type
            dt = entry.device_type
            summary["by_device_type"][dt] = summary["by_device_type"].get(dt, 0) + 1

            # Count by category
            cat = entry.product_category
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1

            # Count by vendor
            vendor = entry.vendor
            summary["by_vendor"][vendor] = summary["by_vendor"].get(vendor, 0) + 1

            # Count calibrated
            if entry.calibrations:
                summary["calibrated_count"] += 1

        return summary

    def find_similar(
        self,
        target_peak_gflops: float,
        target_bandwidth_gbps: float,
        device_type: Optional[str] = None,
        n: int = 5
    ) -> List[Tuple[HardwareEntry, float]]:
        """
        Find hardware similar to target specs.

        Args:
            target_peak_gflops: Target compute peak.
            target_bandwidth_gbps: Target memory bandwidth.
            device_type: Optional device type filter.
            n: Number of results.

        Returns:
            List of (HardwareEntry, similarity_score) tuples.
        """
        import math

        results = []

        for entry in self._hardware_cache.values():
            if device_type and entry.device_type != device_type:
                continue

            # Get calibrated or theoretical peak
            peak = entry.get_calibrated_peak("fp32")
            bandwidth = entry.get_calibrated_bandwidth()

            if peak <= 0 or bandwidth <= 0:
                continue

            # Calculate normalized distance
            peak_ratio = peak / target_peak_gflops if target_peak_gflops > 0 else 0
            bw_ratio = bandwidth / target_bandwidth_gbps if target_bandwidth_gbps > 0 else 0

            # Similarity = 1 / (1 + distance)
            distance = math.sqrt((1 - peak_ratio)**2 + (1 - bw_ratio)**2)
            similarity = 1 / (1 + distance)

            results.append((entry, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: -x[1])
        return results[:n]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_registry(
    registry_path: Optional[Path] = None,
    db_path: Optional[Path] = None
) -> HardwareRegistry:
    """
    Get hardware registry instance.

    Args:
        registry_path: Path to hardware_registry directory.
        db_path: Path to calibration database.

    Returns:
        HardwareRegistry instance.
    """
    return HardwareRegistry(registry_path, db_path)


def get_calibrated_hardware(hardware_id: str) -> Optional[HardwareEntry]:
    """
    Get hardware entry with calibration data.

    Args:
        hardware_id: Registry ID.

    Returns:
        HardwareEntry or None.
    """
    registry = get_registry()
    return registry.get_hardware(hardware_id)


def get_roofline_params(
    hardware_id: str,
    precision: str = "fp32",
    use_calibrated: bool = True
) -> Tuple[float, float, float]:
    """
    Get roofline parameters for hardware.

    Args:
        hardware_id: Registry ID.
        precision: Precision for compute peak.
        use_calibrated: Whether to prefer calibrated values.

    Returns:
        Tuple of (peak_gflops, bandwidth_gbps, ridge_point).
    """
    registry = get_registry()
    entry = registry.get_hardware(hardware_id)

    if entry:
        return entry.get_roofline_params(precision, use_calibrated)

    return 0.0, 0.0, 0.0
